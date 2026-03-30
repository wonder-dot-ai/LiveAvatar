# Plan: Vectorize RoPE computation (step by step)

## Context
The RoPE computation in `forward()` (lines 882-914) has a Python loop over batch items with `.item()`, `np.linspace`, and `random.randint` — all CUDA graph incompatible. The previous attempt to vectorize this had a bug. This plan breaks the work into testable steps.

## Current code structure
```python
for bi in range(b):
    # 1. Get frame offset from current_start
    cs_bi = int(current_start[bi].item())
    frame_offset_bi = cs_bi // frame_seqlen_int

    # 2. Shift grid_sizes by frame_offset
    gs_bi_shifted = rollout_grid_sizes(gs_bi, frame_offset_bi)

    # 3. Call rope_precompute (uses numpy, .item())
    f_bi = rope_precompute(x_bi, gs_bi_shifted, self.freqs, start=None)

    # 4. Rolling RoPE for cond (same structure, different offset)
    relative_dist = random.randint(4, 30)
    num_frames_cond = max(0, frame_offset_bi - (30 - relative_dist))
    cf_bi = rope_precompute(cond_shape, cond_gs_shifted, self.freqs, start=None)
```

## What rope_precompute actually computes (for our inference case)

For a single batch item with grid `[F, H, W]` and frame_offset `fo`:
```
f_indices = [fo, fo+1, ..., fo+F-1]        # temporal positions
h_indices = [0, 1, ..., H-1]                # spatial height
w_indices = [0, 1, ..., W-1]                # spatial width

freqs_output[f, h, w] = cat(freqs_t[f_indices[f]], freqs_h[h_indices[h]], freqs_w[w_indices[w]])
```

The output shape is `[1, F*H*W, 1, freq_dim]` (complex).

Key observations:
- `h_indices` and `w_indices` are ALWAYS `arange(H)` and `arange(W)` — identical for all batch items
- Only `f_indices` varies per batch item (depends on `frame_offset`)
- For cond RoPE, same structure but different `f_indices` (from Rolling RoPE offset)

## Implementation steps

### Step 1: Add `_compute_rope_inference` method (noisy tokens only)
Vectorize ONLY the noisy-token RoPE. Keep cond RoPE in the old loop for now.

Test: run inference, compare output with the loop-based version numerically.

### Step 2: Vectorize cond RoPE for the first cond group (ref tokens)
The first cond group is the ref/motion tokens with Rolling RoPE offset. Handle it with the same batched indexing approach.

Note: the first cond group from prefill has `F=0` (empty). Skip it.

Test: run inference, verify quality unchanged.

### Step 3: Handle additional cond groups (FramePack motion tokens)
The `rope_cache["grid_sizes"]` has multiple groups after `inject_motion` adds motion tokens. Each group has fixed temporal indices (not per-batch). Process in a loop over groups (not batch items — groups are few, typically 3-4).

Test: run inference, verify quality unchanged.

### Step 4: Replace `random.randint` with `torch.randint`
Change `random.randint(4, 30)` to `torch.randint(4, 31, (b,), device=device)`.
This is a one-line change that makes Rolling RoPE CUDA graph compatible.

Test: run inference, verify quality unchanged.

## File to modify
- `liveavatar_nari/models/causal_model_s2v.py` — add `_compute_rope_inference`, update `forward()`

## Verification per step
After each step:
1. `python -c "import liveavatar_nari.models.causal_model_s2v; print('OK')"` — syntax check
2. Run inference with `--num_blocks 10 --seed 4200` — verify video is not degraded
