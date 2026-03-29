# Plan: Implement TPP diagonal scheduling on single GPU

## Context

The current pipeline denoises blocks sequentially: complete all 4 steps for block 0, then all 4 for block 1, etc. This means each KV cache step-slot sees representations at varying noise levels.

TPP (Timestep-forcing Pipeline Parallelism) changes the execution order so that each KV cache step-slot only ever processes blocks at the **same noise level**. On single GPU, we simulate the multi-GPU pipeline with diagonal scheduling:

```
Group  Step0  Step1  Step2  Step3
  0      0     -1     -2     -3     ← warmup (dummy blocks < 0 are skipped)
  1      1      0     -1     -2
  2      2      1      0     -1
  3      3      2      1      0     ← block 0 fully denoised
  4      4      3      2      1     ← steady state: one block completes per group
  5      5      4      3      2
  ...
```

At group `g`, step `s`: block_idx = `g - s`. Skip if `block_idx < 0` (warmup) or `>= N` (drain).

Total groups = `N + S - 1` (N = total_blocks, S = sampling_steps).
Total forward passes = `N * S` (same as sequential — dummies are skipped).

## File to modify
- `liveavatar_nari/pipeline.py` — rewrite the generation loop in `generate()`

## Current loop structure
```python
for block_idx in range(total_blocks):
    block_latents = randn(...)
    for step_idx, t in enumerate(timesteps):
        step_kv = kv_cache[step_idx]
        noise_pred = model(block_latents, t=t, kv_cache=step_kv, current_start=token_start, ...)
        block_latents = scheduler.step(noise_pred, t, block_latents)
    all_block_latents.append(block_latents)
```

## New loop structure

### Data structures for in-flight blocks
```python
S = sampling_steps
# Each step slot holds the latent of the block currently at that denoising stage
inflight_latents = [None] * S       # block latent at each pipeline stage
inflight_audio = [None] * S         # audio for block at each stage
inflight_token_start = [0] * S      # token offset for block at each stage
inflight_block_idx = [-1] * S       # which block is in each stage
```

### Main loop (double loop)
```python
total_groups = total_blocks + S - 1

for group in tqdm(range(total_groups), desc="blocks"):
    for step_idx, t in enumerate(timesteps):
        block_idx = group - step_idx

        # Skip dummy blocks (warmup/drain)
        if block_idx < 0 or block_idx >= total_blocks:
            continue

        if step_idx == 0:
            # New block enters the pipeline — generate noise
            inflight_latents[0] = randn(..., generator=seed_g)
            inflight_audio[0] = audio_embeddings[..., block_idx*afpb:(block_idx+1)*afpb]
            inflight_token_start[0] = block_idx * tokens_per_block
            inflight_block_idx[0] = block_idx

        block_latents = inflight_latents[step_idx]

        # --- KV cache slice for this step ---
        step_kv = [{"k": l["k"][step_idx:step_idx+1], ...} for l in self.kv_cache]
        step_crossattn = [{"k": l["k"][step_idx:step_idx+1], ...} for l in self.crossattn_cache]

        # --- Forward pass ---
        noise_pred = model(
            [block_latents], t=t.unsqueeze(0).expand(1, fpb),
            context=..., cond_states=..., audio_input=inflight_audio[step_idx],
            motion_frames=..., kv_cache=step_kv, crossattn_cache=step_crossattn,
            current_start=inflight_token_start[step_idx],
            current_end=inflight_token_start[step_idx] + tokens_per_block,
        )

        # --- Scheduler step ---
        scheduler.sigmas = saved_sigmas
        scheduler._step_index = step_idx
        scheduler._begin_index = 0
        block_latents = scheduler.step(noise_pred, t, block_latents, generator=seed_g)[0].squeeze(0)

        if step_idx == S - 1:
            # Block completed — output it
            all_block_latents.append(block_latents.detach().cpu())
            inflight_latents[step_idx] = None
            # AAS: first completed block replaces sink
            if inflight_block_idx[step_idx] == 0:
                ref_image_latents = block_latents.unsqueeze(0)[:, :, 0:1]
        else:
            # Advance to next pipeline stage
            inflight_latents[step_idx + 1] = block_latents
            inflight_audio[step_idx + 1] = inflight_audio[step_idx]
            inflight_token_start[step_idx + 1] = inflight_token_start[step_idx]
            inflight_block_idx[step_idx + 1] = inflight_block_idx[step_idx]
            inflight_latents[step_idx] = None
```

### Scheduler handling
`FlowMatchEulerDiscreteScheduler.step()` uses `self._step_index` to look up sigma values. We set it explicitly before each call since steps are no longer sequential:
```python
scheduler.sigmas = saved_sigmas    # restore original sigma schedule
scheduler._step_index = step_idx   # point to current step's sigma
scheduler._begin_index = 0
```

### Noise generation
Noise is generated when a block enters step 0 (`if step_idx == 0`). The `seed_g` generator advances deterministically — blocks enter in order 0, 1, 2, ..., so noise is generated in block order.

### AAS (Adaptive Attention Sink)
Block 0 completes at group `S - 1`, step `S - 1`. At that point, `inflight_block_idx[S-1] == 0`, so we apply AAS. This is later than the current code (which applies after block 0's first pass), but matches the TPP paper's design.

### What stays the same
- `prefill_cond_cache` — unchanged
- KV cache structure — already has per-step slots `[num_steps, kv_cache_size, ...]`
- Streaming VAE decode — unchanged (runs after all blocks generated)
- Model forward call — identical arguments
- Audio encoding, image encoding — unchanged

### What changes
- Block loop: `for block_idx / for step_idx` → `for group / for step_idx` with `block_idx = group - step_idx`
- In-flight state: need to track S blocks simultaneously (one per pipeline stage)
- Scheduler: explicit `_step_index` setting instead of sequential reset
- AAS timing: block 0 completes later (after S-1 groups instead of immediately)

## Verification
1. `python -c "import liveavatar_nari.pipeline"` — syntax check
2. Run inference — verify video quality (should match or improve over sequential, especially for longer videos where KV cache noise-level consistency matters)
3. Verify block output order is correct (block 0 should be first in `all_block_latents`, block 1 second, etc.)
4. Total forward passes = `total_blocks * S` (same as sequential)
