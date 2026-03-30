# Plan: Batched TPP inference (fixed batch_size = S)

## Context

The current TPP scheduling runs S sequential forward passes per group (one per pipeline stage). Since each pass uses a different KV cache slot, they can be batched into a single forward pass with batch_size=S. This improves GPU utilization since the model's linear layers, FFN, and cross-attention scale well with batch size.

## Key insight: batch dim = step dim

The KV cache is already `[num_steps, kv_cache_size, nh, hd]`. When we batch all S steps into one forward pass, the batch dimension naturally maps to the step dimension. No gather/scatter needed — the model reads/writes directly to the correct rows.

## KV cache correctness with batching

### Problem: non-uniform active sizes

Each step processes a different block, so `active_size` differs per batch item:
- Batch item 0 (step 0): newest block, largest active_size
- Batch item S-1 (step S-1): oldest block, smallest active_size

`attention()` (cuDNN SDPA) ignores `k_lens` and attends to ALL of `k_cat[:, :max_active_size]`. This means batch items with smaller active_size attend to extra entries (zeros or stale data), corrupting their output.

### Solution: use `flash_attention` with `k_lens`

`flash_attention` uses the varlen API which respects per-item lengths via `cu_seqlens_k`. Each batch item only attends to its valid entries:
```python
k_lens = torch.tensor([active_size_0, ..., active_size_{S-1}], dtype=int32)
```

This requires changing the self-attention to call `flash_attention` instead of `attention`.

## Fixed batch size with dummy padding

During warmup (groups 0 to S-2) and drain (groups N to N+S-2), some steps don't have real blocks. We pad with dummies:

- Dummy latent: `torch.zeros([C, F, H, W])` (zero noise)
- Dummy audio: `torch.zeros([1, layers, dim, T])` (silence)
- Dummy current_start: `0` (writes to position 0, overwritten by real blocks later)

Dummies are processed by the model (wasting some compute) but their output is discarded. Their KV cache writes are overwritten when real blocks arrive at that step.

**Benefit**: Fixed batch size avoids `torch.compile` recompilation and simplifies the loop.

## Implementation

### File to modify
- `liveavatar_nari/pipeline.py` — rewrite the generation loop
- `liveavatar_nari/models/causal_model_s2v.py` — change self-attention from `attention()` to `flash_attention()` (in the unified forward path)

### Step 1: Gather (before inner loop)

For each group, prepare all S batch items:
```python
S = sampling_steps
# Prepare batch inputs
batch_x = []          # list of S latent tensors [C, F, H, W]
batch_audio = []      # list of S audio tensors [1, layers, dim, T]
batch_cs = []         # list of S current_start values

for step_idx in range(S):
    block_idx = group - step_idx
    if 0 <= block_idx < total_blocks:
        # Real block
        if step_idx == 0:
            # New block enters — generate noise
            inflight_latents[0] = torch.randn(...)
            inflight_audio[0] = audio_embeddings[..., block_idx*afpb:(block_idx+1)*afpb]
            inflight_token_start[0] = block_idx * tokens_per_block
            inflight_block_idx[0] = block_idx
        batch_x.append(inflight_latents[step_idx])
        batch_audio.append(inflight_audio[step_idx])
        batch_cs.append(inflight_token_start[step_idx])
    else:
        # Dummy block
        batch_x.append(dummy_latent)
        batch_audio.append(dummy_audio)
        batch_cs.append(0)

# Construct batched tensors
t_batch = timesteps.unsqueeze(1).expand(S, fpb)                    # [S, F]
audio_batch = torch.cat(batch_audio, dim=0)                         # [S, layers, dim, T]
current_start_batch = torch.tensor(batch_cs, device=self.device)    # [S]
context_batch = text_embeddings[0:1] * S                            # list of S identical embeddings
cond_batch = dummy_cond.expand(S, -1, -1, -1, -1)                  # [S, C, F, H, W]
```

### Step 2: Batched forward pass

```python
# KV cache: pass full [S, ...] tensors (no slicing)
batch_kv = [
    {"k": l["k"], "v": l["v"], "cond_k": l["cond_k"], "cond_v": l["cond_v"], "cond_end": l["cond_end"]}
    for l in self.kv_cache
]
batch_crossattn = [
    {"k": l["k"], "v": l["v"], "is_init": l["is_init"]}
    for l in self.crossattn_cache
]

noise_pred_list = self.noise_model(
    batch_x,
    t=t_batch,
    context=context_batch,
    cond_states=cond_batch,
    audio_input=audio_batch,
    motion_frames=[self.motion_frames, latent_motion_frames],
    kv_cache=batch_kv,
    crossattn_cache=batch_crossattn,
    current_start=current_start_batch,
    current_end=0,  # unused in inference
)
```

### Step 3: Scatter results and advance

```python
# Batched euler step (no scheduler needed — just x + dt * pred)
sigmas = saved_sigmas
for step_idx in range(S):
    block_idx = group - step_idx
    if block_idx < 0 or block_idx >= total_blocks:
        continue  # dummy — discard

    sigma = sigmas[step_idx]
    sigma_next = sigmas[step_idx + 1]
    dt = sigma_next - sigma
    result = inflight_latents[step_idx] + dt * noise_pred_list[step_idx].unsqueeze(0)
    # Note: noise_pred_list[step_idx] shape depends on model output format

    if step_idx == S - 1:
        # Block completed — output
        all_block_latents.append(result.detach().cpu())
        inflight_latents[step_idx] = None
        if inflight_block_idx[step_idx] == 0:
            ref_image_latents = result.unsqueeze(0)[:, :, 0:1]
    else:
        # Advance to next stage
        inflight_latents[step_idx + 1] = result
        inflight_audio[step_idx + 1] = inflight_audio[step_idx]
        inflight_token_start[step_idx + 1] = inflight_token_start[step_idx]
        inflight_block_idx[step_idx + 1] = inflight_block_idx[step_idx]
        inflight_latents[step_idx] = None
```

### Model changes

In `CausalWanS2VSelfAttention.forward()`, replace `attention()` with `flash_attention()`:
```python
x = flash_attention(
    q=roped_query,
    k=k_cat,
    v=v_cat,
    k_lens=k_lens,   # per-batch-item valid lengths — critical for correctness
    window_size=self.window_size,
)
```

This is the ONLY model change. Everything else (QKV projections, output projection, cross-attention, FFN, modulation) already handles variable batch sizes.

### What about the RoPE?

The RoPE loop in `forward()` already handles batched `current_start`:
```python
for bi in range(b):
    cs_bi = int(current_start[bi].item())
    ...
```

With b=S=4, it loops 4 times computing per-item RoPE. This is correct.

## Dummy block details

```python
dummy_latent = torch.zeros(self.latent_channels, fpb, latent_h, latent_w,
                           dtype=self.param_dtype, device=self.device)
dummy_audio = torch.zeros(1, audio_layers, audio_dim, audio_frames_per_block,
                          dtype=self.param_dtype, device=self.device)
```

Dummies at `current_start=0` write to KV cache position 0. When the real block 0 eventually reaches that step, it also writes to position 0, overwriting the dummy data.

## Scheduler elimination

With batched TPP, we don't need `FlowMatchEulerDiscreteScheduler` at all. The euler step is:
```
x_{t-1} = x_t + (sigma_{t-1} - sigma_t) * model_output
```

We compute `dt = sigmas[step_idx + 1] - sigmas[step_idx]` per step and apply directly. No scheduler state management needed.

## Performance estimate

- Forward passes per group: 1 (was S=4)
- Batch size: S=4 (was 1)
- Effective throughput: ~2-3x speedup (not 4x due to memory bandwidth and attention scaling)
- No `torch.compile` recompilation (fixed shapes)

## Verification

1. Syntax check
2. Run inference — compare quality with non-batched TPP (should be identical if flash_attention is numerically close to cuDNN)
3. Measure timing — should see significant speedup
4. Verify total forward passes = total_groups (was total_groups * S)
