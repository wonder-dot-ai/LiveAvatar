# Batched TPP Pipeline — Implementation Plan

> **Date**: 2026-03-20
> **Based on**: `batched_tpp_2gpu_analysis.md`, `repository_analysis.md`

---

## 1. Phasing

| Phase | Scope | Model Changes |
|-------|-------|---------------|
| **Phase 1 (this plan)** | Single-GPU, **batched DiT** (batch=4 pipeline shift register) | Yes — per-batch `current_start`, per-batch RoPE |
| Phase 2 (future) | Extend to 2-GPU (GPU 0 = DiT, GPU 1 = VAE, overlapped) | No |

No SAM2, no TTS, no training code, no FSDP/SP, no `dist.*`.

---

## 2. The Pipeline Shift Register

### 2.1 Concept

Instead of denoising each block through 4 sequential steps, we process 4 **different blocks at different denoising stages** in a single batched forward pass:

```
Iteration 0:  [block0@step0,  —,            —,            —           ]
Iteration 1:  [block1@step0,  block0@step1, —,            —           ]
Iteration 2:  [block2@step0,  block1@step1, block0@step2, —           ]
Iteration 3:  [block3@step0,  block2@step1, block1@step2, block0@step3] → block0 done
Iteration 4:  [block4@step0,  block3@step1, block2@step2, block1@step3] → block1 done
...
Iteration N+2: [—,           —,            blockN@step2,  blockN-1@step3] → blockN-1 done
Iteration N+3: [—,           —,            —,             blockN@step3]   → blockN done
```

At steady state, each iteration:
- Takes 1 batched forward pass (batch=4)
- Produces 1 fully denoised block
- Total iterations: `num_blocks + 3` (vs `num_blocks × 4` sequential)

### 2.2 What Differs Per Batch Element

| Input | batch[0] (step 0) | batch[1] (step 1) | batch[2] (step 2) | batch[3] (step 3) |
|-------|-------------------|-------------------|-------------------|-------------------|
| **Input latent** | Fresh noise for block B | Output of prev iter batch[0] | Output of prev iter batch[1] | Output of prev iter batch[2] |
| **Timestep** | t₀ (highest noise) | t₁ | t₂ | t₃ (lowest noise) |
| **KV cache** | Stage-0 cache | Stage-1 cache | Stage-2 cache | Stage-3 cache |
| **`current_start`** | B × bsl | (B−1) × bsl | (B−2) × bsl | (B−3) × bsl |
| **Audio slice** | block B frames | block B−1 frames | block B−2 frames | block B−3 frames |
| **Cond slice** | block B pose | block B−1 pose | block B−2 pose | block B−3 pose |
| **Text context** | Same | Same | Same | Same |
| **ref_latents** | Same | Same | Same | Same |
| **motion_latents** | Same | Same | Same | Same |

Where `bsl = num_frames_per_block × frame_seq_length` (tokens per block in KV cache).

### 2.3 Ramp-Up and Drain

During ramp-up (first 3 iterations) and drain (last 3), some batch slots are inactive. We handle this by:
- Running batch=4 always, filling inactive slots with dummy noise
- Discarding outputs from inactive slots
- Not writing to KV cache for inactive slots (use a boolean mask)

This wastes ~3 iterations of compute — negligible for any video with >4 blocks.

---

## 3. Model Changes Required

### 3.1 `CausalWanS2VSelfAttention.forward()` (causal_model_s2v.py:235)

Current signature:
```python
def forward(self, x, seq_lens, grid_sizes, freqs, block_mask,
            kv_cache=None, current_start=0, current_end=0, sp_size=None,
            seg_idx=None, freqs_cond=None):
```

**Change `current_start` from `int` to `Tensor[B]` or `int`** (backward compatible).

#### 3.1a KV Cache Write (line 303-304)

Current:
```python
kv_cache["k"][:, current_start:(current_start+seg_len_block)] = roped_key[:,seg_idx[0]:seg_idx[1]]
kv_cache["v"][:, current_start:(current_start+seg_len_block)] = v[:,seg_idx[0]:seg_idx[1]]
```

New (per-batch write, B=4 is small so loop is fine):
```python
if isinstance(current_start, int):
    # Scalar path (backward compatible, used during prefill)
    kv_cache["k"][:, current_start:(current_start+seg_len_block)] = roped_key[:,seg_idx[0]:seg_idx[1]]
    kv_cache["v"][:, current_start:(current_start+seg_len_block)] = v[:,seg_idx[0]:seg_idx[1]]
    active_starts = [current_start] * b
    active_sizes = [current_start + seg_len_block] * b
else:
    # Per-batch path (batched pipeline)
    for bi in range(b):
        cs = current_start[bi].item()
        kv_cache["k"][bi, cs:(cs+seg_len_block)] = roped_key[bi, seg_idx[0]:seg_idx[1]]
        kv_cache["v"][bi, cs:(cs+seg_len_block)] = v[bi, seg_idx[0]:seg_idx[1]]
    active_starts = [cs_i.item() for cs_i in current_start]
    active_sizes = [cs_i.item() + seg_len_block for cs_i in current_start]
```

#### 3.1b KV Cache Read + Attention (lines 305-323)

Current code reads a single contiguous slice and passes uniform `k_lens`. For per-batch variable lengths:

```python
# Pad all batches to max KV length, use per-batch k_lens for masking
max_active_size = max(active_sizes)
max_active_start = 0  # always read from 0 (simplification)

# All batch elements share the same tensor slice [0:max_active_size]
# but k_lens tells flash_attn the actual length per batch
k_lens = torch.tensor([
    (active_sizes[bi] - 0) + active_cond_cache_size
    for bi in range(b)
], device=x.device)

x = attention(
    q=roped_query[:, seg_idx[0]:seg_idx[1]],
    k=torch.cat([
        kv_cache["k"][:, 0:max_active_size],
        causal_rope_apply_cond(kv_cache["cond_k"][:, :active_cond_cache_size], ...)
    ], dim=1),
    v=torch.cat([
        kv_cache["v"][:, 0:max_active_size],
        kv_cache["cond_v"][:, :active_cond_cache_size]
    ], dim=1),
    k_lens=k_lens,
    window_size=self.window_size
)
```

The `attention()` function already uses `flash_attn_varlen_func` which supports per-batch `k_lens` via `cu_seqlens_k`. The padding beyond each batch element's actual KV is masked out by `k_lens`.

**Important**: For batch elements in ramp-up/drain that are inactive, their KV cache has `active_size = 0 + seg_len_block` (they only have the dummy block). The attention will attend to that dummy data, but we discard the output anyway.

#### 3.1c Local Attention Size

The `local_attn_size` branch (line 298-300) computes `active_kv_cache_start` to limit attention window. For per-batch:

```python
if self.local_attn_size != -1:
    active_starts = [max(0, sz - self.local_attn_size * seg_len_block // 3)
                     for sz in active_sizes]
```

This changes `active_kv_cache_start` per batch. We handle it same way: pad to max, use `k_lens`.

### 3.2 `CausalWanModel_S2V._forward_inference()` (causal_model_s2v.py:1054)

#### 3.2a RoPE Computation (lines 1136-1145)

Current:
```python
self.pre_compute_freqs = rope_precompute(
    x.detach().view(b, s, n, d),
    rollout_grid_sizes(grid_sizes, current_start // frame_seqlen),
    self.freqs, start=None)
```

`current_start // frame_seqlen` gives the frame offset for RoPE. For per-batch offsets, we need per-batch RoPE. Two approaches:

**Option A: Compute per-batch, concatenate** (simple, B=4)
```python
if isinstance(current_start, (int, float)):
    frame_offsets = [current_start // frame_seqlen] * b
else:
    frame_offsets = [cs.item() // frame_seqlen for cs in current_start]

# Compute RoPE per batch element
freqs_list = []
for bi in range(b):
    x_bi = x[bi:bi+1].detach().view(1, s, n, d)
    grid_bi = rollout_grid_sizes(grid_sizes, frame_offsets[bi])
    freqs_bi = rope_precompute(x_bi, grid_bi, self.freqs, start=None)
    freqs_list.append(freqs_bi)
self.pre_compute_freqs = torch.cat(freqs_list, dim=0)  # [B, s, n, d//2]
```

**Option B: Vectorized** — modify `rollout_grid_sizes` and `rope_precompute` to accept per-batch offsets. More complex, save for optimization.

**Use Option A** for Phase 1 — B=4 loop overhead is negligible vs DiT forward pass.

#### 3.2b Cond RoPE (lines 1143-1145)

Same approach: per-batch cond RoPE via loop.

```python
cond_freqs_list = []
for bi in range(b):
    num_frames_cond_rollout = max(0, frame_offsets[bi] - start_idx)
    cond_freqs_bi = rope_precompute(
        torch.empty(self.rope_cache['cond_shape']).type_as(x)[0:1],
        rollout_grid_sizes(self.rope_cache['grid_sizes'], num_frames_cond_rollout),
        self.freqs, start=None)
    cond_freqs_list.append(cond_freqs_bi)
cond_pre_compute_freqs = torch.cat(cond_freqs_list, dim=0)
```

#### 3.2c Per-Layer KV Cache Indexing (line 1234)

Current: `kv_cache[idx]` indexes layer `idx` from a list. The KV cache is `list[dict]` with 40 entries.

For batched pipeline, the KV cache structure changes:
- Old: 4 separate caches, each `list[40 × dict]` with tensors `[1, seq, 40, 128]`
- New: 1 cache, `list[40 × dict]` with tensors `[4, seq, 40, 128]`

The batch dimension is already in the tensor. The model code doesn't need to change here — it already operates on `[:, ...]` which handles any batch size.

#### 3.2d Audio, Cond States (per-batch slicing)

The model already handles batched inputs:
```python
x = [self.patch_embedding(u.unsqueeze(0)) for u in x]  # list of B elements
x = torch.cat(x, dim=0)  # [B, seq, dim]
```

So we pass a list of 4 block latents (one per pipeline slot), 4 cond_states slices, and batched audio_input `[4, 25, 1024, frames]` with per-slot frame ranges.

### 3.3 `CausalWanS2VSelfAttention.forward()` — Prefill Path (line 324-338)

The prefill path (cond caching with `sink_flag=True`) runs once with all batch elements at `current_start=0`. This is naturally batchable — no per-batch position issues. Keep as-is.

### 3.4 Summary of Model Changes

| File | Function | Change |
|------|----------|--------|
| `causal_model_s2v.py:235` | `CausalWanS2VSelfAttention.forward` | Per-batch KV cache write (loop), per-batch `k_lens` |
| `causal_model_s2v.py:1054` | `_forward_inference` | Per-batch RoPE via loop, accept tensor `current_start` |
| `causal_model_s2v.py:1054` | `_forward_inference` | Per-batch cond RoPE via loop |

All changes are backward compatible — scalar `current_start` takes the existing code path.

---

## 4. Pipeline Design

### 4.1 KV Cache Structure

```python
# Single batched KV cache: 4 pipeline stages stacked in batch dim
# self.kv_cache = list of 40 layer dicts
self.kv_cache = [{
    "k":      torch.zeros([4, max_seq_len, 40, 128], dtype=dtype, device=device),
    "v":      torch.zeros([4, max_seq_len, 40, 128], dtype=dtype, device=device),
    "cond_k": torch.zeros([4, 2800, 40, 128], dtype=dtype, device=device),
    "cond_v": torch.zeros([4, 2800, 40, 128], dtype=dtype, device=device),
    "cond_end": torch.tensor([0], dtype=torch.long, device=device),
} for _ in range(num_layers)]
```

Memory: identical to existing 4 separate caches. Just `[4, ...]` instead of 4× `[1, ...]`.

### 4.2 Shared Cond Cache

The cond cache (text + ref + motion embeddings) is identical across all 4 pipeline stages. We can share it:

```python
# Prefill once, all 4 batch elements get the same cond cache
# cond_k/cond_v shape: [4, cond_len, 40, 128] — all 4 slices identical
# This is filled during the prefill pass with sink_flag=True
```

Alternatively, keep cond as `[1, ...]` and broadcast in attention. But since the existing code structure puts it in the same dict, we keep `[4, ...]` with identical content.

### 4.3 `generate()` — Batched Denoising Loop

```python
num_steps = 4  # sampling steps = pipeline depth
num_blocks = lat_target_frames // num_frames_per_block
bsl = num_frames_per_block * frame_seq_length  # tokens per block in KV cache
total_iters = num_blocks + num_steps - 1  # pipeline fill + steady + drain

# Pipeline state: what block each slot is currently processing
# slot_block[i] = which block is in pipeline slot i (or -1 if inactive)
# pipeline_latents[i] = current latent for slot i

pipeline_latents = [None] * num_steps  # [slot0, slot1, slot2, slot3]

for iter_idx in range(total_iters):
    # --- Shift pipeline ---
    # Slot 3 output (if active) is the fully denoised block → collect it
    if pipeline_latents[3] is not None and (iter_idx - 3) >= 0 and (iter_idx - 3) < num_blocks:
        completed_block_idx = iter_idx - 3
        clip_output[:, completed_block_idx*nfpb:(completed_block_idx+1)*nfpb] = pipeline_latents[3]

    # Shift: slot[i] = prev slot[i-1]
    for i in range(num_steps - 1, 0, -1):
        pipeline_latents[i] = pipeline_latents[i - 1]

    # Slot 0: new noise block (or None if past last block)
    if iter_idx < num_blocks:
        pipeline_latents[0] = clip_noise[iter_idx]  # [16, nfpb, H, W]
    else:
        pipeline_latents[0] = dummy_noise  # inactive, will be discarded

    # --- Build batched inputs ---
    # Stack 4 latents into batch
    batch_latents = [pipeline_latents[i] for i in range(num_steps)]  # list of 4
    batch_timesteps = torch.stack([timesteps[i].expand(nfpb) for i in range(num_steps)])  # [4, nfpb]

    # Per-batch current_start
    batch_current_start = torch.tensor([
        max(0, iter_idx - i) * bsl for i in range(num_steps)
    ], device=device)

    # Per-batch audio slices
    batch_audio = torch.stack([
        audio_input[..., block_idx*nfpb*4:(block_idx+1)*nfpb*4]
        for block_idx in [max(0, iter_idx - i) for i in range(num_steps)]
    ])  # [4, 25, 1024, frames_per_block]

    # Per-batch cond_states slices (similarly)
    batch_cond = torch.stack([...])

    # Context, ref_latents, motion_latents: same for all → expand to batch=4
    batch_context = context * 4  # list of 4 identical
    batch_ref = ref_latents.expand(4, ...)
    batch_motion = motion_latents.expand(4, ...)

    # --- Batched forward ---
    noise_pred = self.noise_model(
        batch_latents,             # list of 4 tensors
        t=batch_timesteps,         # [4, nfpb]
        context=batch_context,
        ref_latents=batch_ref,
        motion_latents=batch_motion,
        cond_states=batch_cond,
        audio_input=batch_audio,
        kv_cache=self.kv_cache,
        crossattn_cache=self.crossattn_cache,
        current_start=batch_current_start,
        current_end=batch_current_end,
        motion_frames=[motion_frames, lat_motion_frames],
    )

    # --- Per-slot scheduler step ---
    for i in range(num_steps):
        if pipeline_latents[i] is not None:
            block_idx = iter_idx - i
            if 0 <= block_idx < num_blocks:
                # Use per-slot scheduler (each slot has its own step index)
                pipeline_latents[i] = slot_schedulers[i].step(
                    noise_pred[i].unsqueeze(0),
                    timesteps[i],
                    pipeline_latents[i].unsqueeze(0),
                    generator=seed_g
                )[0].squeeze(0)
```

### 4.4 Scheduler Per Slot

Each pipeline slot is at a fixed timestep (slot 0 = step 0, slot 1 = step 1, etc.). We need 4 scheduler instances, each configured to execute only its assigned step:

```python
slot_schedulers = []
for step_idx in range(num_steps):
    sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3)
    sched.set_timesteps(num_steps, device=device)
    sched._step_index = step_idx
    sched._begin_index = 0
    slot_schedulers.append(sched)
```

### 4.5 Memory (unchanged from analysis)

| Component | FP8 |
|-----------|-----|
| Model weights | ~15.6 GB |
| KV cache (batch=4) | ~41.2 GB |
| Peak activations (batch=4) | ~0.9 GB |
| Other + allocator | ~3.4 GB |
| **Total GPU 0** | **~61.1 GB** |

Same as 4 separate caches — the memory is identical, just reshaped.

---

## 5. Removed Features

Compared to `causal_s2v_pipeline.py`:

| Removed | Lines saved |
|---------|-------------|
| SAM2 (`+` audio, mask routing, dilate_mask) | ~100 lines |
| TTS (`tts()`, `load_tts()`) | ~30 lines |
| Training (`is_training`, train scheduler, grad methods) | ~50 lines |
| FSDP / SP (`_configure_model` branches, shard_fn) | ~30 lines |
| `dist.*` (barriers, ranks, broadcast) | ~20 lines |
| Dataset mode | ~10 lines |

---

## 6. New Files

### 6.1 `liveavatar/models/wan/causal_s2v_pipeline_2gpu.py`

Pipeline class with batched DiT. See sections 4.1–4.5 above.

### 6.2 `minimal_inference/s2v_2gpu.py`

Standalone CLI inference script:

```python
"""Standalone inference for batched TPP pipeline (Phase 1: single-GPU)."""
import argparse, os, torch, imageio

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--audio", required=True)
    p.add_argument("--prompt", default="A person is talking")
    p.add_argument("--ckpt_dir", default="ckpt/Wan2.2-S2V-14B/")
    p.add_argument("--task", default="s2v-14B")
    p.add_argument("--load_lora", default=None)
    p.add_argument("--infer_frames", type=int, default=48)
    p.add_argument("--num_clip", type=int, default=1)
    p.add_argument("--max_area", type=int, default=720*400)
    p.add_argument("--sample_steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--n_prompt", default="")
    p.add_argument("--fp8", action="store_true")
    p.add_argument("--offload_model", type=bool, default=True)
    p.add_argument("--offload_kv_cache", action="store_true")
    p.add_argument("--enable_online_decode", action="store_true")
    p.add_argument("--pose_video", default=None)
    p.add_argument("--output", default="output.mp4")
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--torch_trace", action="store_true")
    p.add_argument("--profile_output_dir", default=None)
    return p.parse_args()

def main():
    args = parse_args()
    from liveavatar.models.wan.wan_2_2.configs.wan_s2v_14B_modified import s2v_14B as cfg

    from liveavatar.models.wan.causal_s2v_pipeline_2gpu import WanS2V
    pipeline = WanS2V(config=cfg, checkpoint_dir=args.ckpt_dir,
                      device_id=0, offload_kv_cache=args.offload_kv_cache)

    if args.load_lora:
        pipeline.noise_model = pipeline.add_lora_to_model(
            pipeline.noise_model, pretrained_lora_path=args.load_lora, load_only=True)
    if args.fp8:
        from liveavatar.utils.fp8_linear import convert_model_to_fp8
        convert_model_to_fp8(pipeline.noise_model)

    video, _ = pipeline.generate(
        input_prompt=args.prompt, ref_image_path=args.image,
        audio_path=args.audio, infer_frames=args.infer_frames,
        num_repeat=args.num_clip, max_area=args.max_area,
        sampling_steps=args.sample_steps, seed=args.seed,
        n_prompt=args.n_prompt, offload_model=args.offload_model,
        pose_video=args.pose_video,
        enable_online_decode=args.enable_online_decode,
        profiler=None, torch_trace=args.torch_trace,
        profile_output_dir=args.profile_output_dir)

    if video is not None:
        video_np = ((video.clamp(-1, 1) + 1) / 2 * 255).byte()
        video_np = video_np.permute(1, 2, 3, 0).cpu().numpy()
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        imageio.mimwrite(args.output, video_np, fps=args.fps, codec="libx264")
        print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()
```

### 6.3 `inference_2gpu.sh`

```bash
#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python minimal_inference/s2v_2gpu.py \
    --image examples/sample1.jpg \
    --audio examples/sample1.wav \
    --prompt "A person is talking" \
    --fp8 --offload_model True \
    --infer_frames 48 --num_clip 1 --sample_steps 4 \
    --output output/result.mp4
```

---

## 7. Implementation Checklist

### 7.1 Model Changes (`causal_model_s2v.py`)

- [ ] `CausalWanS2VSelfAttention.forward()`: Add `isinstance(current_start, int)` branch
  - Scalar path: existing code unchanged (backward compat)
  - Tensor path: per-batch KV write loop, per-batch `k_lens`
- [ ] `_forward_inference()`: Accept tensor `current_start`
  - Per-batch RoPE via loop over B=4
  - Per-batch cond RoPE via loop
  - Pass tensor `current_start` through to attention blocks

### 7.2 Pipeline (`causal_s2v_pipeline_2gpu.py`)

- [ ] Copy `causal_s2v_pipeline.py` as base
- [ ] Strip: SAM2, TTS, training, FSDP/SP, `dist.*`, dataset mode
- [ ] Simplify `__init__`: 4 params (`config`, `checkpoint_dir`, `device_id`, `offload_kv_cache`)
- [ ] Change `_initialize_kv_cache`: single cache with `batch_size=4`, shape `[4, seq, 40, 128]`
- [ ] Implement pipeline shift register in `generate()`:
  - `pipeline_latents[4]` state array
  - Per-iteration: shift, insert new noise, build batched inputs, forward, per-slot scheduler step
  - Collect completed blocks from slot 3
  - Handle ramp-up (first 3 iters: partial batch) and drain (last 3 iters)
- [ ] 4 scheduler instances (one per slot/step)
- [ ] Prefill: batch=4, all at `current_start=0`, `sink_flag=True`
- [ ] Keep: profiler hooks, online decode, multi-clip, LoRA, FP8 support

### 7.3 Inference Script (`minimal_inference/s2v_2gpu.py`)

- [ ] Standalone argparse
- [ ] Config loading, FP8 setup, LoRA
- [ ] Pipeline creation and `generate()` call
- [ ] Video save with imageio

### 7.4 Launch Script (`inference_2gpu.sh`)

- [ ] Single-GPU launch command

---

## 8. Testing

1. **Correctness**: Run with same seed as existing pipeline → outputs should match (after accounting for batched RoPE order)
2. **Ramp-up/drain**: Test with small num_blocks (e.g., 4) to exercise all pipeline phases
3. **Multi-clip**: Test num_clip=2 to verify motion latent handoff
4. **Memory**: Verify peak VRAM ≤ 61 GB with FP8 at 704×384

---

## 9. Future: Phase 2 (2-GPU Extension)

Add to `causal_s2v_pipeline_2gpu.py`:
1. `_initialize_comm_group()` for rank roles
2. Move preprocessing to GPU 1, broadcast to GPU 0
3. `dist.send(completed_block, dst=1)` after each pipeline iteration
4. VAE streaming decode on GPU 1 overlapped with DiT
5. Launch with `torchrun --nproc_per_node=2`

No further model changes needed.
