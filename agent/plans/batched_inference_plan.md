# Batched Inference Implementation Plan

> Created: 2026-03-16

---

## Goal

Support batch_size > 1 during inference to improve GPU utilization and throughput, processing multiple (image, audio, prompt) inputs in a single forward pass.

## Constraint

All samples in a batch must share the **same resolution and audio duration**. This avoids the need for per-sample padding/masking in the diffusion loop and attention masks, keeping the change scope manageable.

---

## Phase 1: Data Loading & Preprocessing

Batch the per-sample input loading into collated tensors before entering the pipeline.

### 1.1 Audio Encoding

**Files:**
- `liveavatar/models/wan/wan_2_2/modules/s2v/audio_encoder.py` — `extract_audio_feat()` (line ~66-89)
- `liveavatar/models/wan/causal_s2v_pipeline.py` — `encode_audio()` (line ~443-456)

**Changes:**
- `extract_audio_feat()`: Accept a list of audio paths, load all with librosa, pad to max length, run Wav2Vec2 as a batch
- `encode_audio()`: Accept list of paths, return `[B, feat_dim, T]` instead of `[1, feat_dim, T]`
- Remove `unsqueeze(0)` at line ~451

### 1.2 Reference Image Loading & VAE Encoding

**Files:**
- `liveavatar/models/wan/causal_s2v_pipeline.py` — `generate()` (line ~757, 884-898)
- `liveavatar/models/wan/wan_wrapper.py` — VAE `encode()` / `decode()` (line ~76-91)

**Changes:**
- Load multiple reference images, stack into `[B, C, 1, H, W]`
- `wan_wrapper.py`: Replace per-sample decode loop (`for u in zs`) with a single batched call or chunked batched call to manage VRAM
- VAE encode: pass `[B, C, T, H, W]` directly instead of single samples

### 1.3 Text Encoding

**Files:**
- `liveavatar/models/wan/causal_s2v_pipeline.py` — `generate()` (line ~983, 991)

**Changes:**
- Pass `batch_size=B` to the T5 text encoder instead of hardcoded `1`
- Handle `context[0:1]` slicing (line ~1025, 1067) — change to `context[0:B]`

---

## Phase 2: Model Forward Pass

Convert the list-of-tensors pattern to proper batched tensor operations.

### 2.1 Patch Embedding & Condition Encoding

**Files:**
- `liveavatar/models/wan/causal_model_s2v.py` — `_forward_inference()` (line ~1110-1118), `_forward_sink()` (line ~883-925)
- `liveavatar/models/wan/wan_2_2/modules/s2v/model_s2v.py` — forward (line ~729-785)

**Changes:**
- Replace list comprehensions like `[self.patch_embedding(u.unsqueeze(0)) for u in x]` with a single `self.patch_embedding(x)` operating on `[B, C, T, H, W]`
- Same for `cond_encoder`, ref_latents processing
- `grid_sizes` and `seq_lens` (line ~1116-1118): Compute once (all samples share same resolution per constraint), broadcast

### 2.2 Motion Encoder

**Files:**
- `liveavatar/models/wan/causal_motioner.py` — `forward()` (line ~67-157)
- `liveavatar/models/wan/wan_2_2/modules/s2v/motioner.py` — `forward()` (line ~679-762)
- `liveavatar/models/wan/wan_2_2/modules/s2v/model_s2v.py` — `process_motion()` (line ~485-512)

**Changes:**
- Replace `for m in motion_latents:` loop with batched processing
- Remove `padd_lat.unsqueeze(0)` (line ~87 / ~699) — tensor should already have batch dim
- `process_motion()`: Replace `[self.patch_embedding(m.unsqueeze(0)) for m in motion_latents]` with batched call

### 2.3 Timestep Handling

**Files:**
- `liveavatar/models/wan/causal_s2v_pipeline.py` (line ~1080)
- `liveavatar/models/wan/causal_model_s2v.py` (line ~1152-1153)

**Changes:**
- `timestep`: Shape from `[1, num_frames_per_block]` → `[B, num_frames_per_block]`
- Remove hardcoded `torch.zeros([1, t.shape[1]])` — use `B` for batch dim

### 2.4 RoPE Embeddings

**Files:**
- `liveavatar/models/wan/causal_s2v_utils.py` (line ~125-126, 159)
- `liveavatar/models/wan/causal_model_s2v.py` (line ~1136-1137)

**Changes:**
- RoPE is resolution-dependent, not sample-dependent. With same-resolution constraint, compute once and broadcast across batch. No major change needed beyond ensuring the batch dim propagates.

---

## Phase 3: KV Cache

**Files:**
- `liveavatar/models/wan/causal_s2v_pipeline.py` — `_initialize_kv_cache()` (line ~601-626)
- `liveavatar/models/wan/causal_s2v_pipeline_tpp.py` — same method (line ~605-626)

**Changes:**
- Already accepts `batch_size` param — callers need to pass `B` instead of `1`
- `cond_end` (line ~622): Change from scalar `torch.tensor([0])` to `torch.tensor([0]*B)` for per-sample tracking
- VRAM impact: KV cache scales linearly with B. Profile to find max feasible batch size per GPU config

---

## Phase 4: Diffusion Loop & Output

**Files:**
- `liveavatar/models/wan/causal_s2v_pipeline.py` — generation loop (line ~1020-1102)
- `liveavatar/models/wan/flow_match.py`

**Changes:**
- `clip_latents[0]` indexing (line ~1020-1021, 1062-1063) → handle B samples
- `noise_pred[0].unsqueeze(0)` (line ~1094) → keep batch dim
- `temp_x0.squeeze(0)` (line ~1099) → don't squeeze batch dim
- `clip_output` accumulation: shape from `[C, T, H, W]` → `[B, C, T, H, W]`
- Flow matching solver: Verify it handles batch dim (likely already does since it operates element-wise)

---

## Phase 5: Attention Mask

**Files:**
- `liveavatar/models/wan/causal_model_s2v.py` — `_prepare_blockwise_causal_attn_mask()` (line ~768-825)

**Changes:**
- With same-resolution + same-frame-count constraint, the block mask is identical for all samples — no change needed, just verify it broadcasts correctly across batch dim in FlashAttention calls

---

## Phase 6: Pipeline Orchestration

**Files:**
- `liveavatar/models/wan/causal_s2v_pipeline.py` — `generate()`
- `minimal_inference/batch_eval.py`
- `minimal_inference/s2v_streaming_interact.py`

**Changes:**
- `generate()`: Accept lists of `ref_image_path`, `audio_path`, `input_prompt`; collate into batch tensors
- `batch_eval.py`: Group samples by (resolution, audio_duration) into batches before calling `generate()`
- Add a `--batch_size` CLI argument
- Online decode path (line ~1106-1141): Needs per-sample frame tracking or disable for batched mode initially

---

## File Change Summary

| File | Scope |
|------|-------|
| `causal_s2v_pipeline.py` | Heavy — generate(), encode_audio(), KV cache, diffusion loop |
| `causal_model_s2v.py` | Heavy — forward passes, list→batch conversion, timestep, grid_sizes |
| `wan_2_2/modules/s2v/model_s2v.py` | Heavy — same list→batch pattern |
| `wan_2_2/modules/s2v/motioner.py` | Medium — replace per-sample loop |
| `causal_motioner.py` | Medium — replace per-sample loop |
| `wan_2_2/modules/s2v/audio_encoder.py` | Medium — batch audio loading |
| `wan_wrapper.py` | Light — batch VAE encode/decode |
| `batch_eval.py` | Light — group samples, add batch_size arg |
| `s2v_streaming_interact.py` | Light — add batch_size arg |
| `flow_match.py` | Verify only — likely already batch-compatible |
| `causal_s2v_utils.py` | Verify only — RoPE broadcast |

---

## VRAM Considerations

KV cache and latent tensors scale linearly with batch size. Rough estimates:
- **KV cache**: ~2.7 GB per sample (40 layers × 2 × 13500 seq × 40 heads × 128 dim × bf16)
- **Latents + activations**: ~1-2 GB per sample depending on resolution
- **Practical max batch size**: 2-4 on 80GB H800 (after model weights ~30GB), profile to confirm

## Suggested Implementation Order

1. Phase 2.1 + 2.3 — Core model forward (biggest structural change, test in isolation)
2. Phase 1.1 + 1.2 — Data loading (enables end-to-end testing)
3. Phase 3 — KV cache (straightforward param change)
4. Phase 4 — Diffusion loop (connect everything)
5. Phase 1.3 + 5 + 6 — Text encoding, masks, orchestration (finishing touches)
6. Profile VRAM, determine max batch sizes per GPU config
