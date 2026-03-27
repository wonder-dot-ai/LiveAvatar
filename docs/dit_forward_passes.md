# DiT Forward Passes: `_forward_sink` and `_forward_inference`

This document explains the two inference-time forward passes of `CausalWanModel_S2V` and how each input is integrated into the model.

## Overview

The `forward()` method dispatches to one of three functions:

```
forward()
  ├── kv_cache provided + sink_flag=True  → _forward_sink()
  ├── kv_cache provided + sink_flag=False → _forward_inference()
  └── no kv_cache                         → _forward_train()
```

During inference, generation is a two-phase process:

1. **Sink (prefill)** — called once per generation. Encodes ref image + motion latents into the KV cache as persistent conditioning. Processes zero noisy-latent frames (nf=0). Returns dummy output.
2. **Inference (decode)** — called once per (block, denoising step). Denoises one block of latent frames using the cached conditioning from the sink pass. Returns predicted noise.

```
Pipeline:
  _prefill_cond_cache()  →  _forward_sink()     # one-time: cache ref + motion
  for clip:
    for block:
      for step:
        noise_model()    →  _forward_inference() # per-step: denoise block
```

---

## Shared Signature

Both functions accept the same parameters. Some are consumed differently or ignored depending on the phase.

| Parameter | Type | Description |
|---|---|---|
| `x` | `list[Tensor]` | Noisy latent block(s), each `[C=16, F, H, W]` |
| `t` | `Tensor [B, F]` | Diffusion timestep per frame |
| `context` | `list[Tensor]` | T5 text embeddings, each `[L, 4096]` |
| `seq_len` | unused | — |
| `ref_latents` | `Tensor [B, C, 1, H, W]` | VAE-encoded reference image (1 latent frame) |
| `motion_latents` | `Tensor [B, C, T_m, H, W]` | VAE-encoded motion context frames |
| `cond_states` | `Tensor [B, C, F, H, W]` | Pose condition (e.g. DWPose), same temporal length as `x` |
| `audio_input` | `Tensor [B, 25, 1024, T_a]` | Wav2Vec2 features (25 layers, 1024-dim, T_a audio frames) |
| `motion_frames` | `[int, int]` | `[pixel_motion_frames, latent_motion_frames]`, e.g. `[73, 19]` |
| `add_last_motion` | `int` | FramePack resolution level (0/1/2) |
| `drop_motion_frames` | `bool` | Whether to zero-out motion info (CFG) |
| `kv_cache` | `list[dict]` | Per-layer KV cache with `k, v, cond_k, cond_v, cond_end` |
| `crossattn_cache` | `list[dict]` | Per-layer cross-attention cache for text |
| `current_start` | `int` | Token offset in the KV cache for this block |
| `current_end` | `int` | Token end offset |

---

## Phase 1: `_forward_sink` — Condition Prefill

**Purpose:** Encode ref image and motion latents into the KV cache's `cond_k`/`cond_v` slots. No actual denoising happens.

### Step-by-step

#### 1. Discard noisy latents (create empty sequence)

```python
nf = 0
x = [zeros([1, dim, 0, H/2, W/2])]  # zero-length noisy sequence
```

The sink pass has **no noisy-latent tokens**. `original_seq_len = 0`.

#### 2. Patch-embed the reference image

```python
ref = [self.patch_embedding(r)]  # [1, dim, 1, H_p, W_p]
```

The ref image becomes `H_p * W_p` tokens (e.g. 24×16 = 384 tokens) with a fixed RoPE grid position at temporal index 30 (a sentinel position far from the noisy sequence).

```python
ref_grid_sizes = [[
    [30, 0, 0],   # start
    [31, H_p, W_p],  # end
    [1, H_p, W_p],   # range
]]
```

The ref tokens are concatenated to `x` and marked with `mask_input = 1` (ref token type).

#### 3. Inject motion via FramePack

```python
x, seq_lens, freqs, mask_input = self.inject_motion(
    x, seq_lens, freqs, mask_input, motion_latents, ...)
```

`inject_motion` calls `self.frame_packer` (FramePackMotioner), which:
- Takes `motion_latents [B, C=16, T_m, H, W]`
- Compresses them into 3 temporal buckets via strided 3D convolutions:
  - **1x bucket** (nearest frames): Conv3d with kernel (1,2,2), stride (1,2,2)
  - **2x bucket** (intermediate): Conv3d with kernel (2,4,4), stride (2,4,4)
  - **4x bucket** (farthest): Conv3d with kernel (4,8,8), stride (4,8,8)
- Returns `mot` (motion token embeddings), `mot_remb` (RoPE embeddings), and `motion_rope_cache`

The motion tokens are concatenated to `x` and marked with `mask_input = 2` (motion token type).

After injection, the full sequence is: `[0 noisy tokens | 384 ref tokens | N motion tokens]`.

#### 4. Token-type embedding

```python
x = x + self.trainable_cond_mask(mask_input)
```

`trainable_cond_mask` is `nn.Embedding(3, dim)` with indices:
- **0** = noisy latent tokens
- **1** = ref image tokens
- **2** = motion tokens

This lets the model distinguish the three token types.

#### 5. Timestep embedding

Timesteps are embedded via sinusoidal → MLP → projection to 6 modulation vectors. With `zero_timestep=True`, a zero-timestep embedding is computed and assigned to conditioning tokens, while the actual timestep applies to noisy tokens:

```python
e0 = [e0_tensor, self.original_seq_len]
# e0_tensor: [B, F, 6, 2, dim]  — slot 0 = noisy modulation, slot 1 = zero modulation
# self.original_seq_len: boundary between noisy and conditioning tokens
```

Since `original_seq_len = 0` in the sink pass, **all tokens receive the zero-timestep modulation**.

#### 6. Text embedding

```python
context = self.text_embedding(padded_t5_embeddings)  # [B, text_len, dim]
```

#### 7. Transformer blocks with cond caching

Each transformer block runs self-attention. In the self-attention layer, the `seg_idx` mechanism determines behavior:

```python
seg_idx = [0, seg_idx_boundary, total_len]
#           ↑ noisy segment        ↑ conditioning segment
```

Since `seg_idx[1] - seg_idx[0] = 0` (no noisy tokens) and `seg_idx[2] - seg_idx[1] > 0` (ref + motion tokens exist), the attention enters the **prefill cond caching** branch:

```python
# Store K/V into cond cache
kv_cache["cond_k"][:, :cond_len] = k[:, seg_idx[1]:seg_idx[2]]
kv_cache["cond_v"][:, :cond_len] = v[:, seg_idx[1]:seg_idx[2]]
kv_cache["cond_end"] = cond_len

# Self-attention among conditioning tokens only
x = attention(q=..., k=cond_k, v=cond_v)
```

The ref image, motion tokens, and their positional encodings are now stored in `cond_k`/`cond_v` for all future steps.

#### 8. Return dummy output

```python
return [zeros_like(cond_states)]  # output is discarded
```

### What `_forward_sink` does NOT do

- Does **not** process audio (audio is not relevant until actual denoising)
- Does **not** add pose/cond to noisy latents (there are none)
- Does **not** produce a noise prediction
- Does **not** call `after_transformer_block` (no audio injection needed)
- Does **not** call `self.head` or `self.unpatchify`

---

## Phase 2: `_forward_inference` — Block Denoising

**Purpose:** Denoise one block of latent frames (e.g. 3 latent frames = 12 pixel frames), using the cached conditioning from the sink pass.

### Step-by-step

#### 1. Audio encoding

```python
# Pad audio to cover motion frames (repeat first frame)
audio_input = cat([audio[..., 0:1].repeat(..., motion_frames[0]), audio], dim=-1)

# Encode: wav2vec features → CausalAudioEncoder → per-frame audio tokens
audio_emb_res = self.casual_audio_encoder(audio_input)
# → (global: [B, T_total, 1, dim], local: [B, T_total, 5, dim])

# Slice off motion-frame portion, keep only the denoising-frame portion
self.merged_audio_emb = audio_emb[:, motion_frames[1]:, :]  # [B, F, 5, dim]
self.audio_emb_global = audio_emb_global[:, motion_frames[1]:]  # [B, F, 1, dim]
```

The `CausalAudioEncoder`:
1. Learns a weighted sum across wav2vec's 25 layers
2. Passes through `MotionEncoder_tc` (causal 1D convolutions) to produce per-frame token embeddings
3. Outputs 5 tokens per frame (4 local + 1 padding) at `dim=5120`

Audio is stored on `self` and consumed later by `after_transformer_block`.

#### 2. Patch-embed noisy latents + add pose

```python
x = [self.patch_embedding(u)]     # [1, dim, F, H_p, W_p]
cond = [self.cond_encoder(c)]     # [1, dim, F, H_p, W_p]
x = [x_ + pose for x_, pose in zip(x, cond)]
```

Both noisy latents and pose go through separate Conv3d patch embeddings (kernel/stride = `(1,2,2)`), then **summed element-wise**. The pose signal is baked directly into the token representations.

Flattened to `[B, F*H_p*W_p, dim]`.

#### 3. RoPE computation

```python
frame_offset = current_start // frame_seqlen
grid_sizes_shifted = rollout_grid_sizes(grid_sizes, frame_offset)
self.pre_compute_freqs = rope_precompute(x, grid_sizes_shifted, self.freqs)
```

`rollout_grid_sizes` shifts the temporal index of the grid so that each block's RoPE reflects its true position in the full sequence (block 0 starts at frame 0, block 1 at frame 3, etc.).

A separate `cond_pre_compute_freqs` is computed for the conditioning cache with a randomized temporal offset (between 4-30 frames back) for robustness.

#### 4. Token-type embedding

```python
mask_input = zeros([1, seq_len])  # all zeros → noisy token type
x = x + self.trainable_cond_mask(mask_input)
```

All tokens are type 0 (noisy). Conditioning tokens are already in the KV cache with their own type embeddings from the sink pass.

#### 5. Timestep embedding (same as sink)

```python
e0 = [e0_tensor, self.original_seq_len]
# original_seq_len = F*H_p*W_p (noisy token count)
```

Now `original_seq_len > 0`, so the per-frame timestep modulation applies to the noisy tokens, and zero-timestep applies beyond (though in inference there are no conditioning tokens in the sequence itself — they are in the cache).

#### 6. Text embedding (same as sink)

#### 7. Transformer blocks with KV cache lookup

Each block runs self-attention. Now `seg_idx[1] - seg_idx[0] > 0` (noisy tokens exist), entering the **streaming inference** branch:

```python
# Write current block's K/V into the rolling cache
kv_cache["k"][:, current_start:current_start+block_len] = roped_key[:, noisy_segment]
kv_cache["v"][:, current_start:current_start+block_len] = v[:, noisy_segment]

# Read: concatenate rolling cache + cond cache
k_full = cat([kv_cache["k"][:, :active_size], rope(kv_cache["cond_k"])], dim=1)
v_full = cat([kv_cache["v"][:, :active_size], kv_cache["cond_v"]], dim=1)

# Attend: current block queries against full history + conditioning
x = attention(q=current_block_q, k=k_full, v=v_full)
```

This means each block attends to:
- **All previously generated blocks** (in the rolling KV cache)
- **All conditioning tokens** (ref image + motion, in the cond cache from the sink pass)

This is the causal streaming mechanism: each block can see everything before it, but nothing after.

#### 8. Audio injection (`after_transformer_block`)

After transformer blocks at layers `[0, 4, 8, 12, 16, 20, 24, 27]`:

```python
# Extract noisy-token hidden states only
input_hidden_states = hidden_states[:, :self.original_seq_len]
input_hidden_states = rearrange(input_hidden_states, "b (t n) c -> (b t) n c", t=F)

# Cross-attend: visual tokens (query) × audio tokens (key/value)
audio_emb = rearrange(self.merged_audio_emb, "b t n c -> (b t) n c")
residual = self.audio_injector.injector[layer_id](
    x=norm(input_hidden_states),   # [(B*F), H_p*W_p, dim]
    context=audio_emb,             # [(B*F), 5, dim]
)

# Add residual back (only to noisy tokens)
hidden_states[:, :self.original_seq_len] += residual
```

Audio injection is a per-frame cross-attention: each spatial token attends to 5 audio tokens for its corresponding frame. If AdaIN is enabled, global audio features additionally modulate the LayerNorm.

#### 9. Output head

```python
x = x[:, :self.original_seq_len]  # keep only noisy tokens
x = self.head(x, e)               # project to output dim with timestep modulation
x = self.unpatchify(x, grid_sizes) # reshape to [C_out, F, H_lat, W_lat]
```

The head applies a final LayerNorm + linear, modulated by the timestep embedding. `unpatchify` reverses the patch flattening to produce the noise prediction in latent space.

---

## Input Integration Summary

| Input | Sink | Inference | Integration method |
|---|---|---|---|
| **ref_latents** | patch_embed → cond KV cache | read from cond cache | Self-attention KV cache |
| **motion_latents** | FramePack → cond KV cache | read from cond cache | Self-attention KV cache |
| **x** (noisy latents) | discarded (nf=0) | patch_embed → tokens | Sequence tokens (query + rolling KV) |
| **cond_states** (pose) | not used | patch_embed, summed into x | Element-wise addition to noisy tokens |
| **audio_input** | not used | CausalAudioEncoder → `after_transformer_block` | Cross-attention at 8 layers |
| **t** (timestep) | zero-timestep only | sinusoidal → MLP → 6 AdaLN modulations | AdaLN in each transformer block |
| **context** (text) | text_embedding → cross-attn | text_embedding → cross-attn | Cross-attention in each block |
| **motion_frames** | controls FramePack slicing | controls audio slicing | Boundary index |
| **current_start/end** | 0 (beginning) | token offset | KV cache write position + RoPE offset |
| **trainable_cond_mask** | 0=noisy, 1=ref, 2=motion | 0=noisy | Learned embedding added to tokens |

---

## KV Cache Structure

Each transformer layer maintains:

```
kv_cache[layer] = {
    "k":      [B, max_tokens, num_heads, head_dim]   # rolling cache for noisy blocks
    "v":      [B, max_tokens, num_heads, head_dim]
    "cond_k": [B, max_cond_tokens, num_heads, head_dim]  # static cache for ref + motion
    "cond_v": [B, max_cond_tokens, num_heads, head_dim]
    "cond_end": int  # number of valid conditioning tokens
}
```

- **Cond cache** is filled once by `_forward_sink` and read every step.
- **Rolling cache** grows by `tokens_per_block` each denoising block. The batch dimension is `num_denoising_steps`, so each diffusion step has its own cache slot — they progress through the sequence at different positions simultaneously.

```
After sink:     cond_cache = [ref_tokens | motion_tokens]
                rolling_cache = [empty]

After block 0:  rolling_cache = [block_0_kv]
After block 1:  rolling_cache = [block_0_kv | block_1_kv]
...
```
