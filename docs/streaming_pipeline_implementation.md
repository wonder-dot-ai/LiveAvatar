# Streaming Pipeline — Implementation Plan (v2)

> **Date**: 2026-03-24
> **Base**: `causal_s2v_pipeline_2gpu.py` (sequential denoising, single-GPU)
> **Reference**: `causal_s2v_pipeline_tpp_blockwise.py` (existing per-block VAE + streaming audio)
> **New file**: `causal_s2v_pipeline_2gpu_streaming.py`

---

## Overview

Two features, both already proven in `causal_s2v_pipeline_tpp_blockwise.py` (5-GPU TPP). We adapt them to our single-GPU sequential pipeline.

1. **Per-block VAE decode** using the streaming VAE (`vae_streaming.py`'s `stream_decode()`)
2. **Streaming audio** via per-clip on-demand wav2vec encoding

---

## 1. Per-Block VAE Decode (Streaming VAE)

### 1.1 Key discovery: `vae_streaming.py`

The codebase already has a streaming VAE module (`wan_2_2/modules/vae_streaming.py`) with:

- **`WanVAE_.stream_decode(z, scale)`** (line 612): Maintains persistent decoder feature caches (`self._feat_map`) across calls. First call initializes caches; subsequent calls decode incrementally using cached conv states.
- **`WanVAE_.clear_cache_decode()`** (line 673): Resets decoder caches for a new sequence.
- **`WanVAE.stream_decode(zs)`** (line 768): Public wrapper with `@conditional_compile`.

The `CausalConv3d` layers cache the last `CACHE_T=2` temporal frames, enabling incremental decode without re-processing earlier frames.

### 1.2 Proven pattern from `causal_s2v_pipeline_tpp_blockwise.py`

Lines 995-1012 show per-block VAE decode:

```python
# Prime decoder with motion context (first block of first clip only)
if r == 0 and block_index == 0:
    decode_latents = motion_latents[:, :, :7]
    self.vae.stream_decode(decode_latents)  # warms up caches, output discarded

# Decode this block's latents incrementally
decode_latents = block_latents.unsqueeze(0)  # [1, 16, nfpb, h, w]
image = torch.stack(self.vae.stream_decode(decode_latents))

# Extract new frames
image = image[:, :, -(infer_frames) // num_blocks:]
if r == 0 and block_index == 0:
    image = image[:, :, 3:]  # skip reference padding frames

yield image.cpu()
```

### 1.3 Our implementation

Adapt the above pattern into our sequential denoising loop. The change is minimal:

```python
# BEFORE (current — deferred decode at clip end):
for block_index in range(num_blocks):
    # ... 4-step denoising ...
    clip_output[:, block_index * nfpb:...] = block_latents
# ... after all blocks, decode entire clip_output at once ...

# AFTER (streaming — decode after each block):
for block_index in range(num_blocks):
    # ... 4-step denoising ...
    # Immediately decode this block
    image_block = vae.stream_decode(block_latents.unsqueeze(0))
    yield image_block  # or push to frame_queue
```

### 1.4 Priming the streaming decoder

Before the block loop, the VAE decoder must be primed with motion context:

```python
# Reset decoder caches for new clip
self.vae.model.clear_cache_decode()
self.vae.model.first_decode = True

# Prime with motion latents (first clip only — subsequent clips continue the cache)
if r == 0:
    prime_latents = motion_latents[:, :, :lat_motion_frames]
    self.vae.stream_decode(prime_latents)  # output discarded, caches warmed
```

For subsequent clips (r >= 1), the decoder caches carry over from the previous clip's last block, providing temporal continuity.

### 1.5 Frame extraction per block

Each block produces `nfpb` latent frames → decoded to `nfpb * 4` pixel frames (VAE temporal upsample factor depends on `temperal_upsample` config; for the s2v-14B config with `[False, True, True]`, temporal upsample is 4x at layers 1+2).

Wait — checking the config: `temperal_downsample=[False, True, True]` means encoder downsamples at layers 1 and 2 (2×2 = 4× total). Decoder upsamples correspondingly. So 1 latent frame → 4 pixel frames. With `nfpb=3` latent frames per block → 12 pixel frames per block.

```python
# Per-block frame count
frames_per_block = nfpb * 4  # = 3 * 4 = 12

# First block of first clip: skip 3 padding frames → 9 frames
# All other blocks: 12 frames
```

This matches `causal_s2v_pipeline_tpp_blockwise.py`'s `image[:, :, -(infer_frames)//num_blocks:]`.

### 1.6 Motion latents update at clip boundary

At the end of each clip, the last decoded frames become the motion context for the next clip. With streaming decode, we accumulate pixel frames across blocks:

```python
# After all blocks in a clip:
# Collect the last motion_frames pixel frames from decoded output
# Re-encode them as motion_latents for the next clip
overlap = min(self.motion_frames, total_decoded_frames)
videos_last_frames = ... last `overlap` decoded pixel frames ...
motion_latents = torch.stack(self.vae.encode(videos_last_frames))
```

### 1.7 Import: use streaming VAE

```python
from .wan_2_2.modules.vae_streaming import WanVAE as Wan2_1_VAE
```

Instead of the regular `vae2_1.Wan2_1_VAE`. The streaming VAE has identical `encode()`/`decode()` plus `stream_encode()`/`stream_decode()`.

### 1.8 Queue abstraction (for future 2-GPU)

Even though single-GPU is synchronous, we structure the code with a queue interface:

```python
class LatentQueue:
    """Thin wrapper — swap to ZMQ/mp.Queue for 2-GPU."""
    def __init__(self, maxsize=4):
        self._q = queue.Queue(maxsize=maxsize)

    def put(self, item):   self._q.put(item)
    def get(self):         return self._q.get()
```

On single-GPU, the producer (DiT) puts a block latent, then immediately the consumer (VAE) gets it — synchronous. On 2-GPU, the queue would be inter-process.

---

## 2. Streaming Audio

### 2.1 Current flow (upfront)

```python
# All audio loaded and encoded before generation:
audio_emb, nr = self.encode_audio(audio_path, infer_frames)
# audio_emb: [1, 25, 1024, total_frames]  — entire file encoded at once
```

### 2.2 Proven pattern from `causal_s2v_pipeline_tpp_blockwise.py`

Lines 485-488:

```python
def _streaming_encode_next_audio_block_or_random(self, block_frames):
    chunk = self.get_audio_callback()  # callback returns raw audio numpy array
    audio_embed, _ = self.encode_audio_from_array(chunk, infer_frames=block_frames)
    return audio_embed[..., :block_frames].contiguous()
```

This encodes audio on-demand per block via a callback. We adapt this for per-clip encoding.

### 2.3 Our implementation — per-clip encoding

For streaming, we don't need the full audio file upfront. We encode one clip's audio at a time:

```python
class AudioChunkLoader:
    """Yields clip-sized raw audio chunks from a file or stream."""

    def __init__(self, audio_path, sample_rate=16000, fps=25, infer_frames=48):
        import librosa
        self.waveform, _ = librosa.load(audio_path, sr=sample_rate)
        self.samples_per_clip = int(infer_frames / fps * sample_rate)
        self.clip_idx = 0

    def next_clip(self):
        """Return next clip's raw waveform, or None if exhausted."""
        start = self.clip_idx * self.samples_per_clip
        if start >= len(self.waveform):
            return None
        end = start + self.samples_per_clip
        chunk = self.waveform[start:end]
        if len(chunk) < self.samples_per_clip:
            chunk = np.pad(chunk, (0, self.samples_per_clip - len(chunk)))
        self.clip_idx += 1
        return chunk

    @property
    def num_clips(self):
        return int(np.ceil(len(self.waveform) / self.samples_per_clip))
```

### 2.4 AudioEncoder addition: `extract_audio_feat_from_array()`

The existing `AudioEncoder.extract_audio_feat()` takes a file path. We need a variant that accepts a numpy array:

```python
# In audio_encoder.py:
def extract_audio_feat_from_array(self, audio_array, sr=16000, return_all_layers=False):
    """Same as extract_audio_feat but takes a numpy array instead of file path."""
    # Skip librosa.load(), use the provided array directly
    input_values = self.processor(audio_array, sampling_rate=sr, return_tensors="pt")
    # ... rest identical to extract_audio_feat ...
```

### 2.5 Per-clip encoding in generate()

```python
audio_loader = AudioChunkLoader(audio_path, fps=self.fps, infer_frames=infer_frames)

for r in range(num_clips):
    raw_chunk = audio_loader.next_clip()
    if raw_chunk is None:
        break

    # Encode this clip's audio on demand
    z = self.audio_encoder.extract_audio_feat_from_array(raw_chunk, return_all_layers=True)
    clip_audio_emb, _ = self.audio_encoder.get_audio_embed_bucket_fps(
        z, fps=self.fps, batch_frames=infer_frames, m=self.audio_sample_m)
    clip_audio_emb = clip_audio_emb.to(self.device, self.param_dtype).unsqueeze(0)
    if len(clip_audio_emb.shape) == 4:
        clip_audio_emb = clip_audio_emb.permute(0, 2, 3, 1)

    # Per-block slicing as before:
    for block_index in range(num_blocks):
        la = block_index * (nfpb * 4)
        ra = (block_index + 1) * (nfpb * 4)
        block_audio = clip_audio_emb[..., la:ra]
        # ... DiT forward ...
```

### 2.6 Compatibility

When an audio file path is provided (non-streaming), we still use `AudioChunkLoader` — it loads the file once and yields clip-sized chunks. The downstream code is identical.

For future real-time streaming (microphone input), the `AudioChunkLoader` can be replaced with a callback that reads from a live audio buffer.

---

## 3. New File: `causal_s2v_pipeline_2gpu_streaming.py`

### 3.1 Class hierarchy

```python
class WanS2VStreaming(WanS2V):
    """Streaming pipeline: per-block VAE decode + on-demand audio encoding.

    Inherits model loading, LoRA, prefill from WanS2V.
    Overrides generate() with streaming loop.
    Uses vae_streaming.WanVAE instead of vae2_1.Wan2_1_VAE.
    """
```

### 3.2 Key differences from parent

| Aspect | `WanS2V` (current) | `WanS2VStreaming` (new) |
|--------|--------------------|-----------------------|
| VAE module | `vae2_1.Wan2_1_VAE` | `vae_streaming.WanVAE` |
| Audio encoding | Upfront (entire file) | Per-clip on-demand |
| VAE decode | Deferred (after all blocks) | Per-block streaming |
| Output | Returns full video tensor | Yields per-block frames (generator) |
| Motion update | At clip end from full decode | At clip end from accumulated frames |

### 3.3 generate() as a generator

```python
def generate(self, ...):
    """Yields (image_chunk, metadata) per block."""
    ...
    for r in range(num_clips):
        for block_index in range(num_blocks):
            # DiT denoising (4 steps)
            ...
            # VAE stream decode
            image_block = self._stream_decode_block(block_latents, r, block_index)
            yield image_block, {"clip": r, "block": block_index}

        # Clip boundary: update motion_latents
        self._update_motion_latents(...)
```

The caller consumes frames as they're produced:

```python
for frames, meta in pipeline.generate(...):
    # Write frames, display, or push to output queue
    output_frames.append(frames)
```

### 3.4 Methods to add/override

```python
class WanS2VStreaming(WanS2V):
    def __init__(self, ...):
        super().__init__(...)
        # Replace VAE with streaming version
        self.vae = Wan2_1_VAE_Streaming(...)

    def generate(self, ...):
        # Override: streaming generation loop (generator)
        ...

    def _stream_decode_block(self, block_latents, clip_idx, block_idx):
        """Decode one block using streaming VAE."""
        ...

    def _prime_stream_decoder(self, motion_latents):
        """Reset and prime the streaming VAE decoder."""
        ...

    def _encode_clip_audio(self, raw_audio_chunk):
        """Encode one clip's raw audio → embedding tensor."""
        ...
```

---

## 4. Implementation Order

### Step 1: Audio streaming
1. Add `extract_audio_feat_from_array()` to `AudioEncoder` (small addition)
2. Implement `AudioChunkLoader` in the new file
3. Wire per-clip audio encoding into `generate()`
4. **Test**: verify output matches non-streaming (same audio embeddings)

### Step 2: Streaming VAE decode
1. Switch to `vae_streaming.WanVAE` in `__init__`
2. Add `_prime_stream_decoder()` — reset caches, prime with motion latents
3. Add `_stream_decode_block()` — call `stream_decode()` per block, extract new frames
4. Convert `generate()` to yield per-block frames
5. Handle motion_latents update at clip boundary (accumulate decoded frames, re-encode)
6. **Test**: verify per-block frames concatenated == non-streaming full decode

### Step 3: Inference script
1. Create `minimal_inference/s2v_2gpu_streaming.py`
2. Consume generator, concatenate frames, save video
3. Add `--streaming` flag

---

## 5. Inference Script: `minimal_inference/s2v_2gpu_streaming.py`

```python
from liveavatar.models.wan.causal_s2v_pipeline_2gpu_streaming import WanS2VStreaming

pipeline = WanS2VStreaming(config=cfg, checkpoint_dir=args.ckpt_dir, ...)

all_frames = []
for frames, meta in pipeline.generate(...):
    all_frames.append(frames)
    print(f"clip {meta['clip']} block {meta['block']}: {frames.shape[2]} frames")

video = torch.cat(all_frames, dim=2)
save_video(video[None], ...)
```

---

## 6. Testing

1. **Audio correctness**: Compare `AudioChunkLoader` per-clip output vs upfront `encode_audio()` — embeddings should be identical (same wav2vec model, same chunking)
2. **VAE correctness**: Compare concatenated stream_decode output vs single decode — should be identical (stream_decode caches guarantee this)
3. **End-to-end**: Full pipeline output comparison with `compare_videos.py`
4. **Latency**: Measure time-to-first-frame (should be ~1 block denoising + 1 block decode ≈ 1-2s instead of full clip ≈ 10s)
