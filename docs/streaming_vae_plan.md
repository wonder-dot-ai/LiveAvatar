# Plan: Add streaming VAE decode to blockwise pipeline

## Context

Our blockwise pipeline currently uses **deferred VAE decode** — all blocks are generated first, then decoded in clip-sized chunks. Each chunk requires prepending `motion_latents` and re-encoding motion between chunks, which is slow and adds complexity.

The original TPP pipelines use a **streaming VAE** that decodes per-block by maintaining causal conv caches between blocks. This eliminates the need for motion prepending and re-encoding, since the VAE's temporal context is carried forward in the cache.

## How streaming VAE works

The VAE decoder uses `CausalConv3d` layers that cache the last `CACHE_T=2` frames from each layer's output. On the next block, these cached frames provide temporal context for the causal convolutions, ensuring smooth output at block boundaries.

**API:**
```python
# First call: initialize cache, warm up with initial frames
vae.model.first_decode = True
warmup_latents = motion_latents[:, :, :7]  # first 7 latent frames
vae.stream_decode(warmup_latents)          # populates cache, output discarded

# Per-block: decode with persistent cache
image = vae.stream_decode(block_latents)   # uses cache from previous block
```

**Key differences from regular decode:**
- `clear_cache()` called only once at first decode (not before/after each call)
- `feat_cache` persists between calls → temporal continuity
- No motion prepending or re-encoding needed

## Critical files

- `liveavatar_nari/modules/vae2_1.py` — add `stream_decode` to both inner `WanVAE_` and outer `Wan2_1_VAE`
- `liveavatar_nari/pipeline.py` — replace chunked decode with per-block streaming decode

## Changes

### 1. Add `stream_decode` to `WanVAE_` (inner model)

Add `self.first_decode = True` flag and `stream_decode` method:

```python
def stream_decode(self, z, scale):
    """Decode with persistent causal conv cache across calls."""
    if isinstance(scale[0], torch.Tensor):
        z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(1, self.z_dim, 1, 1, 1)
    else:
        z = z / scale[1] + scale[0]
    t = z.shape[2]
    x = self.conv2(z)

    if self.first_decode:
        self.first_decode = False
        # Initialize decoder cache (not encoder)
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        # First frame
        self._conv_idx = [0]
        out = self.decoder(x[:, :, :1], feat_cache=self._feat_map, feat_idx=self._conv_idx)
        # Remaining frames in first block
        for i in range(1, t):
            self._conv_idx = [0]
            out_ = self.decoder(x[:, :, i:i+1], feat_cache=self._feat_map, feat_idx=self._conv_idx)
            out = torch.cat([out, out_], 2)
    else:
        # Subsequent blocks: persistent cache from previous call
        out_parts = []
        for i in range(t):
            self._conv_idx = [0]
            out_parts.append(self.decoder(x[:, :, i:i+1], feat_cache=self._feat_map, feat_idx=self._conv_idx))
        out = torch.cat(out_parts, 2)
    return out
```

Also add `clear_cache_decode` (decoder-only cache clear, doesn't touch encoder):
```python
def clear_cache_decode(self):
    self._conv_num = count_conv3d(self.decoder)
    self._conv_idx = [0]
    self._feat_map = [None] * self._conv_num
    self.first_decode = True
```

### 2. Add `stream_decode` to `Wan2_1_VAE` (wrapper)

```python
def stream_decode(self, zs):
    """Streaming decode: maintains causal conv cache between calls."""
    with torch.amp.autocast("cuda", dtype=self.dtype):
        return [self.model.stream_decode(u.unsqueeze(0), self.scale).float().clamp_(-1, 1).squeeze(0) for u in zs]
```

### 3. Update pipeline `generate()` — replace chunked decode with streaming

Current (chunked decode):
```python
# After all DiT blocks generated:
for chunk_start in range(0, total_blocks, blocks_per_decode_chunk):
    chunk_latent = cat(all_block_latents[chunk_start:chunk_end])
    decode_input = cat([motion_latents_decode, chunk_latent])
    image = vae.decode(decode_input)
    # re-encode motion for next chunk...
```

New (streaming decode):
```python
# Interleaved: decode each block right after DiT generates it
self.vae.model.first_decode = True

# Warmup: feed initial motion latents to populate VAE cache
warmup_latents = motion_latents[:, :, :7]  # 7 frames for warmup
self.vae.stream_decode(warmup_latents)     # output discarded

for block_index in range(total_blocks):
    # ... DiT denoising (same as before) ...

    # Decode immediately after denoising
    image = torch.stack(self.vae.stream_decode([block_latents]))
    decoded_blocks.append(image.cpu())
```

### 4. Handle model offloading

With streaming decode, VAE and DiT need to coexist on GPU (or we offload between each block). Two options:

**Option A: No offloading during generation** (simpler, needs enough VRAM)
- Keep both DiT and VAE on GPU throughout
- Works if DiT (FP8) + VAE fit in 80GB

**Option B: Offload per-block** (more complex, lower VRAM)
- After each block's DiT denoising: offload DiT → load VAE → stream_decode → offload VAE → load DiT
- Slower but works on smaller GPUs
- VAE cache must persist on GPU even when model is offloaded

For initial implementation, use Option A (keep both on GPU) since H100 has 80GB and FP8 DiT + VAE should fit.

### 5. Frame trimming

With streaming decode, each block outputs `latent_frames_per_block * vae_temporal_stride` pixel frames. The first block's first few frames may have warmup artifacts (same as DECODE_SKIP_PIXEL_FRAMES=3).

```python
if block_index == 0:
    image = image[:, :, DECODE_SKIP_PIXEL_FRAMES:]
```

### 6. Remove motion re-encoding

Streaming VAE eliminates the need for:
- `motion_latents_decode` variable
- `motion_pixel_frames` update loop
- `vae.encode(motion_pixel_frames)` calls between chunks

This is a significant simplification and speedup.

## What stays the same
- DiT block loop (unchanged)
- `_prepare_image` (still needed for initial motion_latents for DiT)
- `_prefill_cond_cache`
- KV cache management
- All model loading / offloading logic (except decode phase)

## Verification
1. Syntax check: `python -c "import liveavatar_nari.modules.vae2_1; import liveavatar_nari.pipeline"`
2. Compare output quality: run with same seed, compare frames
3. Verify no discontinuities at block boundaries
4. Check VRAM usage (DiT + VAE must fit simultaneously)

## Estimated improvement
- Eliminates N-1 VAE encode calls (motion re-encoding) — saves ~10-20s for 100 blocks
- Eliminates motion prepending overhead
- Enables true streaming output (frames available as soon as each block is decoded)
