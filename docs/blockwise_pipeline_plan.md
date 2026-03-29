# Plan: Convert pipeline from clip-based to blockwise generation

## Context
The current `pipeline.py` generates video in "clips" (each clip = `infer_frames` pixel frames = multiple blocks), then iterates over clips. Since motion_latents are static and the KV cache is the only inter-clip signal, the clip abstraction adds unnecessary complexity at the DiT level:
- Per-clip noise allocation for the full clip
- Per-clip `clip_output` accumulation buffer
- `clip_token_offset` calculation
- Nested clip+block loop when a flat block loop suffices

The clip concept is still useful for **VAE decoding** to avoid OOM on long videos, so we keep clip-sized chunked decode but flatten the DiT generation to a simple block loop.

## Critical files
- `liveavatar_nari/pipeline.py` — rewrite `generate()` method
- `liveavatar_nari/cli/inference.py` — update CLI args (replace `--num_clip` with `--max_blocks`)

## Current flow (clip-based)
```
for clip_index in range(total_clips):
    clip_noise = randn(latent_channels, *latent_shape)    # full clip noise
    clip_output = zeros_like(clip_noise)
    for block_index in range(num_blocks_per_clip):
        block_latents = clip_noise[:, block_start:block_end]
        for step_index, t in enumerate(timesteps):          # denoise
            noise_pred = model(block_latents, ...)
            block_latents = scheduler.step(...)
        clip_output[:, block_start:block_end] = block_latents
    clip_latent_outputs.append(clip_output)

# Deferred VAE decode per clip
for clip_latent in clip_latent_outputs:
    decode_input = cat([motion_latents_decode, clip_latent])
    image = vae.decode(decode_input)
    # re-encode motion for next clip
```

## New flow (blockwise DiT + clip-chunked VAE decode)
```
# --- DiT generation: flat block loop ---
total_blocks = total_audio_pixel_frames // audio_frames_per_block
all_block_latents = []

for block_index in range(total_blocks):
    block_noise = randn(latent_channels, fpb, latent_h, latent_w)
    audio_slice = audio_embeddings[..., audio_start:audio_end]
    token_start = block_index * tokens_per_block

    scheduler.reset()
    for step_index, t in enumerate(timesteps):
        noise_pred = model(block_noise, ..., current_start=token_start)
        block_noise = scheduler.step(...)

    all_block_latents.append(block_noise.cpu())

    if block_index == 0:  # AAS
        ref_image_latents = block_noise.unsqueeze(0)[:, :, 0:1]

# --- VAE decode: clip-sized chunks (for OOM safety) ---
blocks_per_decode_chunk = num_blocks_per_clip  # reuse old clip size as decode chunk
for chunk_start in range(0, total_blocks, blocks_per_decode_chunk):
    chunk_end = min(chunk_start + blocks_per_decode_chunk, total_blocks)
    chunk_latent = cat(all_block_latents[chunk_start:chunk_end], dim=1)
    decode_input = cat([motion_latents, chunk_latent.unsqueeze(0)], dim=2)
    image = vae.decode(decode_input)
    # trim: drop motion context frames, keep only generated frames
    decoded_chunks.append(image[..., skip:])
```

## Key changes

### 1. Compute total_blocks from audio length
Replace `total_clips * num_blocks_per_clip` with a flat `total_blocks` count:
```python
total_audio_pixel_frames = audio_embeddings.shape[-1]
audio_frames_per_block = latent_frames_per_block * self.vae_temporal_stride
total_blocks = total_audio_pixel_frames // audio_frames_per_block
```
`total_blocks` need NOT be a multiple of `blocks_per_decode_chunk`. The last decode chunk may be smaller.

### 2. Flat block loop (replaces nested clip+block loop)
- Generate noise per-block, not per-clip (smaller allocation: `[C, fpb, H, W]` vs `[C, T_clip, H, W]`)
- Audio indexing: `audio_start = block_index * audio_frames_per_block`
- Token offset: `token_start = block_index * tokens_per_block` (no clip_token_offset needed)
- Accumulate block latents in a list (moved to CPU immediately to save GPU memory)
- Single progress bar: `tqdm(range(total_blocks))` instead of nested

### 3. Clip-chunked VAE decode (for OOM safety)
- Group block latents into decode chunks of `blocks_per_decode_chunk` blocks
- Last chunk may be smaller (not required to be a multiple)
- Each chunk: concatenate block latents → prepend motion_latents → VAE decode → trim
- motion_latents stays static (no re-encoding between chunks, since the paper confirms it's static)
- `blocks_per_decode_chunk` derived from `infer_frames` parameter to keep backward compatibility

### 4. Update generate() signature
- Replace `num_clips` / `max_clips` with `max_blocks` (optional, default=None means use full audio)
- Keep `infer_frames` as the decode chunk size in pixel frames (determines `blocks_per_decode_chunk`)
- This means `infer_frames` controls VAE memory usage, not DiT behavior

### 5. KV cache sizing
- `max_tokens = total_blocks * tokens_per_block` (total across all blocks, no per-clip reset)

### 6. Keep AAS (Adaptive Attention Sink)
- `if block_index == 0: ref_image_latents = block_noise.unsqueeze(0)[:, :, 0:1]`

### 7. Seed handling
- Single generator with initial seed (blocks are sequential, deterministic)
- No per-clip seed offset needed

### 8. Remove dead code
- Remove `clip_noise`, `clip_output`, `clip_token_offset`, `clip_latent_outputs`
- Remove `input_motion_latents = motion_latents.clone()` per clip (motion is passed directly)
- Remove motion re-encoding in decode loop

## What stays the same
- `__init__`, model loading, `_prepare_image`, `_prepare_text`, `_prepare_audio`
- `prefill_cond_cache` (model method)
- `_prefill_cond_cache` (pipeline method)
- `_initialize_kv_cache`, `_initialize_crossattn_cache`
- Scheduler setup and timestep handling
- Model offloading logic
- The denoising inner loop (per-step logic within a block)
- `dummy_cond` (still needed by DiT's `cond_states` parameter)

## Verification
1. Syntax check: `python -c "import liveavatar_nari.pipeline"`
2. Run inference with same seed, verify output video quality
3. Test with short audio (few blocks) and long audio (many blocks, multiple decode chunks)
4. Verify last decode chunk handles non-multiple block count correctly
