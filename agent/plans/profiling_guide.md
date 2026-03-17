# LiveAvatar Inference Profiling Guide

## Quick Start

### Single GPU (1x H100, 80GB)

```bash
bash profile_inference.sh
```

### Multi-GPU (5x H800/H100, TPP pipeline)

```bash
bash profile_inference_multi_gpu.sh
```

Both scripts disable `torch.compile` for accurate profiling and limit to 2 clips.

---

## What Gets Profiled

Every time-taking step is measured for **wall-clock latency** and **peak VRAM**:

| Phase | Category | Description |
|-------|----------|-------------|
| `pipeline_init` | init | Load T5, VAE, DiT, Audio models from disk |
| `lora_loading` | init | Apply LoRA adapter to DiT |
| `fp8_quantization` | init | Replace linear layers with FP8 |
| `audio_encoding` | encode | Wav2Vec2 feature extraction |
| `text_encoding` | encode | T5-XXL forward pass |
| `vae_encode_ref_image` | vae | Encode reference frame to latent |
| `vae_encode_motion_init` | vae | Encode motion frames to latent |
| `pose_cond_loading` | encode | Pose conditioning + VAE encode |
| `offload_dit_to_gpu` | offload | Move DiT to GPU, others to CPU |
| `kv_cache_init` | cache | Allocate KV cache tensors |
| `prefill_cond_cache_clip{N}` | cache | Warm up condition KV cache |
| `kv_cache_to_gpu_clip{N}_block{B}_step{S}` | cache | Move KV cache to working GPU |
| `dit_forward_clip{N}_block{B}_step{S}` | dit | 14B parameter DiT forward pass |
| `kv_cache_offload_clip{N}_block{B}_step{S}` | cache | Move KV cache back |
| `scheduler_step_clip{N}_block{B}_step{S}` | scheduler | Euler denoising step |
| `vae_decode_*_clip{N}` | vae | Latent-to-pixel decode |
| `vae_encode_motion_clip{N}` | vae | Re-encode last frames for next clip |
| `video_save` | io | Write MP4 to disk |
| `audio_merge` | io | Merge audio track into video |

**Multi-GPU extras**: `dist_recv_*`, `dist_send_*`, `vae_stream_decode_*`, `vae_recv_latents_*`

---

## CLI Arguments

Add these to any existing inference command:

```bash
--enable_profiling              # Enable VRAM + latency profiling
--profile_output_dir DIR        # Output directory (default: profiling_output)
--profile_num_clips N           # Limit clips profiled (default: 2)
--torch_trace                   # Export Chrome traces + memory snapshots
```

Example — add profiling to your existing script:

```bash
bash infinite_inference_single_gpu.sh \
    --enable_profiling \
    --profile_output_dir my_profiling \
    --profile_num_clips 3 \
    --torch_trace
```

> **Note**: Set `export ENABLE_COMPILE=false` for accurate per-kernel profiling. With `torch.compile` enabled, fused kernels make traces harder to interpret.

---

## Output Files

```
profiling_output/
├── vram_profile.json                  # All phase records (JSON)
├── dit_block0_step0_trace.json        # Chrome trace: DiT step 0
├── dit_block0_step1_trace.json        # Chrome trace: DiT step 1
├── dit_block0_step2_trace.json        # Chrome trace: DiT step 2
├── dit_block0_step3_trace.json        # Chrome trace: DiT step 3
├── vae_decode_online_trace.json       # Chrome trace: VAE online decode
├── vae_decode_deferred_trace.json     # Chrome trace: VAE deferred decode
├── dit_memory_snapshot.pickle         # PyTorch memory snapshot
├── chart_timeline.png                 # VRAM peak over wall-clock time
├── chart_latency_waterfall.png        # All phases sorted by duration
├── chart_category_pie.png             # Time breakdown by category
├── chart_dit_detail.png               # Per-block per-step DiT VRAM + latency
├── chart_memory_timeline.png          # Allocated/reserved/peak across phases
├── chart_component_comparison.png     # Component comparison bars
└── profiling_report.html              # Combined HTML report with all charts
```

Multi-GPU mode produces rank-specific files: `vram_profile_rank0.json`, `vram_profile_rank1.json`, etc.

---

## How to View Results

### 1. HTML Report (recommended)
```bash
open profiling_output/profiling_report.html
# or
python -m http.server 8000 --directory profiling_output
```

### 2. Chrome Traces (kernel-level detail)
Open in https://ui.perfetto.dev or `chrome://tracing`:
- `dit_block0_step*_trace.json` — per-kernel GPU timing for DiT
- `vae_decode_*_trace.json` — per-kernel GPU timing for VAE

### 3. Memory Snapshot (allocation timeline)
Upload `dit_memory_snapshot.pickle` to https://pytorch.org/memory_viz
Shows exactly which tensors consume VRAM during a DiT forward pass.

### 4. TensorBoard
```bash
tensorboard --logdir=profiling_output
```

### 5. Re-generate Charts
```bash
python minimal_inference/visualize_profile.py profiling_output/vram_profile.json
```

---

## Console Output Example

```
[PROFILE] pipeline_init: 45231.2ms | peak=28500MB | delta=28500MB
[PROFILE] lora_loading: 3200.5ms | peak=29800MB | delta=1300MB
[PROFILE] fp8_quantization: 890.3ms | peak=29900MB | delta=100MB
[PROFILE] audio_encoding: 1560.7ms | peak=12400MB | delta=2400MB
[PROFILE] text_encoding: 2100.1ms | peak=15600MB | delta=3200MB
[PROFILE] vae_encode_ref_image: 450.2ms | peak=11200MB | delta=1200MB
[PROFILE] offload_dit_to_gpu: 5600.8ms | peak=32000MB | delta=20000MB
[PROFILE] kv_cache_init: 234.5ms | peak=40200MB | delta=8200MB
[PROFILE] dit_forward_clip0_block0_step0: 892.3ms | peak=45200MB | delta=5000MB
...
[PROFILE] vae_decode_deferred_clip0: 3567.8ms | peak=23400MB | delta=12400MB

====================================================================================================
Phase                                              Time(ms)   Peak(MB)  Delta(MB)  Category
----------------------------------------------------------------------------------------------------
pipeline_init                                       45231.2      28500      28500  init
...
TOTAL                                               98542.1
====================================================================================================

Category         Count  Total Time(ms)  Max Peak(MB)     % Time
-----------------------------------------------------------------
dit                 32        28500.0          45200      28.9%
vae                  8        12400.0          23400      12.6%
init                 3        49321.0          29900      50.0%
...
```

---

## Architecture Notes

### Single GPU Pipeline
```
Audio → Wav2Vec2 → features
Image → VAE encode → ref latents
Text  → T5-XXL → context

For each clip:
  DiT → GPU, VAE → CPU
  For each block (4 blocks):
    For each step (4 steps):
      KV cache → GPU
      DiT forward (14B params)
      KV cache → offload
      Scheduler step
  DiT → CPU, VAE → GPU
  VAE decode → pixels
```

### Multi-GPU TPP Pipeline (5 GPUs)
```
GPU 0-3: DiT pipeline parallelism (1 timestep each)
GPU 4:   VAE streaming decode

For each clip:
  For each block:
    GPU 0: noise → DiT step 0 → send to GPU 1
    GPU 1: recv → DiT step 1 → send to GPU 2
    GPU 2: recv → DiT step 2 → send to GPU 3
    GPU 3: recv → DiT step 3 → send to GPU 4
    GPU 4: recv → streaming VAE decode → pixels
```
