# Plan: Comprehensive VRAM & Latency Profiling for LiveAvatar Inference

## Context

Profile **every time-taking step** in the LiveAvatar inference pipeline on both single-GPU and multi-GPU (5x TPP) configurations. Measure per-component VRAM consumption and latency using `torch.profiler`, custom VRAM tracking, and generate rich visualizations.

---

## Files Created/Modified

| File | Action | Purpose |
|------|--------|---------|
| `minimal_inference/profiler_utils.py` | **Created** | `VRAMProfiler` class — tracks peak VRAM, allocated/reserved memory, wall-clock time per phase with category support |
| `minimal_inference/visualize_profile.py` | **Created** | 6 matplotlib charts + HTML report generator |
| `profile_inference.sh` | **Created** | Single-GPU profiling launcher (`ENABLE_COMPILE=false`, 2 clips) |
| `profile_inference_multi_gpu.sh` | **Created** | Multi-GPU (5x) profiling launcher |
| `liveavatar/models/wan/causal_s2v_pipeline.py` | **Modified** | Added `profiler=`, `torch_trace=`, `profile_output_dir=` to `generate()`, ~20 instrumentation points |
| `liveavatar/models/wan/causal_s2v_pipeline_tpp.py` | **Modified** | Same profiler params, ~14 instrumentation points including `dist.send/recv` and streaming VAE |
| `minimal_inference/s2v_streaming_interact.py` | **Modified** | 4 new CLI args, wraps A1-A3 (init/LoRA/FP8), rank-aware output, auto-visualization |

---

## Profiled Phases

### Phase A: Initialization (entry point)
- A1: Pipeline init (T5 + VAE + DiT + Audio model loading)
- A2: LoRA loading
- A3: FP8 quantization

### Phase B: Conditional Input Preparation (pipeline)
- B1: Audio encoding (Wav2Vec2) + offload
- B2: Text encoding (T5-XXL)
- B3: VAE encode ref image
- B4: VAE encode motion latents
- B5: Pose cond loading

### Phase C: Per-Clip Generation (pipeline)
- C1: Model offload (DiT→GPU, others→CPU)
- C2: KV cache init
- C3: Prefill cond cache
- C4: KV cache move to GPU (single-GPU) / dist.recv (multi-GPU)
- C5: DiT forward pass
- C6: KV cache offload (single-GPU) / dist.send (multi-GPU)
- C7: Scheduler step
- C8: Model offload for decode
- C9: VAE decode (full-clip single-GPU / streaming multi-GPU)
- C10: VAE encode motion (for next clip)
- C11: Video save + audio merge

---

## Profiling Levels

1. **VRAMProfiler** — every phase: peak VRAM + wall time → JSON
2. **torch.profiler Chrome traces** — first clip DiT steps + first VAE decode → `*_trace.json`
3. **Memory snapshot** — first DiT forward → `dit_memory_snapshot.pickle` (pytorch.org/memory_viz)
4. **Custom charts** — 6 matplotlib charts + HTML report

---

## Bugs Found & Fixed

1. **Redundant `import os as _os`** inside conditional blocks — replaced with module-level `os`
2. **VAE trace filename collision** — online and deferred both wrote `vae_decode_trace.json` → renamed to `vae_decode_online_trace.json` / `vae_decode_deferred_trace.json`
3. **TPP `t=None` placeholder + duplicated recv/send logic** — refactored to compute timestep and do recv/send outside the profiler branch, ensuring identical control flow with/without profiling

---

## Verification

1. `bash profile_inference.sh` — single-GPU profiling
2. `bash profile_inference_multi_gpu.sh` — multi-GPU profiling
3. Check `profiling_output/vram_profile.json` has records for all phases
4. Check `profiling_output/profiling_report.html` renders all charts
5. Chrome traces loadable in Perfetto UI
6. Output video unchanged with/without profiling
