# Batched TPP on 2× H100: Feasibility Analysis

> **Proposal**: Replace the 5-GPU TPP pipeline (4 DiT + 1 VAE) with a 2-GPU setup where GPU 0 runs all 4 diffusion timesteps as a **batch_size=4** forward pass on shared model weights, and GPU 1 runs VAE decode.

---

## 1. The Idea

### Current TPP (5 GPUs)

```
GPU 0: DiT step 0 (t=750)  ──send──►  GPU 1: DiT step 1 (t=500)  ──send──►  ...  ──send──►  GPU 4: VAE decode
        14B weights                           14B weights                                      VAE weights
        1× KV cache                           1× KV cache
```

- **4 copies** of 14B model weights across 4 GPUs
- Each GPU holds 1 KV cache and processes 1 timestep
- Latents passed via `dist.send/recv` between GPUs
- Total VRAM: ~42 GB × 4 + ~13 GB = **181 GB**

### Proposed Batched TPP (2 GPUs)

```
GPU 0: DiT batch=4 (all 4 timesteps simultaneously)  ──►  GPU 1: VAE decode
       1× 14B weights (shared across batch)
       4× KV cache (one per timestep, as batch dim)
```

- **1 copy** of model weights, shared across batch dimension
- 4 KV caches stored as batch dimension `[4, seq, heads, dim]`
- No inter-GPU communication for DiT (all local)
- Total VRAM: ~61 GB + ~13 GB = **74 GB**

### Why It Works

The 4 diffusion timesteps in TPP operate on **different blocks at different stages**:

| Batch slot | Block | Timestep | Input |
|------------|-------|----------|-------|
| 0 | B | t₀ (noise) | Fresh noise for block B |
| 1 | B-1 | t₁ | Output of slot 0 from previous iteration |
| 2 | B-2 | t₂ | Output of slot 1 from previous iteration |
| 3 | B-3 | t₃ | Output of slot 2 from previous iteration |

Each batch element has:
- A **different timestep embedding** (t₀, t₁, t₂, t₃)
- A **different input latent** (different noise levels)
- Its **own KV cache** (accumulated from different blocks)
- The **same model weights, text conditioning, and audio features**

The forward pass is identical for all 4 — only the inputs differ. This is a textbook batching opportunity.

---

## 2. Memory Analysis

### 2.1 Model Weights

| Component | Parameters | BF16 | FP8 |
|-----------|-----------|------|-----|
| 40× Transformer blocks | ~14.06B | 26.24 GB | 13.12 GB |
| Audio encoder + injector | ~1.31B | 2.44 GB | 1.22 GB |
| Patch/text/time embeddings | ~0.23B | 0.43 GB | kept BF16 |
| **Total** | **~15.6B** | **29.11 GB** | **~15.6 GB** |

FP8 quantization (via `fp8_linear.py`) replaces `nn.Linear` weights with `float8_e4m3fn` and **deletes BF16 copies** — true 50% savings. Layers excluded from FP8: `text_embedding`, `time_embedding`, `time_projection`, `head.head`, `casual_audio_encoder.encoder.final_linear` (~0.5B params, ~1 GB BF16).

### 2.2 KV Cache (The Critical Component)

**Resolution: 704×384** (single-GPU default)

Latent space: 88×48 (÷8 VAE stride). After patch embedding (1,2,2): 44×24 spatial tokens.

```
lat_target_frames = (48 + 3 + 73) // 4 - 19 = 12 latent frames
tokens_per_frame = 44 × 24 = 1,056
tokens_per_block = 3 × 1,056 = 3,168  (num_frames_per_block=3)
max_seq_len = 12 × 1,056 = 12,672     (full clip capacity)
```

**Per-layer KV cache allocation:**

| Tensor | Shape (batch=4) | BF16 bytes | MB |
|--------|----------------|------------|-----|
| Self-attn k | `[4, 12672, 40, 128]` | 519,045,120 | 494.9 |
| Self-attn v | `[4, 12672, 40, 128]` | 519,045,120 | 494.9 |
| **Per layer** | | | **989.9** |
| **40 layers** | | | **38,674** |

**Shared cond cache** (text/ref conditioning — same for all timesteps):

| Tensor | Shape (batch=1) | BF16 bytes | MB |
|--------|----------------|------------|-----|
| cond_k | `[1, 2800, 40, 128]` | 28,672,000 | 27.3 |
| cond_v | `[1, 2800, 40, 128]` | 28,672,000 | 27.3 |
| **Per layer** | | | **54.7** |
| **40 layers** | | | **2,187** |

**Cross-attention cache** (audio): ~0.4 GB (40 layers, grows dynamically)

| KV Component | Memory |
|-------------|--------|
| Self-attn KV (4× batch) | **38.67 GB** |
| Shared cond cache (1× batch) | **2.14 GB** |
| Cross-attn cache | **~0.4 GB** |
| **Total KV** | **~41.2 GB** |

> **Critical insight**: The current single-GPU code *already* allocates 4 KV caches on `cuda:0` (one per timestep, line 1021-1028 in `causal_s2v_pipeline.py`). The memory footprint is **identical** — we're just restructuring from 4×[1,...] to 1×[4,...].

### 2.3 Peak Activations (Per Block, batch=4)

During DiT forward pass on one block (3,168 tokens × batch 4):

| Tensor | Shape | BF16 MB |
|--------|-------|---------|
| Input x | `[4, 3168, 5120]` | 123.7 |
| q, k, v (each) | `[4, 3168, 40, 128]` | 62.3 × 3 |
| Flash Attention output | `[4, 3168, 5120]` | 123.7 |
| FFN intermediate | `[4, 3168, 13824]` | 333.7 |
| Residuals, norms | ~100 |
| **Peak per block** | | **~870** |

Note: Flash Attention avoids materializing the full `[4, 40, 3168, 12672]` attention matrix. Peak memory is dominated by the FFN intermediate.

### 2.4 Other Tensors

| Item | Size |
|------|------|
| Audio embeddings | ~100 MB |
| Ref latents, motion latents, cond_states | ~200 MB |
| Clip noise `[4, 16, 12, 48, 88]` | ~50 MB |
| Scheduler state | ~50 MB |
| PyTorch allocator overhead (~5%) | ~3 GB |
| **Total** | **~3.4 GB** |

### 2.5 GPU 0 (DiT) Total

| Component | FP8 Weights | BF16 Weights |
|-----------|-------------|--------------|
| Model weights | 15.6 GB | 29.1 GB |
| KV cache (batch=4) | 41.2 GB | 41.2 GB |
| Peak activations | 0.9 GB | 0.9 GB |
| Other + allocator | 3.4 GB | 3.4 GB |
| **Total** | **61.1 GB** | **74.6 GB** |
| H100 headroom | **18.9 GB** | **5.4 GB** |
| **Verdict** | **Fits comfortably** | **Fits, but tight** |

### 2.6 GPU 1 (VAE + Preprocessing) Total

| Phase | Component | Memory |
|-------|-----------|--------|
| Preprocessing | T5-XXL encoder | 10.9 GB |
| Preprocessing | Audio encoder (Wav2Vec2) | 0.6 GB |
| Preprocessing | VAE (for encode) | 1.1 GB |
| **Preprocessing peak** | | **~12.6 GB** |
| Decode | VAE model | 1.1 GB |
| Decode | Full-clip decode intermediates | 10.3 GB |
| Decode | Feat cache (causal) | 1.2 GB |
| **Decode peak** | | **~12.6 GB** |

T5 and audio encoder are offloaded after preprocessing, before decode begins.

**GPU 1 peak: ~12.6 GB** — fits trivially on H100 with 67 GB headroom.

### 2.7 Resolution Comparison

| Resolution | max_seq_len | KV Cache (4×) | GPU 0 Total (FP8) |
|------------|-------------|---------------|-------------------|
| 576×320 | 8,640 | 28.0 GB | 47.9 GB |
| **704×384** | **12,672** | **41.2 GB** | **61.1 GB** |
| 720×400 | 13,500 | 43.9 GB | 63.8 GB |
| 832×480 | 19,920 | 64.7 GB | 84.6 GB ❌ |

832×480 exceeds H100 80 GB even with FP8.

---

## 3. Comparison: Current vs Proposed

| Metric | Single-GPU (current) | TPP 5-GPU (current) | **Batched 2-GPU (proposed)** |
|--------|---------------------|--------------------|-----------------------------|
| GPUs | 1 | 5 | **2** |
| Total VRAM used | ~60 GB | ~181 GB | **~74 GB** |
| Model weight copies | 1 | 4 | **1** |
| KV caches on DiT GPU | 4 (sequential) | 1 per GPU | **4 (batched)** |
| DiT passes per block | 4 sequential | 4 parallel (pipelined) | **1 batched (batch=4)** |
| Inter-GPU comms (DiT) | 0 (local KV swap) | 3× dist.send/recv | **0** |
| DiT↔VAE overlap | No (offloading) | Yes (pipeline) | **Yes (2 GPUs)** |
| Expected DiT speedup | 1× | ~3.5× | **~3-4×** |
| Memory efficiency | Low (weight loaded, 3/4 KV idle) | Low (4× weight copies) | **High (shared weights)** |

### Key Advantages

1. **Same memory footprint as current single-GPU** — the 4 KV caches are already allocated; we just batch the compute
2. **~3-4× DiT throughput** — batch=4 on H100 saturates tensor cores; no sequential overhead
3. **Pipeline parallelism with VAE** — GPU 1 decodes previous block while GPU 0 denoises current block
4. **No inter-GPU DiT communication** — eliminates `dist.send/recv` latency entirely
5. **75% fewer GPUs than TPP** — 2 vs 5, massive cost savings

### Potential Risks

1. **Batch=4 compute is ~4× FLOPs per call** — single H100 must handle it. At 989 TFLOPS (BF16), a single DiT forward at batch=4 should take ~4× the batch=1 time. Pipeline parallelism hides this by overlapping with VAE.
2. **Memory fragmentation** — the 18.9 GB headroom (FP8) should handle PyTorch allocator overhead, but needs real testing.
3. **Pipeline ramp-up** — first 3 iterations have partial batches (1, 2, 3 active items). Negligible for long videos.

---

## 4. Further Optimizations (If Needed)

| Optimization | Savings | Complexity |
|-------------|---------|------------|
| **KV cache in FP8/INT8** | KV: 41.2 → 20.6 GB (saves ~20 GB) | Medium — modify attention code |
| **Dynamic KV allocation** | Saves ~30% early in clip | Medium — replace pre-allocation |
| **Streaming VAE** (existing) | VAE peak: 12.6 → 2.5 GB | Already implemented |
| **Lower resolution** (576×320) | KV: 41.2 → 28.0 GB | Just change CLI arg |
| **GQA (grouped-query attention)** | KV: 41.2 → 10.3 GB (4 groups) | High — retrain model |

With KV cache quantization alone, GPU 0 drops to ~41 GB — enough to run **batch=4 at 832×480** or even **batch=8 at 704×384**.

---

## 5. Implementation Roadmap

### Phase 1: Batched DiT Forward (Core Change)

Modify `causal_s2v_pipeline.py` to:
1. Initialize KV cache as `[4, max_seq_len, 40, 128]` instead of 4 separate `[1, ...]`
2. Stack the 4 block latents (at different pipeline stages) into batch dim
3. Stack the 4 timestep values into a batch
4. Call `self.noise_model()` once with batch=4
5. Unstack outputs and route to the pipeline shift register

**Key files**: `causal_s2v_pipeline.py` (denoising loop), `causal_model_s2v.py` (forward pass)

### Phase 2: 2-GPU Pipeline

1. GPU 0: DiT (batched) — processes blocks in steady-state pipeline
2. GPU 1: VAE decode + preprocessing — decodes completed blocks as they arrive
3. Single `dist.send/recv` between GPU 0→1 for decoded block latents
4. Overlap DiT and VAE compute

**Key files**: New pipeline file (e.g., `causal_s2v_pipeline_batched.py`)

### Phase 3: Optimization

1. KV cache quantization (FP8/INT8) for higher resolution support
2. Dynamic KV allocation to reduce peak memory
3. Profiling and tuning batch efficiency

---

## 6. Conclusion

| Question | Answer |
|----------|--------|
| Does it work theoretically? | **Yes** — identical to TPP but batched instead of distributed |
| Does it fit on 2× H100 (80 GB)? | **Yes with FP8** — 61.1 GB DiT + 12.6 GB VAE |
| Does it fit without FP8? | **Barely** — 74.6 GB, risky with fragmentation |
| Expected performance vs 5-GPU TPP? | **Comparable** — ~3-4× DiT speedup + pipeline overlap |
| Expected performance vs 1-GPU? | **~3-4× faster** — batched compute + no model offloading |
| Implementation effort | **Medium** — mainly pipeline restructuring, model code unchanged |
| Cost savings vs 5-GPU TPP | **60% fewer GPUs** (2 vs 5), **59% less VRAM** |
