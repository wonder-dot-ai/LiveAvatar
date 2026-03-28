# H100 Inference Optimization Opportunities

Analysis of `liveavatar_nari/models/causal_model_s2v.py` for GPU inefficiencies during inference on NVIDIA H100.

**Model**: CausalWanModel_S2V (14B params, dim=5120, 40 heads, 40 layers, head_dim=128)
**Typical shapes**: seq_len≈8000 tokens, batch_size=1-4, 4 denoising steps per block

---

## HIGH impact

### 1. Separate Q, K, V projections

**Location**: `CausalWanS2VSelfAttention.forward` (line ~115)

```python
q = self.norm_q(self.q(x)).view(b, s, n, d)
k = self.norm_k(self.k(x)).view(b, s, n, d)
v = self.v(x).view(b, s, n, d)
```

Three separate `nn.Linear` projections reading the same input `x`. Each launches a separate GEMM kernel and reads the full input (dim=5120, ~40MB at seq_len=8000).

**Fix**: Fuse into a single `self.qkv = nn.Linear(dim, 3*dim)` followed by `.chunk(3, dim=-1)`. 1 kernel instead of 3, 1 read of `x` instead of 3.

**Estimated speedup**: 5-10%
**Effort**: Low
**Risk**: Requires weight migration for existing checkpoints.

---

### 2. KV-cond concatenation every block

**Location**: `CausalWanS2VSelfAttention.forward` (line ~150)

```python
k_cat = torch.cat([kv_cache["k"][:, :max_active_size], cond_k_roped], dim=1)
v_cat = torch.cat([kv_cache["v"][:, :max_active_size], kv_cache["cond_v"][:, ...]], dim=1)
```

Every block (40×) allocates new tensors and copies the KV cache + conditioning cache. For head_dim=128, 40 heads, ~8000 tokens: ~80MB per concat × 2 (k+v) × 40 blocks ≈ **6.4GB of wasted memory bandwidth per forward**.

**Fix**: Pre-allocate a combined `[B, max_kv + max_cond, H, D]` buffer. Write KV and cond into fixed offsets. Pass the unified buffer to attention — no concat needed.

**Estimated speedup**: 10-15%
**Effort**: Medium
**Risk**: Low — logic change is straightforward but touches hot path.

---

### 3. Per-batch Python loop with CUDA syncs for RoPE

**Location**: `_forward_inference` (line ~1021)

```python
for bi in range(b):
    cs_bi = int(current_start[bi].item())  # GPU→CPU sync!
    ...
    f_bi = rope_precompute(x_bi, gs_bi_shifted, self.freqs, start=None)
```

Each `.item()` forces a CUDA synchronization (pipeline stall). For batch_size=4 with 4 denoising steps, that's 16 stalls per forward. The `rope_precompute` calls are sequential when they could be batched.

**Fix**: Vectorize `rope_precompute` to accept batched grid_sizes. Transfer `current_start` to CPU once with a single `.tolist()` instead of per-element `.item()`.

**Estimated speedup**: 3-8% (depends on batch size)
**Effort**: Medium
**Risk**: Moderate — `rope_precompute` needs refactoring.

---

### 4. Redundant text embedding every denoising step

**Location**: `_embed_context` / `_forward_inference` (line ~857)

```python
context = self.text_embedding(torch.stack([...]))
```

The `text_embedding` MLP (Linear 4096→5120, GELU, Linear 5120→5120) is applied to the same T5 embeddings on every denoising step. The text prompt doesn't change between steps.

**Fix**: Compute `text_embedding(context)` once in `pipeline.py` after `encode_prompt`, pass the result through. The DiT never needs to re-embed.

**Estimated speedup**: 3-5%
**Effort**: Low
**Risk**: Low — clean separation of concerns.

---

### 5. Audio injection `.clone()` 12 times per forward

**Location**: `after_transformer_block` (line ~674)

```python
input_hidden_states = hidden_states[:, : self.original_seq_len].clone()
```

At 12 audio injection layers, each clones the full noisy sequence (dim=5120, ~8000 tokens ≈ 160MB). Total: **~1.9GB of copies per forward**.

**Fix**: Pre-allocate a scratch buffer (`self._audio_scratch`) once. Copy into it instead of allocating new memory each time. Or, restructure to avoid the clone entirely by reading from a view and writing the residual to a separate buffer before adding.

**Estimated speedup**: 5-8%
**Effort**: Medium
**Risk**: Low.

---

## MEDIUM impact

### 6. Modulation expansion with `repeat_interleave` per block

**Location**: `CausalWanS2VAttentionBlock.forward` (line ~298)

```python
for element in e:  # 6 iterations
    element_noisy = element[:, :, 0].repeat_interleave(int(frame_seqlen), dim=1)
    element_cond = element[:, 0:1, 1].repeat(1, seq_lens - ..., 1)
    element = torch.cat([element_noisy, element_cond], dim=1)
```

6 × (repeat_interleave + repeat + cat) × 40 blocks = 720 small kernel launches. `repeat_interleave` is particularly slow on GPU (not a simple view).

**Fix**: The modulation values depend only on the timestep embedding, not per-block state. Precompute the expanded modulations once before the block loop and index into them.

**Estimated speedup**: 3-5%
**Effort**: Medium
**Risk**: Moderate — e0 is passed as block kwargs, need to restructure.

---

### 7. fp32 upcasting for modulation in every block

**Location**: `CausalWanS2VAttentionBlock.forward` (lines ~313, ~330, ~346)

```python
norm_x = self.norm1(x).float()  # bf16 → fp32
with torch.amp.autocast("cuda", dtype=torch.float32):
    y = y * e[2]
    x = x + y
```

6 dtype conversions per block × 40 blocks = 240 upcasts/downcasts. On H100, bf16 compute is 2× faster than fp32.

**Fix**: Use a fused AdaLN kernel (available in xformers `fused_norm_and_linear` or write a custom Triton kernel) that does norm + scale + shift + cast in a single kernel in bf16.

**Estimated speedup**: 5-10%
**Effort**: High
**Risk**: Moderate — needs numerical validation.

---

### 8. cuDNN attention path with unnecessary transposes

**Location**: `attention.py` → `cudnn_attention_forward_with_lse` (line ~34)

```python
q = q.transpose(1, 2)  # [B,L,H,D] → [B,H,L,D]
k = k.transpose(1, 2)
v = v.transpose(1, 2)
```

The scalar inference path uses cuDNN SDPA which requires [B,H,L,D] layout. The transpose on non-contiguous tensors forces a copy. Flash Attention 3 uses [B,L,H,D] natively and is available.

**Fix**: Force the scalar path to use `flash_attention` (FA3) directly, bypassing the cuDNN fallback. The `attention()` function dispatches to cuDNN when `window_size == (-1,-1)`, which is always true in inference.

**Estimated speedup**: 2-5%
**Effort**: Low
**Risk**: Low.

---

### 9. einops `rearrange` in audio injection

**Location**: `after_transformer_block` (lines ~678, ~689, ~701, ~706, ~711)

```python
input_hidden_states = rearrange(input_hidden_states, "b (t n) c -> (b t) n c", t=num_frames)
```

`einops.rearrange` is not `torch.compile`-friendly and adds Python-level overhead per call.

**Fix**: Replace with explicit `.view()` / `.reshape()` calls.

**Estimated speedup**: 1-2%
**Effort**: Low
**Risk**: None.

---

## LOW impact (easy wins)

### 10. `deepcopy(grid_sizes)` in `_flatten_to_sequence`

```python
original_grid_sizes = deepcopy(grid_sizes)
```

Python `deepcopy` has serialization overhead. Use `grid_sizes.clone()`.

### 11. Dummy tensor for dtype casting

```python
bf_dtype_tensor = torch.zeros([1]).type_as(x)  # allocates every block
```

Replace with `x.dtype` directly: `norm_x.to(x.dtype)`.

### 12. Small tensor creation on GPU per block

```python
k_lens = torch.tensor([...], dtype=torch.int32, device=x.device)
```

Repeated `torch.tensor()` on GPU triggers `cudaMalloc`. Pre-allocate or use CUDA graphs.

---

## Summary

| Priority | Issue | Est. speedup | Effort | Risk |
|----------|-------|-------------|--------|------|
| **1** | Fused QKV projection | 5-10% | Low | Low (weight migration) |
| **2** | Eliminate KV-cond concat | 10-15% | Medium | Low |
| **3** | Cache text embeddings | 3-5% | Low | Low |
| **4** | Pre-alloc audio injection buffer | 5-8% | Medium | Low |
| **5** | Fused AdaLN kernel | 5-10% | High | Moderate |
| **6** | Vectorize per-batch RoPE | 3-8% | Medium | Moderate |
| **7** | Precompute modulation expansion | 3-5% | Medium | Moderate |
| **8** | Force FA3 in scalar attention path | 2-5% | Low | Low |
| **9** | Replace einops with view/reshape | 1-2% | Low | None |

**Estimated total if all applied: 30-50% speedup** (not additive — some overlap in memory-bandwidth-bound operations).

**Recommended first batch** (high impact, low effort): #1, #3, #8, #9 — collectively ~10-20% with minimal risk.
