# Plan: Prepare for CUDA Graphs (fixed image size)

## Context
With fixed image size, all tensor shapes are constant across forward calls. This is the key prerequisite for CUDA graph capture. We need to eliminate all remaining graph breaks inside the 40-block loop.

## Remaining blockers inside the block loop

### Self-attention (`forward` in CausalWanS2VSelfAttention)
| Line | Issue | Fix |
|------|-------|-----|
| 136 | `int(kv_cache["cond_end"])` — `.item()` via `int()` | Use full cond cache (shape is constant) |
| 140-146 | Python loop with `.item()` for KV cache write | Vectorize with batch indexing |
| 148 | `max(active_sizes)` — Python max on list | Use `torch.max` or attend to full cache |
| 150 | `[:active_cond_cache_size]` dynamic slice | Use full cond cache shape |
| 156 | `[:max_active_size]` dynamic slice | Use full KV cache (cuDNN attends to all anyway) |

### rope_apply / rope_apply_cond (`modules/s2v/model_s2v.py`)
| Line | Issue | Fix |
|------|-------|-----|
| 69 | `for i, _ in enumerate(x):` Python loop | Vectorize batch complex multiply |
| 87 | Same loop in rope_apply_cond | Same fix |

### Block attention block (`_expand_modulation`)
| Line | Issue | Fix |
|------|-------|-----|
| 211 | `e[1].item() if isinstance(...)` | Cache `original_seq_len` as Python int at prefill |
| 224 | `int(frame_seqlen)` in repeat_interleave | Cache as Python int at prefill |

### after_transformer_block
| Line | Issue | Fix |
|------|-------|-----|
| 576 | `block_idx in dict.keys()` | Pre-compute as set at init |
| 606-611 | `torch.ones(...) * shape` tensor creation | Pre-allocate buffer |
| 590,594,601,613 | `einops.rearrange` | Replace with `.reshape()` |

### quant_fp8 (`utils/fp8_linear.py`)
| Line | Issue | Fix |
|------|-------|-----|
| 9 | `@torch.compile(mode="max-autotune-no-cudagraphs")` | Remove — let outer compile handle |

## Implementation steps

### Step 1: Eliminate dynamic slicing in self-attention
Since cuDNN attends to ALL keys anyway, use full KV cache and full cond cache:
```python
# Before: kv_cache["k"][:, :max_active_size]
# After:  kv_cache["k"]  (full cache, zeros get negligible weight)
```
This eliminates `.item()` for `active_cond_cache_size`, the Python loop for `active_sizes`, and the `max()` call.

### Step 2: Vectorize KV cache write
```python
cs_all = current_start % kv_max  # [B] tensor
offsets = cs_all.unsqueeze(1) + torch.arange(seq_len, device=dev)
batch_idx = torch.arange(b, device=dev).unsqueeze(1).expand_as(offsets)
kv_cache["k"][batch_idx, offsets] = roped_key
kv_cache["v"][batch_idx, offsets] = v
```

### Step 3: Vectorize rope_apply
```python
# Before: for i in enumerate(x): x_i * freqs_i
# After:
x_complex = torch.view_as_complex(x.to(float64).reshape(b, s, n, -1, 2))
result = torch.view_as_real(x_complex * freqs[:, :s]).flatten(3).float()
```

### Step 4: Cache static values during prefill
In `prefill_cond_cache`, store:
- `self._seq_len` (int)
- `self._frame_seqlen` (int)
- `self._num_frames` (int)
- `self._injected_block_set` (set)
- `self._audio_context_lens` (pre-allocated tensor)

Use these in `forward()` and block methods instead of computing/extracting per call.

### Step 5: Fix after_transformer_block
- `block_idx in self._injected_block_set` (pre-computed set)
- Pre-allocate `context_lens` buffer
- Replace `rearrange` with `.reshape()`

### Step 6: Remove @torch.compile from quant_fp8
Let the outer `@conditional_compile` on `forward()` compile it as part of the graph.

### Step 7: Enable CUDA graphs
Change `inference_utils.py`:
```python
torch.compile(mode="reduce-overhead", backend="inductor")
```

## Testing strategy
- After each step: run inference, verify output quality
- After step 7: check `TORCH_LOGS=graph_breaks` for zero breaks in block loop
- Benchmark: measure forward pass time before/after

## Files to modify
- `liveavatar_nari/models/causal_model_s2v.py` — steps 1, 2, 4, 5
- `liveavatar_nari/modules/s2v/model_s2v.py` — step 3
- `liveavatar_nari/utils/fp8_linear.py` — step 6
- `liveavatar_nari/modules/inference_utils.py` — step 7
