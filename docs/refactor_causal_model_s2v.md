# Refactor `causal_model_s2v.py` for Readability

## Context

`liveavatar_nari/models/causal_model_s2v.py` is 1658 lines with three forward methods (`_forward_sink`, `_forward_inference`, `_forward_train`) that share ~70% of their code via copy-paste. The file also has ~145 lines of dead code (`sp_attn_forward_s2v`, `assert False` branches, debug prints, commented-out blocks), ~20 Chinese comments, and inline imports. The goal is to reduce duplication, remove dead code, and make the file scannable — while preserving all behavior.

**Only `CausalWanModel_S2V` is imported externally** (by `pipeline.py`). All other classes are internal. No tests exist, so every step must be small and verifiable by reading the diff.

**Target: ~1050-1100 lines** (down from 1658).

## Critical files

- `liveavatar_nari/models/causal_model_s2v.py` — all edits here
- `liveavatar_nari/pipeline.py` — sole caller, verify `forward()` dispatch still works

## Plan

### Step 1: Remove dead code

Remove entirely:
- `sp_attn_forward_s2v` function (lines 56-200) — disabled, only in commented-out code
- Debug test code in `_prepare_blockwise_causal_attn_mask` (lines 852-868: hardcoded test tensor creation + print)
- All `assert False` branches with their dead code bodies:
  - Self-forcing paths in `CausalWanS2VSelfAttention.forward` (lines 252-280)
  - Motioner paths in `inject_motion` (lines 697-702, 713-716)
  - Non-zero-timestep paths in all three forwards (lines 1012, 1252, 1508)
- All commented-out code blocks (context parallel toggles in `_forward_sink`, debug prints, alternative masks)
- `"not checkpoint!!"` debug prints (lines 1081, 1329, 1584)
- Unused `token_len` assignments (lines 1244, 1500)
- Shadowed `create_custom_forward` definitions (the `return_dict` variant is always overwritten)

### Step 2: Cleanup imports and comments

- Move `import random` to top-level imports
- Remove redundant `import torch.distributed as dist` inside `_prepare_blockwise_causal_attn_mask` (already at line 47)
- Translate all Chinese comments to English (or remove if they just restate the code)
- Remove noisy inline shape annotations (e.g. `#torch.Size([1, 25, 1024, 80])->torch.Size([1, 25, 1024, 153])`) — keep only those that clarify non-obvious logic

### Step 3: Clean up `forward()` dispatcher

Pop `sink_flag` from kwargs before dispatching so sub-methods don't need `**extra_kwargs`:

```python
def forward(self, *args, **kwargs):
    sink_flag = kwargs.pop('sink_flag', False)
    if kwargs.get('kv_cache') is not None:
        return self._forward_sink(*args, **kwargs) if sink_flag else self._forward_inference(*args, **kwargs)
    return self._forward_train(*args, **kwargs)
```

Remove `*extra_args, **extra_kwargs` from all three sub-forward signatures.

### Step 4: Extract shared helpers

Each helper below is extracted from textually identical (or near-identical) code duplicated across the forward methods. Listed in dependency order.

#### 4a. `_create_custom_forward` → static method
Defined 6 times as nested function. Extract once.

#### 4b. `_encode_audio(self, audio_input, motion_frames)`
Shared by `_forward_inference` and `_forward_train`. Sets `self.merged_audio_emb` and `self.audio_emb_global`.

```python
def _encode_audio(self, audio_input, motion_frames):
    audio_input = torch.cat([audio_input[..., 0:1].repeat(1, 1, 1, motion_frames[0]), audio_input], dim=-1)
    audio_emb_res = self.casual_audio_encoder(audio_input)
    audio_emb_res = tuple(aa.to(self.dtype) for aa in audio_emb_res)
    if self.enbale_adain:
        audio_emb_global, audio_emb = audio_emb_res
        self.audio_emb_global = audio_emb_global[:, motion_frames[1]:].clone()
    else:
        audio_emb = audio_emb_res
    self.merged_audio_emb = audio_emb[:, motion_frames[1]:, :]
```

#### 4c. `_embed_patches_with_pose(self, x, cond_states)`
Shared by `_forward_inference` and `_forward_train`. Patch-embeds noisy latents and adds pose.

```python
def _embed_patches_with_pose(self, x, cond_states):
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    cond = [self.cond_encoder(c.unsqueeze(0)) for c in cond_states]
    x = [x_ + c for x_, c in zip(x, cond)]
    return x
```

#### 4d. `_flatten_to_sequence(self, x)` → returns `x, seq_lens, grid_sizes, original_grid_sizes`
Shared by all three. Computes grid sizes, flattens patch-embedded tensors to sequence form.

#### 4e. `_prepare_ref_tokens(self, ref_latents, x, seq_lens, grid_sizes)`
Shared by `_forward_sink` and `_forward_train`. Patch-embeds ref image, creates ref_grid_sizes at temporal index 30, concatenates to sequence, updates seq_lens and grid_sizes.

#### 4f. `_compute_timestep_embeddings(self, t)`
Shared by all three (textually identical). Returns `(e, e0)` where `e0 = [tensor, original_seq_len]`.

#### 4g. `_embed_context(self, context)`
Shared by all three (textually identical). Pads and embeds T5 text.

#### 4h. `_apply_context_parallel(self, x, e0, pre_compute_freqs, use_pad_chunk=True)`
Shared by all three. Sink/inference use `pad_chunk`, train uses `torch.chunk`. Controlled by `use_pad_chunk` flag.

#### 4i. `_run_transformer_blocks(self, x, block_kwargs, *, kv_cache, crossattn_cache, ...)`
The block loop. Differences controlled by parameters:
- `kv_cache`/`crossattn_cache`: present for sink/inference, absent for train
- `freqs_cond`: present for inference only
- `run_audio_injection`: False for sink, True for inference/train
- `audio_mask`: passed in inference, None for train
- `in_sink_forward`: True for sink only (passed via block_kwargs)

#### 4j. `_postprocess_output(self, x, e, original_grid_sizes)`
Shared by `_forward_inference` and `_forward_train`. Gather context-parallel, slice to `original_seq_len`, apply head, unpatchify.

### Step 5: Rewrite forward methods using helpers

After extraction, each forward becomes a thin orchestrator:

- **`_forward_sink`** (~40 lines): create empty x → `_flatten_to_sequence` → `_prepare_ref_tokens` → RoPE → `inject_motion` → `trainable_cond_mask` → `_compute_timestep_embeddings` → `_embed_context` → `_apply_context_parallel` → `_run_transformer_blocks(run_audio_injection=False)` → return zeros
- **`_forward_inference`** (~55 lines): `_encode_audio` → `_embed_patches_with_pose` → `_flatten_to_sequence` → RoPE (with per-batch + cond_freqs logic) → `trainable_cond_mask` → `_compute_timestep_embeddings` → `_embed_context` → `_apply_context_parallel` → `_run_transformer_blocks(run_audio_injection=True)` → `_postprocess_output`
- **`_forward_train`** (~65 lines): `_encode_audio` → `_embed_patches_with_pose` → `_flatten_to_sequence` → `_prepare_ref_tokens` → RoPE → `inject_motion` → `trainable_cond_mask` → build block_mask → `_compute_timestep_embeddings` → `_embed_context` → `_apply_context_parallel(use_pad_chunk=False)` → `_run_transformer_blocks(run_audio_injection=True)` → `_postprocess_output`

Note: RoPE computation stays inline in each forward method because each variant has meaningfully different logic (sink stores to rope_cache, inference has per-batch + random cond offset, train has no rollout).

### Step 6: Organize file structure

Final file layout with section comments:

```
Imports
CausalHead_S2V
CausalWanS2VSelfAttention
CausalWanS2VAttentionBlock
CausalWanModel_S2V:
    # --- Config & initialization ---
    __init__, init_weights, enable_gradient_checkpointing, zero_init_weights
    # --- Motion & audio processing ---
    process_motion_frame_pack, inject_motion, after_transformer_block
    _prepare_blockwise_causal_attn_mask
    # --- Shared forward helpers ---
    _create_custom_forward, _encode_audio, _embed_patches_with_pose,
    _flatten_to_sequence, _prepare_ref_tokens, _compute_timestep_embeddings,
    _embed_context, _apply_context_parallel, _run_transformer_blocks,
    _postprocess_output
    # --- Forward methods ---
    _forward_sink, _forward_inference, _forward_train, forward
    # --- Output ---
    unpatchify
```

## Notes

- **Do NOT rename `enbale_adain`** — it's baked into saved model configs via `@register_to_config` / ConfigMixin. Renaming breaks `from_pretrained`.
- **Do NOT rename `casual_audio_encoder`** — same reason, it's a registered attribute name.
- Keep `@conditional_compile` on `_forward_inference` (performance-critical).
- The `CausalWanS2VSelfAttention` class has internal duplication (per-batch vs scalar paths) but this is harder to extract cleanly and the two paths differ in non-trivial ways. Leave for a future pass.

## Verification

Since there are no tests:
1. After each step, verify the file is syntactically valid: `python -c "import liveavatar_nari.models.causal_model_s2v"`
2. After all steps, run inference end-to-end if a checkpoint is available, or at minimum verify `CausalWanModel_S2V.from_pretrained` still works
3. Diff review: every extracted helper should be a strict subset of the original code with no logic changes
