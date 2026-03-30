# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange
from ..distributed.sequence_parallel import (
    gather_forward,
    get_rank,
    get_world_size,
)
from ..modules.model import (
    Head,
    WanAttentionBlock,
    WanLayerNorm,
    WanModel,
    WanSelfAttention,
    rope_params,
    sinusoidal_embedding_1d,
    flash_attention,
)
from ..modules.s2v.model_s2v import zero_module, torch_dfs
from ..modules.s2v.model_s2v import (
    rope_apply as rope_apply,
    rope_apply as causal_rope_apply,
    rope_apply_cond as causal_rope_apply_cond,
)
from ..modules.s2v.model_s2v import (
    rope_apply_usp as rope_apply_usp,
    rope_apply_usp as causal_rope_apply_usp,
)

from ..modules.attention import attention
from ..modules.s2v.audio_utils import AudioInjector_WAN, CausalAudioEncoder
from .causal_motioner import FramePackMotioner
from .causal_s2v_utils import rollout_grid_sizes, causal_distributed_attention
from ..modules.s2v.s2v_utils import rope_precompute
from ..distributed import util as dist_util
from ..distributed.util import all_to_all, pad_chunk
import torch.distributed as dist
from ..modules.inference_utils import conditional_compile


class CausalHead_S2V(Head):
    def forward(self, x, e):
        """
        Args:
            x(Tensor): Shape [B, L1, C], L1 covers only the noisy_latent portion
            e(Tensor): Shape [B*F, C]
        """
        assert e.dtype == torch.float32
        original_dtype = x.dtype
        batch_size, num_frames = x.shape[0], e.shape[0] // x.shape[0]
        frame_seqlen = x.shape[1] // num_frames
        with torch.amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = self.head(
                (
                    self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen))
                    * (1 + e[1].unflatten(dim=0, sizes=(batch_size, num_frames)))
                    + e[0].unflatten(dim=0, sizes=(batch_size, num_frames))
                ).flatten(1, 2)
            )
        return x.to(original_dtype)


class CausalWanS2VSelfAttention(WanSelfAttention):
    def __init__(
        self,
        dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        eps=1e-6,
        local_attn_size=-1,
    ):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)
        self.local_attn_size = local_attn_size

    def fuse_qkv(self):
        """Fuse separate Q, K, V projections into a single QKV linear."""
        if hasattr(self, "qkv"):
            return  # already fused
        dim = self.q.in_features
        has_bias = self.q.bias is not None
        self.qkv = nn.Linear(dim, 3 * dim, bias=has_bias, device=self.q.weight.device, dtype=self.q.weight.dtype)
        self.qkv.weight.data[:dim] = self.q.weight.data
        self.qkv.weight.data[dim : 2 * dim] = self.k.weight.data
        self.qkv.weight.data[2 * dim :] = self.v.weight.data
        if has_bias:
            self.qkv.bias.data[:dim] = self.q.bias.data
            self.qkv.bias.data[dim : 2 * dim] = self.k.bias.data
            self.qkv.bias.data[2 * dim :] = self.v.bias.data
        del self.q, self.k, self.v

    def _qkv(self, x):
        b, s = x.shape[:2]
        n, d = self.num_heads, self.head_dim
        if hasattr(self, "qkv"):
            q, k, v = self.qkv(x).chunk(3, dim=-1)
            q = self.norm_q(q).view(b, s, n, d)
            k = self.norm_k(k).view(b, s, n, d)
            v = v.view(b, s, n, d)
        else:
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
        return q, k, v

    def _output(self, x):
        return self.o(x.flatten(2))

    def forward(self, x, grid_sizes, freqs, kv_cache, current_start, freqs_cond):
        """Streaming inference: write to rolling KV cache and attend to cached history + cond.

        Args:
            x: [B, L, dim] normalized hidden states.
            grid_sizes: Post-patch grid sizes for RoPE.
            freqs: Precomputed RoPE frequencies for noisy tokens.
            kv_cache: Rolling KV cache dict {k, v, cond_k, cond_v, cond_end}.
            current_start: Token offset (scalar int or [B] tensor), wraps via modulo.
            freqs_cond: Rolling RoPE frequencies for sink conditioning cache.
        """
        b = x.shape[0]
        q, k, v = self._qkv(x)

        roped_query = causal_rope_apply(q, grid_sizes, freqs).type_as(v)
        roped_key = causal_rope_apply(k, grid_sizes, freqs).type_as(v)
        seq_len = x.shape[1]

        active_cond_cache_size = int(kv_cache["cond_end"])
        kv_max = kv_cache["k"].shape[1]

        active_sizes = []
        for bi in range(b):
            cs_raw = int(current_start[bi].item()) if isinstance(current_start, torch.Tensor) else current_start
            wrapped = cs_raw >= kv_max
            cs = cs_raw % kv_max if wrapped else cs_raw
            kv_cache["k"][bi, cs : cs + seq_len] = roped_key[bi]
            kv_cache["v"][bi, cs : cs + seq_len] = v[bi]
            active_sizes.append(kv_max if wrapped else cs + seq_len)

        max_active_size = max(active_sizes)

        cond_k_roped = causal_rope_apply_cond(kv_cache["cond_k"][:, :active_cond_cache_size], None, freqs_cond).type_as(
            v
        )

        x = attention(
            q=roped_query,
            k=torch.cat([kv_cache["k"][:, :max_active_size], cond_k_roped], dim=1),
            v=torch.cat(
                [kv_cache["v"][:, :max_active_size], kv_cache["cond_v"][:, :active_cond_cache_size]],
                dim=1,
            ),
            window_size=self.window_size,
        )
        return self._output(x)

    def prefill_cond(self, x, grid_sizes, freqs, kv_cache):
        """Prefill cond cache: store ref/motion tokens into cond_k/cond_v.

        Args:
            x: [B, L, dim] normalized hidden states (ref + motion tokens).
            grid_sizes: Post-patch grid sizes for RoPE.
            freqs: Precomputed RoPE frequencies for cond tokens.
            kv_cache: KV cache dict — writes to cond_k, cond_v, cond_end.
        """
        b = x.shape[0]
        q, k, v = self._qkv(x)
        seq_len = x.shape[1]

        roped_query = causal_rope_apply_cond(q, grid_sizes, freqs).type_as(v)
        kv_cache["cond_end"][0] = max(int(kv_cache["cond_end"]), seq_len)
        kv_cache["cond_k"][:, :seq_len] = k
        kv_cache["cond_v"][:, :seq_len] = v
        x = attention(
            q=roped_query,
            k=causal_rope_apply_cond(k, grid_sizes, freqs).type_as(v)[:, :seq_len],
            v=kv_cache["cond_v"][:, :seq_len],
            k_lens=torch.tensor(seq_len).repeat(b),
            window_size=self.window_size,
        )
        return self._output(x)


class CausalWanS2VAttentionBlock(WanAttentionBlock):
    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        local_attn_size=-1,
    ):
        super().__init__(dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
        self.self_attn = CausalWanS2VSelfAttention(dim, num_heads, window_size, qk_norm, eps, local_attn_size)

    def _expand_modulation(self, x, e, seq_lens, frame_seqlen, is_prefill, use_context_parallel=False, sp_size=None):
        """Expand timestep modulation [B,F,6,2,dim] → tuple of 6 [B,L,dim] tensors."""
        assert e[0].dtype == torch.float32
        original_seq_len = e[1].item() if isinstance(e[1], torch.Tensor) else e[1]
        e = e[0]  # [B, F, 6, 2, C]

        modulation = self.modulation.unsqueeze(1).unsqueeze(3)
        with torch.amp.autocast("cuda", dtype=torch.float32):
            e = (modulation + e).chunk(6, dim=2)
        e = [el.squeeze(2) for el in e]  # tuple(6)*[B,F,2,dim]

        e_cache = []
        for element in e:
            if is_prefill:
                element_noisy = element[:, :0, 0]  # empty — no noisy tokens in prefill
            else:
                element_noisy = element[:, :, 0].repeat_interleave(int(frame_seqlen), dim=1)
            element_cond = element[:, 0:1, 1].repeat(1, seq_lens - element_noisy.shape[1], 1)
            element = torch.cat([element_noisy, element_cond], dim=1)
            if use_context_parallel:
                global_rank = get_rank()
                model_sp_size = sp_size if sp_size is not None else self.sp_size
                sp_rank = global_rank % model_sp_size
                element, _ = pad_chunk(element, model_sp_size, dim=1)
            e_cache.append(element)
        return tuple(e_cache), original_seq_len

    def _cross_attn_ffn(self, x, context, e, dtype):
        """Cross-attention to text context + feed-forward."""
        x = x + self.cross_attn(self.norm3(x.to(torch.bfloat16)).to(dtype), context, None)
        norm2_x = self.norm2(x).float()
        norm2_x = norm2_x * (1 + e[4]) + e[3]
        y = self.ffn(norm2_x.to(dtype))
        with torch.amp.autocast("cuda", dtype=torch.float32):
            y = y * e[5]
            x = x + y
        return x.to(dtype)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        frame_seqlen,
        kv_cache,
        current_start,
        freqs_cond,
        use_context_parallel=False,
        sp_size=None,
    ):
        """Streaming inference block: self-attn with KV cache → cross-attn → FFN."""
        dtype = x.dtype  # capture entry dtype (bfloat16) before fp32 operations
        e, _ = self._expand_modulation(
            x, e, seq_lens, frame_seqlen, is_prefill=False, use_context_parallel=use_context_parallel, sp_size=sp_size
        )
        norm_x = self.norm1(x).float()
        norm_x = norm_x * (1 + e[1]) + e[0]

        y = self.self_attn(norm_x.to(dtype), grid_sizes, freqs, kv_cache, current_start, freqs_cond)

        with torch.amp.autocast("cuda", dtype=torch.float32):
            x = x + y * e[2]
        return self._cross_attn_ffn(x, context, e, dtype)

    def prefill_cond(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        frame_seqlen,
        kv_cache,
        use_context_parallel=False,
        sp_size=None,
    ):
        """Prefill block: populate cond KV cache with ref/motion tokens."""
        dtype = x.dtype
        e, _ = self._expand_modulation(
            x, e, seq_lens, frame_seqlen, is_prefill=True, use_context_parallel=use_context_parallel, sp_size=sp_size
        )
        norm_x = self.norm1(x).float()
        norm_x = norm_x * (1 + e[1]) + e[0]

        y = self.self_attn.prefill_cond(norm_x.to(dtype), grid_sizes, freqs, kv_cache)

        with torch.amp.autocast("cuda", dtype=torch.float32):
            x = x + y * e[2]
        return self._cross_attn_ffn(x, context, e, dtype)


class CausalWanModel_S2V(ModelMixin, ConfigMixin):
    ignore_for_config = [
        "args",
        "kwargs",
        "patch_size",
        "cross_attn_norm",
        "qk_norm",
        "text_dim",
        "window_size",
    ]
    _no_split_modules = ["CausalWanS2VAttentionBlock"]

    @register_to_config
    def __init__(
        self,
        cross_attn_type="t2v_cross_attn",
        cond_dim=0,
        audio_dim=5120,
        num_audio_token=4,
        enable_adain=False,
        adain_mode="attn_norm",
        audio_inject_layers=[0, 4, 8, 12, 16, 20, 24, 27],
        zero_init=False,
        zero_timestep=False,
        enable_motioner=True,
        add_last_motion=True,
        enable_tsm=False,
        trainable_token_pos_emb=False,
        motion_token_num=1024,
        enable_framepack=False,  # Mutually exclusive with enable_motioner
        framepack_drop_mode="drop",
        model_type="s2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        slide_motion_frames=False,
        local_attn_size=-1,
        *args,
        **kwargs,
    ):
        super().__init__()

        assert model_type == "s2v"
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.slide_motion_frames = slide_motion_frames
        self.local_attn_size = local_attn_size
        # embeddings
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList(
            [
                CausalWanS2VAttentionBlock(
                    cross_attn_type,
                    dim,
                    ffn_dim,
                    num_heads,
                    window_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    local_attn_size,
                )
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = CausalHead_S2V(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat(
            [
                rope_params(45000, d - 4 * (d // 6)),
                rope_params(45000, 2 * (d // 6)),
                rope_params(45000, 2 * (d // 6)),
            ],
            dim=1,
        ).to("cuda")
        self.rope_cache = {}

        # initialize weights
        self.init_weights()

        self.use_context_parallel = False  # will modify in _configure_model func
        self.sp_size = None  # will be set in _configure_model func

        if cond_dim > 0:
            self.cond_encoder = nn.Conv3d(cond_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.enbale_adain = enable_adain
        self.casual_audio_encoder = CausalAudioEncoder(
            dim=audio_dim,
            out_dim=self.dim,
            num_token=num_audio_token,
            need_global=enable_adain,
        )
        all_modules, all_modules_names = torch_dfs(self.blocks, parent_name="root.transformer_blocks")
        self.audio_injector = AudioInjector_WAN(
            all_modules,
            all_modules_names,
            dim=self.dim,
            num_heads=self.num_heads,
            inject_layer=audio_inject_layers,
            root_net=self,
            enable_adain=enable_adain,
            adain_dim=self.dim,
            need_adain_ont=adain_mode != "attn_norm",
        )
        self.adain_mode = adain_mode

        self.trainable_cond_mask = nn.Embedding(3, self.dim)

        if zero_init:
            self.zero_init_weights()

        self.zero_timestep = zero_timestep  # Whether to assign 0 value timestep to ref/motion

        self.add_last_motion = add_last_motion

        self.enable_framepack = enable_framepack
        if enable_framepack:
            self.frame_packer = FramePackMotioner(
                inner_dim=self.dim,
                num_heads=self.num_heads,
                zip_frame_buckets=[1, 2, 16],
                drop_mode=framepack_drop_mode,
                slide_motion_frames=slide_motion_frames,
            )

    def fuse_projections(self):
        """Fuse Q/K/V into single QKV linear for all attention layers. Call before FP8."""
        for block in self.blocks:
            block.self_attn.fuse_qkv()
            block.cross_attn.fuse_kv()
        print(f"Fused QKV/KV projections in {len(self.blocks)} blocks.")

    def init_weights(self):
        """Initialize model parameters using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        nn.init.zeros_(self.head.head.weight)

    def zero_init_weights(self):
        with torch.no_grad():
            self.trainable_cond_mask = zero_module(self.trainable_cond_mask)
            if hasattr(self, "cond_encoder"):
                self.cond_encoder = zero_module(self.cond_encoder)

            for i in range(self.audio_injector.injector.__len__()):
                self.audio_injector.injector[i].o = zero_module(self.audio_injector.injector[i].o)
                if self.enbale_adain:
                    self.audio_injector.injector_adain_layers[i].linear = zero_module(
                        self.audio_injector.injector_adain_layers[i].linear
                    )

    # --- Motion & audio processing ---

    def process_motion_frame_pack(
        self,
        motion_latents,
        drop_motion_frames=False,
        add_last_motion=2,
        rollout_num_frames=0,
        sequence_current_start_frames=0,
    ):
        flattern_mot, mot_remb, motion_rope_cache = self.frame_packer(
            motion_latents,
            add_last_motion,
            rollout_num_frames,
            sequence_current_start_frames,
        )
        if drop_motion_frames:
            # return [m[:, :0] for m in flattern_mot
            #        ], [m[:, :0] for m in mot_remb]
            return (
                [torch.zeros_like(m) for m in flattern_mot],
                [torch.zeros_like(m) for m in mot_remb],
                motion_rope_cache,
            )
        else:
            return flattern_mot, mot_remb, motion_rope_cache

    def inject_motion(
        self,
        x,
        seq_lens,
        rope_embs,
        mask_input,
        motion_latents,
        drop_motion_frames=False,
        add_last_motion=True,
        rollout_num_frames=0,
        sequence_current_start=0,
    ):
        # Encode motion latents via FramePack into compressed tokens
        assert self.enable_framepack, "only FramePack motion injection is supported"
        mot, mot_remb, motion_rope_cache = self.process_motion_frame_pack(
            motion_latents,
            drop_motion_frames=drop_motion_frames,
            add_last_motion=add_last_motion,
            rollout_num_frames=rollout_num_frames,
            sequence_current_start_frames=sequence_current_start,
        )

        if len(mot) > 0:
            x = [torch.cat([u, m], dim=1) for u, m in zip(x, mot)]
            seq_lens = seq_lens + torch.tensor([r.size(1) for r in mot], dtype=torch.long)
            rope_embs = [torch.cat([u, m], dim=1) for u, m in zip(rope_embs, mot_remb)]
            self.rope_cache["cond_shape"] = torch.cat(
                [
                    torch.empty(self.rope_cache["cond_shape"]),
                    torch.empty(motion_rope_cache["cond_shape"]),
                ],
                dim=1,
            ).shape
            self.rope_cache["grid_sizes"] = self.rope_cache["grid_sizes"] + motion_rope_cache["grid_sizes"]
            mask_input = [
                torch.cat(
                    [
                        m,
                        2 * torch.ones([1, u.shape[1] - m.shape[1]], device=m.device, dtype=m.dtype),
                    ],
                    dim=1,
                )
                for m, u in zip(mask_input, x)
            ]
        return x, seq_lens, rope_embs, mask_input

    def after_transformer_block(self, block_idx, hidden_states, mask=None):
        if block_idx in self.audio_injector.injected_block_id.keys():
            audio_attn_id = self.audio_injector.injected_block_id[block_idx]
            audio_emb = self.merged_audio_emb  # b f n c
            num_actors = audio_emb.shape[0]
            num_frames = audio_emb.shape[1]

            if self.use_context_parallel:
                sp_size = self.sp_size
                hidden_states = gather_forward(hidden_states, dim=1, sp_size=sp_size)

            input_hidden_states = hidden_states[:, : self.original_seq_len].clone()  # b (f h w) c

            input_hidden_states = input_hidden_states.expand(num_actors, -1, -1)

            input_hidden_states = rearrange(input_hidden_states, "b (t n) c -> (b t) n c", t=num_frames)

            if self.enbale_adain and self.adain_mode == "attn_norm":
                audio_emb_global = self.audio_emb_global.to(self.dtype)
                audio_emb_global = rearrange(audio_emb_global, "b t n c -> (b t) n c")
                adain_hidden_states = self.audio_injector.injector_adain_layers[audio_attn_id](
                    input_hidden_states, temb=audio_emb_global[:, 0]
                )
                attn_hidden_states = adain_hidden_states
            else:
                attn_hidden_states = self.audio_injector.injector_pre_norm_feat[audio_attn_id](input_hidden_states)
            audio_emb = rearrange(audio_emb, "b t n c -> (b t) n c", t=num_frames)
            attn_audio_emb = audio_emb
            residual_out = self.audio_injector.injector[audio_attn_id](
                x=attn_hidden_states,
                context=attn_audio_emb,
                context_lens=torch.ones(
                    attn_hidden_states.shape[0],
                    dtype=torch.long,
                    device=attn_hidden_states.device,
                )
                * attn_audio_emb.shape[1],
            )
            residual_out = rearrange(residual_out, "(b t) n c -> b (t n) c", t=num_frames)

            if mask is not None:
                h = self.h_patches
                w = self.w_patches
                residual_out = rearrange(residual_out, "b (t h w) c -> b t h w c", t=num_frames, h=h, w=w)
                mask_cropped = mask[:, :num_frames]
                residual_out = residual_out * mask_cropped

                residual_out = residual_out.sum(dim=0, keepdim=True)
                residual_out = rearrange(residual_out, "b t h w c -> b (t h w) c")
            elif num_actors > 1 and num_actors != hidden_states.shape[0]:
                # Multi-actor SAM2 case requires a mask; batched pipeline
                # (num_actors == batch_size) is fine without one.
                assert False, "num_actors should be equal to num_mask, but no mask is provided."

            hidden_states[:, : self.original_seq_len] = hidden_states[:, : self.original_seq_len] + residual_out

            if self.use_context_parallel:
                sp_size = self.sp_size
                hidden_states, _ = pad_chunk(hidden_states, sp_size, dim=1)

        return hidden_states

    # ── shared helpers (Step 4) ──────────────────────────────────────────

    # --- Shared forward helpers ---

    def _encode_audio(self, audio_input, motion_frames):
        """Encode audio input via CausalAudioEncoder, store results as instance attrs."""
        audio_input = torch.cat(
            [audio_input[..., 0:1].repeat(1, 1, 1, int(motion_frames[0])), audio_input],
            dim=-1,
        )
        audio_emb_res = self.casual_audio_encoder(audio_input)
        audio_emb_res = tuple(aa.to(self.dtype) for aa in audio_emb_res)
        if self.enbale_adain:
            audio_emb_global, audio_emb = audio_emb_res
            self.audio_emb_global = audio_emb_global[:, motion_frames[1] :].clone()
        else:
            audio_emb = audio_emb_res
        self.merged_audio_emb = audio_emb[:, motion_frames[1] :, :]

    def _embed_patches_with_pose(self, x, cond_states):
        """Patch-embed noisy latents and add pose conditioning. x: [B, C, F, H, W]."""
        return self.patch_embedding(x) + self.cond_encoder(cond_states)

    def _flatten_to_sequence(self, x):
        """Flatten patch-embedded [B, dim, F', H', W'] to [B, L, dim] with grid metadata."""
        b = x.shape[0]
        grid_size = torch.tensor(x.shape[2:], dtype=torch.long)  # [F', H', W'] — same for all
        x = x.flatten(2).transpose(1, 2)  # [B, F'*H'*W', dim]
        seq_len = x.shape[1]
        original_grid_sizes = grid_size.unsqueeze(0).expand(b, -1)  # [B, 3]
        grid_sizes = [[torch.zeros_like(original_grid_sizes), original_grid_sizes, original_grid_sizes]]
        num_frames = grid_size[0].item()
        frame_seqlen = seq_len // num_frames if num_frames > 0 else grid_size[1:].prod()
        return x, seq_len, grid_sizes, original_grid_sizes, num_frames, frame_seqlen

    def _prepare_ref_tokens(self, ref_latents, x, seq_lens, grid_sizes):
        """Patch-embed ref image, create ref_grid_sizes, concatenate to sequence."""
        ref = [self.patch_embedding(r.unsqueeze(0)) for r in ref_latents]
        batch_size = len(ref)
        height, width = ref[0].shape[3], ref[0].shape[4]
        ref_grid_sizes = [
            [
                torch.tensor([30, 0, 0]).unsqueeze(0).repeat(batch_size, 1),
                torch.tensor([31, height, width]).unsqueeze(0).repeat(batch_size, 1),
                torch.tensor([1, height, width]).unsqueeze(0).repeat(batch_size, 1),
            ]
        ]
        ref = [r.flatten(2).transpose(1, 2) for r in ref]
        self.original_seq_len = seq_lens[0]
        seq_lens = seq_lens + torch.tensor([r.size(1) for r in ref], dtype=torch.long)
        grid_sizes = grid_sizes + ref_grid_sizes
        x = [torch.cat([u, r], dim=1) for u, r in zip(x, ref)]
        return x, seq_lens, grid_sizes

    def _compute_timestep_embeddings(self, t):
        """Compute time embeddings and projections. Returns (e, e0)."""
        if self.zero_timestep:
            t = torch.cat([t, torch.zeros([1, t.shape[1]], dtype=t.dtype, device=t.device)])
        with torch.amp.autocast("cuda", dtype=torch.float32):
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            assert e.dtype == torch.float32 and e0.dtype == torch.float32
        if self.zero_timestep:
            e = e[: -1 * t.shape[1]]
            zero_e0 = e0[-1:]
            e0 = e0[:-1]
            e0 = torch.cat(
                [
                    e0.unsqueeze(3),
                    zero_e0.unsqueeze(3).repeat(e0.size(0), 1, 1, 1, 1),
                ],
                dim=3,
            )
            e0 = [e0, self.original_seq_len]
        return e, e0

    def _embed_context(self, context):
        """Pad and embed T5 text context."""
        return self.text_embedding(
            torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
        )

    def _postprocess_output(self, x, e, original_grid_sizes):
        """Gather context-parallel, slice to original seq_len, apply head, unpatchify."""
        if self.use_context_parallel:
            x = gather_forward(x.contiguous(), dim=1)
        x = x[:, : self.original_seq_len]
        x = self.head(x, e)
        return self.unpatchify(x, original_grid_sizes)

    # ── end shared helpers ───────────────────────────────────────────────

    # --- Forward methods ---

    def prefill_cond_cache(
        self,
        *,
        ref_latents,
        motion_latents,
        context,
        motion_frames,
        kv_cache,
        crossattn_cache,
        latent_shape,
        current_end,
        latent_frames_per_block,
        drop_motion_frames=False,
        add_last_motion=2,
    ):
        """Prefill KV cache with conditioning tokens (ref image, motion, text, RoPE).

        Called once at the start of autoregressive generation. Populates the
        cond_k/cond_v slots in the KV cache and stores rope_cache for inference.

        Args:
            ref_latents: Sink frame latent [B, C, 1, H, W].
            motion_latents: Static VAE-encoded reference frames [B, C, T_m, H, W] for FramePack.
            context: T5 text embeddings, list of [L, 4096] tensors.
            motion_frames: [pixel_motion_frames, latent_motion_frames] sizes for FramePack.
            kv_cache: Per-layer KV cache dicts (slot 0 only, broadcast to all slots after).
            crossattn_cache: Per-layer cross-attention cache dicts.
            latent_shape: (latent_h, latent_w) spatial dims in latent space.
            current_end: Token count for one block (latent_frames_per_block * tokens_per_frame).
            latent_frames_per_block: Number of latent frames per block (used for timestep shape).
            drop_motion_frames: Zero out motion tokens (ablation only, always False).
            add_last_motion: FramePack bucket control (0/1/2). 2 = use all buckets.
        """
        device = ref_latents.device
        latent_h, latent_w = latent_shape
        bs = ref_latents.shape[0]

        # Create empty latent (no noisy content — this is prefill only)
        x = [torch.zeros([1, self.dim, 0, latent_h, latent_w], dtype=torch.bfloat16, device=device)] * bs

        # Flatten (list-based for prefill since _prepare_ref_tokens works with lists)
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        original_grid_sizes = grid_sizes.clone()
        grid_sizes = [[torch.zeros_like(grid_sizes), grid_sizes, grid_sizes]]
        frame_seqlen = grid_sizes[0][-1][:, 1:].prod()
        self.h_patches = original_grid_sizes[0][1].item()
        self.w_patches = original_grid_sizes[0][2].item()
        self.lat_motion_frames = motion_latents[0].shape[1]
        x, seq_lens, grid_sizes = self._prepare_ref_tokens(ref_latents, x, seq_lens, grid_sizes)

        mask_input = [torch.ones([1, u.shape[1]], dtype=torch.long, device=device) for u in x]

        # RoPE (stores to rope_cache for inference to use later)
        x = torch.cat(x)
        b, s, n, d = x.size(0), x.size(1), self.num_heads, self.dim // self.num_heads
        self.rope_cache["cond_shape"] = x.detach().view(b, s, n, d).shape
        self.rope_cache["grid_sizes"] = grid_sizes
        self.pre_compute_freqs = rope_precompute(
            x.detach().view(b, s, n, d),
            rollout_grid_sizes(grid_sizes, 0),
            self.freqs,
            start=None,
        )

        x = [u.unsqueeze(0) for u in x]
        self.pre_compute_freqs = [u.unsqueeze(0) for u in self.pre_compute_freqs]

        x, seq_lens, self.pre_compute_freqs, mask_input = self.inject_motion(
            x,
            seq_lens,
            self.pre_compute_freqs,
            mask_input,
            motion_latents,
            drop_motion_frames=drop_motion_frames,
            add_last_motion=add_last_motion,
            rollout_num_frames=0,
            sequence_current_start=0,
        )

        x = torch.cat(x, dim=0)
        self.pre_compute_freqs = torch.cat(self.pre_compute_freqs, dim=0)
        mask_input = torch.cat(mask_input, dim=0)

        x = x + self.trainable_cond_mask(mask_input).to(x.dtype)

        t = torch.zeros([1, latent_frames_per_block], dtype=torch.bfloat16, device=device)
        e, e0 = self._compute_timestep_embeddings(t)
        context = self._embed_context(context)

        if self.use_context_parallel:
            global_rank = get_rank()
            model_sp_size = self.sp_size
            sp_rank = global_rank % model_sp_size
            x, orig_seq_len = pad_chunk(x, model_sp_size, dim=1)
            e0[1] = e0[1] - int(x.shape[1] * sp_rank)
            self.pre_compute_freqs, _ = pad_chunk(self.pre_compute_freqs, model_sp_size, dim=1)

        for idx, block in enumerate(self.blocks):
            x = block.prefill_cond(
                x,
                e=e0,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=self.pre_compute_freqs,
                context=context,
                frame_seqlen=frame_seqlen,
                kv_cache=kv_cache[idx],
                use_context_parallel=self.use_context_parallel,
                sp_size=self.sp_size,
            )

    @conditional_compile
    def forward(
        self,
        x,
        t,
        context,
        cond_states,
        audio_input,
        motion_frames,
        kv_cache,
        crossattn_cache,
        current_start,
        current_end,
        mask=None,
    ):
        """Streaming inference forward pass (blockwise autoregressive with cached KV).

        Args:
            x: Noisy latent block, list of [C, F, H, W] tensors (F=latent_frames_per_block).
            t: Denoising timestep [B, F], same value broadcast to all frames.
            context: T5 text embeddings, list of [L, 4096] tensors.
            cond_states: Pose conditioning [B, C, F, H, W] (zeros when no pose control).
            audio_input: Wav2Vec2 features [B, 25, 1024, T_a] for this block's audio segment.
            motion_frames: [pixel_motion_frames, latent_motion_frames] sizes for FramePack.
            kv_cache: Per-layer rolling KV cache dicts {k, v, cond_k, cond_v, cond_end}.
            crossattn_cache: Per-layer cross-attention cache dicts {k, v, is_init}.
            current_start: Token offset for this block in KV cache (wraps via modulo).
            current_end: Token offset where this block ends (current_start + tokens_per_block).
            mask: Per-actor spatial mask for multi-actor audio injection, or None.
        """
        self._encode_audio(audio_input, motion_frames)
        x = self._embed_patches_with_pose(x, cond_states)
        x, seq_len, grid_sizes, original_grid_sizes, num_frames, frame_seqlen = self._flatten_to_sequence(x)
        self.original_seq_len = seq_len

        x = x  # already [B, L, dim] — no cat needed
        b, s, n, d = x.size(0), x.size(1), self.num_heads, self.dim // self.num_heads

        # RoPE per batch item (handles both scalar and tensor current_start)
        frame_seqlen_int = int(frame_seqlen)
        freqs_list = []
        cond_freqs_list = []
        for bi in range(b):
            cs_bi = int(current_start[bi].item()) if isinstance(current_start, torch.Tensor) else int(current_start)
            frame_offset_bi = cs_bi // frame_seqlen_int

            gs_bi = [[g[bi : bi + 1].clone() for g in group] for group in grid_sizes]
            gs_bi_shifted = rollout_grid_sizes(gs_bi, frame_offset_bi)

            x_bi = x[bi : bi + 1].detach().view(1, s, n, d)
            f_bi = rope_precompute(x_bi, gs_bi_shifted, self.freqs, start=None)
            freqs_list.append(f_bi)

            # Rolling RoPE for sink cond
            relative_dist = random.randint(4, 30)
            start_idx = 30 - relative_dist
            num_frames_cond = max(0, frame_offset_bi - start_idx)
            cond_shape_bi = list(self.rope_cache["cond_shape"])
            cond_shape_bi[0] = 1
            cond_gs_bi = [[g[0:1].clone() for g in group] for group in self.rope_cache["grid_sizes"]]
            cond_gs_shifted = rollout_grid_sizes(cond_gs_bi, num_frames_cond)
            cf_bi = rope_precompute(
                torch.empty(cond_shape_bi).type_as(x),
                cond_gs_shifted,
                self.freqs,
                start=None,
            )
            cond_freqs_list.append(cf_bi)

        self.pre_compute_freqs = torch.cat(freqs_list, dim=0)
        cond_pre_compute_freqs = torch.cat(cond_freqs_list, dim=0)
        x = x + self.trainable_cond_mask(torch.zeros([1, x.shape[1]], dtype=torch.long, device=x.device)).to(x.dtype)

        e, e0 = self._compute_timestep_embeddings(t)
        context = self._embed_context(context)

        if self.use_context_parallel:
            global_rank = get_rank()
            model_sp_size = self.sp_size
            sp_rank = global_rank % model_sp_size
            x, orig_seq_len = pad_chunk(x, model_sp_size, dim=1)
            e0[1] = e0[1] - int(x.shape[1] * sp_rank)
            self.pre_compute_freqs, _ = pad_chunk(self.pre_compute_freqs, model_sp_size, dim=1)

        for idx, block in enumerate(self.blocks):
            x = block(
                x,
                e=e0,
                seq_lens=seq_len,
                grid_sizes=grid_sizes,
                freqs=self.pre_compute_freqs,
                context=context,
                frame_seqlen=frame_seqlen,
                kv_cache=kv_cache[idx],
                current_start=current_start,
                freqs_cond=cond_pre_compute_freqs,
                use_context_parallel=self.use_context_parallel,
                sp_size=self.sp_size,
            )
            x = self.after_transformer_block(idx, x, mask)

        return self._postprocess_output(x, e, original_grid_sizes)

    # --- Output ---

    def unpatchify(self, x, grid_sizes):
        """
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return torch.stack(out)  # [B, C, F, H, W]
