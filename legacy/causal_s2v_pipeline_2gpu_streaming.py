# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Streaming pipeline — per-block VAE decode + on-demand audio encoding.
# Based on causal_s2v_pipeline_2gpu.py (sequential denoising).
import gc
import logging
import math
import os
import random
import sys

import librosa
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from diffusers import FlowMatchEulerDiscreteScheduler
from .causal_audio_encoder import AudioEncoder
from .causal_model_s2v import CausalWanModel_S2V
from .wan_2_2.modules.t5 import T5EncoderModel
from .wan_2_2.modules.vae_streaming import WanVAE as Wan2_1_VAE
from ...utils.load_weight_utils import load_state_dict


class AudioChunkLoader:
    """Yields clip-sized raw audio chunks from a file."""

    def __init__(self, audio_path, sample_rate=16000, fps=25, infer_frames=48):
        self.waveform, _ = librosa.load(audio_path, sr=sample_rate)
        self.samples_per_clip = int(infer_frames / fps * sample_rate)
        self.clip_idx = 0

    def next_clip(self):
        """Return next clip's raw waveform numpy array, or None if exhausted."""
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
        return max(1, int(np.ceil(len(self.waveform) / self.samples_per_clip)))


class WanS2VStreaming:
    """Streaming pipeline: per-block VAE decode + on-demand audio encoding.

    Uses vae_streaming.WanVAE for incremental decode with persistent caches.
    generate() is a generator that yields per-block decoded frames.
    """

    def __init__(self, config, checkpoint_dir, device_id=0, offload_kv_cache=False):
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.num_train_timesteps = config.num_train_timesteps
        self.num_frames_per_block = config.num_frames_per_block
        self.param_dtype = config.param_dtype
        self.checkpoint_dir = checkpoint_dir
        self.offload_kv_cache = offload_kv_cache

        self.text_encoder = T5EncoderModel(
            text_len=config.text_len, dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer))

        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device, dtype=self.param_dtype)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.noise_model = CausalWanModel_S2V.from_pretrained(
            checkpoint_dir, torch_dtype=self.param_dtype, device_map=self.device)
        self.noise_model.freqs.to(device=self.device)
        self.noise_model.eval().requires_grad_(False)
        self.noise_model.num_frame_per_block = self.num_frames_per_block

        self.audio_encoder = AudioEncoder(
            model_id=os.path.join(checkpoint_dir, "wav2vec2-large-xlsr-53-english"))

        self.sample_neg_prompt = config.sample_neg_prompt
        self.motion_frames = config.transformer.motion_frames
        self.fps = config.sample_fps
        self.audio_sample_m = 0

    # ---- Model setup ----

    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4,
                          lora_target_modules="q,k,v,o,ffn.0,ffn.2",
                          init_lora_weights="kaiming",
                          pretrained_lora_path=None,
                          state_dict_converter=None,
                          load_only=False,
                          load_lora_weight_only=False):
        if not load_only:
            self.lora_alpha = lora_alpha
            lora_config = LoraConfig(
                r=lora_rank, lora_alpha=lora_alpha,
                init_lora_weights=True if init_lora_weights == "kaiming" else init_lora_weights,
                target_modules=lora_target_modules.split(","))
            model = get_peft_model(model, lora_config)

        if pretrained_lora_path is not None:
            state_dict = load_state_dict(pretrained_lora_path)
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            first_key = next(iter(state_dict.keys()))
            if not first_key.startswith("base_model.model."):
                state_dict = {f"base_model.model.{k}": v for k, v in state_dict.items()}
            if load_lora_weight_only:
                state_dict = {k: v for k, v in state_dict.items() if 'lora' in k}
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            n_loaded = sum(1 for _ in model.named_parameters()) - len(missing)
            print(f"{n_loaded} params loaded from {pretrained_lora_path}. {len(unexpected)} unexpected.")
            print(f"Merging LoRA weights from {pretrained_lora_path}...")
            model = model.merge_and_unload()
            print("LoRA merged successfully.")
        return model

    def get_size_less_than_area(self, height, width, target_area=1024 * 704, divisor=64):
        if height * width <= target_area:
            max_upper_area = target_area
            min_scale = 0.1
            max_scale = 1.0
        else:
            max_upper_area = target_area
            d = divisor - 1
            b = d * (height + width)
            a = height * width
            c = d**2 - max_upper_area
            min_scale = (-b + math.sqrt(b**2 - 2 * a * c)) / (2 * a)
            max_scale = math.sqrt(max_upper_area / (height * width))

        for i in range(100):
            scale = max_scale - (max_scale - min_scale) * i / 100
            new_h, new_w = int(height * scale), int(width * scale)
            pad_h = (64 - new_h % 64) % 64
            pad_w = (64 - new_w % 64) % 64
            if (new_h + pad_h) * (new_w + pad_w) <= max_upper_area:
                return new_h + pad_h, new_w + pad_w

        ar = width / height
        tw = int((target_area * ar)**0.5 // divisor * divisor)
        th = int((target_area / ar)**0.5 // divisor * divisor)
        if tw >= width or th >= height:
            tw = int(width // divisor * divisor)
            th = int(height // divisor * divisor)
        return th, tw

    # ---- KV cache ----

    def _initialize_kv_cache(self, batch_size, dtype, device, kv_cache_size=13500):
        cache_device = "cpu" if self.offload_kv_cache else device
        self.kv_cache = [{
            "k": torch.zeros([batch_size, kv_cache_size, 40, 128], dtype=dtype, device=cache_device),
            "v": torch.zeros([batch_size, kv_cache_size, 40, 128], dtype=dtype, device=cache_device),
            "cond_k": torch.zeros([batch_size, 2800, 40, 128], dtype=dtype, device=cache_device),
            "cond_v": torch.zeros([batch_size, 2800, 40, 128], dtype=dtype, device=cache_device),
            "cond_end": torch.tensor([0], dtype=torch.long, device=cache_device),
        } for _ in range(self.noise_model.num_layers)]

    def _move_kv_cache_to_device(self, device):
        for layer in self.kv_cache:
            for key in ["k", "v", "cond_k", "cond_v", "cond_end"]:
                layer[key] = layer[key].to(device)

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        self.crossattn_cache = [{
            "k": torch.zeros([batch_size, 0, 40, 128], dtype=dtype, device=device),
            "v": torch.zeros([batch_size, 0, 40, 128], dtype=dtype, device=device),
            "is_init": False,
        } for _ in range(self.noise_model.num_layers)]

    def _prefill(self, block_latents, cond, audio, context,
                 motion_latents, ref_latents, lat_motion_frames,
                 nfpb, frame_seq_length, num_steps):
        """Run prefill with batch=1, broadcast cond cache to all pipeline slots."""
        prefill_kv = [{
            "k": l["k"][0:1], "v": l["v"][0:1],
            "cond_k": l["cond_k"][0:1], "cond_v": l["cond_v"][0:1],
            "cond_end": l["cond_end"],
        } for l in self.kv_cache]
        prefill_crossattn = [{
            "k": l["k"][0:1], "v": l["v"][0:1], "is_init": l["is_init"],
        } for l in self.crossattn_cache]

        if self.offload_kv_cache:
            self._move_kv_cache_to_device(self.device)

        self.noise_model(
            [block_latents],
            t=torch.zeros([1, nfpb], device=self.device, dtype=self.param_dtype),
            context=context[0:1], seq_len=None,
            cond_states=cond,
            motion_latents=motion_latents, ref_latents=ref_latents,
            audio_input=audio,
            motion_frames=[self.motion_frames, lat_motion_frames],
            drop_motion_frames=False,
            sink_flag=True,
            kv_cache=prefill_kv, crossattn_cache=prefill_crossattn,
            current_start=0, current_end=nfpb * frame_seq_length)

        for li in range(len(self.kv_cache)):
            for key in ["k", "v", "cond_k", "cond_v"]:
                self.kv_cache[li][key][:] = prefill_kv[li][key]
            self.kv_cache[li]["cond_end"] = prefill_kv[li]["cond_end"]
        for li in range(len(self.crossattn_cache)):
            self.crossattn_cache[li]["k"] = prefill_crossattn[li]["k"].expand(num_steps, -1, -1, -1).contiguous()
            self.crossattn_cache[li]["v"] = prefill_crossattn[li]["v"].expand(num_steps, -1, -1, -1).contiguous()
            self.crossattn_cache[li]["is_init"] = prefill_crossattn[li]["is_init"]

        if self.offload_kv_cache:
            self._move_kv_cache_to_device("cpu")

    # ---- Streaming audio ----

    def _encode_clip_audio(self, raw_audio_chunk, infer_frames):
        """Encode one clip's raw audio → [1, 25, 1024, infer_frames]."""
        z = self.audio_encoder.extract_audio_feat_from_array(
            raw_audio_chunk, return_all_layers=True)
        bucket, _ = self.audio_encoder.get_audio_embed_bucket_fps(
            z, fps=self.fps, batch_frames=infer_frames, m=self.audio_sample_m)
        bucket = bucket.to(self.device, self.param_dtype).unsqueeze(0)
        if bucket.dim() == 3:
            bucket = bucket.permute(0, 2, 1)
        elif bucket.dim() == 4:
            bucket = bucket.permute(0, 2, 3, 1)
        return bucket

    # ---- Streaming VAE decode ----

    def _prime_stream_decoder(self, motion_latents):
        """Reset and prime streaming VAE decoder with motion context."""
        self.vae.model.first_decode = True
        prime_latents = motion_latents[:, :, :min(7, motion_latents.shape[2])]
        self.vae.stream_decode(prime_latents)

    def _stream_decode_block(self, block_latents, clip_idx, block_idx,
                             frames_per_block):
        """Decode one block incrementally. Returns [1, 3, N, H, W]."""
        image = torch.stack(self.vae.stream_decode(block_latents.unsqueeze(0)))
        image = image[:, :, -frames_per_block:]
        if clip_idx == 0 and block_idx == 0:
            image = image[:, :, 3:]
        return image

    # ---- Denoising step (shared logic) ----

    def _denoise_block(self, block_latents, timesteps, context, cond_block,
                       motion_latents, ref_latents, audio_block,
                       lat_motion_frames, cs, ce, nfpb,
                       sample_scheduler, seed_g):
        """Run 4-step denoising on one block. Returns denoised block_latents."""
        sample_scheduler.timesteps = self._sampler_timesteps
        sample_scheduler.sigmas = self._sampler_sigmas
        sample_scheduler._step_index = 0
        sample_scheduler._begin_index = 0

        for step_i, t in enumerate(timesteps):
            if self.offload_kv_cache:
                self._move_kv_cache_to_device(self.device)

            step_kv = [{
                "k": l["k"][step_i:step_i+1], "v": l["v"][step_i:step_i+1],
                "cond_k": l["cond_k"][step_i:step_i+1], "cond_v": l["cond_v"][step_i:step_i+1],
                "cond_end": l["cond_end"],
            } for l in self.kv_cache]
            step_crossattn = [{
                "k": l["k"][step_i:step_i+1], "v": l["v"][step_i:step_i+1],
                "is_init": l["is_init"],
            } for l in self.crossattn_cache]

            noise_pred = self.noise_model(
                [block_latents],
                t=t.unsqueeze(0).expand(1, nfpb),
                context=context[0:1], seq_len=None,
                cond_states=cond_block,
                motion_latents=motion_latents, ref_latents=ref_latents,
                audio_input=audio_block,
                motion_frames=[self.motion_frames, lat_motion_frames],
                drop_motion_frames=False,
                kv_cache=step_kv, crossattn_cache=step_crossattn,
                current_start=cs, current_end=ce)

            if self.offload_kv_cache:
                self._move_kv_cache_to_device("cpu")

            block_latents = sample_scheduler.step(
                noise_pred[0].unsqueeze(0), t, block_latents.unsqueeze(0),
                return_dict=False, generator=seed_g
            )[0].squeeze(0)

        return block_latents

    # ---- Main generate (generator) ----

    def generate(
        self,
        input_prompt=None,
        ref_image_path=None,
        audio_path=None,
        num_repeat=1,
        max_area=720 * 1280,
        infer_frames=80,
        sampling_steps=4,
        n_prompt="",
        seed=-1,
        offload_model=True,
        max_repeat=1000000,
    ):
        """Streaming generation — yields (image_chunk, metadata) per block.

        Yields:
            image_chunk: [1, 3, N, H, W] decoded pixel frames for one block
            metadata: {"clip": int, "block": int}
        """
        num_steps = sampling_steps

        # ---- 1. Prepare fixed inputs ----
        ref_image = np.array(Image.open(ref_image_path).convert('RGB'))
        HEIGHT, WIDTH = self.get_size_less_than_area(
            ref_image.shape[0], ref_image.shape[1], target_area=max_area)

        resize_op = transforms.Resize(min(HEIGHT, WIDTH))
        crop_op = transforms.CenterCrop((HEIGHT, WIDTH))

        self.audio_encoder.model.to(device=self.device, dtype=self.param_dtype)
        self.audio_encoder.model.requires_grad_(False)
        self.audio_encoder.model.eval()
        self.vae.model.to(self.device)

        audio_loader = AudioChunkLoader(
            audio_path, fps=self.fps, infer_frames=infer_frames)
        if num_repeat is None or num_repeat > audio_loader.num_clips:
            num_repeat = audio_loader.num_clips

        lat_motion_frames = (self.motion_frames + 3) // 4
        model_pic = crop_op(resize_op(Image.fromarray(ref_image)))
        ref_pv = transforms.ToTensor()(model_pic).unsqueeze(1).unsqueeze(0) * 2 - 1.0
        ref_pv = ref_pv.to(dtype=self.vae.dtype, device=self.vae.device)
        ref_pv = ref_pv.repeat(1, 1, 5, 1, 1)
        ref_latents = torch.stack(self.vae.encode(ref_pv))[:, :, 1:]

        motion_latents = torch.stack(self.vae.encode(
            ref_pv.repeat(1, 1, self.motion_frames, 1, 1)))

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        self.text_encoder.model.to(self.device)
        context = self.text_encoder([input_prompt], self.device)
        if offload_model:
            self.text_encoder.model.cpu()

        print("complete prepare conditional inputs")
        sample_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=self.num_train_timesteps, shift=3)

        # ---- Precompute clip-invariant values ----
        lat_target_frames = (infer_frames + 3 + self.motion_frames) // 4 - lat_motion_frames
        target_shape = [lat_target_frames, HEIGHT // 8, WIDTH // 8]
        frame_seq_length = HEIGHT // 8 * WIDTH // 8 // 2 // 2
        nfpb = self.num_frames_per_block
        num_blocks = target_shape[0] // nfpb
        bsl = nfpb * frame_seq_length
        max_seq_len = np.prod(target_shape) // 4
        frames_per_block = infer_frames // num_blocks

        cond_latents = torch.zeros(
            1, 16, *target_shape, dtype=self.param_dtype, device=self.device)

        # ---- 2. Streaming generation ----
        with torch.amp.autocast('cuda', dtype=self.param_dtype), torch.no_grad():
            try:
                self.kv_cache = None
                active_nr = min(max_repeat, num_repeat)

                for r in range(active_nr):
                    seed_g = torch.Generator(device=self.device)
                    seed_g.manual_seed(seed + r)

                    clip_noise = torch.randn(
                        16, *target_shape,
                        dtype=self.param_dtype, device=self.device, generator=seed_g)

                    # First clip: init caches
                    if self.kv_cache is None:
                        if offload_model:
                            self.noise_model.to(self.device)
                            self.vae.model.cpu()
                            self.text_encoder.model.cpu()
                            self.audio_encoder.model.cpu()
                            torch.cuda.empty_cache()
                        self._initialize_kv_cache(num_steps, self.param_dtype, self.device, max_seq_len)
                        self._initialize_crossattn_cache(num_steps, self.param_dtype, self.device)

                    # Streaming audio: encode this clip on demand
                    self.audio_encoder.model.to(device=self.device, dtype=self.param_dtype)
                    raw_chunk = audio_loader.next_clip()
                    if raw_chunk is None:
                        break
                    audio_input = self._encode_clip_audio(raw_chunk, infer_frames)
                    self.audio_encoder.model.to("cpu")

                    if offload_model:
                        self.noise_model.to(self.device)
                        self.vae.model.cpu()
                        torch.cuda.empty_cache()

                    # Prefill cond cache (first clip only)
                    if r == 0:
                        self._prefill(
                            block_latents=clip_noise[:, :nfpb],
                            cond=cond_latents[:, :, :nfpb],
                            audio=audio_input[..., :nfpb * 4],
                            context=context,
                            motion_latents=motion_latents,
                            ref_latents=ref_latents,
                            lat_motion_frames=lat_motion_frames,
                            nfpb=nfpb,
                            frame_seq_length=frame_seq_length,
                            num_steps=num_steps)

                    # Scheduler setup (once)
                    if getattr(self, '_sampler_timesteps', None) is None:
                        sample_scheduler.set_timesteps(sampling_steps, device=self.device)
                        self._sampler_timesteps = sample_scheduler.timesteps
                        self._sampler_sigmas = sample_scheduler.sigmas
                    timesteps = self._sampler_timesteps
                    clip_base = r * num_blocks * bsl

                    # Per-block: denoise → stream decode → yield
                    for block_idx in tqdm(range(num_blocks), desc=f"clip {r}"):
                        cs = block_idx * bsl + clip_base
                        ce = cs + bsl

                        block_latents = self._denoise_block(
                            block_latents=clip_noise[:, block_idx * nfpb:(block_idx + 1) * nfpb],
                            timesteps=timesteps,
                            context=context,
                            cond_block=cond_latents[:, :, block_idx * nfpb:(block_idx + 1) * nfpb],
                            motion_latents=motion_latents,
                            ref_latents=ref_latents,
                            audio_block=audio_input[..., block_idx * nfpb * 4:(block_idx + 1) * nfpb * 4],
                            lat_motion_frames=lat_motion_frames,
                            cs=cs, ce=ce, nfpb=nfpb,
                            sample_scheduler=sample_scheduler,
                            seed_g=seed_g)

                        # Stream decode
                        if offload_model:
                            self.noise_model.cpu()
                            self.vae.model.to(self.device)
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()

                        if r == 0 and block_idx == 0:
                            self._prime_stream_decoder(motion_latents)

                        image = self._stream_decode_block(
                            block_latents, r, block_idx, frames_per_block)

                        if offload_model:
                            self.vae.model.cpu()
                            self.noise_model.to(self.device)
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()

                        yield image.cpu(), {"clip": r, "block": block_idx}

            finally:
                self._sampler_timesteps = None
                self._sampler_sigmas = None
                self.kv_cache = None
                self.crossattn_cache = None
                if offload_model:
                    gc.collect()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
