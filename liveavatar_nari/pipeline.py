# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Sequential denoising pipeline — single-GPU inference.
import gc
import logging
import math
import os
import random
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


class WanS2V:

    # VAE encoding constants
    REF_WARMUP_FRAMES = 5      # repeat ref image N times for stable VAE encoding
    REF_SKIP_LATENT_FRAMES = 1 # drop first N latent frames (VAE cold start)
    DECODE_SKIP_PIXEL_FRAMES = 3  # drop first N decoded frames (ref padding artifact)

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        offload_kv_cache=False,
    ):
        import time

        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype
        self.checkpoint_dir = checkpoint_dir
        self.offload_kv_cache = offload_kv_cache

        # Lazy imports — deferred to avoid 30s+ module loading at import time
        from .models.causal_model_s2v import CausalWanModel_S2V
        from .models.causal_audio_encoder import AudioEncoder
        from .modules.t5 import T5EncoderModel
        from .modules.vae2_1 import Wan2_1_VAE

        t0 = time.perf_counter()

        # CUDA init (one-time cost, attributed to first GPU operation)
        torch.cuda.init()
        t_cuda = time.perf_counter()
        print(f"[TIMING] CUDA init: {t_cuda - t0:.2f}s")

        # DiT — largest model, loads to GPU
        self.noise_model = CausalWanModel_S2V.from_pretrained(
            checkpoint_dir,
            torch_dtype=self.param_dtype,
            device_map=self.device)
        self.noise_model.freqs.to(device=self.device)
        self.noise_model.eval().requires_grad_(False)
        self.noise_model.num_frame_per_block = config.num_frames_per_block
        t_dit = time.perf_counter()
        print(f"[TIMING] DiT (from_pretrained): {t_dit - t_cuda:.2f}s")

        # VAE — loads to GPU
        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device, dtype=self.param_dtype)
        t_vae = time.perf_counter()
        print(f"[TIMING] VAE: {t_vae - t_dit:.2f}s")

        # T5 text encoder — loads to CPU (safetensors if available, else .pth)
        t5_path = os.path.join(checkpoint_dir, config.t5_checkpoint)
        t5_safetensors_path = t5_path.rsplit('.', 1)[0] + '.safetensors'
        if os.path.exists(t5_safetensors_path):
            self.text_encoder = self._load_t5_from_safetensors(
                t5_safetensors_path, config, checkpoint_dir)
        else:
            self.text_encoder = T5EncoderModel(
                text_len=config.text_len, dtype=config.t5_dtype,
                device=torch.device('cpu'),
                checkpoint_path=t5_path,
                tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer))
        t_t5 = time.perf_counter()
        print(f"[TIMING] T5 text encoder: {t_t5 - t_vae:.2f}s")

        # Wav2Vec audio encoder — lightweight
        self.audio_encoder = AudioEncoder(
            model_id=os.path.join(checkpoint_dir, "wav2vec2-large-xlsr-53-english"))
        t_audio = time.perf_counter()
        print(f"[TIMING] Wav2Vec audio encoder: {t_audio - t_t5:.2f}s")
        print(f"[TIMING] Total __init__: {t_audio - t0:.2f}s")

        self.fps = config.sample_fps
        self.audio_sample_m = 0

        # Dimensions from config
        self.latent_channels = config.transformer.cond_dim           # 16 (= VAE z_dim)
        self.motion_frames = config.transformer.motion_frames        # 73 pixel frames
        self.latent_frames_per_block = config.num_frames_per_block   # 3

        # Strides
        vae_t, vae_s, _ = config.vae_stride                         # (4, 8, 8)
        _, patch_h, patch_w = config.transformer.patch_size          # (1, 2, 2)
        self.vae_temporal_stride = vae_t       # pixel frames per latent frame
        self.vae_spatial_stride = vae_s        # pixel pixels per latent pixel
        self.patch_spatial_stride = patch_h * patch_w  # latent pixels per DiT token

        # DiT dimensions (from loaded model)
        self.dit_num_heads = config.transformer.num_heads                        # 40
        self.dit_head_dim = config.transformer.dim // self.dit_num_heads             # 128
        self.dit_num_layers = config.transformer.num_layers          # 40
        self.dit_max_cond_cache_tokens = 2800  # hardcoded max cond sequence length

        # Audio encoder dimensions (wav2vec2-large-xlsr-53)
        self.audio_feature_dim = config.transformer.audio_dim        # 1024
        self.audio_num_layers = 25  # wav2vec2-large: 24 transformer + 1 feature extraction

    @staticmethod
    def _load_t5_from_safetensors(safetensors_path, config, checkpoint_dir):
        """Load T5 from safetensors with meta-device init (skips parameter allocation)."""
        from safetensors.torch import load_file
        from .modules.t5 import T5EncoderModel, umt5_xxl, HuggingfaceTokenizer

        # Construct model on meta device (no memory allocated)
        with torch.device('meta'):
            model = umt5_xxl(
                encoder_only=True, return_tokenizer=False,
                dtype=config.t5_dtype, device='meta')

        # Load weights directly into the model (assign=True skips copy)
        state_dict = load_file(safetensors_path, device='cpu')
        model.load_state_dict(state_dict, assign=True)
        model = model.eval().requires_grad_(False)

        encoder = T5EncoderModel.__new__(T5EncoderModel)
        encoder.model = model
        encoder.text_len = config.text_len
        encoder.dtype = config.t5_dtype
        encoder.device = torch.device('cpu')
        encoder.tokenizer = HuggingfaceTokenizer(
            name=os.path.join(checkpoint_dir, config.t5_tokenizer),
            seq_len=config.text_len, clean='whitespace')
        return encoder

    def load_lora(self, lora_path, lora_rank=128, lora_alpha=64.0,
                  lora_target_modules="q,k,v,o,ffn.0,ffn.2",
                  init_lora_weights="kaiming"):
        """Load LoRA weights into the DiT, merge, and discard adapter structure."""
        from peft import LoraConfig, get_peft_model
        from .utils.load_weight_utils import load_state_dict

        lora_config = LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha,
            init_lora_weights=True if init_lora_weights == "kaiming" else init_lora_weights,
            target_modules=lora_target_modules.split(","))
        model = get_peft_model(self.noise_model, lora_config)

        state_dict = load_state_dict(lora_path)
        first_key = next(iter(state_dict.keys()))
        if not first_key.startswith("base_model.model."):
            state_dict = {f"base_model.model.{k}": v for k, v in state_dict.items()}

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        n_loaded = sum(1 for _ in model.named_parameters()) - len(missing)
        print(f"{n_loaded} params loaded from {lora_path}. {len(unexpected)} unexpected.")
        print(f"Merging LoRA weights...")
        self.noise_model = model.merge_and_unload()
        print("LoRA merged successfully.")

    def save_model(self, output_dir):
        """Save the current DiT weights (with merged LoRA) for fast reload."""
        os.makedirs(output_dir, exist_ok=True)
        self.noise_model.save_pretrained(output_dir)
        print(f"Saved merged DiT to {output_dir}")

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
            new_height, new_width = int(height * scale), int(width * scale)
            pad_height = (64 - new_height % 64) % 64
            pad_width = (64 - new_width % 64) % 64
            padded_height, padded_width = new_height + pad_height, new_width + pad_width
            if padded_height * padded_width <= max_upper_area:
                return padded_height, padded_width

        aspect_ratio = width / height
        target_width = int((target_area * aspect_ratio)**0.5 // divisor * divisor)
        target_height = int((target_area / aspect_ratio)**0.5 // divisor * divisor)
        if target_width >= width or target_height >= height:
            target_width = int(width // divisor * divisor)
            target_height = int(height // divisor * divisor)
        return target_height, target_width

    def encode_audio(self, audio_path, infer_frames):
        z = self.audio_encoder.extract_audio_feat(audio_path, return_all_layers=True)
        audio_embed_bucket, num_clips = self.audio_encoder.get_audio_embed_bucket_fps(
            z, fps=self.fps, batch_frames=infer_frames, m=self.audio_sample_m)
        audio_embed_bucket = audio_embed_bucket.to(self.device, self.param_dtype)
        audio_embed_bucket = audio_embed_bucket.unsqueeze(0)
        if len(audio_embed_bucket.shape) == 3:
            audio_embed_bucket = audio_embed_bucket.permute(0, 2, 1)
        elif len(audio_embed_bucket.shape) == 4:
            audio_embed_bucket = audio_embed_bucket.permute(0, 2, 3, 1)
        return audio_embed_bucket, num_clips

    def encode_prompt(self, input_prompt, offload_model=True):
        self.text_encoder.model.to(self.device)
        text_prompt_embeddings = self.text_encoder([input_prompt], self.device)
        if offload_model:
            self.text_encoder.model.cpu()
        return text_prompt_embeddings

    def _initialize_kv_cache(self, batch_size, dtype, device, kv_cache_size=13500):
        cache_device = "cpu" if self.offload_kv_cache else device
        nh, hd = self.dit_num_heads, self.dit_head_dim
        self.kv_cache = [{
            "k": torch.zeros([batch_size, kv_cache_size, nh, hd], dtype=dtype, device=cache_device),
            "v": torch.zeros([batch_size, kv_cache_size, nh, hd], dtype=dtype, device=cache_device),
            "cond_k": torch.zeros([batch_size, self.dit_max_cond_cache_tokens, nh, hd], dtype=dtype, device=cache_device),
            "cond_v": torch.zeros([batch_size, self.dit_max_cond_cache_tokens, nh, hd], dtype=dtype, device=cache_device),
            "cond_end": torch.tensor([0], dtype=torch.long, device=cache_device),
        } for _ in range(self.dit_num_layers)]

    def _move_kv_cache_to_device(self, device):
        for layer in self.kv_cache:
            for key in ["k", "v", "cond_k", "cond_v", "cond_end"]:
                layer[key] = layer[key].to(device)

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        nh, hd = self.dit_num_heads, self.dit_head_dim
        self.crossattn_cache = [{
            "k": torch.zeros([batch_size, 0, nh, hd], dtype=dtype, device=device),
            "v": torch.zeros([batch_size, 0, nh, hd], dtype=dtype, device=device),
            "is_init": False,
        } for _ in range(self.dit_num_layers)]

    def _prefill_cond_cache(self, text_prompt_embeddings, motion_latents, ref_image_latents,
                           latent_motion_frames, latent_frames_per_block,
                           tokens_per_latent_frame, num_denoising_steps):
        """Cache conditioning (ref, motion, text) into KV cache, broadcast to all slots.

        The model's _forward_sink requires block_latents, cond, and audio in its
        signature, but discards their values. We pass zeros for those.
        """
        prefill_kv = [{
            "k": layer["k"][0:1], "v": layer["v"][0:1],
            "cond_k": layer["cond_k"][0:1], "cond_v": layer["cond_v"][0:1],
            "cond_end": layer["cond_end"],
        } for layer in self.kv_cache]
        prefill_crossattn = [{
            "k": layer["k"][0:1], "v": layer["v"][0:1],
            "is_init": layer["is_init"],
        } for layer in self.crossattn_cache]

        if self.offload_kv_cache:
            self._move_kv_cache_to_device(self.device)

        # Dummies — required by model signature but values are discarded
        latent_h, latent_w = motion_latents.shape[3], motion_latents.shape[4]
        fpb = latent_frames_per_block
        dummy_block = torch.zeros(self.latent_channels, fpb, latent_h, latent_w,
                                  dtype=self.param_dtype, device=self.device)
        dummy_cond = torch.zeros(1, self.latent_channels, fpb, latent_h, latent_w,
                                 dtype=self.param_dtype, device=self.device)
        dummy_audio = torch.zeros(1, self.audio_num_layers, self.audio_feature_dim,
                                  fpb * self.vae_temporal_stride,
                                  dtype=self.param_dtype, device=self.device)

        self.noise_model(
            [dummy_block],
            t=torch.zeros([1, fpb], device=self.device, dtype=self.param_dtype),
            context=text_prompt_embeddings[0:1], seq_len=None,
            ref_latents=ref_image_latents,
            motion_latents=motion_latents,
            cond_states=dummy_cond,
            audio_input=dummy_audio,
            motion_frames=[self.motion_frames, latent_motion_frames],
            drop_motion_frames=False,
            sink_flag=True,
            kv_cache=prefill_kv, crossattn_cache=prefill_crossattn,
            current_start=0,
            current_end=fpb * tokens_per_latent_frame)

        # Broadcast cond cache from slot 0 to all slots
        for li in range(len(self.kv_cache)):
            for key in ["cond_k", "cond_v"]:
                self.kv_cache[li][key][:] = prefill_kv[li][key]
            self.kv_cache[li]["cond_end"] = prefill_kv[li]["cond_end"]

        if self.offload_kv_cache:
            self._move_kv_cache_to_device("cpu")

    def _prepare_image(self, ref_image_path, max_area):
        """Encode reference image → ref_image_latents + motion_latents."""
        ref_image = np.array(Image.open(ref_image_path).convert('RGB'))
        HEIGHT, WIDTH = self.get_size_less_than_area(
            ref_image.shape[0], ref_image.shape[1], target_area=max_area)

        resize_op = transforms.Resize(min(HEIGHT, WIDTH))
        crop_op = transforms.CenterCrop((HEIGHT, WIDTH))

        self.vae.model.to(self.device)

        # Preprocess reference image → pixel tensor [-1, 1]
        ref_pixel = transforms.ToTensor()(crop_op(resize_op(Image.fromarray(ref_image))))
        ref_pixel = ref_pixel.unsqueeze(1).unsqueeze(0) * 2 - 1.0
        ref_pixel = ref_pixel.to(dtype=self.vae.dtype, device=self.vae.device)

        # VAE encode: ref image repeated as motion context (5 * 73 = 365 pixel frames)
        # First REF_WARMUP_FRAMES copies warm up the VAE's causal convolutions.
        # ref_image_latents is one stable latent frame sliced from the result.
        motion_pixel_frames = ref_pixel.repeat(1, 1, self.REF_WARMUP_FRAMES * self.motion_frames, 1, 1)
        motion_latents = torch.stack(self.vae.encode(motion_pixel_frames))
        ref_image_latents = motion_latents[:, :, self.REF_SKIP_LATENT_FRAMES:self.REF_SKIP_LATENT_FRAMES + 1]

        return ref_image_latents, motion_latents, motion_pixel_frames.detach(), HEIGHT, WIDTH

    def _prepare_text(self, input_prompt, offload_model):
        """Encode text prompt → T5 embeddings."""
        return self.encode_prompt(input_prompt, offload_model)

    def _prepare_audio(self, audio_path, infer_frames, num_clips):
        """Encode full audio file and determine clip count."""
        self.audio_encoder.model.to(device=self.device, dtype=self.param_dtype)
        self.audio_encoder.model.requires_grad_(False)
        self.audio_encoder.model.eval()

        audio_embeddings, max_clips = self.encode_audio(audio_path, infer_frames=infer_frames)
        self.audio_encoder.model.to("cpu")

        if num_clips is None or num_clips > max_clips:
            num_clips = max_clips

        return audio_embeddings, num_clips

    def generate(
        self,
        input_prompt=None,
        ref_image_path=None,
        audio_path=None,
        num_clips=1,
        max_area=720 * 1280,
        infer_frames=80,
        sampling_steps=4,
        seed=-1,
        offload_model=True,
        max_clips=1000000,
    ):
        num_denoising_steps = sampling_steps
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)

        # ---- 1. Prepare conditional inputs ----
        ref_image_latents, motion_latents, motion_pixel_frames, HEIGHT, WIDTH = \
            self._prepare_image(ref_image_path, max_area)
        text_prompt_embeddings = self._prepare_text(input_prompt, offload_model)
        audio_embeddings, num_clips = self._prepare_audio(
            audio_path, infer_frames, num_clips)

        # Latent space dimensions
        latent_motion_frames = math.ceil(self.motion_frames / self.vae_temporal_stride)
        latent_h = HEIGHT // self.vae_spatial_stride
        latent_w = WIDTH // self.vae_spatial_stride
        latent_target_frames = (
            math.ceil((infer_frames + self.motion_frames) / self.vae_temporal_stride)
            - latent_motion_frames)
        latent_shape = [latent_target_frames, latent_h, latent_w]

        # Block/token dimensions
        latent_frames_per_block = self.latent_frames_per_block
        num_blocks = latent_target_frames // latent_frames_per_block
        tokens_per_latent_frame = latent_h * latent_w // self.patch_spatial_stride
        tokens_per_block = latent_frames_per_block * tokens_per_latent_frame
        max_tokens = np.prod(latent_shape) // self.patch_spatial_stride
        audio_frames_per_block = latent_frames_per_block * self.vae_temporal_stride

        from diffusers import FlowMatchEulerDiscreteScheduler
        sample_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=self.num_train_timesteps, shift=3)

        # ---- 2. Generate clips ----
        with torch.amp.autocast('cuda', dtype=self.param_dtype), torch.no_grad():
            decoded_clips = []
            clip_latent_outputs = []
            total_clips = min(max_clips, num_clips)

            dummy_cond = torch.zeros(
                1, self.latent_channels, latent_frames_per_block, latent_h, latent_w,
                dtype=self.param_dtype, device=self.device)

            if offload_model:
                self.noise_model.to(self.device)
                self.vae.model.cpu()
                self.text_encoder.model.cpu()
                self.audio_encoder.model.cpu()
                torch.cuda.empty_cache()

            self._initialize_kv_cache(num_denoising_steps, self.param_dtype, self.device, max_tokens)
            self._initialize_crossattn_cache(num_denoising_steps, self.param_dtype, self.device)

            # ---- Prefill cond cache ----
            self._prefill_cond_cache(
                text_prompt_embeddings=text_prompt_embeddings,
                motion_latents=motion_latents,
                ref_image_latents=ref_image_latents,
                latent_motion_frames=latent_motion_frames,
                latent_frames_per_block=latent_frames_per_block,
                tokens_per_latent_frame=tokens_per_latent_frame,
                num_denoising_steps=num_denoising_steps)

            # ---- Setup scheduler ----
            sample_scheduler.set_timesteps(sampling_steps, device=self.device)
            self._sampler_timesteps = sample_scheduler.timesteps
            self._sampler_sigmas = sample_scheduler.sigmas
            timesteps = self._sampler_timesteps

            for clip_index in range(total_clips):
                # ---- Clip-level setup ----
                seed_g = torch.Generator(device=self.device)
                seed_g.manual_seed(seed + clip_index)
                clip_noise = torch.randn(
                    self.latent_channels, *latent_shape,
                    dtype=self.param_dtype, device=self.device, generator=seed_g)
                clip_audio = audio_embeddings[..., clip_index * infer_frames:(clip_index + 1) * infer_frames]
                input_motion_latents = motion_latents.clone()
                clip_output = torch.zeros_like(clip_noise)

                if offload_model:
                    self.noise_model.to(self.device)
                    self.vae.model.cpu()
                    torch.cuda.empty_cache()

                # ---- Denoising ----
                clip_token_offset = clip_index * num_blocks * tokens_per_block

                for block_index in tqdm(range(num_blocks), desc=f"clip {clip_index}"):
                    block_start = block_index * latent_frames_per_block
                    block_end = block_start + latent_frames_per_block
                    block_latents = clip_noise[:, block_start:block_end]

                    audio_start = block_index * audio_frames_per_block
                    audio_end = audio_start + audio_frames_per_block

                    token_start = block_index * tokens_per_block + clip_token_offset
                    token_end = token_start + tokens_per_block

                    sample_scheduler.timesteps = self._sampler_timesteps
                    sample_scheduler.sigmas = self._sampler_sigmas
                    sample_scheduler._step_index = 0
                    sample_scheduler._begin_index = 0

                    for step_index, t in enumerate(timesteps):
                        if self.offload_kv_cache:
                            self._move_kv_cache_to_device(self.device)

                        step_kv = [{
                            "k": layer["k"][step_index:step_index+1],
                            "v": layer["v"][step_index:step_index+1],
                            "cond_k": layer["cond_k"][step_index:step_index+1],
                            "cond_v": layer["cond_v"][step_index:step_index+1],
                            "cond_end": layer["cond_end"],
                        } for layer in self.kv_cache]
                        step_crossattn = [{
                            "k": layer["k"][step_index:step_index+1],
                            "v": layer["v"][step_index:step_index+1],
                            "is_init": layer["is_init"],
                        } for layer in self.crossattn_cache]

                        noise_pred = self.noise_model(
                            [block_latents],
                            t=t.unsqueeze(0).expand(1, latent_frames_per_block),
                            context=text_prompt_embeddings[0:1], seq_len=None,
                            cond_states=dummy_cond,
                            motion_latents=input_motion_latents,
                            ref_latents=ref_image_latents,
                            audio_input=clip_audio[..., audio_start:audio_end],
                            motion_frames=[self.motion_frames, latent_motion_frames],
                            drop_motion_frames=False,
                            kv_cache=step_kv, crossattn_cache=step_crossattn,
                            current_start=token_start, current_end=token_end)

                        if self.offload_kv_cache:
                            self._move_kv_cache_to_device("cpu")

                        block_latents = sample_scheduler.step(
                            noise_pred[0].unsqueeze(0), t,
                            block_latents.unsqueeze(0),
                            return_dict=False, generator=seed_g
                        )[0].squeeze(0)

                    clip_output[:, block_start:block_end] = block_latents

                    # AAS: after first block, replace sink with generated latent
                    if clip_index == 0 and block_index == 0:
                        ref_image_latents = block_latents.unsqueeze(0)[:, :, 0:1]

                clip_latent_outputs.append(clip_output.detach().cpu())

        # ---- 3. Deferred VAE decode ----
        print("complete full-sequence generation")
        if clip_latent_outputs:
            if offload_model:
                print("loading VAE for final decode")
                self.kv_cache = None
                self.vae.model.to(self.device)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            motion_latents_decode = motion_latents
            for clip_idx, clip_latent_cpu in enumerate(clip_latent_outputs):
                clip_latent = clip_latent_cpu.to(
                    device=self.vae.device, dtype=self.vae.dtype)
                decode_input = torch.cat(
                    [motion_latents_decode, clip_latent.unsqueeze(0)], dim=2)
                image = torch.stack(self.vae.decode(decode_input))
                image = image[:, :, -(infer_frames):]
                if clip_idx == 0:
                    image = image[:, :, self.DECODE_SKIP_PIXEL_FRAMES:]

                overlap = min(self.motion_frames, image.shape[2])
                motion_pixel_frames = torch.cat([
                    motion_pixel_frames[:, :, overlap:],
                    image[:, :, -overlap:],
                ], dim=2)
                motion_pixel_frames = motion_pixel_frames.to(
                    dtype=motion_latents_decode.dtype, device=motion_latents_decode.device)
                motion_latents_decode = torch.stack(
                    self.vae.encode(motion_pixel_frames)
                ).type_as(clip_latent)
                decoded_clips.append(image.cpu())

        video = torch.cat(decoded_clips, dim=2)
        del clip_noise, clip_output
        self._sampler_timesteps = None
        self._sampler_sigmas = None
        self.kv_cache = None
        self.crossattn_cache = None
        if offload_model:
            self.vae.model.cpu()
            self.noise_model.to(self.device)
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        return video[0], {}
