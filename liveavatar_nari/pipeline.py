# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Sequential denoising pipeline — single-GPU inference.
import gc
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
    REF_WARMUP_FRAMES = 3  # repeat ref image N times for stable VAE encoding
    REF_SKIP_LATENT_FRAMES = 1  # drop first N latent frames (VAE cold start)
    DECODE_SKIP_PIXEL_FRAMES = 3  # drop first N decoded frames (ref padding artifact)

    def __init__(self, config, checkpoint_dir, device_id=0, offload_kv_cache=False):
        import time

        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype
        self.checkpoint_dir = checkpoint_dir
        self.offload_kv_cache = offload_kv_cache

        from .models.causal_model_s2v import CausalWanModel_S2V
        from .modules.s2v.audio_encoder import AudioEncoder
        from .modules.t5 import T5EncoderModel
        from .modules.vae_streaming import WanVAE

        t0 = time.perf_counter()
        torch.cuda.init()
        t_cuda = time.perf_counter()
        print(f"[TIMING] CUDA init: {t_cuda - t0:.2f}s")

        # DiT
        self.noise_model = CausalWanModel_S2V.from_pretrained(
            checkpoint_dir, torch_dtype=self.param_dtype, device_map=self.device
        )
        self.noise_model.freqs.to(device=self.device)
        self.noise_model.eval().requires_grad_(False)
        t_dit = time.perf_counter()
        print(f"[TIMING] DiT: {t_dit - t_cuda:.2f}s")

        # VAE
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device,
            dtype=self.param_dtype,
        )
        t_vae = time.perf_counter()
        print(f"[TIMING] VAE: {t_vae - t_dit:.2f}s")

        # T5
        t5_path = os.path.join(checkpoint_dir, config.t5_checkpoint)
        t5_safetensors_path = t5_path.rsplit(".", 1)[0] + ".safetensors"
        if os.path.exists(t5_safetensors_path):
            self.text_encoder = self._load_t5_from_safetensors(t5_safetensors_path, config, checkpoint_dir)
        else:
            self.text_encoder = T5EncoderModel(
                text_len=config.text_len,
                dtype=config.t5_dtype,
                device=torch.device("cpu"),
                checkpoint_path=t5_path,
                tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            )
        t_t5 = time.perf_counter()
        print(f"[TIMING] T5: {t_t5 - t_vae:.2f}s")

        # Wav2Vec
        self.audio_encoder = AudioEncoder(model_id=os.path.join(checkpoint_dir, "wav2vec2-large-xlsr-53-english"))
        t_audio = time.perf_counter()
        print(f"[TIMING] Wav2Vec: {t_audio - t_t5:.2f}s")
        print(f"[TIMING] Total __init__: {t_audio - t0:.2f}s")

        self.fps = config.sample_fps

        # Dimensions from config
        self.latent_channels = config.transformer.cond_dim
        self.motion_frames = config.transformer.motion_frames
        self.latent_frames_per_block = config.num_frames_per_block

        vae_t, vae_s, _ = config.vae_stride
        _, patch_h, patch_w = config.transformer.patch_size
        self.vae_temporal_stride = vae_t
        self.vae_spatial_stride = vae_s
        self.patch_spatial_stride = patch_h * patch_w

        self.dit_num_heads = config.transformer.num_heads
        self.dit_head_dim = config.transformer.dim // self.dit_num_heads
        self.dit_num_layers = config.transformer.num_layers
        self.dit_max_cond_cache_tokens = 2800

    @staticmethod
    def _load_t5_from_safetensors(safetensors_path, config, checkpoint_dir):
        """Load T5 from safetensors with meta-device init (skips parameter allocation)."""
        from safetensors.torch import load_file
        from .modules.t5 import T5EncoderModel, umt5_xxl, HuggingfaceTokenizer

        with torch.device("meta"):
            model = umt5_xxl(encoder_only=True, return_tokenizer=False, dtype=config.t5_dtype, device="meta")

        state_dict = load_file(safetensors_path, device="cpu")
        model.load_state_dict(state_dict, assign=True)
        model = model.eval().requires_grad_(False)

        encoder = T5EncoderModel.__new__(T5EncoderModel)
        encoder.model = model
        encoder.text_len = config.text_len
        encoder.dtype = config.t5_dtype
        encoder.device = torch.device("cpu")
        encoder.tokenizer = HuggingfaceTokenizer(
            name=os.path.join(checkpoint_dir, config.t5_tokenizer),
            seq_len=config.text_len,
            clean="whitespace",
        )
        return encoder

    def load_lora(
        self,
        lora_path,
        lora_rank=128,
        lora_alpha=64.0,
        lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        init_lora_weights="kaiming",
    ):
        """Load LoRA weights into the DiT, merge, and discard adapter structure."""
        from peft import LoraConfig, get_peft_model
        from .utils.load_weight_utils import load_state_dict

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=True if init_lora_weights == "kaiming" else init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = get_peft_model(self.noise_model, lora_config)
        state_dict = load_state_dict(lora_path)
        first_key = next(iter(state_dict.keys()))
        if not first_key.startswith("base_model.model."):
            state_dict = {f"base_model.model.{k}": v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        n_loaded = sum(1 for _ in model.named_parameters()) - len(missing)
        print(f"{n_loaded} params loaded from {lora_path}. {len(unexpected)} unexpected.")
        self.noise_model = model.merge_and_unload()
        print("LoRA merged.")

    def save_model(self, output_dir):
        """Save the current DiT weights (with merged LoRA) for fast reload."""
        os.makedirs(output_dir, exist_ok=True)
        self.noise_model.save_pretrained(output_dir)
        print(f"Saved merged DiT to {output_dir}")

    # --- Encoding helpers ---

    def _encode_image(self, ref_image_path, max_area):
        """Encode reference image → (ref_image_latents, motion_latents, HEIGHT, WIDTH)."""
        ref_image = np.array(Image.open(ref_image_path).convert("RGB"))
        HEIGHT, WIDTH = self._fit_to_area(ref_image.shape[0], ref_image.shape[1], target_area=max_area)

        ref_pixel = transforms.ToTensor()(
            transforms.CenterCrop((HEIGHT, WIDTH))(transforms.Resize(min(HEIGHT, WIDTH))(Image.fromarray(ref_image)))
        )
        ref_pixel = ref_pixel.unsqueeze(1).unsqueeze(0) * 2 - 1.0
        ref_pixel = ref_pixel.to(dtype=self.vae.dtype, device=self.vae.device)

        motion_pixel = ref_pixel.repeat(1, 1, self.REF_WARMUP_FRAMES * self.motion_frames, 1, 1)
        motion_latents = torch.stack(self.vae.encode(motion_pixel))
        ref_image_latents = motion_latents[:, :, self.REF_SKIP_LATENT_FRAMES : self.REF_SKIP_LATENT_FRAMES + 1]

        return ref_image_latents, motion_latents, HEIGHT, WIDTH

    def _encode_text(self, input_prompt, offload_model=True):
        """Encode text prompt → T5 embeddings."""
        self.text_encoder.model.to(self.device)
        embeddings = self.text_encoder([input_prompt], self.device)
        if offload_model:
            self.text_encoder.model.cpu()
        return embeddings

    def _encode_audio(self, audio_path, infer_frames):
        """Encode full audio file → audio embeddings [1, layers, dim, pixel_frames]."""
        self.audio_encoder.model.to(device=self.device, dtype=self.param_dtype)
        self.audio_encoder.model.requires_grad_(False)
        self.audio_encoder.model.eval()

        z = self.audio_encoder.extract_audio_feat(audio_path, return_all_layers=True)
        audio_embed, _ = self.audio_encoder.get_audio_embed_bucket_fps(z, fps=self.fps, batch_frames=infer_frames, m=0)
        audio_embed = audio_embed.to(self.device, self.param_dtype).unsqueeze(0)
        if audio_embed.dim() == 3:
            audio_embed = audio_embed.permute(0, 2, 1)
        elif audio_embed.dim() == 4:
            audio_embed = audio_embed.permute(0, 2, 3, 1)

        self.audio_encoder.model.to("cpu")
        return audio_embed

    @staticmethod
    def _fit_to_area(height, width, target_area=1024 * 704, divisor=64):
        """Find largest (H, W) that fits in target_area, divisible by divisor."""
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

        aspect_ratio = width / height
        tw = int((target_area * aspect_ratio) ** 0.5 // divisor * divisor)
        th = int((target_area / aspect_ratio) ** 0.5 // divisor * divisor)
        if tw >= width or th >= height:
            tw = int(width // divisor * divisor)
            th = int(height // divisor * divisor)
        return th, tw

    # --- KV cache ---

    def _initialize_kv_cache(self, num_steps, dtype, device, kv_cache_size):
        cache_device = "cpu" if self.offload_kv_cache else device
        nh, hd = self.dit_num_heads, self.dit_head_dim
        self.kv_cache = [
            {
                "k": torch.zeros([num_steps, kv_cache_size, nh, hd], dtype=dtype, device=cache_device),
                "v": torch.zeros([num_steps, kv_cache_size, nh, hd], dtype=dtype, device=cache_device),
                "cond_k": torch.zeros(
                    [num_steps, self.dit_max_cond_cache_tokens, nh, hd], dtype=dtype, device=cache_device
                ),
                "cond_v": torch.zeros(
                    [num_steps, self.dit_max_cond_cache_tokens, nh, hd], dtype=dtype, device=cache_device
                ),
                "cond_end": torch.tensor([0], dtype=torch.long, device=cache_device),
            }
            for _ in range(self.dit_num_layers)
        ]

    def _initialize_crossattn_cache(self, num_steps, dtype, device):
        nh, hd = self.dit_num_heads, self.dit_head_dim
        self.crossattn_cache = [
            {
                "k": torch.zeros([num_steps, 0, nh, hd], dtype=dtype, device=device),
                "v": torch.zeros([num_steps, 0, nh, hd], dtype=dtype, device=device),
                "is_init": False,
            }
            for _ in range(self.dit_num_layers)
        ]

    def _move_kv_cache_to_device(self, device):
        for layer in self.kv_cache:
            for key in layer:
                layer[key] = layer[key].to(device)

    def _prefill_cond_cache(
        self,
        text_embeddings,
        motion_latents,
        ref_image_latents,
        latent_motion_frames,
        latent_frames_per_block,
        tokens_per_latent_frame,
        num_steps,
    ):
        """Cache conditioning tokens into KV cache, broadcast to all timestep slots."""
        prefill_kv = [
            {
                "k": l["k"][0:1],
                "v": l["v"][0:1],
                "cond_k": l["cond_k"][0:1],
                "cond_v": l["cond_v"][0:1],
                "cond_end": l["cond_end"],
            }
            for l in self.kv_cache
        ]
        prefill_crossattn = [
            {"k": l["k"][0:1], "v": l["v"][0:1], "is_init": l["is_init"]} for l in self.crossattn_cache
        ]

        if self.offload_kv_cache:
            self._move_kv_cache_to_device(self.device)

        latent_h, latent_w = motion_latents.shape[3], motion_latents.shape[4]
        self.noise_model.prefill_cond_cache(
            ref_latents=ref_image_latents,
            motion_latents=motion_latents,
            context=text_embeddings[0:1],
            motion_frames=[self.motion_frames, latent_motion_frames],
            kv_cache=prefill_kv,
            crossattn_cache=prefill_crossattn,
            latent_shape=(latent_h, latent_w),
            current_end=latent_frames_per_block * tokens_per_latent_frame,
            latent_frames_per_block=latent_frames_per_block,
        )

        for li in range(len(self.kv_cache)):
            for key in ["cond_k", "cond_v"]:
                self.kv_cache[li][key][:] = prefill_kv[li][key]
            self.kv_cache[li]["cond_end"] = prefill_kv[li]["cond_end"]

        if self.offload_kv_cache:
            self._move_kv_cache_to_device("cpu")

    # --- Generation ---

    def generate(
        self,
        input_prompt=None,
        ref_image_path=None,
        audio_path=None,
        max_blocks=None,
        max_area=720 * 1280,
        infer_frames=80,
        sampling_steps=4,
        seed=-1,
        offload_model=True,
    ):
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)

        # ---- 1. Encode inputs ----
        self.vae.model.to(self.device)
        ref_image_latents, motion_latents, HEIGHT, WIDTH = self._encode_image(ref_image_path, max_area)
        text_embeddings = self._encode_text(input_prompt, offload_model)
        audio_embeddings = self._encode_audio(audio_path, infer_frames)

        # ---- 2. Compute dimensions ----
        latent_motion_frames = math.ceil(self.motion_frames / self.vae_temporal_stride)
        latent_h = HEIGHT // self.vae_spatial_stride
        latent_w = WIDTH // self.vae_spatial_stride
        fpb = self.latent_frames_per_block
        tokens_per_frame = latent_h * latent_w // self.patch_spatial_stride
        tokens_per_block = fpb * tokens_per_frame
        audio_frames_per_block = fpb * self.vae_temporal_stride
        pixel_frames_per_block = fpb * self.vae_temporal_stride

        # Total blocks from audio length; KV cache sized per rolling window
        total_blocks = audio_embeddings.shape[-1] // audio_frames_per_block
        if max_blocks is not None:
            total_blocks = min(total_blocks, max_blocks)
        kv_window_blocks = (
            math.ceil((infer_frames + self.motion_frames) / self.vae_temporal_stride) - latent_motion_frames
        ) // fpb
        max_tokens = kv_window_blocks * tokens_per_block

        from diffusers import FlowMatchEulerDiscreteScheduler

        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=self.num_train_timesteps, shift=3)

        # ---- 3. DiT blockwise generation ----
        with torch.amp.autocast("cuda", dtype=self.param_dtype), torch.no_grad():
            if offload_model:
                self.noise_model.to(self.device)
                self.vae.model.cpu()
                self.text_encoder.model.cpu()
                self.audio_encoder.model.cpu()
                torch.cuda.empty_cache()

            self._initialize_kv_cache(sampling_steps, self.param_dtype, self.device, max_tokens)
            self._initialize_crossattn_cache(sampling_steps, self.param_dtype, self.device)
            self._prefill_cond_cache(
                text_embeddings,
                motion_latents,
                ref_image_latents,
                latent_motion_frames,
                fpb,
                tokens_per_frame,
                sampling_steps,
            )

            scheduler.set_timesteps(sampling_steps, device=self.device)
            timesteps = scheduler.timesteps
            saved_sigmas = scheduler.sigmas.clone()

            seed_g = torch.Generator(device=self.device)
            seed_g.manual_seed(seed)

            dummy_cond = torch.zeros(
                1, self.latent_channels, fpb, latent_h, latent_w, dtype=self.param_dtype, device=self.device
            )

            S = sampling_steps
            # In-flight pipeline state: one slot per denoising step
            inflight_latents = [None] * S
            inflight_audio = [None] * S
            inflight_token_start = [0] * S
            inflight_block_idx = [-1] * S

            all_block_latents = []
            total_groups = total_blocks + S - 1

            for group in tqdm(range(total_groups), desc="blocks"):
                for step_idx, t in reversed(list(enumerate(timesteps))):
                    block_idx = group - step_idx

                    # Skip dummy blocks (warmup/drain)
                    if block_idx < 0 or block_idx >= total_blocks:
                        continue

                    if step_idx == 0:
                        # New block enters pipeline — generate noise
                        inflight_latents[0] = torch.randn(
                            self.latent_channels,
                            fpb,
                            latent_h,
                            latent_w,
                            dtype=self.param_dtype,
                            device=self.device,
                            generator=seed_g,
                        )
                        inflight_audio[0] = audio_embeddings[
                            ..., block_idx * audio_frames_per_block : (block_idx + 1) * audio_frames_per_block
                        ]
                        inflight_token_start[0] = block_idx * tokens_per_block
                        inflight_block_idx[0] = block_idx

                    block_latents = inflight_latents[step_idx]

                    if self.offload_kv_cache:
                        self._move_kv_cache_to_device(self.device)

                    step_kv = [
                        {
                            "k": l["k"][step_idx : step_idx + 1],
                            "v": l["v"][step_idx : step_idx + 1],
                            "cond_k": l["cond_k"][step_idx : step_idx + 1],
                            "cond_v": l["cond_v"][step_idx : step_idx + 1],
                            "cond_end": l["cond_end"],
                        }
                        for l in self.kv_cache
                    ]
                    step_crossattn = [
                        {
                            "k": l["k"][step_idx : step_idx + 1],
                            "v": l["v"][step_idx : step_idx + 1],
                            "is_init": l["is_init"],
                        }
                        for l in self.crossattn_cache
                    ]

                    token_start = inflight_token_start[step_idx]
                    noise_pred = self.noise_model(
                        [block_latents],
                        t=t.unsqueeze(0).expand(1, fpb),
                        context=text_embeddings[0:1],
                        cond_states=dummy_cond,
                        audio_input=inflight_audio[step_idx],
                        motion_frames=[self.motion_frames, latent_motion_frames],
                        kv_cache=step_kv,
                        crossattn_cache=step_crossattn,
                        current_start=torch.tensor([token_start], device=self.device),
                        current_end=token_start + tokens_per_block,
                    )

                    if self.offload_kv_cache:
                        self._move_kv_cache_to_device("cpu")

                    scheduler.sigmas = saved_sigmas
                    scheduler._step_index = step_idx
                    scheduler._begin_index = 0
                    block_latents = scheduler.step(
                        noise_pred[0].unsqueeze(0),
                        t,
                        block_latents.unsqueeze(0),
                        return_dict=False,
                        generator=seed_g,
                    )[0].squeeze(0)

                    if step_idx == S - 1:
                        # Block completed all steps — output it
                        all_block_latents.append(block_latents.detach().cpu())
                        inflight_latents[step_idx] = None
                        # AAS: first completed block replaces sink
                        if inflight_block_idx[step_idx] == 0:
                            ref_image_latents = block_latents.unsqueeze(0)[:, :, 0:1]
                    else:
                        # Advance to next pipeline stage
                        inflight_latents[step_idx + 1] = block_latents
                        inflight_audio[step_idx + 1] = inflight_audio[step_idx]
                        inflight_token_start[step_idx + 1] = inflight_token_start[step_idx]
                        inflight_block_idx[step_idx + 1] = inflight_block_idx[step_idx]
                        inflight_latents[step_idx] = None

        # ---- 4. Streaming VAE decode ----
        print(f"DiT complete ({total_blocks} blocks). Decoding...")
        decoded_blocks = []
        if all_block_latents:
            if offload_model:
                self.kv_cache = None
                self.crossattn_cache = None
                self.vae.model.to(self.device)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            self.vae.model.clear_cache_decode()
            warmup = motion_latents[:, :, :7].to(device=self.vae.device, dtype=self.vae.dtype)
            self.vae.stream_decode(warmup)

            for i, blk in enumerate(all_block_latents):
                image = torch.stack(
                    self.vae.stream_decode(blk.unsqueeze(0).to(device=self.vae.device, dtype=self.vae.dtype))
                )
                image = image[:, :, -pixel_frames_per_block:]
                if i == 0:
                    image = image[:, :, self.DECODE_SKIP_PIXEL_FRAMES :]
                decoded_blocks.append(image.cpu())

        video = torch.cat(decoded_blocks, dim=2)

        if offload_model:
            self.vae.model.cpu()
            self.noise_model.to(self.device)
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        return video[0], {}
