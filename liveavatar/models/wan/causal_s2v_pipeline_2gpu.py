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
from peft import LoraConfig, get_peft_model
from diffusers import FlowMatchEulerDiscreteScheduler
from .causal_audio_encoder import AudioEncoder
from .causal_model_s2v import CausalWanModel_S2V
from .wan_2_2.modules.t5 import T5EncoderModel
from .wan_2_2.modules.vae2_1 import Wan2_1_VAE
from ...utils.load_weight_utils import load_state_dict


class WanS2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        offload_kv_cache=False,
    ):
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.num_train_timesteps = config.num_train_timesteps  # 1000
        self.num_frames_per_block = config.num_frames_per_block
        self.param_dtype = config.param_dtype
        self.checkpoint_dir = checkpoint_dir
        self.offload_kv_cache = offload_kv_cache

        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
        )

        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device, dtype=self.param_dtype)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.noise_model = CausalWanModel_S2V.from_pretrained(
            checkpoint_dir,
            torch_dtype=self.param_dtype,
            device_map=self.device)

        self.noise_model.freqs.to(device=self.device)
        self.noise_model.eval().requires_grad_(False)
        self.noise_model.num_frame_per_block = self.num_frames_per_block

        self.audio_encoder = AudioEncoder(
            model_id=os.path.join(checkpoint_dir, "wav2vec2-large-xlsr-53-english"))

        self.sample_neg_prompt = config.sample_neg_prompt
        self.motion_frames = config.transformer.motion_frames
        # config.drop_first_motion is True (training augmentation), but at inference
        # we keep motion frames so the first clip stays faithful to the reference image
        self.drop_first_motion = False
        self.fps = config.sample_fps
        self.audio_sample_m = 0

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
                target_modules=lora_target_modules.split(","),
            )
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
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            all_keys = [i for i, _ in model.named_parameters()]
            print(f"{len(all_keys) - len(missing_keys)} params loaded from {pretrained_lora_path}. "
                  f"{len(unexpected_keys)} unexpected.")
            print(f"Merging LoRA weights from {pretrained_lora_path}...")
            model = model.merge_and_unload()
            print(f"LoRA merged successfully.")
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
        audio_embed_bucket, num_repeat = self.audio_encoder.get_audio_embed_bucket_fps(
            z, fps=self.fps, batch_frames=infer_frames, m=self.audio_sample_m)
        audio_embed_bucket = audio_embed_bucket.to(self.device, self.param_dtype)
        audio_embed_bucket = audio_embed_bucket.unsqueeze(0)
        if len(audio_embed_bucket.shape) == 3:
            audio_embed_bucket = audio_embed_bucket.permute(0, 2, 1)
        elif len(audio_embed_bucket.shape) == 4:
            audio_embed_bucket = audio_embed_bucket.permute(0, 2, 3, 1)
        return audio_embed_bucket, num_repeat

    def encode_prompt(self, input_prompt, n_prompt=None, offload_model=True):
        context_null = None
        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        self.text_encoder.model.to(self.device)
        context = self.text_encoder([input_prompt], self.device)
        if n_prompt is not None:
            context_null = self.text_encoder([n_prompt], self.device)
        if offload_model:
            self.text_encoder.model.cpu()
        return context, context_null

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

    def _prefill_cond_cache(self, context, motion_latents, ref_latents,
                           lat_motion_frames, nfpb, frame_seq_length, num_steps):
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
        h, w = motion_latents.shape[3], motion_latents.shape[4]
        dummy_block = torch.zeros(16, nfpb, h, w,
                                  dtype=self.param_dtype, device=self.device)
        dummy_cond = torch.zeros(1, 16, nfpb, h, w,
                                 dtype=self.param_dtype, device=self.device)
        dummy_audio = torch.zeros(1, 25, 1024, nfpb * 4,
                                  dtype=self.param_dtype, device=self.device)

        self.noise_model(
            [dummy_block],
            t=torch.zeros([1, nfpb], device=self.device, dtype=self.param_dtype),
            context=context[0:1], seq_len=None,
            cond_states=dummy_cond,
            motion_latents=motion_latents, ref_latents=ref_latents,
            audio_input=dummy_audio,
            motion_frames=[self.motion_frames, lat_motion_frames],
            drop_motion_frames=False,
            sink_flag=True,
            kv_cache=prefill_kv, crossattn_cache=prefill_crossattn,
            current_start=0, current_end=nfpb * frame_seq_length)

        # Broadcast cond cache from slot 0 to all slots
        # (k, v are untouched by prefill — still zeros; crossattn_cache likewise)
        for li in range(len(self.kv_cache)):
            for key in ["cond_k", "cond_v"]:
                self.kv_cache[li][key][:] = prefill_kv[li][key]
            self.kv_cache[li]["cond_end"] = prefill_kv[li]["cond_end"]

        if self.offload_kv_cache:
            self._move_kv_cache_to_device("cpu")

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
        torch_trace=False,
        profile_output_dir=None,
    ):
        num_steps = sampling_steps  # pipeline depth = number of denoising steps

        # ---- 1. Prepare conditional inputs ----
        ref_image = np.array(Image.open(ref_image_path).convert('RGB'))
        HEIGHT, WIDTH = self.get_size_less_than_area(
            ref_image.shape[0], ref_image.shape[1], target_area=max_area)
        size = (HEIGHT, WIDTH)

        resize_op = transforms.Resize(min(HEIGHT, WIDTH))
        crop_op = transforms.CenterCrop((HEIGHT, WIDTH))
        tensor_trans = transforms.ToTensor()

        self.audio_encoder.model.to(device=self.device, dtype=self.param_dtype)
        self.audio_encoder.model.requires_grad_(False)
        self.audio_encoder.model.eval()
        self.vae.model.to(self.device)

        audio_emb, nr = self.encode_audio(audio_path, infer_frames=infer_frames)
        self.audio_encoder.model.to("cpu")
        if num_repeat is None or num_repeat > nr:
            num_repeat = nr

        lat_motion_frames = (self.motion_frames + 3) // 4
        model_pic = crop_op(resize_op(Image.fromarray(ref_image)))

        ref_pixel_values = tensor_trans(model_pic)
        ref_pixel_values = ref_pixel_values.unsqueeze(1).unsqueeze(0) * 2 - 1.0
        ref_pixel_values = ref_pixel_values.to(dtype=self.vae.dtype, device=self.vae.device)
        ref_pixel_values = ref_pixel_values.repeat(1, 1, 5, 1, 1)
        ref_latents = torch.stack(self.vae.encode(ref_pixel_values))[:, :, 1:]

        motion_latents = ref_pixel_values.repeat(1, 1, self.motion_frames, 1, 1)
        videos_last_frames = motion_latents.detach()
        motion_latents = torch.stack(self.vae.encode(motion_latents))

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        context, context_null = self.encode_prompt(input_prompt, n_prompt, offload_model)

        print("complete prepare conditional inputs")
        sample_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=self.num_train_timesteps, shift=3)

        # ---- 2. Generate clips ----
        lat_target_frames = (infer_frames + 3 + self.motion_frames) // 4 - lat_motion_frames
        target_shape = [lat_target_frames, HEIGHT // 8, WIDTH // 8]
        frame_seq_length = HEIGHT // 8 * WIDTH // 8 // 2 // 2
        nfpb = self.num_frames_per_block
        num_blocks = target_shape[0] // nfpb
        bsl = nfpb * frame_seq_length  # tokens per block in KV cache
        max_seq_len = np.prod(target_shape) // 4

        with torch.amp.autocast('cuda', dtype=self.param_dtype), torch.no_grad():
            out = []
            clip_outputs = []
            active_nr = min(max_repeat, num_repeat)

            if offload_model:
                self.noise_model.to(self.device)
                self.vae.model.cpu()
                self.text_encoder.model.cpu()
                self.audio_encoder.model.cpu()
                torch.cuda.empty_cache()

            self._initialize_kv_cache(num_steps, self.param_dtype, self.device, max_seq_len)
            self._initialize_crossattn_cache(num_steps, self.param_dtype, self.device)

            # ---- Prefill cond cache ----
            dummy_cond = torch.zeros(
                1, 16, nfpb, HEIGHT // 8, WIDTH // 8,
                dtype=self.param_dtype, device=self.device)
            self._prefill_cond_cache(
                context=context,
                motion_latents=motion_latents,
                ref_latents=ref_latents,
                lat_motion_frames=lat_motion_frames,
                nfpb=nfpb,
                frame_seq_length=frame_seq_length,
                num_steps=num_steps)

            # ---- Setup scheduler ----
            sample_scheduler.set_timesteps(sampling_steps, device=self.device)
            self._sampler_timesteps = sample_scheduler.timesteps
            self._sampler_sigmas = sample_scheduler.sigmas
            timesteps = self._sampler_timesteps

            for r in range(active_nr):
                # ---- Clip-level setup ----
                seed_g = torch.Generator(device=self.device)
                seed_g.manual_seed(seed + r)
                clip_noise = torch.randn(
                    16, target_shape[0], target_shape[1], target_shape[2],
                    dtype=self.param_dtype, device=self.device, generator=seed_g)
                audio_input = audio_emb[..., r * infer_frames:(r + 1) * infer_frames]
                input_motion_latents = motion_latents.clone()
                clip_output = torch.zeros_like(clip_noise)

                if offload_model:
                    self.noise_model.to(self.device)
                    self.vae.model.cpu()
                    torch.cuda.empty_cache()

                # ---- Denoising ----
                clip_base = r * num_blocks * bsl

                for block_index in tqdm(range(num_blocks), desc=f"clip {r}"):
                    block_latents = clip_noise[:, block_index * nfpb:(block_index + 1) * nfpb]
                    la = block_index * (nfpb * 4)
                    ra = (block_index + 1) * (nfpb * 4)
                    cs = block_index * bsl + clip_base
                    ce = cs + bsl

                    sample_scheduler.timesteps = self._sampler_timesteps
                    sample_scheduler.sigmas = self._sampler_sigmas
                    sample_scheduler._step_index = 0
                    sample_scheduler._begin_index = 0

                    for step_i, t in enumerate(timesteps):
                        if self.offload_kv_cache:
                            self._move_kv_cache_to_device(self.device)

                        step_kv = [{
                            "k": layer["k"][step_i:step_i+1],
                            "v": layer["v"][step_i:step_i+1],
                            "cond_k": layer["cond_k"][step_i:step_i+1],
                            "cond_v": layer["cond_v"][step_i:step_i+1],
                            "cond_end": layer["cond_end"],
                        } for layer in self.kv_cache]
                        step_crossattn = [{
                            "k": layer["k"][step_i:step_i+1],
                            "v": layer["v"][step_i:step_i+1],
                            "is_init": layer["is_init"],
                        } for layer in self.crossattn_cache]

                        noise_pred = self.noise_model(
                            [block_latents],
                            t=t.unsqueeze(0).expand(1, nfpb),
                            context=context[0:1], seq_len=None,
                            cond_states=dummy_cond,
                            motion_latents=input_motion_latents,
                            ref_latents=ref_latents,
                            audio_input=audio_input[..., la:ra],
                            motion_frames=[self.motion_frames, lat_motion_frames],
                            drop_motion_frames=False,
                            kv_cache=step_kv, crossattn_cache=step_crossattn,
                            current_start=cs, current_end=ce)

                        if self.offload_kv_cache:
                            self._move_kv_cache_to_device("cpu")

                        block_latents = sample_scheduler.step(
                            noise_pred[0].unsqueeze(0), t,
                            block_latents.unsqueeze(0),
                            return_dict=False, generator=seed_g
                        )[0].squeeze(0)

                    clip_output[:, block_index * nfpb:(block_index + 1) * nfpb] = block_latents

                    # AAS: after first block, replace sink with generated latent
                    if r == 0 and block_index == 0:
                        ref_latents = block_latents.unsqueeze(0)[:, :, 0:1]

                clip_outputs.append(clip_output.detach().cpu())

        # ---- 3. Deferred VAE decode ----
        print(f"complete full-sequence generation")
        if clip_outputs:
            if offload_model:
                print(f"loading VAE for final decode")
                self.kv_cache = None
                self.vae.model.to(self.device)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            motion_latents_pp = motion_latents
            for clip_idx, clip_output_cpu in enumerate(clip_outputs):
                clip_output = clip_output_cpu.to(
                    device=self.vae.device, dtype=self.vae.dtype)
                decode_latents = torch.cat(
                    [motion_latents_pp, clip_output.unsqueeze(0)], dim=2)
                image = torch.stack(self.vae.decode(decode_latents))
                image = image[:, :, -(infer_frames):]
                if clip_idx == 0:
                    image = image[:, :, 3:]

                overlap_frames_num = min(self.motion_frames, image.shape[2])
                videos_last_frames = torch.cat([
                    videos_last_frames[:, :, overlap_frames_num:],
                    image[:, :, -overlap_frames_num:],
                ], dim=2)
                videos_last_frames = videos_last_frames.to(
                    dtype=motion_latents_pp.dtype, device=motion_latents_pp.device)
                motion_latents_pp = torch.stack(
                    self.vae.encode(videos_last_frames)
                ).type_as(clip_output)
                out.append(image.cpu())

        videos = torch.cat(out, dim=2)
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

        return videos[0], {}
