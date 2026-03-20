"""Standalone inference script for the batched TPP pipeline (Phase 1: single-GPU)."""
import argparse
import os
import sys
import warnings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

import torch
import imageio
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Batched TPP inference for LiveAvatar")
    # Required
    p.add_argument("--image", required=True, help="Reference image path")
    p.add_argument("--audio", required=True, help="Audio WAV path")
    p.add_argument("--prompt", default="A person is talking", help="Text prompt")
    # Model
    p.add_argument("--ckpt_dir", default="ckpt/Wan2.2-S2V-14B/", help="Checkpoint directory")
    p.add_argument("--load_lora", default=None, help="LoRA checkpoint path")
    # Generation
    p.add_argument("--infer_frames", type=int, default=48, help="Frames per clip (must be 4n)")
    p.add_argument("--num_clip", type=int, default=1, help="Number of clips")
    p.add_argument("--max_area", type=int, default=720 * 400, help="Max pixel area")
    p.add_argument("--sample_steps", type=int, default=4, help="Diffusion sampling steps")
    p.add_argument("--seed", type=int, default=-1, help="Random seed (-1=random)")
    p.add_argument("--n_prompt", default="", help="Negative prompt")
    # Hardware
    p.add_argument("--fp8", action="store_true", help="Enable FP8 quantization")
    p.add_argument("--offload_model", default=True, type=lambda x: x.lower() != 'false',
                    help="Offload models to CPU between stages")
    p.add_argument("--offload_kv_cache", action="store_true",
                    help="Offload KV cache to CPU between forward passes")
    p.add_argument("--enable_online_decode", action="store_true",
                    help="Online VAE decode after clip 0")
    # Pose
    p.add_argument("--pose_video", default=None, help="Pose driving video path")
    # Output
    p.add_argument("--output", default="output/result.mp4", help="Output video path")
    p.add_argument("--fps", type=int, default=16, help="Output video FPS")
    return p.parse_args()


def main():
    args = parse_args()

    # Load config
    from liveavatar.models.wan.wan_2_2.configs.wan_s2v_14B_modified import s2v_14B as cfg

    # Create pipeline
    from liveavatar.models.wan.causal_s2v_pipeline_2gpu import WanS2V
    print(f"Creating pipeline (offload_model={args.offload_model}, "
          f"offload_kv_cache={args.offload_kv_cache}, fp8={args.fp8})")
    pipeline = WanS2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=0,
        offload_kv_cache=args.offload_kv_cache,
    )

    # Load LoRA if specified
    if args.load_lora:
        print(f"Loading LoRA from {args.load_lora}")
        pipeline.noise_model = pipeline.add_lora_to_model(
            pipeline.noise_model,
            pretrained_lora_path=args.load_lora,
            load_only=True)

    # FP8 conversion
    if args.fp8:
        if hasattr(torch, "_scaled_mm"):
            print("Applying FP8 quantization...")
            from liveavatar.utils.fp8_linear import replace_linear_with_scaled_fp8
            replace_linear_with_scaled_fp8(
                pipeline.noise_model,
                ignore_keys=[
                    'text_embedding', 'time_embedding',
                    'time_projection', 'head.head',
                    'casual_audio_encoder.encoder.final_linear',
                ])
        else:
            print("WARNING: torch._scaled_mm not available, skipping FP8")

    # Generate
    print(f"Generating: image={args.image}, audio={args.audio}")
    print(f"  infer_frames={args.infer_frames}, num_clip={args.num_clip}, "
          f"sample_steps={args.sample_steps}, seed={args.seed}")

    video, _ = pipeline.generate(
        input_prompt=args.prompt,
        ref_image_path=args.image,
        audio_path=args.audio,
        infer_frames=args.infer_frames,
        num_repeat=args.num_clip,
        max_area=args.max_area,
        sampling_steps=args.sample_steps,
        seed=args.seed,
        n_prompt=args.n_prompt,
        offload_model=args.offload_model,
        pose_video=args.pose_video,
        enable_online_decode=args.enable_online_decode,
    )

    # Save video
    if video is not None:
        video_np = ((video.clamp(-1, 1) + 1) / 2 * 255).byte()
        video_np = video_np.permute(1, 2, 3, 0).cpu().numpy()
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        imageio.mimwrite(args.output, video_np, fps=args.fps, codec="libx264")
        print(f"Saved: {args.output} ({video_np.shape[0]} frames, {video_np.shape[1]}x{video_np.shape[2]})")
    else:
        print("No video generated.")


if __name__ == "__main__":
    main()
