"""CLI entry point for LiveAvatar inference."""

import argparse
import os
import time


def parse_args():
    p = argparse.ArgumentParser(description="LiveAvatar inference")
    # Required
    p.add_argument("--image", required=True, help="Reference image path")
    p.add_argument("--audio", required=True, help="Audio WAV path")
    p.add_argument("--prompt", default="A person is talking", help="Text prompt")
    # Model
    p.add_argument("--ckpt_dir", default="ckpt/merged-s2v-14b-lora/", help="Checkpoint directory")
    p.add_argument("--config", default="configs/s2v_14B.yaml", help="Model config YAML")
    p.add_argument("--load_lora", default=None, help="LoRA checkpoint path (skip for merged)")
    p.add_argument("--lora_config", default="configs/s2v_inference.yaml", help="LoRA config YAML")
    p.add_argument("--save_merged", default=None, help="Save merged DiT to this dir")
    # Generation
    p.add_argument("--infer_frames", type=int, default=48, help="Pixel frames per VAE decode chunk")
    p.add_argument("--num_blocks", type=int, default=None, help="Max blocks to generate (None=full audio)")
    p.add_argument("--max_area", type=int, default=720 * 400, help="Max pixel area")
    p.add_argument("--sample_steps", type=int, default=4, help="Diffusion sampling steps")
    p.add_argument("--seed", type=int, default=-1, help="Random seed (-1=random)")
    # Hardware
    p.add_argument("--fp8", action="store_true", help="Enable FP8 quantization")
    p.add_argument(
        "--offload_model",
        default=True,
        type=lambda x: x.lower() != "false",
        help="Offload models to CPU between stages",
    )
    p.add_argument(
        "--offload_kv_cache",
        action="store_true",
        help="Offload KV cache to CPU between forward passes",
    )
    # Output
    p.add_argument("--output", default="output/result.mp4", help="Output video path")
    p.add_argument("--fps", type=int, default=25, help="Output video FPS")
    return p.parse_args()


def main():
    t_start = time.perf_counter()
    args = parse_args()

    import yaml
    import torch

    from liveavatar_nari.config import load_config

    cfg = load_config(args.config)
    t_config = time.perf_counter()
    print(f"[TIMING] Load config: {t_config - t_start:.2f}s")

    from liveavatar_nari.pipeline import WanS2V

    t_import = time.perf_counter()
    print(f"[TIMING] Import pipeline: {t_import - t_config:.2f}s")

    print(f"Creating pipeline (offload_model={args.offload_model}, fp8={args.fp8})")
    pipeline = WanS2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=0,
        offload_kv_cache=args.offload_kv_cache,
    )

    # Load LoRA if specified (skip for pre-merged checkpoints)
    if args.load_lora:
        with open(args.lora_config) as f:
            lora_cfg = yaml.safe_load(f)
        print(f"Loading LoRA from {args.load_lora} (rank={lora_cfg['lora_rank']})")
        pipeline.load_lora(
            lora_path=args.load_lora,
            lora_rank=lora_cfg["lora_rank"],
            lora_alpha=lora_cfg["lora_alpha"],
            lora_target_modules=lora_cfg["lora_target_modules"],
            init_lora_weights=lora_cfg["init_lora_weights"],
        )

    # Save merged model if requested
    if args.save_merged:
        pipeline.save_model(args.save_merged)
        print(f"Re-run with --ckpt_dir {args.save_merged} (no --load_lora) for fast startup.")

    # Fuse QKV/KV projections (before FP8 so fused weights get quantized together)
    pipeline.noise_model.fuse_projections()

    # FP8 conversion
    if args.fp8:
        if hasattr(torch, "_scaled_mm"):
            print("Applying FP8 quantization...")
            from liveavatar_nari.utils.fp8_linear import replace_linear_with_scaled_fp8

            replace_linear_with_scaled_fp8(
                pipeline.noise_model,
                ignore_keys=[
                    "text_embedding",
                    "time_embedding",
                    "time_projection",
                    "head.head",
                    "casual_audio_encoder.encoder.final_linear",
                ],
            )
        else:
            print("WARNING: torch._scaled_mm not available, skipping FP8")

    # Generate
    print(f"Generating: image={args.image}, audio={args.audio}")
    print(
        f"  infer_frames={args.infer_frames}, num_blocks={args.num_blocks}, "
        f"sample_steps={args.sample_steps}, seed={args.seed}"
    )

    video, _ = pipeline.generate(
        input_prompt=args.prompt,
        ref_image_path=args.image,
        audio_path=args.audio,
        infer_frames=args.infer_frames,
        max_blocks=args.num_blocks,
        max_area=args.max_area,
        sampling_steps=args.sample_steps,
        seed=args.seed,
        offload_model=args.offload_model,
    )

    # Save video
    if video is not None:
        from liveavatar_nari.utils.video import save_video, merge_video_audio

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        save_video(
            tensor=video[None],
            save_file=args.output,
            fps=args.fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        merge_video_audio(video_path=args.output, audio_path=args.audio)
        print(f"Saved: {args.output}")
    else:
        print("No video generated.")


if __name__ == "__main__":
    main()
