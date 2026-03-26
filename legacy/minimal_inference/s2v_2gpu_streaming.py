"""Streaming inference script — per-block VAE decode + on-demand audio."""
import argparse
import os
import sys
import warnings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

import yaml
import torch


def parse_args():
    p = argparse.ArgumentParser(description="Streaming inference for LiveAvatar")
    # Required
    p.add_argument("--image", required=True, help="Reference image path")
    p.add_argument("--audio", required=True, help="Audio WAV path")
    p.add_argument("--prompt", default="A person is talking", help="Text prompt")
    # Model
    p.add_argument("--ckpt_dir", default="ckpt/Wan2.2-S2V-14B/", help="Checkpoint directory")
    p.add_argument("--load_lora", default=None, help="LoRA checkpoint path")
    p.add_argument("--config", default="configs/s2v_inference.yaml", help="Inference config (LoRA settings)")
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
    # Output
    p.add_argument("--output", default="output/result_streaming.mp4", help="Output video path")
    p.add_argument("--fps", type=int, default=25, help="Output video FPS")
    return p.parse_args()


def main():
    args = parse_args()

    from liveavatar.models.wan.wan_2_2.configs.wan_s2v_14B_modified import s2v_14B as cfg

    from liveavatar.models.wan.causal_s2v_pipeline_2gpu_streaming import WanS2VStreaming
    print(f"Creating streaming pipeline (offload_model={args.offload_model}, "
          f"offload_kv_cache={args.offload_kv_cache}, fp8={args.fp8})")
    pipeline = WanS2VStreaming(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=0,
        offload_kv_cache=args.offload_kv_cache,
    )

    # Load LoRA if specified
    if args.load_lora:
        with open(args.config) as f:
            lora_cfg = yaml.safe_load(f)
        print(f"Loading LoRA from {args.load_lora} (rank={lora_cfg['lora_rank']})")
        pipeline.noise_model = pipeline.add_lora_to_model(
            pipeline.noise_model,
            lora_rank=lora_cfg['lora_rank'],
            lora_alpha=lora_cfg['lora_alpha'],
            lora_target_modules=lora_cfg['lora_target_modules'],
            init_lora_weights=lora_cfg['init_lora_weights'],
            pretrained_lora_path=args.load_lora)

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

    # Generate — streaming
    print(f"Generating (streaming): image={args.image}, audio={args.audio}")
    print(f"  infer_frames={args.infer_frames}, num_clip={args.num_clip}, "
          f"sample_steps={args.sample_steps}, seed={args.seed}")

    all_frames = []
    total_pixel_frames = 0
    for image_chunk, meta in pipeline.generate(
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
    ):
        n = image_chunk.shape[2]
        total_pixel_frames += n
        print(f"  clip {meta['clip']} block {meta['block']}: {n} frames (total: {total_pixel_frames})")
        all_frames.append(image_chunk)

    # Save video
    if all_frames:
        from liveavatar.models.wan.wan_2_2.utils.utils import save_video, merge_video_audio
        video = torch.cat(all_frames, dim=2)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        save_video(
            tensor=video,
            save_file=args.output,
            fps=args.fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))
        merge_video_audio(video_path=args.output, audio_path=args.audio)
        print(f"Saved: {args.output} ({total_pixel_frames} frames)")
    else:
        print("No frames generated.")


if __name__ == "__main__":
    main()
