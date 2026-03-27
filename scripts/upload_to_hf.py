#!/usr/bin/env python3
"""Upload a merged LiveAvatar checkpoint directory to Hugging Face Hub.

Usage:
    python scripts/upload_to_hf.py --repo_id YOUR_USERNAME/LiveAvatar-merged \
        --ckpt_dir ckpt/merged-s2v-14b-lora/

    # Dry-run (list files without uploading):
    python scripts/upload_to_hf.py --repo_id YOUR_USERNAME/LiveAvatar-merged \
        --ckpt_dir ckpt/merged-s2v-14b-lora/ --dry_run
"""

import argparse
import os
import sys


# Files/dirs that make up a complete checkpoint (matches pipeline.py expectations):
#   - diffusion_pytorch_model*.safetensors + index.json  (DiT)
#   - config.json                                        (DiT config)
#   - Wan2.1_VAE.pth                                     (VAE)
#   - models_t5_umt5-xxl-enc-bf16.{pth,safetensors}      (T5 text encoder)
#   - google/umt5-xxl/                                   (T5 tokenizer)
#   - wav2vec2-large-xlsr-53-english/                     (audio encoder)

IGNORE_PATTERNS = [
    "*.pyc",
    "__pycache__",
    ".git",
    "*.bin",           # prefer safetensors when both exist
    "flax_model.*",    # flax weights in wav2vec dir
    "*.txt",           # eval logs in wav2vec dir
    "eval.py",
    "full_eval.sh",
]


def list_files(ckpt_dir: str) -> list[str]:
    """Walk the checkpoint dir and return relative paths."""
    paths = []
    for root, _, files in os.walk(ckpt_dir):
        for f in files:
            paths.append(os.path.relpath(os.path.join(root, f), ckpt_dir))
    paths.sort()
    return paths


def main():
    parser = argparse.ArgumentParser(description="Upload merged checkpoint to HF Hub")
    parser.add_argument("--repo_id", required=True,
                        help="HF repo id, e.g. your-username/LiveAvatar-merged")
    parser.add_argument("--ckpt_dir", default="ckpt/merged-s2v-14b-lora/",
                        help="Local checkpoint directory")
    parser.add_argument("--private", action="store_true",
                        help="Create a private repo")
    parser.add_argument("--dry_run", action="store_true",
                        help="List files that would be uploaded, without uploading")
    parser.add_argument("--revision", default="main",
                        help="Branch to upload to (default: main)")
    args = parser.parse_args()

    if not os.path.isdir(args.ckpt_dir):
        print(f"Error: checkpoint directory not found: {args.ckpt_dir}")
        sys.exit(1)

    files = list_files(args.ckpt_dir)
    print(f"Found {len(files)} files in {args.ckpt_dir}")
    for f in files:
        size_mb = os.path.getsize(os.path.join(args.ckpt_dir, f)) / 1e6
        print(f"  {f}  ({size_mb:.1f} MB)")

    if args.dry_run:
        print("\n[dry run] Would upload the above files. Exiting.")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model",
                    private=args.private, exist_ok=True)

    print(f"\nUploading to https://huggingface.co/{args.repo_id} ...")
    api.upload_folder(
        folder_path=args.ckpt_dir,
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        ignore_patterns=IGNORE_PATTERNS,
        commit_message="Upload merged LiveAvatar checkpoint",
    )
    print(f"Done! https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
