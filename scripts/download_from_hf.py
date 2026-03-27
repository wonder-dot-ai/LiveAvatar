#!/usr/bin/env python3
"""Download a LiveAvatar checkpoint from Hugging Face Hub.

Usage:
    python scripts/download_from_hf.py --repo_id YOUR_USERNAME/LiveAvatar-merged

    # Custom output directory:
    python scripts/download_from_hf.py --repo_id YOUR_USERNAME/LiveAvatar-merged \
        --output_dir ckpt/merged-s2v-14b-lora/
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Download checkpoint from HF Hub")
    parser.add_argument("--repo_id", required=True,
                        help="HF repo id, e.g. your-username/LiveAvatar-merged")
    parser.add_argument("--output_dir", default="ckpt/merged-s2v-14b-lora/",
                        help="Local directory to save to (default: ckpt/merged-s2v-14b-lora/)")
    parser.add_argument("--revision", default="main",
                        help="Branch or commit to download (default: main)")
    parser.add_argument("--token", default=None,
                        help="HF token (for private repos; or use huggingface-cli login)")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    print(f"Downloading {args.repo_id} -> {args.output_dir} ...")
    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        local_dir=args.output_dir,
        token=args.token,
    )
    print(f"Done! Checkpoint saved to {path}")


if __name__ == "__main__":
    main()
