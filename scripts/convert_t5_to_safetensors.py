"""Convert T5 .pth checkpoint to safetensors format."""
import argparse
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="ckpt/Wan2.2-S2V-14B/models_t5_umt5-xxl-enc-bf16.pth")
    parser.add_argument("--output", default="ckpt/merged-s2v-14b-lora/models_t5_umt5-xxl-enc-bf16.safetensors")
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    t0 = time.perf_counter()
    state_dict = torch.load(args.input, map_location="cpu")
    t1 = time.perf_counter()
    print(f"  Loaded in {t1 - t0:.2f}s ({len(state_dict)} tensors)")

    print(f"Saving {args.output}...")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_file(state_dict, args.output)
    t2 = time.perf_counter()
    print(f"  Saved in {t2 - t1:.2f}s")

    # Verify
    input_size = os.path.getsize(args.input)
    output_size = os.path.getsize(args.output)
    print(f"\n  Input:  {input_size / 1e9:.2f} GB (.pth)")
    print(f"  Output: {output_size / 1e9:.2f} GB (.safetensors)")
    print("Done.")


if __name__ == "__main__":
    main()
