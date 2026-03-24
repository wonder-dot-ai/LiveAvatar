"""Compare two video files frame-by-frame using decord."""
import sys
import torch
import numpy as np
from decord import VideoReader

def main():
    path1 = sys.argv[1] if len(sys.argv) > 1 else "output/baseline.mp4"
    path2 = sys.argv[2] if len(sys.argv) > 2 else "output/result_2gpu.mp4"

    vr1 = VideoReader(path1)
    vr2 = VideoReader(path2)
    n1, n2 = len(vr1), len(vr2)

    v1 = torch.from_numpy(vr1.get_batch(range(n1)).asnumpy()).float()
    v2 = torch.from_numpy(vr2.get_batch(range(n2)).asnumpy()).float()

    print(f"Video 1: {path1}  shape={tuple(v1.shape)}")
    print(f"Video 2: {path2}  shape={tuple(v2.shape)}")

    n = min(n1, n2)
    if n1 != n2:
        print(f"WARNING: frame count differs ({n1} vs {n2}), comparing first {n}")

    diff = (v1[:n] - v2[:n]).abs()
    print(f"\nFrames compared: {n}")
    print(f"  Mean pixel diff: {diff.mean():.2f}")
    print(f"  Max pixel diff:  {diff.max():.2f}")
    mse = (diff / 255).pow(2).mean()
    psnr = -10 * torch.log10(mse) if mse > 0 else float('inf')
    print(f"  PSNR:            {psnr:.1f} dB")

    per_frame = diff.mean(dim=(1, 2, 3))
    print(f"\nPer-frame mean diff (first 10):")
    for i in range(min(10, n)):
        print(f"  frame {i:3d}: {per_frame[i]:.2f}")

    if diff.mean() < 1.0:
        print("\nVERDICT: Videos are essentially identical (numerical noise only)")
    elif diff.mean() < 5.0:
        print("\nVERDICT: Minor differences (likely fp precision / attention backend)")
    else:
        print("\nVERDICT: Significant differences — there is a logic bug")

if __name__ == "__main__":
    main()
