#!/bin/bash
# Batched TPP pipeline — single-GPU inference
# Uses batch=4 pipeline shift register for DiT denoising

CUDA_VISIBLE_DEVICES=0 python minimal_inference/s2v_2gpu.py \
    --image examples/sample1.jpg \
    --audio examples/sample1.wav \
    --prompt "A person is talking" \
    --fp8 \
    --offload_model True \
    --infer_frames 48 \
    --num_clip 1 \
    --sample_steps 4 \
    --seed 42 \
    --output output/result_2gpu.mp4
