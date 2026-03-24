#!/bin/bash
# Batched TPP pipeline — single-GPU inference
# Uses batch=4 pipeline shift register for DiT denoising

CUDA_VISIBLE_DEVICES=0 python minimal_inference/s2v_2gpu.py \
    --image "examples/dwarven_blacksmith.jpg" \
    --audio "examples/dwarven_blacksmith.wav" \
    --prompt "A stout, cheerful dwarf with a magnificent braided beard adorned with metal rings, wearing a heavy leather apron. He's standing in his fiery, cluttered forge, laughing heartily as he explains the mastery of his craft, holding up a glowing hammer. Style of Blizzard Entertainment cinematics (like World of Warcraft), warm, dynamic lighting from the forge." \
    --ckpt_dir ckpt/Wan2.2-S2V-14B/ \
    --load_lora "Quark-Vision/Live-Avatar" \
    --config configs/s2v_inference.yaml \
    --fp8 \
    --offload_model True \
    --infer_frames 48 \
    --num_clip 1 \
    --sample_steps 4 \
    --seed 4200 \
    --output output/result_2gpu.mp4
