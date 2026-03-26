#!/bin/bash
# Optimized pipeline — streaming VAE decode + on-demand audio

CUDA_VISIBLE_DEVICES=0 python minimal_inference/s2v_2gpu_optimized.py \
    --image "examples/dwarven_blacksmith.jpg" \
    --audio "examples/dwarven_blacksmith.wav" \
    --prompt "A stout, cheerful dwarf with a magnificent braided beard adorned with metal rings, wearing a heavy leather apron. He's standing in his fiery, cluttered forge, laughing heartily as he explains the mastery of his craft, holding up a glowing hammer. Style of Blizzard Entertainment cinematics (like World of Warcraft), warm, dynamic lighting from the forge." \
    --ckpt_dir ckpt/merged-s2v-14b-lora/ \
    --fp8 \
    --offload_model True \
    --infer_frames 48 \
    --num_clip 1 \
    --sample_steps 4 \
    --seed 4200 \
    --output output/result_optimized.mp4
