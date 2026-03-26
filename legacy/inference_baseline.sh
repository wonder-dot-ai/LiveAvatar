#!/bin/bash
# Baseline: original single-GPU pipeline for A/B comparison
# Must match inference_2gpu.sh parameters exactly (same seed, image, audio, etc.)

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29101 minimal_inference/s2v_streaming_interact.py \
    --task s2v-14B \
    --size "704*384" \
    --base_seed 1234 \
    --training_config liveavatar/configs/s2v_causal_sft.yaml \
    --offload_model True \
    --convert_model_dtype \
    --prompt "A stout, cheerful dwarf with a magnificent braided beard adorned with metal rings, wearing a heavy leather apron. He's standing in his fiery, cluttered forge, laughing heartily as he explains the mastery of his craft, holding up a glowing hammer. Style of Blizzard Entertainment cinematics (like World of Warcraft), warm, dynamic lighting from the forge." \
    --image "examples/dwarven_blacksmith.jpg" \
    --audio "examples/dwarven_blacksmith.wav" \
    --infer_frames 48 \
    --load_lora \
    --lora_path_dmd "Quark-Vision/Live-Avatar" \
    --sample_steps 4 \
    --sample_guide_scale 0 \
    --num_clip 1 \
    --num_gpus_dit 1 \
    --sample_solver euler \
    --single_gpu \
    --ckpt_dir ckpt/Wan2.2-S2V-14B/ \
    --fp8 \
    --save_file output/baseline.mp4
