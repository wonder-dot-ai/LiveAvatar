#!/bin/bash
# LiveAvatar multi-GPU (5x GPU) inference profiling script.
# Profiles per-rank: DiT forward, dist.send/recv, streaming VAE decode, etc.
# Each rank writes its own profiling data with rank-specific phase names.

CUDA_VISIBLE_DEVICES=0,1,2,3,4
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=OFF
export ENABLE_COMPILE=false  # Disable torch.compile for accurate profiling

mkdir -p profiling_output_multi_gpu

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun \
    --nproc_per_node=5 --master_port=29102 \
    minimal_inference/s2v_streaming_interact.py \
    --ulysses_size 1 \
    --task s2v-14B \
    --size "720*400" \
    --base_seed 420 \
    --training_config liveavatar/configs/s2v_causal_sft.yaml \
    --offload_model False \
    --convert_model_dtype \
    --prompt "A stout, cheerful dwarf with a magnificent braided beard adorned with metal rings, wearing a heavy leather apron. He's standing in his fiery, cluttered forge, laughing heartily as he explains the mastery of his craft, holding up a glowing hammer. Style of Blizzard Entertainment cinematics (like World of Warcraft), warm, dynamic lighting from the forge." \
    --image "examples/dwarven_blacksmith.jpg" \
    --audio "examples/dwarven_blacksmith.wav" \
    --infer_frames 48 \
    --load_lora \
    --lora_path_dmd "Quark-Vision/Live-Avatar" \
    --sample_steps 4 \
    --sample_guide_scale 0 \
    --num_clip 2 \
    --num_gpus_dit 4 \
    --sample_solver euler \
    --enable_vae_parallel \
    --ckpt_dir ckpt/Wan2.2-S2V-14B/ \
    --fp8 \
    --enable_profiling \
    --profile_output_dir profiling_output_multi_gpu \
    --profile_num_clips 2 \
    --torch_trace
