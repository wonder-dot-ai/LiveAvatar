"""Benchmark model loading: base+LoRA vs pre-merged checkpoint."""
import os
import sys
import time
import warnings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

import yaml
import torch


def time_it(label, fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"  {label}: {elapsed:.2f}s")
    return result, elapsed


def benchmark_base_plus_lora():
    print("\n=== Base + LoRA (original flow) ===")
    from liveavatar.models.wan.wan_2_2.configs.wan_s2v_14B_modified import s2v_14B as cfg

    from liveavatar.models.wan.causal_s2v_pipeline_2gpu import WanS2V

    pipeline, t_init = time_it("Pipeline init (T5 + VAE + DiT + wav2vec)", lambda: WanS2V(
        config=cfg, checkpoint_dir="ckpt/Wan2.2-S2V-14B/", device_id=0))

    with open("configs/s2v_inference.yaml") as f:
        lora_cfg = yaml.safe_load(f)

    def _load_and_merge():
        pipeline.noise_model = pipeline.add_lora_to_model(
            pipeline.noise_model,
            lora_rank=lora_cfg['lora_rank'],
            lora_alpha=lora_cfg['lora_alpha'],
            lora_target_modules=lora_cfg['lora_target_modules'],
            init_lora_weights=lora_cfg['init_lora_weights'],
            pretrained_lora_path="Quark-Vision/Live-Avatar")

    _, t_lora = time_it("LoRA load + merge", _load_and_merge)

    total = t_init + t_lora
    print(f"  TOTAL: {total:.2f}s")
    del pipeline
    torch.cuda.empty_cache()
    return total


def benchmark_merged():
    print("\n=== Pre-merged checkpoint (fast flow) ===")
    from liveavatar.models.wan.wan_2_2.configs.wan_s2v_14B_modified import s2v_14B as cfg

    from liveavatar.models.wan.causal_s2v_pipeline_2gpu_optimized import WanS2V

    pipeline, t_init = time_it("Pipeline init (T5 + VAE + merged DiT + wav2vec)", lambda: WanS2V(
        config=cfg, checkpoint_dir="ckpt/merged-s2v-14b-lora/", device_id=0))

    total = t_init
    print(f"  TOTAL: {total:.2f}s")
    del pipeline
    torch.cuda.empty_cache()
    return total


if __name__ == "__main__":
    t_original = benchmark_base_plus_lora()
    t_merged = benchmark_merged()

    print(f"\n=== Summary ===")
    print(f"  Base + LoRA: {t_original:.2f}s")
    print(f"  Pre-merged:  {t_merged:.2f}s")
    print(f"  Saved:       {t_original - t_merged:.2f}s ({(t_original - t_merged) / t_original * 100:.0f}%)")
