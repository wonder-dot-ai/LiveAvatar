# Migrate Optimized Pipeline to `liveavatar_nari` Package

## Context

Importing the optimized pipeline from `liveavatar.models.wan` triggers `liveavatar/models/__init__.py`, which imports `wan_wrapper` → `transformers` → `sklearn` → `deepspeed` → etc., costing ~50s of cold start. The pipeline itself only needs ~25 files from the liveavatar codebase.

Goal: Create a clean, `pip install`-able package `liveavatar_nari/` with proper entry points. No `sys.path` hacks, no heavy init chains, deployment-ready for Cerebrium.

## Package Structure

```
LiveAvatar/
├── pyproject.toml                    # package definition + entry points
├── configs/
│   ├── s2v_14B.yaml                  # model config (included as package data)
│   └── s2v_inference.yaml            # LoRA config
├── liveavatar_nari/
│   ├── __init__.py                   # version only
│   ├── pipeline.py                   # WanS2V (main pipeline class)
│   ├── config.py                     # YAML config loader (_DotDict)
│   ├── cli/
│   │   ├── __init__.py
│   │   └── inference.py              # CLI entry point → `liveavatar-infer`
│   ├── models/
│   │   ├── __init__.py               # empty
│   │   ├── causal_model_s2v.py       # DiT model
│   │   ├── causal_audio_encoder.py   # wav2vec wrapper
│   │   ├── causal_motioner.py        # FramePackMotioner
│   │   └── causal_s2v_utils.py       # rollout_grid_sizes etc.
│   ├── modules/
│   │   ├── __init__.py               # empty
│   │   ├── t5.py                     # T5 encoder
│   │   ├── vae.py                    # VAE (renamed from vae2_1.py)
│   │   ├── attention.py              # flash/cudnn attention
│   │   ├── model.py                  # WanModel base
│   │   ├── tokenizers.py             # HuggingfaceTokenizer
│   │   ├── inference_utils.py        # COMPILE flag, conditional_compile
│   │   └── s2v/
│   │       ├── __init__.py           # empty
│   │       ├── model_s2v.py          # WanModel_S2V
│   │       ├── audio_utils.py        # CausalAudioEncoder
│   │       ├── audio_encoder.py      # AudioEncoder (wav2vec)
│   │       ├── s2v_utils.py          # rope_precompute
│   │       ├── motioner.py           # FramePackMotioner (original)
│   │       └── auxi_blocks.py        # MotionEncoder_tc
│   ├── distributed/
│   │   ├── __init__.py               # empty
│   │   ├── util.py                   # get_rank, pad_chunk etc.
│   │   ├── sequence_parallel.py
│   │   └── ulysses.py
│   └── utils/
│       ├── __init__.py               # empty
│       ├── load_weights.py           # load_state_dict (from load_weight_utils.py)
│       └── fp8_linear.py             # FP8 quantization (copy from liveavatar/utils/)
├── scripts/                          # dev tools (not packaged)
│   ├── benchmark_imports.py
│   ├── benchmark_loading.py
│   ├── benchmark_import_chain.py
│   ├── compare_videos.py
│   └── convert_t5_to_safetensors.py
└── examples/                         # example configs (not packaged)
    ├── dwarven_blacksmith.jpg
    └── dwarven_blacksmith.wav
```

## pyproject.toml

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "liveavatar-nari"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
    "numpy",
    "Pillow",
    "torchvision",
    "tqdm",
    "safetensors",
    "diffusers>=0.37",
    "pyyaml",
    "librosa",
    "einops",
    "ftfy",
    "regex",
]

[project.optional-dependencies]
lora = ["peft>=0.10"]
fp8 = []  # requires torch._scaled_mm (built-in)

[project.scripts]
liveavatar-infer = "liveavatar_nari.cli.inference:main"

[tool.setuptools.package-data]
liveavatar_nari = ["../configs/*.yaml"]
```

## CLI Entry Point

`liveavatar_nari/cli/inference.py`:
```python
def main():
    args = parse_args()
    from liveavatar_nari.config import load_config
    cfg = load_config(args.config)
    from liveavatar_nari.pipeline import WanS2V
    pipeline = WanS2V(cfg, args.ckpt_dir, ...)
    ...
```

Usage:
```bash
# After pip install -e .
liveavatar-infer --image ref.jpg --audio speech.wav --ckpt_dir ckpt/merged/

# Or directly
python -m liveavatar_nari.cli.inference --image ref.jpg ...
```

## Import Rewriting Rules

| Old path | New path |
|----------|----------|
| `liveavatar.models.wan.causal_model_s2v` | `liveavatar_nari.models.causal_model_s2v` |
| `liveavatar.models.wan.causal_audio_encoder` | `liveavatar_nari.models.causal_audio_encoder` |
| `liveavatar.models.wan.causal_motioner` | `liveavatar_nari.models.causal_motioner` |
| `liveavatar.models.wan.causal_s2v_utils` | `liveavatar_nari.models.causal_s2v_utils` |
| `liveavatar.models.wan.inference_utils` | `liveavatar_nari.modules.inference_utils` |
| `liveavatar.models.wan.wan_2_2.modules.t5` | `liveavatar_nari.modules.t5` |
| `liveavatar.models.wan.wan_2_2.modules.vae2_1` | `liveavatar_nari.modules.vae` |
| `liveavatar.models.wan.wan_2_2.modules.attention` | `liveavatar_nari.modules.attention` |
| `liveavatar.models.wan.wan_2_2.modules.model` | `liveavatar_nari.modules.model` |
| `liveavatar.models.wan.wan_2_2.modules.tokenizers` | `liveavatar_nari.modules.tokenizers` |
| `liveavatar.models.wan.wan_2_2.modules.s2v.*` | `liveavatar_nari.modules.s2v.*` |
| `liveavatar.models.wan.wan_2_2.distributed.*` | `liveavatar_nari.distributed.*` |
| `liveavatar.utils.load_weight_utils` | `liveavatar_nari.utils.load_weights` |
| `liveavatar.utils.fp8_linear` | `liveavatar_nari.utils.fp8_linear` |

## What NOT to Copy

- `liveavatar/models/__init__.py` (the 50s import chain)
- `liveavatar/models/wan/wan_wrapper.py` (training wrapper)
- `liveavatar/models/wan/wan_base/` (old base package)
- `liveavatar/models/wan/wan_2_2/configs/` (replaced by YAML)
- `liveavatar/models/wan/wan_2_2/utils/` (fm_solvers — via diffusers)
- `liveavatar/models/model_interface.py` (training)
- `liveavatar/scheduler.py` (training)
- `wan_2_2/distributed/fsdp.py` (unused)

## Steps

1. Create `pyproject.toml`
2. Create directory structure with empty `__init__.py` files
3. Copy and rename files per the structure above
4. Rewrite all relative imports using the mapping table
5. Create `liveavatar_nari/config.py` (move from `configs/load_config.py`)
6. Create `liveavatar_nari/cli/inference.py` (from `s2v_2gpu_optimized.py`)
7. `pip install -e .`
8. Verify: `liveavatar-infer --help`
9. Verify: `python scripts/benchmark_import_chain.py` (update to new package)
10. Verify: `liveavatar-infer --ckpt_dir ckpt/merged-s2v-14b-lora/ ...` produces correct video

## Expected Result

- `from liveavatar_nari.pipeline import WanS2V` — no init chain, ~15s import
- `liveavatar-infer` CLI works without `sys.path` hacks
- `pip install .` in Cerebrium's `cerebrium.toml` just works
- Total cold start: ~25s (torch+dynamo import + model loading) instead of ~100s+
