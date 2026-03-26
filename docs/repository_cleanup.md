# Repository Cleanup — Separate New Package from Legacy Files

## Context

The repo has old pipeline files, new `liveavatar_nari` package, streaming experiments, scripts, and plans scattered at the root level. Need to organize into a clean structure while preserving all old files.

## Target Structure

```
LiveAvatar/
├── pyproject.toml
├── README.md
│
├── liveavatar_nari/                        # New package (clean, self-contained)
│   ├── __init__.py
│   ├── pipeline.py
│   ├── config.py
│   ├── cli/
│   ├── models/
│   ├── modules/
│   ├── distributed/
│   └── utils/
│
├── configs/                                # Runtime configs (YAML)
│   ├── s2v_14B.yaml
│   └── s2v_inference.yaml
│
├── examples/                               # Example assets + launch script
│   ├── dwarven_blacksmith.jpg
│   ├── dwarven_blacksmith.wav
│   └── inference.sh                        # ← moved from inference_nari.sh
│
├── scripts/                                # Dev/utility tools
│   ├── benchmark_imports.py
│   ├── benchmark_loading.py
│   ├── benchmark_import_chain.py
│   ├── compare_videos.py
│   └── convert_t5_to_safetensors.py
│
├── docs/                                   # Plans and documentation
│   ├── liveavatar_nari_migration.md        # ← moved from agent/plans/
│   └── streaming_pipeline_implementation.md
│
├── legacy/                                 # Old pipeline scripts (preserved, not used)
│   ├── inference_2gpu.sh
│   ├── inference_2gpu_optimized.sh
│   ├── inference_2gpu_streaming.sh
│   ├── inference_baseline.sh
│   └── minimal_inference/                  # ← moved from root minimal_inference/
│       ├── s2v_2gpu.py
│       ├── s2v_2gpu_optimized.py
│       ├── s2v_2gpu_streaming.py
│       └── s2v_streaming_interact.py       # (and any other files in there)
│
├── liveavatar/                             # OLD package (untouched, for training)
│   └── ...
│
├── ckpt/                                   # Checkpoints (gitignored)
└── output/                                 # Generated videos (gitignored)
```

## Moves

| From | To |
|------|----|
| `inference_nari.sh` | `examples/inference.sh` |
| `inference_2gpu.sh` | `legacy/inference_2gpu.sh` |
| `inference_2gpu_optimized.sh` | `legacy/inference_2gpu_optimized.sh` |
| `inference_2gpu_streaming.sh` | `legacy/inference_2gpu_streaming.sh` |
| `inference_baseline.sh` | `legacy/inference_baseline.sh` |
| `minimal_inference/` | `legacy/minimal_inference/` |
| `agent/plans/liveavatar_nari_migration.md` | `docs/liveavatar_nari_migration.md` |
| `agent/plans/streaming_pipeline_implementation.md` | `docs/streaming_pipeline_implementation.md` |

## Deletes

None — everything preserved in `legacy/` or `docs/`.

## Cleanup

- Remove empty `agent/plans/` directory after moving files
- Remove `agent/` if empty
- Ensure `.gitignore` covers `ckpt/`, `output/`, `__pycache__/`

## Verification

- `bash examples/inference.sh` works
- `liveavatar-infer --help` works
- `git status` shows clean moves (git tracks renames)
- Old files accessible under `legacy/`
