# LiveAvatar Repository Analysis

> Generated: 2026-03-16

---

## 1. Project Overview

**LiveAvatar** is an algorithm-system co-designed framework for **real-time, streaming, infinite-length interactive avatar video generation**. It synthesizes expressive avatar videos driven by audio input, reference images, and optional text prompts.

### Key Capabilities

- Real-time streaming video generation at **45 FPS** on multi-card H800 GPUs
- Infinite-length (10,000+ second) autoregressive video generation via block-wise processing
- Audio-driven avatar synthesis with high quality and generalization
- Support for diverse character types (cartoon, realistic, etc.)
- Support for singing, human-AI conversation scenarios

### Technical Highlights

- **14B-parameter** diffusion model (WanS2V)
- 4-step distribution-matching distillation for efficiency
- Timestep-forcing pipeline parallelism (TPP)
- Block-wise autoregressive processing for infinite streaming
- FP8 quantization support (enables 48GB GPU inference)
- Torch compilation for 2.5x peak / 3x average FPS improvements

---

## 2. Architecture

The system follows a **modular, pipeline-based architecture** with clear separation between:

| Component | Role |
|-----------|------|
| **Diffusion Models** | Core generation engine (DiT with 40 layers, 5120 dim, 40 heads) |
| **Audio Processing** | Wav2Vec2-based audio encoding and temporal alignment |
| **Distributed Inference** | Multi-GPU orchestration via TPP |
| **VAE Codecs** | Latent encoding/decoding (8x8 spatial, 4x temporal compression) |
| **Text Encoding** | T5-XXL semantic conditioning |

### Speech-to-Video Pipeline Flow

```
Audio (WAV) ──► Wav2Vec2 ──► Audio Features (1024-dim @ 50Hz)
                                    │
Reference Image ──► VAE Encode ──►  │
                                    ▼
Text Prompt ──► T5-XXL ──► Diffusion Transformer (14B params, 4-step flow matching)
                                    │
                                    ▼
                           VAE Decode ──► Video Frames (MP4)
```

### Audio Injection

Cross-attention injection at **12 strategic transformer layers**: 0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39.

### Block-wise Generation

The diffusion operates in **latent space** where temporal compression is 4x. Each clip of `infer_frames` pixel frames (default 48) becomes ~12 latent frames. These latent frames are grouped into blocks of **3 latent frames** (`num_frames_per_block = 3`, set in `wan_s2v_14B_modified.py:65`). The blockwise causal attention mask ensures each block can attend to all previous blocks but not future ones, enabling autoregressive infinite-length streaming.

### Distributed Inference (TPP - 5 GPUs)

- **4 GPUs** for DiT inference (timestep-forcing pipeline parallelism)
- **1 GPU** for VAE decoding
- Latency hiding through overlapped computation

---

## 3. Tech Stack

### Core

| Category | Technology |
|----------|------------|
| **Language** | Python 3.10 |
| **Deep Learning** | PyTorch 2.8.0, TorchVision 0.23.0 |
| **Diffusion** | Diffusers 0.31.0, Flow Matching |
| **Transformers** | Transformers 4.49.0–4.51.3, PEFT 0.17.1 (LoRA) |
| **Distributed** | PyTorch Distributed (DDP, FSDP), DeepSpeed ZeRO-2 |
| **Compilation** | torch.compile + Inductor backend |
| **Attention** | FlashAttention 2/3 |

### Supporting Libraries

| Category | Libraries |
|----------|-----------|
| Audio Processing | librosa, Wav2Vec2, Whisper, CosyVoice |
| Video I/O | OpenCV, imageio, decord, ffmpeg |
| Vision Models | SAM2, InsightFace, DW-Pose, Qwen-VL |
| Optimization | FP8 quantization, torch.compile |
| UI | Gradio |
| Logging | Weights & Biases, TensorBoard |
| Config | OmegaConf, Hydra, HyperPyYAML |
| Model Hub | Hugging Face Hub, safetensors |

### Model Dependencies

| Model | Parameters | Purpose |
|-------|-----------|---------|
| **Wan2.2-S2V-14B** | 14B | Base diffusion model |
| **Quark-Vision/Live-Avatar** | LoRA | Fine-tuned avatar generation weights |
| **T5-XXL** | ~11B | Text encoder (50 layers) |
| **Wav2Vec2-Large-XLSR-53** | 315M | Audio feature extraction |
| **Wan2.1 VAE** | — | Latent space codec |

---

## 4. Directory Structure

```
/root/LiveAvatar/
├── README.md                              # Documentation
├── LICENSE                                # Apache 2.0
├── requirements.txt                       # 44 dependencies
│
├── infinite_inference_multi_gpu.sh        # Multi-GPU launcher (5 GPUs)
├── infinite_inference_single_gpu.sh       # Single-GPU launcher (80GB VRAM)
├── gradio_multi_gpu.sh                    # Web UI multi-GPU
├── gradio_single_gpu.sh                   # Web UI single-GPU
│
├── examples/                              # Sample inputs (~49 MB)
│   └── *.jpg, *.png, *.wav               # 16 example files
│
├── assets/                                # Logo, demo images
│
├── liveavatar/                            # Main package
│   ├── configs/
│   │   └── s2v_causal_sft.yaml           # Training config
│   │
│   ├── models/
│   │   ├── model_interface.py            # Abstract interfaces
│   │   └── wan/                          # WAN framework
│   │       ├── causal_s2v_pipeline.py              # Single-GPU pipeline
│   │       ├── causal_s2v_pipeline_tpp.py          # Multi-GPU TPP pipeline
│   │       ├── causal_s2v_pipeline_tpp_blockwise.py # Block-wise infinite gen
│   │       ├── causal_model_s2v.py                 # S2V diffusion model
│   │       ├── causal_audio_encoder.py             # Audio features
│   │       ├── causal_motioner.py                  # Motion encoder
│   │       ├── wan_wrapper.py                      # Model loading
│   │       ├── flow_match.py                       # Flow matching scheduler
│   │       ├── inference_utils.py                  # Compilation flags
│   │       ├── wan_2_2/                            # WAN 2.2 (primary)
│   │       │   ├── configs/                        # Model configs (S2V, I2V, T2V, Ti2V)
│   │       │   ├── modules/                        # Core modules
│   │       │   │   ├── model.py                    # WanModel (DiT)
│   │       │   │   ├── attention.py                # Attention implementations
│   │       │   │   ├── t5.py                       # T5 encoder
│   │       │   │   ├── vae2_1.py / vae2_2.py       # VAE decoders
│   │       │   │   ├── vae_streaming.py            # Streaming VAE
│   │       │   │   └── s2v/                        # Speech-to-Video modules
│   │       │   │       ├── model_s2v.py            # CausalWanModel_S2V
│   │       │   │       ├── audio_encoder.py        # Wav2Vec2 encoder
│   │       │   │       ├── audio_utils.py          # Audio cross-attention
│   │       │   │       ├── motioner.py             # FramePackMotioner
│   │       │   │       └── s2v_utils.py            # RoPE precomputation
│   │       │   ├── distributed/                    # FSDP, sequence parallel, context parallel
│   │       │   └── utils/                          # Solvers, prompt expansion, video utils
│   │       └── wan_base/                           # WAN 1.x (legacy)
│   │
│   └── utils/
│       ├── args_config.py                # Config parsing (YAML + argparse)
│       ├── checkpoint_utils.py           # FSDP checkpoint saving
│       ├── audio_preprocess.py           # Audio preprocessing, vocal separation
│       ├── fp8_linear.py                 # FP8 quantization
│       ├── model_manager.py              # Model loading/management
│       ├── detectors/s3fd/               # Face detection
│       ├── fvd/                          # Fréchet Video Distance metrics
│       ├── router/                       # SAM2 tools, TTS integration
│       └── sync_net/                     # Audio-visual synchronization
│
├── minimal_inference/
│   ├── s2v_streaming_interact.py         # Main CLI inference (40+ args)
│   ├── gradio_app.py                     # Web UI (Gradio)
│   └── batch_eval.py                     # Batch evaluation
│
└── agent/plans/                          # Planning directory
```

---

## 5. Entry Points & Usage

### CLI Inference

**Multi-GPU (5x H800, real-time 45 FPS):**
```bash
bash infinite_inference_multi_gpu.sh
# Uses torchrun with 5 processes for TPP
```

**Single-GPU (80GB VRAM, offline):**
```bash
bash infinite_inference_single_gpu.sh
```

### Web UI (Gradio)

```bash
bash gradio_multi_gpu.sh   # or gradio_single_gpu.sh
```

### Key CLI Arguments

| Argument | Default | Purpose |
|----------|---------|---------|
| `--task` | s2v-14B | Model task selection |
| `--size` | 720*400 | Output resolution |
| `--infer_frames` | 48/80 | Frames per clip |
| `--sample_steps` | 4 | Diffusion steps (4=distilled, 20/40=standard) |
| `--num_clip` | 1 | Number of clips for long videos |
| `--load_lora` | — | LoRA checkpoint path |
| `--single_gpu` | false | Single-GPU mode |
| `--num_gpus_dit` | 4 | GPUs for DiT |
| `--fp8` | false | FP8 quantization |
| `--offload_model` | false | CPU offload |
| `--enable_online_decode` | false | Streaming VAE |

---

## 6. Configuration

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `CUDA_VISIBLE_DEVICES` | — | GPU assignment |
| `ENABLE_COMPILE` | true | torch.compile optimization |
| `ENABLE_FP8` | false | FP8 quantization (48GB GPUs) |
| `NCCL_DEBUG` | WARN | Distributed debugging |
| `HF_ENDPOINT` | huggingface.co | Model download mirror |

### Training Config (`s2v_causal_sft.yaml`)

- Frames per sample: 48 (73 motion frames)
- LoRA: rank=128, alpha=64
- Optimizer: AdamW, lr=5e-5
- Mixed precision: BF16
- DeepSpeed ZeRO-2
- Max steps: 100,000

---

## 7. System Requirements

| Configuration | GPUs | VRAM | Performance |
|---------------|------|------|-------------|
| **Real-time (TPP)** | 5x H800 | 5x 80GB | 45 FPS |
| **Single-GPU** | 1x A100/H100 | 80GB | Offline |
| **FP8 Mode** | 1x | 48GB | Offline (reduced quality) |

**Software:**
- CUDA 12.4.1
- Python 3.10
- FFmpeg (system package)

---

## 8. Code Quality Assessment

### Strengths

- **Well-structured**: Clear separation of concerns (models, utils, configs, inference)
- **Abstract interfaces**: `model_interface.py` defines clean `DiffusionModelInterface`, `VAEInterface`, etc.
- **Comprehensive README**: Detailed setup, usage, performance benchmarks
- **Multiple inference modes**: Single-GPU, multi-GPU, Gradio UI, batch eval
- **Optimization depth**: FP8, torch.compile, streaming VAE, model offloading

### Areas for Improvement

- **No unit tests**: No test directory or test files observed
- **No CI/CD**: No GitHub Actions, Dockerfile, or containerization
- **Mixed-language comments**: English and Chinese comments throughout
- **No type annotations**: Most functions lack type hints
- **No linting config**: No pyproject.toml, .flake8, or ruff.toml
- **Large file sizes**: Some modules exceed 800+ lines (motioner.py, model_s2v.py)

### Planned Work (from README)

- [ ] UI integration for streaming interaction
- [ ] TTS integration
- [ ] Training code release
- [ ] LiveAvatar v1.2

---

## 9. Summary Statistics

| Metric | Value |
|--------|-------|
| Python files | ~104 |
| Repository size | ~313 MB |
| Dependencies | 44 packages |
| Model parameters | 14B (Wan2.2-S2V) |
| Inference speed | 45 FPS (multi-GPU) |
| Min GPU VRAM | 48GB (FP8) / 80GB (standard) |
| Max video length | 10,000+ seconds |
| Diffusion steps | 4 (distilled) / 20–40 (standard) |
| Supported tasks | S2V (primary), T2V, I2V, Ti2V |
| License | Apache 2.0 |

---

## 10. Critical Files for Understanding the System

Read in this order for fastest comprehension:

1. `README.md` — Overview and usage
2. `minimal_inference/s2v_streaming_interact.py` — Main entry point
3. `liveavatar/models/wan/causal_s2v_pipeline.py` — Single-GPU pipeline
4. `liveavatar/models/wan/causal_s2v_pipeline_tpp.py` — Multi-GPU pipeline
5. `liveavatar/models/wan/causal_model_s2v.py` — Model architecture
6. `liveavatar/models/wan/wan_2_2/modules/model.py` — DiT implementation
7. `liveavatar/models/wan/wan_2_2/modules/s2v/model_s2v.py` — S2V transformer
8. `liveavatar/configs/s2v_causal_sft.yaml` — Training configuration
9. `liveavatar/utils/fp8_linear.py` — Optimization details
