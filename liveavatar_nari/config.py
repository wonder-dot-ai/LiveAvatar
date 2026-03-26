"""Lightweight config loader — no torch/diffusers/transformers imports."""
import yaml
import torch


class _DotDict(dict):
    """Dict with attribute access (EasyDict replacement)."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


def _to_dotdict(d):
    """Recursively convert nested dicts to _DotDict."""
    if isinstance(d, dict):
        return _DotDict({k: _to_dotdict(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_to_dotdict(v) for v in d]
    return d


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def load_config(yaml_path):
    """Load config from YAML, return EasyDict-compatible object."""
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    # Convert dtype strings to torch dtypes
    for key in ["t5_dtype", "param_dtype"]:
        if key in raw and isinstance(raw[key], str):
            raw[key] = _DTYPE_MAP[raw[key]]

    # Convert tuple-like lists
    if "vae_stride" in raw:
        raw["vae_stride"] = tuple(raw["vae_stride"])
    if "transformer" in raw and "patch_size" in raw["transformer"]:
        raw["transformer"]["patch_size"] = tuple(raw["transformer"]["patch_size"])
    if "transformer" in raw and "window_size" in raw["transformer"]:
        raw["transformer"]["window_size"] = tuple(raw["transformer"]["window_size"])

    return _to_dotdict(raw)
