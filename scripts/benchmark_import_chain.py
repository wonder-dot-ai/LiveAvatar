"""Trace which modules cause slow imports in the liveavatar package."""
import os
import time
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_original_import = __builtins__.__import__
_timings = []
_active = set()

def _timed_import(name, *args, **kwargs):
    if name in _active or name in sys.modules:
        return _original_import(name, *args, **kwargs)
    _active.add(name)
    start = time.perf_counter()
    result = _original_import(name, *args, **kwargs)
    elapsed = time.perf_counter() - start
    _active.discard(name)
    if elapsed > 0.3:
        _timings.append((name, elapsed))
    return result

print("=== Timing: import liveavatar.models.wan.causal_s2v_pipeline_2gpu_optimized ===\n")

__builtins__.__import__ = _timed_import
t0 = time.perf_counter()
from liveavatar.models.wan.causal_s2v_pipeline_2gpu_optimized import WanS2V
t1 = time.perf_counter()
__builtins__.__import__ = _original_import

print(f"\nTotal import time: {t1 - t0:.2f}s\n")
print("Slow modules (>0.3s, sorted by time):")
for name, elapsed in sorted(_timings, key=lambda x: -x[1]):
    print(f"  {elapsed:6.2f}s  {name}")
