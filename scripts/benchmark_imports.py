"""Measure import time for each dependency."""
import time

def time_import(name):
    start = time.perf_counter()
    __import__(name)
    elapsed = time.perf_counter() - start
    print(f"  {name}: {elapsed:.2f}s")
    return elapsed

print("=== Import benchmarks ===")
total = 0
total += time_import("torch")
total += time_import("torch.nn.attention.flex_attention")
total += time_import("torch._dynamo")
total += time_import("diffusers")
total += time_import("transformers")
# total += time_import("peft")
total += time_import("safetensors")
total += time_import("torchvision")
total += time_import("PIL")
print(f"\n  Total: {total:.2f}s")
