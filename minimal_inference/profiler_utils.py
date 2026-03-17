"""
VRAM & Latency Profiler for LiveAvatar inference pipeline.

Usage:
    profiler = VRAMProfiler(device="cuda:0", log_path="profiling_output/vram_profile.json")
    with profiler.track("dit_forward_step0", "dit"):
        model(x)
    profiler.save()
    profiler.summary_table()
"""

import json
import os
import time
from collections import defaultdict
from contextlib import contextmanager

import torch


class VRAMProfiler:
    """Tracks peak VRAM, allocated/reserved memory, and wall-clock time per phase."""

    def __init__(self, device="cuda:0", log_path="profiling_results.json"):
        self.device = device
        self.log_path = log_path
        self.records = []
        self._global_start = time.perf_counter()

    @contextmanager
    def track(self, phase_name, category="other"):
        """Context manager that records memory + timing for a named phase.

        Args:
            phase_name: Unique name for this phase.
            category: One of "init", "encode", "dit", "vae", "cache",
                      "offload", "scheduler", "io", "other".
        """
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)

        mem_before = torch.cuda.memory_allocated(self.device)
        reserved_before = torch.cuda.memory_reserved(self.device)
        t0 = time.perf_counter()

        yield

        torch.cuda.synchronize(self.device)
        t1 = time.perf_counter()

        mem_after = torch.cuda.memory_allocated(self.device)
        peak = torch.cuda.max_memory_allocated(self.device)
        reserved_after = torch.cuda.memory_reserved(self.device)

        record = {
            "phase": phase_name,
            "category": category,
            "wall_time_ms": (t1 - t0) * 1000,
            "timestamp_s": t0 - self._global_start,
            "mem_before_MB": mem_before / 1e6,
            "mem_after_MB": mem_after / 1e6,
            "peak_MB": peak / 1e6,
            "delta_MB": (mem_after - mem_before) / 1e6,
            "peak_delta_MB": (peak - mem_before) / 1e6,
            "reserved_before_MB": reserved_before / 1e6,
            "reserved_after_MB": reserved_after / 1e6,
        }
        self.records.append(record)

        print(
            f"[PROFILE] {phase_name}: "
            f"{record['wall_time_ms']:.1f}ms | "
            f"peak={record['peak_MB']:.0f}MB | "
            f"delta={record['peak_delta_MB']:.0f}MB"
        )

    def save(self):
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        with open(self.log_path, "w") as f:
            json.dump(self.records, f, indent=2)
        print(f"[PROFILE] Saved {len(self.records)} records to {self.log_path}")

    def summary_table(self):
        """Print a formatted summary table to console."""
        if not self.records:
            print("[PROFILE] No records to display.")
            return

        print("\n" + "=" * 105)
        print(
            f"{'Phase':<55} {'Time(ms)':>10} {'Peak(MB)':>10} "
            f"{'Delta(MB)':>10} {'Category':<12}"
        )
        print("-" * 105)

        total_time = 0
        for r in self.records:
            total_time += r["wall_time_ms"]
            phase_display = r["phase"][:54]
            print(
                f"{phase_display:<55} {r['wall_time_ms']:>10.1f} "
                f"{r['peak_MB']:>10.0f} {r['peak_delta_MB']:>10.0f} "
                f"{r['category']:<12}"
            )

        print("-" * 105)
        print(f"{'TOTAL':<55} {total_time:>10.1f}")
        print("=" * 105)

        # Category summary
        cat_time = defaultdict(float)
        cat_peak = defaultdict(float)
        cat_count = defaultdict(int)
        for r in self.records:
            cat_time[r["category"]] += r["wall_time_ms"]
            cat_peak[r["category"]] = max(cat_peak[r["category"]], r["peak_MB"])
            cat_count[r["category"]] += 1

        print(
            f"\n{'Category':<15} {'Count':>6} {'Total Time(ms)':>15} "
            f"{'Max Peak(MB)':>15} {'% Time':>10}"
        )
        print("-" * 65)
        for cat in sorted(cat_time, key=cat_time.get, reverse=True):
            pct = cat_time[cat] / total_time * 100 if total_time > 0 else 0
            print(
                f"{cat:<15} {cat_count[cat]:>6} {cat_time[cat]:>15.1f} "
                f"{cat_peak[cat]:>15.0f} {pct:>9.1f}%"
            )
        print()
