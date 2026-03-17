#!/usr/bin/env python3
"""
Visualize LiveAvatar profiling results.

Reads vram_profile.json and generates:
  1. chart_timeline.png       — VRAM peak over wall-clock time
  2. chart_latency_waterfall.png — Phase durations sorted longest-first
  3. chart_category_pie.png   — Time breakdown by category
  4. chart_dit_detail.png     — Per-block per-step DiT VRAM + latency
  5. chart_memory_timeline.png — Allocated/reserved/peak across phases
  6. profiling_report.html    — Combined HTML report

Usage:
    python visualize_profile.py [path/to/vram_profile.json]
"""

import base64
import json
import os
import re
import sys
from collections import defaultdict
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

CATEGORY_COLORS = {
    "init": "#4e79a7",
    "encode": "#f28e2b",
    "dit": "#e15759",
    "vae": "#76b7b2",
    "cache": "#59a14f",
    "offload": "#edc948",
    "scheduler": "#b07aa1",
    "io": "#ff9da7",
    "other": "#9c755f",
}


def _color(cat):
    return CATEGORY_COLORS.get(cat, "#9c755f")


def load_data(json_path):
    with open(json_path) as f:
        return json.load(f)


def _fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ── Chart 1: Timeline ────────────────────────────────────────────────────────

def plot_timeline(records, output_dir):
    """VRAM peak over wall-clock time, color-coded by category."""
    fig, ax = plt.subplots(figsize=(16, 6))

    for r in records:
        x = r["timestamp_s"]
        w = r["wall_time_ms"] / 1000.0
        ax.bar(
            x + w / 2, r["peak_MB"], width=max(w, 0.05),
            color=_color(r["category"]), edgecolor="none", alpha=0.85,
        )

    ax.axhline(y=81920, color="red", linestyle="--", linewidth=1, label="H100 80GB limit")
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Peak VRAM (MB)")
    ax.set_title("VRAM Peak Over Time — Full Pipeline")

    handles = [mpatches.Patch(color=c, label=k) for k, c in CATEGORY_COLORS.items()]
    ax.legend(handles=handles, loc="upper left", fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(output_dir, "chart_timeline.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"  Saved {path}")
    b64 = _fig_to_base64(fig)
    plt.close(fig)
    return b64


# ── Chart 2: Latency waterfall ───────────────────────────────────────────────

def plot_latency_waterfall(records, output_dir):
    """Horizontal bar chart of all phases sorted by duration (top-20 + rest)."""
    sorted_recs = sorted(records, key=lambda r: r["wall_time_ms"], reverse=True)

    max_bars = 30
    if len(sorted_recs) > max_bars:
        top = sorted_recs[:max_bars]
        rest_time = sum(r["wall_time_ms"] for r in sorted_recs[max_bars:])
        top.append({"phase": f"... {len(sorted_recs) - max_bars} others",
                     "wall_time_ms": rest_time, "category": "other"})
        sorted_recs = top

    fig, ax = plt.subplots(figsize=(12, max(6, len(sorted_recs) * 0.35)))

    phases = [r["phase"] for r in reversed(sorted_recs)]
    times = [r["wall_time_ms"] for r in reversed(sorted_recs)]
    colors = [_color(r["category"]) for r in reversed(sorted_recs)]

    ax.barh(range(len(phases)), times, color=colors, edgecolor="none")
    ax.set_yticks(range(len(phases)))
    ax.set_yticklabels(phases, fontsize=7)
    ax.set_xlabel("Latency (ms)")
    ax.set_title("Latency Waterfall — All Phases")
    ax.grid(axis="x", alpha=0.3)

    path = os.path.join(output_dir, "chart_latency_waterfall.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"  Saved {path}")
    b64 = _fig_to_base64(fig)
    plt.close(fig)
    return b64


# ── Chart 3: Category pie ────────────────────────────────────────────────────

def plot_category_pie(records, output_dir):
    """Time breakdown by category as a pie/donut chart."""
    cat_time = defaultdict(float)
    for r in records:
        cat_time[r["category"]] += r["wall_time_ms"]

    labels = sorted(cat_time, key=cat_time.get, reverse=True)
    sizes = [cat_time[l] for l in labels]
    colors = [_color(l) for l in labels]
    total = sum(sizes)

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct=lambda p: f"{p:.1f}%\n({p*total/100/1000:.1f}s)",
        startangle=90, pctdistance=0.75, textprops={"fontsize": 9},
    )
    centre_circle = plt.Circle((0, 0), 0.50, fc="white")
    ax.add_artist(centre_circle)
    ax.set_title("Time Breakdown by Category")

    path = os.path.join(output_dir, "chart_category_pie.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"  Saved {path}")
    b64 = _fig_to_base64(fig)
    plt.close(fig)
    return b64


# ── Chart 4: DiT detail ──────────────────────────────────────────────────────

def plot_dit_detail(records, output_dir):
    """Per-block per-step DiT peak VRAM + latency (first clip only)."""
    dit_recs = []
    pattern = re.compile(r"dit_forward_clip(\d+)_block(\d+)_step(\d+)")
    for r in records:
        m = pattern.match(r["phase"])
        if m:
            dit_recs.append({
                "clip": int(m.group(1)),
                "block": int(m.group(2)),
                "step": int(m.group(3)),
                **r,
            })

    if not dit_recs:
        print("  No DiT records found, skipping chart_dit_detail.png")
        return ""

    # Use first clip
    clip0 = [d for d in dit_recs if d["clip"] == 0]
    if not clip0:
        clip0 = dit_recs

    blocks = sorted(set(d["block"] for d in clip0))
    steps = sorted(set(d["step"] for d in clip0))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    n_steps = len(steps)
    width = 0.8 / max(n_steps, 1)
    x = np.arange(len(blocks))

    for si, step in enumerate(steps):
        peaks = []
        latencies = []
        for block in blocks:
            rec = next((d for d in clip0 if d["block"] == block and d["step"] == step), None)
            peaks.append(rec["peak_delta_MB"] if rec else 0)
            latencies.append(rec["wall_time_ms"] if rec else 0)

        offset = (si - n_steps / 2 + 0.5) * width
        ax1.bar(x + offset, peaks, width, label=f"Step {step}",
                color=plt.cm.Reds(0.3 + 0.7 * si / max(n_steps - 1, 1)))
        ax2.bar(x + offset, latencies, width,
                color=plt.cm.Blues(0.3 + 0.7 * si / max(n_steps - 1, 1)))

    ax1.set_ylabel("Peak VRAM Delta (MB)")
    ax1.set_title("DiT Forward Pass — VRAM per Block x Step (Clip 0)")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    ax2.set_ylabel("Latency (ms)")
    ax2.set_xlabel("Block Index")
    ax2.set_title("DiT Forward Pass — Latency per Block x Step (Clip 0)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Block {b}" for b in blocks])
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "chart_dit_detail.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"  Saved {path}")
    b64 = _fig_to_base64(fig)
    plt.close(fig)
    return b64


# ── Chart 5: Memory timeline ─────────────────────────────────────────────────

def plot_memory_timeline(records, output_dir):
    """Allocated / reserved / peak memory across all phases."""
    fig, ax = plt.subplots(figsize=(16, 6))

    indices = range(len(records))
    mem_before = [r["mem_before_MB"] for r in records]
    mem_after = [r["mem_after_MB"] for r in records]
    peaks = [r["peak_MB"] for r in records]
    reserved = [r["reserved_after_MB"] for r in records]

    ax.fill_between(indices, 0, reserved, alpha=0.15, color="gray", label="Reserved")
    ax.plot(indices, peaks, "r-", linewidth=1.5, label="Peak allocated", alpha=0.9)
    ax.plot(indices, mem_before, "b--", linewidth=1, label="Before", alpha=0.7)
    ax.plot(indices, mem_after, "g-", linewidth=1, label="After", alpha=0.7)

    ax.axhline(y=81920, color="red", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Phase index")
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Memory Timeline — Allocated / Reserved / Peak Across All Phases")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Add category color bar at bottom
    for i, r in enumerate(records):
        ax.axvspan(i - 0.4, i + 0.4, ymin=0, ymax=0.02,
                   color=_color(r["category"]), alpha=0.8)

    path = os.path.join(output_dir, "chart_memory_timeline.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"  Saved {path}")
    b64 = _fig_to_base64(fig)
    plt.close(fig)
    return b64


# ── Chart 6: Component comparison ────────────────────────────────────────────

def plot_component_comparison(records, output_dir):
    """Bar chart comparing avg latency + peak VRAM for key components."""
    components = {
        "DiT Forward": "dit",
        "VAE Decode": "vae",
        "KV Cache Ops": "cache",
        "Model Offload": "offload",
        "Encoding": "encode",
        "Scheduler": "scheduler",
        "Init": "init",
    }

    comp_data = {}
    for label, cat in components.items():
        cat_recs = [r for r in records if r["category"] == cat]
        if cat_recs:
            comp_data[label] = {
                "avg_time_ms": np.mean([r["wall_time_ms"] for r in cat_recs]),
                "max_peak_MB": max(r["peak_MB"] for r in cat_recs),
                "count": len(cat_recs),
                "total_time_ms": sum(r["wall_time_ms"] for r in cat_recs),
                "color": _color(cat),
            }

    if not comp_data:
        return ""

    labels = list(comp_data.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: total time
    total_times = [comp_data[l]["total_time_ms"] for l in labels]
    colors = [comp_data[l]["color"] for l in labels]
    bars1 = ax1.barh(labels, total_times, color=colors, edgecolor="none")
    ax1.set_xlabel("Total Time (ms)")
    ax1.set_title("Total Latency by Component")
    ax1.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars1, total_times):
        ax1.text(bar.get_width() + max(total_times) * 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.0f}ms", va="center", fontsize=8)

    # Right: max peak VRAM
    max_peaks = [comp_data[l]["max_peak_MB"] for l in labels]
    bars2 = ax2.barh(labels, max_peaks, color=colors, edgecolor="none")
    ax2.set_xlabel("Max Peak VRAM (MB)")
    ax2.set_title("Peak VRAM by Component")
    ax2.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars2, max_peaks):
        ax2.text(bar.get_width() + max(max_peaks) * 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.0f}MB", va="center", fontsize=8)

    fig.tight_layout()
    path = os.path.join(output_dir, "chart_component_comparison.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"  Saved {path}")
    b64 = _fig_to_base64(fig)
    plt.close(fig)
    return b64


# ── HTML Report ───────────────────────────────────────────────────────────────

def generate_html_report(records, output_dir, chart_b64s):
    """Generate a single HTML file embedding all charts + summary stats."""
    total_time = sum(r["wall_time_ms"] for r in records)

    # Category stats
    cat_stats = defaultdict(lambda: {"time": 0, "count": 0, "peak": 0})
    for r in records:
        c = r["category"]
        cat_stats[c]["time"] += r["wall_time_ms"]
        cat_stats[c]["count"] += 1
        cat_stats[c]["peak"] = max(cat_stats[c]["peak"], r["peak_MB"])

    cat_rows = ""
    for cat in sorted(cat_stats, key=lambda c: cat_stats[c]["time"], reverse=True):
        s = cat_stats[cat]
        pct = s["time"] / total_time * 100 if total_time > 0 else 0
        cat_rows += (
            f'<tr><td style="color:{_color(cat)};font-weight:bold">{cat}</td>'
            f'<td>{s["count"]}</td>'
            f'<td>{s["time"]:.1f}</td>'
            f'<td>{s["time"]/1000:.2f}</td>'
            f'<td>{pct:.1f}%</td>'
            f'<td>{s["peak"]:.0f}</td></tr>\n'
        )

    # Top-10 phases
    top10 = sorted(records, key=lambda r: r["wall_time_ms"], reverse=True)[:10]
    top10_rows = ""
    for r in top10:
        top10_rows += (
            f'<tr><td>{r["phase"]}</td>'
            f'<td style="color:{_color(r["category"])}">{r["category"]}</td>'
            f'<td>{r["wall_time_ms"]:.1f}</td>'
            f'<td>{r["peak_MB"]:.0f}</td>'
            f'<td>{r["peak_delta_MB"]:.0f}</td></tr>\n'
        )

    charts_html = ""
    chart_titles = [
        "VRAM Peak Over Time",
        "Latency Waterfall",
        "Category Breakdown",
        "DiT Step Detail",
        "Memory Timeline",
        "Component Comparison",
    ]
    for title, b64 in zip(chart_titles, chart_b64s):
        if b64:
            charts_html += f'<h2>{title}</h2>\n<img src="data:image/png;base64,{b64}" style="max-width:100%">\n'

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>LiveAvatar Profiling Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1200px; margin: 0 auto; padding: 20px; background: #fafafa; }}
  h1 {{ color: #333; border-bottom: 2px solid #e15759; padding-bottom: 10px; }}
  h2 {{ color: #555; margin-top: 40px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; font-size: 13px; }}
  th {{ background: #f5f5f5; text-align: center; }}
  td:first-child {{ text-align: left; }}
  .summary {{ display: flex; gap: 20px; flex-wrap: wrap; }}
  .stat-box {{ background: white; border: 1px solid #ddd; border-radius: 8px;
               padding: 15px 25px; text-align: center; }}
  .stat-box .value {{ font-size: 28px; font-weight: bold; color: #e15759; }}
  .stat-box .label {{ font-size: 12px; color: #888; margin-top: 5px; }}
  img {{ border: 1px solid #eee; border-radius: 4px; margin: 10px 0; }}
</style>
</head><body>
<h1>LiveAvatar Profiling Report</h1>

<div class="summary">
  <div class="stat-box"><div class="value">{total_time/1000:.1f}s</div><div class="label">Total Time</div></div>
  <div class="stat-box"><div class="value">{len(records)}</div><div class="label">Profiled Phases</div></div>
  <div class="stat-box"><div class="value">{max(r['peak_MB'] for r in records):.0f} MB</div><div class="label">Peak VRAM</div></div>
  <div class="stat-box"><div class="value">{len(cat_stats)}</div><div class="label">Categories</div></div>
</div>

<h2>Category Summary</h2>
<table>
<tr><th>Category</th><th>Count</th><th>Total (ms)</th><th>Total (s)</th><th>% Time</th><th>Max Peak (MB)</th></tr>
{cat_rows}
</table>

<h2>Top 10 Slowest Phases</h2>
<table>
<tr><th>Phase</th><th>Category</th><th>Time (ms)</th><th>Peak (MB)</th><th>Delta (MB)</th></tr>
{top10_rows}
</table>

{charts_html}

<hr>
<p style="color:#aaa;font-size:11px">Generated by LiveAvatar profiler &bull; {len(records)} phases recorded</p>
</body></html>"""

    path = os.path.join(output_dir, "profiling_report.html")
    with open(path, "w") as f:
        f.write(html)
    print(f"  Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    json_path = sys.argv[1] if len(sys.argv) > 1 else "profiling_output/vram_profile.json"
    output_dir = os.path.dirname(json_path) or "profiling_output"
    os.makedirs(output_dir, exist_ok=True)

    records = load_data(json_path)
    print(f"Loaded {len(records)} profiling records from {json_path}")
    print("Generating charts...")

    b64s = []
    b64s.append(plot_timeline(records, output_dir))
    b64s.append(plot_latency_waterfall(records, output_dir))
    b64s.append(plot_category_pie(records, output_dir))
    b64s.append(plot_dit_detail(records, output_dir))
    b64s.append(plot_memory_timeline(records, output_dir))
    b64s.append(plot_component_comparison(records, output_dir))

    generate_html_report(records, output_dir, b64s)
    print(f"\nAll outputs saved to {output_dir}/")
    print(f"Open {output_dir}/profiling_report.html in a browser for the full report.")


if __name__ == "__main__":
    main()
