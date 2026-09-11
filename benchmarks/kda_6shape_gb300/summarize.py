"""Recompute the six-shape geometric mean from saved GPU measurements."""

import argparse
import json
import math
import statistics
from pathlib import Path


def summarize(payload):
    rows = payload["results"]
    expected = [
        (h, layout) for h in (96, 64) for layout in ("fixed", "mixed", "uniform")
    ]
    if [(r["heads"], r["layout"]) for r in rows] != expected:
        raise ValueError("Expected all six shapes in reference order")
    if any(r["correctness"] != "passed" for r in rows):
        raise ValueError("Every output and final-state check must pass")
    ratios = [r["flashkda_ms"] / r["ours_ms"] for r in rows]
    if any(not math.isfinite(x) or x <= 0 for x in ratios):
        raise ValueError("Timings must be finite and positive")
    gm = math.exp(statistics.mean(map(math.log, ratios)))
    hardware = payload["hardware"]
    lines = [
        f"GPU: {hardware['gpu_name']}; SMs: {hardware['sm_count']}; "
        f"driver: {hardware['driver']}; state: BF16; timing: CUPTI cold-L2.",
        "",
        f"FlashInfer: `{payload['sources']['flashinfer']['commit']}`; "
        f"FlashKDA: `{payload['sources']['flashkda']['commit']}`.",
        "",
        "| H | Layout | Ours (ms) | FlashKDA (ms) | Speedup |",
        "|--:|:--|--:|--:|--:|",
    ]
    for row, ratio in zip(rows, ratios, strict=True):
        lines.append(
            f"| {row['heads']} | {row['layout']} | {row['ours_ms']:.6f} | "
            f"{row['flashkda_ms']:.6f} | {ratio:.4f}x |"
        )
    lines.extend(["", f"Geomean (FlashKDA / ours): **{gm:.6f}x**", ""])
    return gm, "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "json", type=Path, nargs="?", default=Path(__file__).with_name("results.json")
    )
    args = parser.parse_args()
    print(summarize(json.loads(args.json.read_text()))[1])
