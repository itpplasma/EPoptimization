#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def promotion_points(summary: dict) -> dict[str, np.ndarray]:
    labels = summary["ranked_candidates"]
    candidates = summary["candidates"]
    points: dict[str, np.ndarray] = {"labels": np.asarray(labels)}
    for metric in ("total", "prompt", "late"):
        rows = [candidates[label]["aggregate"][metric] for label in labels]
        points[metric] = np.asarray([row["change"] for row in rows], dtype=float)
        points[f"{metric}_error"] = 2.0 * np.asarray(
            [row["paired_se"] for row in rows], dtype=float
        )
    return points


def plot_promotion(summary_path: Path, out: Path) -> None:
    summary = json.loads(summary_path.read_text())
    points = promotion_points(summary)
    x = np.arange(len(points["labels"]))
    figure, axis = plt.subplots(figsize=(7.2, 4.8), layout="constrained")
    styles = (
        ("total", "#276FBF", "total"),
        ("prompt", "#F28E2B", "prompt, $t \\leq 1$ ms"),
        ("late", "#D1495B", "late, $1$ ms $< t < 0.3$ s"),
    )
    for offset, (metric, color, label) in zip((-0.16, 0.0, 0.16), styles):
        axis.errorbar(
            x + offset,
            points[metric],
            yerr=points[f"{metric}_error"],
            fmt="o",
            color=color,
            capsize=3,
            label=label,
        )
    axis.axhline(0.0, color="#555555", linewidth=0.9)
    axis.axhline(-0.02, color="#276FBF", linestyle=":", linewidth=0.9)
    axis.axhline(-0.01, color="#D1495B", linestyle=":", linewidth=0.9)
    axis.set_xticks(x, points["labels"])
    axis.set_xlabel("candidate ID, ranked by total loss")
    axis.set_ylabel("paired loss-fraction change from ALPES")
    axis.set_title(r"$s=0.25$: late-loss reductions transfer into prompt loss")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(ncols=3, loc="upper center")
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=200, facecolor="white")
    plt.close(figure)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--summary", type=Path, required=True)
    root.add_argument("--out", type=Path, required=True)
    return root


if __name__ == "__main__":
    args = parser().parse_args()
    plot_promotion(args.summary, args.out)
