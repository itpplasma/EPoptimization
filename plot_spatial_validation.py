#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def validation_points(summary: dict) -> dict[str, np.ndarray]:
    configurations = summary["configurations"]
    labels = sorted(configurations, key=lambda label: (label.lower() != "alpes", label))
    x = np.array([configurations[label]["spatial_score_change"] for label in labels])
    y = np.array([configurations[label]["direct_late_change"] for label in labels])
    yerr = np.array(
        [configurations[label]["direct_late_paired_se"] for label in labels]
    )
    if np.any(yerr < 0.0):
        raise ValueError("direct paired standard errors must be nonnegative")
    lower = np.zeros(len(labels))
    upper = np.zeros(len(labels))
    shift_changes = summary.get("shift_score_changes", {})
    for index, label in enumerate(labels):
        shifts = np.asarray(shift_changes.get(label, [x[index]]), dtype=float)
        lower[index] = x[index] - np.min(shifts)
        upper[index] = np.max(shifts) - x[index]
    if np.any(lower < -1.0e-15) or np.any(upper < -1.0e-15):
        raise ValueError("mean spatial change lies outside its lattice-shift range")
    return {
        "labels": np.asarray(
            ["ALPES" if label.lower() == "alpes" else label for label in labels]
        ),
        "x": x,
        "xerr": np.stack((np.maximum(lower, 0.0), np.maximum(upper, 0.0))),
        "y": y,
        "yerr": yerr,
    }


def plot_validation(summary_path: Path, out: Path, title: str) -> None:
    summary = json.loads(summary_path.read_text())
    points = validation_points(summary)
    figure, axis = plt.subplots(figsize=(6.4, 5.2), layout="constrained")
    axis.errorbar(
        points["x"],
        points["y"],
        xerr=points["xerr"],
        yerr=points["yerr"],
        fmt="o",
        color="#276FBF",
        ecolor="#555555",
        capsize=4,
        markersize=7,
    )
    for label, x, y in zip(points["labels"], points["x"], points["y"]):
        axis.annotate(label, (x, y), xytext=(6, 6), textcoords="offset points")
    axis.axhline(0.0, color="#777777", linewidth=0.8)
    axis.axvline(0.0, color="#777777", linewidth=0.8)
    axis.set_xlabel("spatial barrier score change from ALPES (lower is better)")
    axis.set_ylabel("direct late-loss change from ALPES (lower is better)")
    axis.set_title(title)
    axis.grid(alpha=0.2)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=200, facecolor="white")
    plt.close(figure)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--summary", type=Path, required=True)
    root.add_argument("--out", type=Path, required=True)
    root.add_argument("--title", default="Spatial barrier proxy validation")
    return root


if __name__ == "__main__":
    args = parser().parse_args()
    plot_validation(args.summary, args.out, args.title)
