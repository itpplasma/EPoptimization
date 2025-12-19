#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


_RE_LOSS = re.compile(r"Loss =\s*(?P<loss>[0-9.]+)%")
_RE_SIMPLE = re.compile(
    r"SIMPLE proxy metrics:\s*.*?score=(?P<score>[0-9.]+),\s*objective=(?P<objective>[0-9.]+)"
)


def parse_log(path: Path) -> dict[str, list[float]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    loss: list[float] = []
    score: list[float] = []
    objective: list[float] = []

    for line in text.splitlines():
        m = _RE_LOSS.search(line)
        if m:
            loss.append(float(m.group("loss")))
            continue

        m = _RE_SIMPLE.search(line)
        if m:
            score.append(float(m.group("score")))
            objective.append(float(m.group("objective")))

    return {"loss_pct": loss, "score": score, "objective": objective}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot SIMPLE proxy metric evolution from an EPoptimization stdout log."
    )
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    series = parse_log(args.log)
    if not series["objective"]:
        raise RuntimeError(f"No SIMPLE proxy metrics found in {args.log}")

    fig, ax1 = plt.subplots(figsize=(10, 5))
    x_obj = list(range(len(series["objective"])))
    ax1.plot(x_obj, series["objective"], label="objective = 1 - score", linewidth=1.2)
    ax1.set_xlabel("Evaluation index")
    ax1.set_ylabel("Objective")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    if series["loss_pct"]:
        x_loss = list(range(len(series["loss_pct"])))
        ax2.plot(
            x_loss,
            series["loss_pct"],
            label="loss (%)",
            linewidth=1.0,
            color="tab:orange",
            alpha=0.8,
        )
    ax2.set_ylabel("Loss (%)")
    ax2.set_ylim(0.0, 100.0)

    best = min(series["objective"])
    ax1.axhline(best, linestyle="--", linewidth=1.0, color="tab:green", alpha=0.7)

    title = args.title or f"{args.log.name} (best objective={best:.6g})"
    fig.suptitle(title)

    lines, labels = [], []
    for ax in (ax1, ax2):
        l, lab = ax.get_legend_handles_labels()
        lines += l
        labels += lab
    if lines:
        ax1.legend(lines, labels, loc="upper right")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)


if __name__ == "__main__":
    main()

