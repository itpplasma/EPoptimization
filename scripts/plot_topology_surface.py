#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter


COLORS = ("#8A8A8A", "#276FBF", "#D1495B", "#F28E2B")
TOPOLOGY_LABELS = (
    "passing / unclassified",
    "ideal",
    "non-ideal",
    "lost, no topology",
)
JPAR_LABELS = (
    "passing / unclassified",
    "regular",
    "stochastic",
    "lost, unclassified",
)
B_TARGET = 5.865


def _periodic_edges(count: int) -> np.ndarray:
    return np.linspace(0.0, 2.0 * np.pi, count + 1)


def periodic_critical_points(
    field: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    field = np.asarray(field, dtype=float)
    if field.ndim != 2 or min(field.shape) < 3:
        raise ValueError("field must be a two-dimensional periodic grid")
    maxima = field == maximum_filter(field, size=3, mode="wrap")
    minima = field == minimum_filter(field, size=3, mode="wrap")
    dtheta = 0.5 * (np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0))
    dzeta = 0.5 * (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1))
    htheta = np.roll(field, -1, axis=0) - 2.0 * field + np.roll(field, 1, axis=0)
    hzeta = np.roll(field, -1, axis=1) - 2.0 * field + np.roll(field, 1, axis=1)
    hcross = 0.25 * (
        np.roll(np.roll(field, -1, axis=0), -1, axis=1)
        - np.roll(np.roll(field, -1, axis=0), 1, axis=1)
        - np.roll(np.roll(field, 1, axis=0), -1, axis=1)
        + np.roll(np.roll(field, 1, axis=0), 1, axis=1)
    )
    gradient = dtheta**2 + dzeta**2
    stationary = gradient == minimum_filter(gradient, size=3, mode="wrap")
    saddles = stationary & (htheta * hzeta - hcross**2 < 0.0)
    return np.argwhere(maxima), np.argwhere(minima), np.argwhere(saddles)


def _layout(nlambda: int):
    if nlambda <= 4:
        return 2, nlambda, lambda mu, sign: sign * nlambda + mu
    columns = 6
    rows = int(np.ceil(2 * nlambda / columns))
    return rows, columns, lambda mu, sign: 2 * mu + sign


def _scatter_points(
    axis, points: np.ndarray, theta: np.ndarray, zeta: np.ndarray, **style
) -> None:
    if len(points):
        axis.scatter(zeta[points[:, 1]], theta[points[:, 0]], **style)


def _legend_handles(labels):
    return (
        [Patch(facecolor=color, label=label) for color, label in zip(COLORS, labels)]
        + [Patch(facecolor="white", edgecolor="black", label="forbidden")]
        + [
            Line2D([], [], color="#7A3E9D", linestyle="--", label="turning contour"),
            Line2D(
                [], [], color="black", marker="^", linestyle="", label="$B$ maximum"
            ),
            Line2D(
                [], [], color="black", marker="v", linestyle="", label="$B$ minimum"
            ),
            Line2D(
                [], [], color="#7A3E9D", marker="x", linestyle="", label="$B$ saddle"
            ),
        ]
    )


def plot_surface(
    topology_file: Path,
    out: Path,
    label: str,
    shift: int = 0,
    classifier: str = "topology",
) -> None:
    data = np.load(topology_file)
    if classifier not in ("topology", "jpar"):
        raise ValueError("classifier must be topology or jpar")
    topology = data[classifier]
    particle_index = data["particle_index"]
    lost = data["lost"] if "lost" in data else np.zeros_like(topology, dtype=bool)
    lambdas = data["lambda_values"]
    signs = data["signs"] if "signs" in data else np.array([-1.0, 1.0])
    shifts = data["shifts"] if "shifts" in data else np.arange(topology.shape[2])
    surface = float(data["surface"])
    if topology.ndim != 5 or topology.shape[1:3] != (2, 2):
        raise ValueError("topology must have shape (lambda, sign, shift, theta, zeta)")
    if not 0 <= shift < topology.shape[2]:
        raise ValueError("lattice shift index is outside the topology grid")
    field = data["b"][shift]

    nlambda, _, _, ntheta, nzeta = topology.shape
    nrows, ncolumns, panel_index = _layout(nlambda)
    figure, axes = plt.subplots(
        nrows,
        ncolumns,
        figsize=(2.75 * ncolumns, 2.65 * nrows + 2.0),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    xedges = _periodic_edges(nzeta)
    yedges = _periodic_edges(ntheta)
    cmap = ListedColormap(COLORS)
    cmap.set_bad("white")
    norm = BoundaryNorm((-0.5, 0.5, 1.5, 2.5, 3.5), cmap.N)
    zeta = np.linspace(0.0, 2.0 * np.pi, nzeta, endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    maxima, minima, saddles = periodic_critical_points(field)
    for mu_index, lambda_value in enumerate(lambdas):
        for sign_index, sign_value in enumerate(signs):
            sign = r"$v_\parallel > 0$" if sign_value > 0.0 else r"$v_\parallel < 0$"
            axis = axes.flat[panel_index(mu_index, sign_index)]
            values = topology[mu_index, sign_index, shift].copy()
            unresolved_loss = lost[mu_index, sign_index, shift] & (values == 0)
            values[unresolved_loss] = 3
            values = np.ma.masked_where(
                particle_index[mu_index, sign_index, shift] < 0,
                values,
            )
            axis.pcolormesh(
                xedges, yedges, values, cmap=cmap, norm=norm, shading="flat"
            )
            if np.ptp(field) > 0.0:
                axis.contour(
                    zeta,
                    theta,
                    field,
                    levels=5,
                    colors="black",
                    linewidths=0.35,
                    alpha=0.45,
                )
                turning = B_TARGET / lambda_value
                if np.min(field) < turning < np.max(field):
                    axis.contour(
                        zeta,
                        theta,
                        field,
                        levels=[turning],
                        colors="#7A3E9D",
                        linestyles="--",
                        linewidths=0.8,
                    )
            _scatter_points(
                axis,
                maxima,
                theta,
                zeta,
                marker="^",
                facecolors="none",
                edgecolors="black",
                s=18,
                linewidths=0.6,
            )
            _scatter_points(
                axis,
                minima,
                theta,
                zeta,
                marker="v",
                facecolors="white",
                edgecolors="black",
                s=18,
                linewidths=0.6,
            )
            _scatter_points(
                axis,
                saddles,
                theta,
                zeta,
                marker="x",
                c="#7A3E9D",
                s=14,
                linewidths=0.6,
            )
            axis.set_title(rf"$\mu B_0/E={lambda_value:.3f}$, {sign}", fontsize=9)
            axis.set_aspect("equal")
            axis.set_xticks((0.0, np.pi, 2.0 * np.pi), ("0", r"$\pi$", r"$2\pi$"))
            axis.set_yticks((0.0, np.pi, 2.0 * np.pi), ("0", r"$\pi$", r"$2\pi$"))
    for axis in axes.flat[2 * nlambda :]:
        axis.set_visible(False)
    for axis in axes[-1]:
        axis.set_xlabel(r"field-period Boozer angle $N_{FP}\zeta$")
    for axis in axes[:, 0]:
        axis.set_ylabel(r"Boozer poloidal angle $\theta$")
    figure.legend(
        handles=_legend_handles(
            TOPOLOGY_LABELS if classifier == "topology" else JPAR_LABELS
        ),
        loc="upper center",
        ncol=3,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, 0.94),
    )
    classifier_label = "topology" if classifier == "topology" else "J-parallel"
    figure.suptitle(
        rf"{label} {classifier_label} classes on $s={surface:g}$, "
        rf"lattice shift {shifts[shift]:g}; solid color: class, contours: $|B|$",
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.78))
    figure.subplots_adjust(hspace=0.32, wspace=0.12)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=200, facecolor="white")
    plt.close(figure)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--topology", type=Path, required=True)
    root.add_argument("--out", type=Path, required=True)
    root.add_argument("--label", default="configuration")
    root.add_argument("--shift-index", type=int, default=0)
    root.add_argument("--classifier", choices=("topology", "jpar"), default="topology")
    return root


if __name__ == "__main__":
    args = parser().parse_args()
    plot_surface(
        args.topology,
        args.out,
        args.label,
        args.shift_index,
        args.classifier,
    )
