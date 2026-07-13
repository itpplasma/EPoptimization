#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np


COLORS = ("#8A8A8A", "#276FBF", "#D1495B", "#F28E2B")
LABELS = ("passing / unclassified", "ideal", "non-ideal", "lost unresolved")


def _periodic_edges(count: int) -> np.ndarray:
    return np.linspace(0.0, 2.0 * np.pi, count + 1)


def plot_surface(topology_file: Path, out: Path) -> None:
    data = np.load(topology_file)
    topology = data["topology"]
    particle_index = data["particle_index"]
    lost = data["lost"] if "lost" in data else np.zeros_like(topology, dtype=bool)
    lambdas = data["lambda_values"]
    signs = data["signs"] if "signs" in data else np.array([-1.0, 1.0])
    surface = float(data["surface"])
    field = data["b"][0]
    if topology.ndim != 5 or topology.shape[1:3] != (2, 2):
        raise ValueError("topology must have shape (lambda, sign, shift, theta, zeta)")

    nlambda, _, _, ntheta, nzeta = topology.shape
    figure, axes = plt.subplots(
        2,
        nlambda,
        figsize=(3.15 * nlambda, 5.8),
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
    for mu_index, lambda_value in enumerate(lambdas):
        for sign_index, sign_value in enumerate(signs):
            sign = r"$v_\parallel > 0$" if sign_value > 0.0 else r"$v_\parallel < 0$"
            axis = axes[sign_index, mu_index]
            values = topology[mu_index, sign_index, 0].copy()
            unresolved_loss = lost[mu_index, sign_index, 0] & (values == 0)
            values[unresolved_loss] = 3
            values = np.ma.masked_where(
                particle_index[mu_index, sign_index, 0] < 0,
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
            axis.set_title(rf"$\mu B_0/E={lambda_value:.3f}$, {sign}")
            axis.set_aspect("equal")
            axis.set_xticks((0.0, np.pi, 2.0 * np.pi), ("0", r"$\pi$", r"$2\pi$"))
            axis.set_yticks((0.0, np.pi, 2.0 * np.pi), ("0", r"$\pi$", r"$2\pi$"))
    for axis in axes[-1]:
        axis.set_xlabel(r"field-period Boozer angle $N_{FP}\zeta$")
    for axis in axes[:, 0]:
        axis.set_ylabel(r"Boozer poloidal angle $\theta$")
    figure.legend(
        [Patch(facecolor=color, label=label) for color, label in zip(COLORS, LABELS)]
        + [Patch(facecolor="white", edgecolor="black", label="forbidden")],
        LABELS + ("forbidden",),
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.925),
    )
    figure.suptitle(
        rf"ALPES topology on $s={surface:.2f}$; solid color: topology, contours: $|B|$",
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=200, facecolor="white")
    plt.close(figure)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--topology", type=Path, required=True)
    root.add_argument("--out", type=Path, required=True)
    return root


if __name__ == "__main__":
    args = parser().parse_args()
    plot_surface(args.topology, args.out)
