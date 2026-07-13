#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spatial_atlas import load_spatial_atlas
from spatial_barrier import periodic_components, periodic_kernel_risk, strongest_channel


def evolution_fields(
    topology: np.ndarray, weights: np.ndarray, shift: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    topology = np.asarray(topology)
    weights = np.asarray(weights, dtype=float)
    if topology.ndim != 6:
        raise ValueError(
            "topology must have shape (surface, mu, sign, shift, theta, zeta)"
        )
    expected_weights = (topology.shape[0], topology.shape[3], *topology.shape[-2:])
    if weights.shape != expected_weights:
        raise ValueError("angular weights do not match topology")
    if not 0 <= shift < topology.shape[3]:
        raise ValueError("shift index is outside the topology grid")

    selected = topology[:, :, :, shift]
    angular_weights = weights[:, shift]
    valid = selected > 0
    nonideal = selected == 2
    support = valid * angular_weights[:, None, None]
    denominator = np.sum(support, axis=(-2, -1))
    area = np.divide(
        np.sum(nonideal * support, axis=(-2, -1)),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0.0,
    )
    components = np.zeros(selected.shape[:3], dtype=int)
    for surface in range(selected.shape[0]):
        for mu_index in range(selected.shape[1]):
            for sign_index in range(selected.shape[2]):
                _, components[surface, mu_index, sign_index] = periodic_components(
                    nonideal[surface, mu_index, sign_index]
                )

    risk = periodic_kernel_risk(
        nonideal,
        valid,
        angular_weights[:, None, None],
    )
    capacity = np.zeros(selected.shape[:3])
    for stop in range(1, selected.shape[0] + 1):
        for mu_index in range(selected.shape[1]):
            for sign_index in range(selected.shape[2]):
                capacity[stop - 1, mu_index, sign_index] = strongest_channel(
                    nonideal[:stop, mu_index, sign_index],
                    risk[:stop, mu_index, sign_index],
                    angular_weights[:stop],
                )
    return area, components, capacity


def _edges(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1 or not len(centers):
        raise ValueError("grid centers must be a nonempty vector")
    if len(centers) == 1:
        width = max(abs(float(centers[0])) * 0.1, 0.05)
        return np.array([centers[0] - width, centers[0] + width])
    midpoints = 0.5 * (centers[:-1] + centers[1:])
    return np.concatenate(
        ([centers[0] - (midpoints[0] - centers[0])], midpoints, [centers[-1] + (centers[-1] - midpoints[-1])])
    )


def plot_evolution(
    topology_files: list[Path], out: Path, label: str, shift: int = 0
) -> None:
    atlas = load_spatial_atlas(topology_files)
    area, components, capacity = evolution_fields(
        atlas["topology"], atlas["weights"], shift
    )
    surfaces = atlas["surfaces"]
    lambdas = atlas["lambda_values"]
    signs = atlas["signs"]
    fields = (
        (area, "non-ideal trapped fraction", "magma", 0.0, 1.0),
        (components, "periodic hole components", "viridis", 0.0, None),
        (capacity, "inner-to-radius path capacity", "cividis", 0.0, None),
    )
    figure, axes = plt.subplots(
        3,
        len(signs),
        figsize=(5.0 * len(signs), 9.2),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    xedges = _edges(surfaces)
    yedges = _edges(lambdas)
    tick_indices = np.unique(
        np.linspace(0, len(lambdas) - 1, min(5, len(lambdas)), dtype=int)
    )
    for row, (field, name, cmap, lower, upper) in enumerate(fields):
        images = []
        if upper is None:
            upper = max(float(np.max(field)), 1.0e-12)
        for sign_index, sign_value in enumerate(signs):
            axis = axes[row, sign_index]
            image = axis.pcolormesh(
                xedges,
                yedges,
                field[:, :, sign_index].T,
                cmap=cmap,
                vmin=lower,
                vmax=upper,
                shading="flat",
            )
            images.append(image)
            sign = "> 0" if sign_value > 0.0 else "< 0"
            axis.set_title(rf"$v_\parallel {sign}$")
            if len(surfaces) > 4:
                axis.set_xticks(
                    surfaces,
                    [f"{surface:g}" for surface in surfaces],
                    rotation=45,
                    ha="right",
                )
            else:
                axis.set_xticks(surfaces)
            axis.set_yticks(lambdas[tick_indices])
            if sign_index == 0:
                axis.set_ylabel(r"$\mu B_0/E$")
        figure.colorbar(
            images[-1], ax=axes[row].tolist(), label=name, pad=0.02, shrink=0.85
        )
    for axis in axes[-1]:
        axis.set_xlabel("barrier surface s")
    figure.suptitle(
        f"{label}: radial evolution of non-ideal topology (lattice shift {shift})"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=200, facecolor="white")
    plt.close(figure)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--topology", type=Path, nargs="+", required=True)
    root.add_argument("--out", type=Path, required=True)
    root.add_argument("--label", default="configuration")
    root.add_argument("--shift-index", type=int, default=0)
    return root


if __name__ == "__main__":
    args = parser().parse_args()
    plot_evolution(
        [path.resolve() for path in args.topology],
        args.out,
        args.label,
        args.shift_index,
    )
