#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from reactor_proxy_calibration import write_json
from simple_barrier import loss_windows


FEATURES = ("barrier_overlap_topology", "barrier_overlap_jpar")


def load_results(root: Path) -> dict[str, dict]:
    results = {}
    for path in sorted(root.glob("*/result.json")):
        result = json.loads(path.read_text())
        results[result["case"]] = result
    return results


def load_direct(root: Path, case: str) -> tuple[np.ndarray, float]:
    data = np.loadtxt(root / case / "direct" / "times_lost.dat", ndmin=2)
    if data.shape[1] < 2:
        raise ValueError(f"invalid times_lost.dat for {case}")
    curve = np.loadtxt(
        root / case / "direct" / "confined_fraction.dat", ndmin=2
    )
    if curve.shape[0] == 0 or curve.shape[1] < 2:
        raise ValueError(f"invalid confined_fraction.dat for {case}")
    return data[:, 1], float(curve[-1, 0])


def paired_se(
    first: np.ndarray,
    second: np.ndarray,
    window: str,
    first_final: float,
    second_final: float,
) -> float:
    if first.shape != second.shape:
        raise ValueError("common-random-number arrays have different shapes")
    if window == "prompt":
        a = (first > 0.0) & (first <= 1.0e-3)
        b = (second > 0.0) & (second <= 1.0e-3)
    elif window == "late":
        a = (first > 1.0e-3) & (first < first_final)
        b = (second > 1.0e-3) & (second < second_final)
    else:
        raise ValueError(window)
    difference = a.astype(float) - b.astype(float)
    return float(difference.std(ddof=1) / np.sqrt(difference.size))


def feature_row(result: dict) -> np.ndarray:
    return np.array([result["barrier"][name] for name in FEATURES], dtype=float)


def fit_ridge(x: np.ndarray, y: np.ndarray, penalty: float) -> dict:
    mean = x.mean(axis=0)
    scale = x.std(axis=0, ddof=1)
    if np.any(scale == 0.0):
        raise ValueError("constant proxy feature in calibration")
    z = (x - mean) / scale
    design = np.column_stack((np.ones(len(z)), z))
    regularizer = np.diag([0.0] + [penalty] * x.shape[1])
    coefficients = np.linalg.solve(design.T @ design + regularizer, design.T @ y)
    prediction = design @ coefficients
    return {
        "mean": mean,
        "scale": scale,
        "intercept": float(coefficients[0]),
        "weights": coefficients[1:],
        "prediction": prediction,
    }


def predict(model: dict, x: np.ndarray) -> np.ndarray:
    z = (x - model["mean"]) / model["scale"]
    return model["intercept"] + z @ model["weights"]


def direction_pairs(names: list[str]) -> list[tuple[str, str]]:
    directions = sorted(
        {name.removesuffix("_minus") for name in names if name.endswith("_minus")}
    )
    pairs = [(f"{name}_minus", f"{name}_plus") for name in directions]
    if any(minus not in names or plus not in names for minus, plus in pairs):
        raise ValueError("incomplete perturbation direction")
    return pairs


def cross_validate(
    names: list[str], x: np.ndarray, y: np.ndarray, penalty: float
) -> tuple[int, int]:
    index = {name: position for position, name in enumerate(names)}
    agreements = 0
    pairs = direction_pairs(names)
    for minus, plus in pairs:
        hold = {index[minus], index[plus]}
        train = np.array([i not in hold for i in range(len(names))])
        model = fit_ridge(x[train], y[train], penalty)
        predicted = predict(model, x[[index[minus], index[plus]]])
        observed = y[[index[minus], index[plus]]]
        agreements += int(np.sign(predicted[1] - predicted[0]) == np.sign(observed[1] - observed[0]))
    return agreements, len(pairs)


def proposal(
    names: list[str], results: dict[str, dict], predicted: np.ndarray
) -> np.ndarray:
    index = {name: position for position, name in enumerate(names)}
    gradient = None
    minor_radius = None
    for minus, plus in direction_pairs(names):
        vector = np.asarray(
            results[plus]["candidate_metadata"]["perturbation"], dtype=float
        )
        norm = np.linalg.norm(vector)
        derivative = (predicted[index[plus]] - predicted[index[minus]]) / (2.0 * norm)
        contribution = derivative * vector / norm
        gradient = contribution if gradient is None else gradient + contribution
        amplitude = results[plus]["candidate_metadata"]["amplitude_over_a"]
        minor_radius = norm / amplitude
    if gradient is None or minor_radius is None or np.linalg.norm(gradient) == 0.0:
        raise ValueError("calibration produced no proxy gradient")
    return -0.005 * minor_radius * gradient / np.linalg.norm(gradient)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--penalty", type=float, default=0.25)
    args = parser.parse_args()
    results = load_results(args.results)
    expected = {
        "base",
        *(f"d{index:02d}_{sign}" for index in range(4) for sign in ("minus", "plus")),
    }
    if set(results) != expected:
        raise ValueError(f"calibration cases differ: {sorted(expected - set(results))}")
    accepted = {
        name: result for name, result in results.items() if result["status"] == "accepted"
    }
    if set(accepted) != expected:
        raise ValueError("geometry rejection left calibration incomplete")

    names = ["base"] + sorted(name for name in accepted if name != "base")
    x = np.vstack([feature_row(accepted[name]) for name in names])
    direct = {name: load_direct(args.results, name) for name in names}
    windows = {
        name: loss_windows(times, final_time=endpoint)
        for name, (times, endpoint) in direct.items()
    }
    y = np.array([windows[name]["late_loss"] for name in names])
    model = fit_ridge(x, y, args.penalty)
    predicted = model["prediction"]
    rho = float(spearmanr(predicted, y).statistic)
    agreements, directions = cross_validate(names, x, y, args.penalty)

    base = accepted["base"]
    base_times, base_endpoint = direct["base"]
    base_prompt = windows["base"]["prompt_loss"]
    base_late = windows["base"]["late_loss"]
    base_gamma = base["gamma_c"]["s03"]
    base_ripple = base["effective_ripple"]["s03"]
    gamma_limit = max(1.5 * base_gamma, base_gamma + 0.002)
    ripple_limit = max(1.25 * base_ripple, base_ripple + 0.002)
    rows = []
    for name in names:
        result = accepted[name]
        times, endpoint = direct[name]
        case_windows = windows[name]
        prompt_se = paired_se(
            times, base_times, "prompt", endpoint, base_endpoint
        )
        late_se = paired_se(times, base_times, "late", endpoint, base_endpoint)
        prompt_change = case_windows["prompt_loss"] - base_prompt
        rows.append(
            {
                "case": name,
                "prompt_loss": case_windows["prompt_loss"],
                "late_loss": case_windows["late_loss"],
                "prompt_change": prompt_change,
                "prompt_paired_se": prompt_se,
                "late_change": case_windows["late_loss"] - base_late,
                "late_paired_se": late_se,
                "barrier_overlap_topology": result["barrier"]["barrier_overlap_topology"],
                "barrier_overlap_jpar": result["barrier"]["barrier_overlap_jpar"],
                "gamma_c_s03": result["gamma_c"]["s03"],
                "effective_ripple_s03": result["effective_ripple"]["s03"],
                "prompt_protected": prompt_change <= 2.0 * prompt_se,
                "gamma_c_protected": result["gamma_c"]["s03"] <= gamma_limit,
                "thermal_protected": result["effective_ripple"]["s03"] <= ripple_limit,
                "proxy_prediction": float(predicted[names.index(name)]),
            }
        )

    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "calibration.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    proxy_accepted = rho >= 0.5 and agreements >= 3
    decision = {
        "proxy_accepted": proxy_accepted,
        "spearman_predicted_vs_late": rho,
        "directional_cross_validation_agreements": agreements,
        "direction_count": directions,
        "feature_order": FEATURES,
        "feature_weights_standardized": model["weights"].tolist(),
        "feature_mean": model["mean"].tolist(),
        "feature_scale": model["scale"].tolist(),
        "intercept": model["intercept"],
        "gamma_c_limit_s03": gamma_limit,
        "effective_ripple_limit_s03": ripple_limit,
    }
    if proxy_accepted:
        vector = proposal(names, accepted, predicted)
        np.save(output / "proposal_perturbation.npy", vector)
        decision["proposal_norm"] = float(np.linalg.norm(vector))
    write_json(output / "decision.json", decision)


if __name__ == "__main__":
    main()
