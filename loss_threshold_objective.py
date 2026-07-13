from __future__ import annotations

import numpy as np


def threshold_crossing(curve: np.ndarray, threshold: float) -> float | None:
    values = np.asarray(curve, dtype=float)
    if values.ndim != 2 or values.shape[1] < 3 or len(values) == 0:
        raise ValueError("confinement curve must have time and two confined fractions")
    if not 0.0 < threshold < 1.0:
        raise ValueError("loss threshold must lie in (0,1)")
    time = values[:, 0]
    loss = 1.0 - values[:, 1] - values[:, 2]
    if (
        not np.isfinite(time).all()
        or not np.isfinite(loss).all()
        or time[0] < 0.0
        or np.any(np.diff(time) <= 0.0)
        or np.any((loss < -1.0e-12) | (loss > 1.0 + 1.0e-12))
        or np.any(np.diff(loss) < -1.0e-12)
    ):
        raise ValueError("confinement curve is not finite and monotone")
    reached = np.flatnonzero(loss >= threshold)
    if not len(reached):
        return None
    upper = int(reached[0])
    lower_time, lower_loss = (0.0, 0.0) if upper == 0 else (time[upper - 1], loss[upper - 1])
    upper_time, upper_loss = time[upper], loss[upper]
    if upper_loss == lower_loss:
        return float(upper_time)
    fraction = (threshold - lower_loss) / (upper_loss - lower_loss)
    return float(lower_time + fraction * (upper_time - lower_time))


def threshold_objective(
    loss: float,
    trace_time: float,
    threshold: float,
    *,
    crossing_time: float | None,
    epsilon: float = 1.0e-6,
) -> float:
    values = np.asarray([loss, trace_time, threshold, epsilon], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("threshold-objective inputs must be finite")
    if not 0.0 <= loss <= 1.0 or trace_time <= 0.0:
        raise ValueError("loss and trace time are outside their ranges")
    if not 0.0 < threshold < 1.0 or epsilon <= 0.0:
        raise ValueError("threshold and epsilon are outside their ranges")
    if crossing_time is not None:
        if not 0.0 < crossing_time <= trace_time or loss < threshold:
            raise ValueError("threshold crossing is inconsistent with the trace")
        return float(-np.log10(crossing_time))
    if loss > threshold:
        raise ValueError("a trace above threshold requires a crossing time")
    return float(
        np.log10(loss + epsilon)
        - np.log10(trace_time)
        - np.log10(threshold + epsilon)
    )
