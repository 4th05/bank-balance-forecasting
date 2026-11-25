"""Combine model paths and fit percentage-error calibration on early origins."""

import numpy as np
import polars as pl

from bank_balance_forecasting.forecast_pilot import EVALUATION_START

BALANCE_BANDS = (100, 1000, 10000)
QUANTILE_POOL = "Chronos quantile pool (MAPE)"


def weighted_quantile_pool(quantiles):
    """Approximate the MAPE-optimal point forecast from equally weighted predictive quantiles."""
    values = np.sort(np.concatenate(quantiles, axis=-1), axis=-1)
    weights = 1 / np.maximum(abs(values), 0.01)
    half = weights.sum(axis=-1, keepdims=True) / 2
    index = (weights.cumsum(axis=-1) < half).sum(axis=-1, keepdims=True)
    return np.take_along_axis(values, index, axis=-1)[..., 0]


def _scales(cases):
    """Assign each account-origin to a euro scale band using its last 90 observed calendar days."""
    levels = [
        np.nanmedian(abs(np.asarray(history, dtype=float)[-90:])) for history in cases["history"]
    ]
    return np.digitize(levels, BALANCE_BANDS)


def _mape_scale(actual, forecast):
    """Find the nonnegative multiplier up to one that minimizes pooled MAPE."""
    valid = np.isfinite(actual) & (actual != 0) & np.isfinite(forecast) & (forecast != 0)
    if not valid.any():
        return 1.0
    ratios = actual[valid] / forecast[valid]
    weights = abs(forecast[valid] / actual[valid])
    order = np.argsort(ratios, kind="stable")
    cumulative = weights[order].cumsum()
    best = ratios[order[np.searchsorted(cumulative, cumulative[-1] / 2)]]
    return float(np.clip(best, 0, 1))


def combine_candidates(
    cases, classical, foundation, calibration=None, evaluation_start=EVALUATION_START
):
    """Build fixed averages and apply calibration learned only from selection origins."""
    all_paths = pl.concat([classical, foundation])
    ordered = cases.sort("case_id")
    ids = ordered["case_id"]
    paths = {}
    for (model,), table in all_paths.group_by("model"):
        table = table.sort("case_id")
        if not table["case_id"].equals(ids):
            raise ValueError(f"{model} must forecast the same case IDs before combining")
        paths[model] = np.asarray(table["prediction"].to_list(), dtype=float)

    recipes = {
        "Mean classical": ("Last balance", "SMA 14", "Local level"),
        "Mean MAPE medians": ("MAPE median 30", "MAPE median 90", "MAPE median 365"),
        "Mean two Chronos": ("Chronos-Bolt Tiny", "Chronos-2 Small"),
        "Mean Chronos and persistence": ("Chronos-Bolt Tiny", "Chronos-2 Small", "Last balance"),
        "Mean mixed": ("Last balance", "Chronos-2 Small", "MAPE median 365"),
        "Median mixed": ("Last balance", "Chronos-2 Small", "MAPE median 365"),
    }
    combined = {}
    for name, members in recipes.items():
        if all(member in paths for member in members):
            reducer = np.median if name == "Median mixed" else np.mean
            combined[name] = reducer([paths[member] for member in members], axis=0)

    bands = _scales(ordered)
    fitting = calibration is None
    calibration = {} if fitting else calibration
    if fitting:
        if ordered["actual"].null_count():
            raise ValueError("Supply fitted calibration when generating future forecasts")
        selection = (ordered["origin"] < evaluation_start).to_numpy()
        actual = np.asarray(ordered["actual"].to_list(), dtype=float)
    candidates = paths | combined
    for base in ("MAPE median 365", "Mean MAPE medians", "Median mixed", QUANTILE_POOL):
        if base not in candidates:
            continue
        point = candidates[base]
        if fitting:
            parameters = []
            for band in range(len(BALANCE_BANDS) + 1):
                eligible = selection & (bands == band)
                parameters.append(_mape_scale(actual[eligible], point[eligible]))
            calibration[base] = parameters
        if base in calibration:
            multipliers = np.asarray(calibration[base])[bands, None]
            combined[f"Calibrated {base}"] = point * multipliers

    extra = [
        pl.DataFrame(
            {"case_id": ids, "model": [name] * len(ids), "prediction": prediction.tolist()},
            schema_overrides={"prediction": pl.List(pl.Float64)},
        )
        for name, prediction in combined.items()
    ]
    return pl.concat([all_paths, *extra]), calibration
