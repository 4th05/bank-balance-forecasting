"""Prepare, forecast and score 30-day customer balance paths."""

import os
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from functools import partial
from importlib.util import find_spec
from multiprocessing import get_context

import numpy as np
import pandas as pd
import polars as pl
from threadpoolctl import threadpool_limits

from bank_balance_forecasting.forecast_pilot import (
    FOUNDATION_MODELS,
    HORIZON,
    classical_forecasts,
    fit_groups,
    predict_chronos,
)

from .ensembles import QUANTILE_POOL, combine_candidates, weighted_quantile_pool

ZERO_CONTROL = "Zero control"


def _parallel(function, jobs, workers):
    """Run independent fold jobs in fresh worker processes with bounded resources."""
    if workers < 2:
        return list(map(function, jobs))
    os.environ.setdefault("POLARS_MAX_THREADS", "2")
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        return list(pool.map(function, jobs))


def _cases(balances, origin, groups, backtest):
    """Build matched account histories and target paths on the original daily grid."""
    panel = balances.pivot(on="date", index="account_id", values="balance").sort("account_id")
    dates = sorted(panel.columns[1:])
    cutoff = dates.index(str(origin)) + 1
    history = panel.select(dates[:cutoff]).to_numpy()
    targets = panel.select(dates[cutoff : cutoff + HORIZON]).to_numpy() if backtest else None
    if backtest and len(dates) < cutoff + HORIZON:
        raise ValueError(f"Missing target days after {origin}")
    rows = []
    for i, account in enumerate(panel["account_id"]):
        past = history[i]
        if not np.isfinite(past).any():
            if backtest:
                continue
            raise ValueError("An account has no observed balances for forecasting")
        actual = targets[i].tolist() if backtest else None
        if backtest and not np.isfinite(targets[i]).any():
            continue
        rows.append((account, origin, groups.get(account, "Review"), past.tolist(), actual))
    return pl.DataFrame(
        rows,
        schema={
            "account_id": pl.String,
            "origin": pl.Date,
            "group": pl.String,
            "history": pl.List(pl.Float64),
            "actual": pl.List(pl.Float64),
        },
        orient="row",
    )


def _mape_level(history, lookback):
    """Use an inverse-balance-weighted median to minimize historical MAPE on a flat path."""
    past = history[-lookback:]
    values = np.sort(past[np.isfinite(past) & (past != 0)])
    if not len(values):
        return 0.0
    # Normalize weights without changing the weighted median, including for very small balances.
    weights = np.min(abs(values)) / abs(values)
    return float(values[np.searchsorted(weights.cumsum(), weights.sum() / 2)])


def _cpu_predictions(cases, tolerance):
    """Forecast classical methods and MAPE-aware baselines for one batch of accounts."""
    rows = []
    with threadpool_limits(limits=1):
        for case in cases.iter_rows(named=True):
            history = np.asarray(case["history"], dtype=float)
            valid = history[np.isfinite(history)]
            dates = pd.date_range(end=case["origin"], periods=len(history), freq="D")
            if len(valid) >= 91:
                paths, _ = classical_forecasts(history, dates, tolerance)
            else:
                # Preserve a common comparison sample and fall back to persistence for short histories.
                paths, _ = classical_forecasts(
                    np.full(91, valid[-1]),
                    dates[-1] - pd.to_timedelta(np.arange(90, -1, -1), unit="D"),
                    tolerance,
                )
            paths.update(
                {
                    f"MAPE median {window}": np.full(HORIZON, _mape_level(history, window))
                    for window in (30, 90, 365)
                }
            )
            paths[ZERO_CONTROL] = np.zeros(HORIZON)
            rows.extend((case["case_id"], model, path.tolist()) for model, path in paths.items())
    return pl.DataFrame(
        rows,
        schema={"case_id": pl.UInt32, "model": pl.String, "prediction": pl.List(pl.Float64)},
        orient="row",
    )


def forecast_cpu(cases, stage3):
    """Produce classical candidates using independent account batches across CPU processes."""
    workers = stage3["workers"]
    batch_size = max(1, (cases.height + 2 * workers - 1) // (2 * workers))
    jobs = cases.iter_slices(n_rows=batch_size)
    return pl.concat(
        _parallel(partial(_cpu_predictions, tolerance=stage3["tolerance"]), jobs, workers)
    )


def forecast_foundations(cases, stage3, models=None):
    """Batch zero-shot forecasts locally and use MPS when PyTorch detects it."""
    models = stage3["foundation_models"] if models is None else models
    if not models:
        return pl.DataFrame(
            schema={"case_id": pl.UInt32, "model": pl.String, "prediction": pl.List(pl.Float64)}
        )
    if find_spec("chronos") is None:
        raise RuntimeError(
            "Install the optional Chronos package or set foundation_models to an empty list"
        )
    import torch

    device = stage3["device"]
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    device_used = device
    contexts = []
    for past in cases["history"].to_list():
        context = np.asarray(past[-512:], dtype=np.float32)
        contexts.append(np.pad(context, (512 - len(context), 0), constant_values=np.nan))
    contexts = np.stack(contexts)
    outputs, quantiles = [], []
    levels = [i / 10 for i in range(1, 10)] if len(models) > 1 else None
    for model in models:
        if model not in FOUNDATION_MODELS:
            raise ValueError(f"Unknown foundation model {model}")
        prediction = predict_chronos(contexts, model, device=device_used, quantile_levels=levels)
        if not np.isfinite(prediction).all() and device_used == "mps":
            device_used = "cpu"
            prediction = predict_chronos(
                contexts, model, device=device_used, quantile_levels=levels
            )
        if not np.isfinite(prediction).all():
            raise ValueError(f"{model} returned nonfinite forecasts")
        if levels is not None:
            quantiles.append(prediction)
            prediction = prediction[..., 4]
        print(f"{model} · {device_used} · {len(prediction)} forecasts", flush=True)
        outputs.append(
            pl.DataFrame(
                {
                    "case_id": cases["case_id"],
                    "model": [model] * cases.height,
                    "prediction": prediction.tolist(),
                },
                schema_overrides={"prediction": pl.List(pl.Float64)},
            )
        )
    if quantiles:
        outputs.append(
            pl.DataFrame(
                {
                    "case_id": cases["case_id"],
                    "model": [QUANTILE_POOL] * cases.height,
                    "prediction": weighted_quantile_pool(quantiles).tolist(),
                },
                schema_overrides={"prediction": pl.List(pl.Float64)},
            )
        )
    return pl.concat(outputs)


def forecast_submission(prepared_balances, policy, stage0, stage1, stage3):
    """Refit the chosen policy on the full history and return one row per account."""
    balances = (
        prepared_balances.collect()
        if isinstance(prepared_balances, pl.LazyFrame)
        else prepared_balances
    )
    cutoff = balances["date"].max()
    groups = (
        fit_groups(balances, {"stage0": stage0, "stage1": stage1}, cutoff)[0]
        if policy["policy"] == "Routed"
        else {}
    )
    cases = _cases(balances, cutoff, groups, backtest=False).with_row_index("case_id")
    case_models = [
        policy["group_models"].get(group, policy["common_model"])
        if policy["policy"] == "Routed"
        else policy["common_model"]
        for group in cases["group"]
    ]
    cases = cases.with_columns(pl.Series("model", case_models))
    candidates, _ = combine_candidates(
        cases,
        forecast_cpu(cases, stage3),
        forecast_foundations(cases, stage3),
        calibration=policy.get("ensemble_calibration", {}),
    )
    result = (
        cases.select("case_id", "account_id", "model")
        .join(candidates, on=["case_id", "model"], validate="1:1")
        .sort("account_id")
    )
    paths = np.asarray(result["prediction"].to_list(), dtype=float)
    if paths.shape != (cases.height, HORIZON) or not np.isfinite(paths).all():
        raise ValueError("Submission must provide 30 finite balances for every account")
    cap = policy.get("absolute_cap")
    if cap is not None:
        paths = np.clip(paths, -cap, cap)
    return pl.DataFrame(
        {
            "ACCOUNT_ID": result["account_id"],
            **{
                str(cutoff + timedelta(days=day)): paths[:, day - 1]
                for day in range(1, HORIZON + 1)
            },
        }
    )
