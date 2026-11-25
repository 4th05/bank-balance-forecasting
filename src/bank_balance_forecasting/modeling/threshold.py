"""Select a balance forecast subject to a 200% monthly MAPE requirement."""

from datetime import date

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

from bank_balance_forecasting.forecast_pilot import EVALUATION_START, ORIGINS

from .ensembles import combine_candidates
from .forecasting import ZERO_CONTROL, _cases

PRIMARY = "Original 2022 windows"
REPLAY = "Earlier chronological replay"
EARLY_START = date(2021, 11, 30)
EARLY_ORIGINS = (
    date(2021, 6, 30),
    date(2021, 7, 31),
    date(2021, 8, 31),
    date(2021, 9, 30),
    date(2021, 10, 31),
    EARLY_START,
    date(2021, 12, 31),
    date(2022, 1, 30),
    date(2022, 3, 1),
    date(2022, 3, 31),
)
TARGET_MAPE = 200.0
SELECTION_LIMIT = 150.0
STRESS_LIMIT = 160.0
STRESS_BALANCE = 1.0
STRESS_DAYS = 7


def prepare_threshold_cases(prepared_balances):
    """Build two chronological evaluations with no group refits for a common policy."""
    balances = (
        prepared_balances.collect()
        if isinstance(prepared_balances, pl.LazyFrame)
        else prepared_balances
    )
    cases = []
    for name, origins in ((PRIMARY, ORIGINS), (REPLAY, EARLY_ORIGINS)):
        cases.extend(
            _cases(balances, origin, {}, backtest=True).with_columns(
                pl.lit(name).alias("study")
            )
            for origin in origins
        )
    return pl.concat(cases).with_row_index("case_id")


def _errors(actual, predictions):
    """Return official pooled MAPE, euro errors and a conservative one-week stress bound."""
    observed = np.isfinite(actual)
    scored = observed & (actual != 0)
    days = int(scored.sum())
    if not days:
        raise ValueError(
            "MAPE is undefined when a window has no observed nonzero balances"
        )
    error = np.abs(actual - predictions)
    mape = float(100 * np.mean(error[scored] / np.abs(actual[scored])))
    mae = float(np.mean(error[observed]))
    wape = float(100 * np.sum(error[observed]) / np.sum(np.abs(actual[observed])))
    exposure = (
        sliding_window_view(np.abs(predictions), STRESS_DAYS, axis=1).sum(axis=-1).max()
    )
    # Replace one account-week by ±€1. Adding its entire error without subtracting
    # the original error gives a conservative bound for the replacement scenario.
    bound = float(mape + 100 * (STRESS_DAYS + exposure / STRESS_BALANCE) / days)
    return {
        "MAPE (%)": mape,
        "MAE (€)": mae,
        "WAPE (%)": wape,
        "Stress MAPE bound (%)": bound,
        "MAPE days": days,
        "Zero days": int(observed.sum()) - days,
    }


def _fold_scores(masks, actual, forecast):
    """Compute matched metrics separately for each 30-day origin."""
    return [
        {"origin": origin, **_errors(actual[mask], forecast[mask])}
        for origin, mask in masks.items()
    ]


def _study(cases, classical, foundation, start, cap):
    """Select a model on early windows and compare all models at the same cap."""
    candidates, calibration = combine_candidates(
        cases, classical, foundation, evaluation_start=start
    )
    ordered = cases.sort("case_id")
    origins = np.asarray(ordered["origin"].to_list())
    selection = origins < start
    actual = np.asarray(ordered["actual"].to_list(), dtype=float)
    before = {day: origins == day for day in sorted(set(origins[selection]))}
    after = {day: origins == day for day in sorted(set(origins[~selection]))}
    trials, summaries, fold_table = [], [], []
    evaluation = ~selection
    for (model,), frame in candidates.group_by("model"):
        raw = np.asarray(frame.sort("case_id")["prediction"].to_list(), dtype=float)
        forecast = np.clip(raw, -cap, cap)
        if model != ZERO_CONTROL:
            folds = _fold_scores(before, actual, forecast)
            worst = max(row["MAPE (%)"] for row in folds)
            stressed = max(row["Stress MAPE bound (%)"] for row in folds)
            pooled = _errors(actual[selection], forecast[selection])
            trials.append(
                {
                    "model": model,
                    "cap": cap,
                    "Worst selection MAPE (%)": worst,
                    "Worst selection stress (%)": stressed,
                    "Selection WAPE (%)": pooled["WAPE (%)"],
                    "Selection MAPE (%)": pooled["MAPE (%)"],
                    "eligible": worst <= SELECTION_LIMIT and stressed <= STRESS_LIMIT,
                }
            )
        folds = _fold_scores(after, actual, forecast)
        fold_table.extend({"model": model, **row} for row in folds)
        summaries.append(
            {
                "model": model,
                "Cap (€)": cap,
                **_errors(actual[evaluation], forecast[evaluation]),
                "Worst window MAPE (%)": max(row["MAPE (%)"] for row in folds),
                "Capped forecasts (%)": 100
                * float(np.mean(abs(raw[evaluation]) > cap)),
            }
        )
    table = pl.DataFrame(trials).sort(
        "Selection WAPE (%)", "Worst selection MAPE (%)", "model", "cap"
    )
    eligible = table.filter("eligible")
    if eligible.is_empty():
        raise ValueError(
            "No history-based forecast passed the MAPE and stress budgets on selection"
        )
    selected = eligible.row(0, named=True)
    model = selected["model"]
    policy = {
        "policy": "Common",
        "common_model": model,
        "group_models": {},
        "ensemble_calibration": calibration,
        "absolute_cap": cap,
    }
    selected_model = (pl.col("model") == model).alias("Selected")
    return (
        policy,
        pl.DataFrame([selected]),
        pl.DataFrame(summaries).with_columns(selected_model).sort("MAPE (%)", "model"),
        pl.DataFrame(fold_table).with_columns(selected_model).sort("origin", "model"),
        candidates.with_columns(selected_model),
    )


def select_threshold_policy(cases, classical, foundation, cap):
    """Return the locked policy, matched score tables and uncapped candidate paths."""
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError("The shared forecast cap must be finite and positive")
    main, selections, results, dates, paths = None, [], [], [], []
    for name, start in ((PRIMARY, EVALUATION_START), (REPLAY, EARLY_START)):
        cohort = cases.filter(pl.col("study") == name)
        policy, selected, scores, folds, candidate_paths = _study(
            cohort,
            classical.join(cohort.select("case_id"), on="case_id", how="semi"),
            foundation.join(cohort.select("case_id"), on="case_id", how="semi"),
            start,
            cap,
        )
        if name == PRIMARY:
            main = policy
        selections.append(selected.with_columns(pl.lit(name).alias("study")))
        results.append(scores.with_columns(pl.lit(name).alias("study")))
        dates.append(folds.with_columns(pl.lit(name).alias("study")))
        paths.append(candidate_paths)
    return (
        main,
        pl.concat(selections),
        pl.concat(results),
        pl.concat(dates),
        pl.concat(paths),
    )
