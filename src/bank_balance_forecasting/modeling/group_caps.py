"""Compare common and group-specific caps on the same model forecasts."""

from itertools import product

import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits

from bank_balance_forecasting.forecast_pilot import EVALUATION_START, fit_groups

from .forecasting import _parallel
from .threshold import (
    EARLY_START,
    PRIMARY,
    REPLAY,
    SELECTION_LIMIT,
    STRESS_LIMIT,
    _errors,
    _fold_scores,
)

GROUPS = ("G01", "G02", "G03")
CAP_MULTIPLIERS = (0.5, 0.75, 1.0, 1.25, 1.5)


def _origin_groups(job):
    """Refit the three clustering roles using only data through the forecast origin."""
    origin, balances, settings = job
    with threadpool_limits(limits=1):
        groups, _ = fit_groups(
            balances.filter(pl.col("date") <= origin), settings, origin
        )
    print(f"Fitted groups through {origin} · {len(groups)} accounts", flush=True)
    return pl.DataFrame(
        {"account_id": list(groups), "group": list(groups.values()), "origin": origin}
    )


def assign_origin_groups(balances, cases, settings):
    """Produce training-only group labels for each origin used in the backtests."""
    origins = cases["origin"].unique().sort()
    jobs = ((origin, balances, settings) for origin in origins)
    return pl.concat(_parallel(_origin_groups, jobs, settings["stage3"]["workers"]))


def _capped(raw, group_index, caps):
    """Clip each forecast at its group's cap. The last cap covers unassigned accounts."""
    limits = np.asarray(caps)[group_index, None]
    return np.clip(raw, -limits, limits)


def _select_caps(actual, raw, groups, masks, choices):
    """Minimize selection WAPE under the same monthly MAPE and stress limits as S3."""
    selection = np.logical_or.reduce(list(masks.values()))
    eligible = []
    for caps in choices:
        paths = _capped(raw, groups, caps)
        folds = _fold_scores(masks, actual, paths)
        if all(
            row["MAPE (%)"] <= SELECTION_LIMIT
            and row["Stress MAPE bound (%)"] <= STRESS_LIMIT
            for row in folds
        ):
            metrics = _errors(actual[selection], paths[selection])
            eligible.append((metrics["WAPE (%)"], metrics["MAPE (%)"], caps))
    return min(eligible)[2]


def compare_group_caps(cases, labels, selected_paths, shared_cap):
    """Tune caps on early origins and evaluate identical forecasts with common and group caps."""
    matched = (
        cases.select("case_id", "account_id", "study", "origin", "actual")
        .join(selected_paths, on="case_id", validate="1:1")
        .join(labels, on=["account_id", "origin"], how="left", validate="m:1")
        .with_columns(pl.col("group").fill_null("Review"))
    )
    if matched.height != cases.height:
        raise ValueError(
            "The selected model must forecast every case before comparing caps"
        )

    grid = tuple(shared_cap * multiplier for multiplier in CAP_MULTIPLIERS)
    common_grid = tuple((cap,) * (len(GROUPS) + 1) for cap in grid)
    group_grid = tuple((*caps, shared_cap) for caps in product(grid, repeat=len(GROUPS)))
    group_positions = {group: i for i, group in enumerate((*GROUPS, "Review"))}
    score_rows, group_rows = [], []
    for study, start in ((PRIMARY, EVALUATION_START), (REPLAY, EARLY_START)):
        cohort = matched.filter(pl.col("study") == study)
        origins = np.asarray(cohort["origin"].to_list())
        groups = np.asarray(cohort["group"].to_list())
        group_index = np.array([group_positions[group] for group in groups])
        actual = np.asarray(cohort["actual"].to_list(), dtype=float)
        raw = np.asarray(cohort["prediction"].to_list(), dtype=float)
        before = {
            origin: origins == origin
            for origin in sorted(set(origins))
            if origin < start
        }
        after = {
            origin: origins == origin
            for origin in sorted(set(origins))
            if origin >= start
        }
        evaluation = origins >= start
        for policy, choices in (
            ("Tuned shared cap", common_grid),
            ("Tuned group caps", group_grid),
        ):
            caps = _select_caps(actual, raw, group_index, before, choices)
            paths = _capped(raw, group_index, caps)
            folds = _fold_scores(after, actual, paths)
            score_rows.append(
                {
                    "study": study,
                    "policy": policy,
                    **dict(zip(GROUPS, caps[:-1], strict=True)),
                    **_errors(actual[evaluation], paths[evaluation]),
                    "Worst window MAPE (%)": max(row["MAPE (%)"] for row in folds),
                    "Worst window stress (%)": max(
                        row["Stress MAPE bound (%)"] for row in folds
                    ),
                }
            )
            for group, cap in zip(GROUPS, caps[:-1], strict=True):
                mask = evaluation & (groups == group)
                if (np.isfinite(actual[mask]) & (actual[mask] != 0)).any():
                    group_rows.append(
                        {
                            "study": study,
                            "policy": policy,
                            "group": group,
                            "Cap (€)": cap,
                            **_errors(actual[mask], paths[mask]),
                        }
                    )
    return pl.DataFrame(score_rows), pl.DataFrame(group_rows)
