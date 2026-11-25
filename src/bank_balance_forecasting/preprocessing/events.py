"""Screen financial events without editing observations or inferring fraud."""

import warnings

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

from .profiles import with_changes

EVENT_SCHEMA = {
    "account_id": pl.String,
    "date": pl.Date,
    "candidate_type": pl.String,
    "score": pl.Float64,
    "return_after_days": pl.Int64,
}


def flag_extreme_changes(balances, parameters):
    """Find movements unusually large for this account; retain every balance."""
    records, diagnostics = [], []
    for (account,), group in with_changes(balances).group_by("account_id", maintain_order=True):
        delta = group["change"].to_numpy()
        changes = delta[np.isfinite(delta) & (np.abs(delta) > parameters["tolerance"])]
        enough = len(changes) >= parameters["min_changes"]
        center = float(np.median(changes)) if enough else 0.0
        scale = float(1.4826 * np.median(np.abs(changes - center))) if enough else 0.0
        if enough and scale <= parameters["tolerance"]:
            scale = float(np.subtract(*np.percentile(changes, [75, 25])) / 1.349)
        status = "ok" if scale > parameters["tolerance"] else "insufficient_scale"
        diagnostics.append(
            {
                "account_id": account,
                "change_screen_status": status,
                "nonzero_changes": len(changes),
                "change_scale": scale,
            }
        )
        if status == "ok":
            scores = np.abs(delta - center) / scale
            dates, x = group["date"].to_list(), group["balance"].to_numpy()
            for i in np.flatnonzero(scores > parameters["change_threshold"]):
                records.append(
                    {
                        "account_id": account,
                        "date": dates[i],
                        "change": float(delta[i]),
                        "previous_balance": float(x[i - 1]),
                        "balance": float(x[i]),
                        "score": float(scores[i]),
                    }
                )
    schema = {
        "account_id": pl.String,
        "date": pl.Date,
        "change": pl.Float64,
        "previous_balance": pl.Float64,
        "balance": pl.Float64,
        "score": pl.Float64,
    }
    return pl.DataFrame(records, schema=schema).sort("account_id", "date"), pl.DataFrame(
        diagnostics
    )


def classify_reversals(balances, large_changes, parameters):
    """Separate temporary excursions from movements that have not returned by the cutoff."""
    records = []
    candidates = large_changes.collect().partition_by("account_id", as_dict=True)
    for key, group in (
        balances.collect().sort("account_id", "date").group_by("account_id", maintain_order=True)
    ):
        x = group["balance"].to_numpy()
        dates = {day: i for i, day in enumerate(group["date"].to_list())}
        for row in candidates.get(key, pl.DataFrame()).iter_rows(named=True):
            i = dates[row["date"]]
            horizon = min(len(x), i + parameters["return_days"] + 1)
            returns = np.flatnonzero(
                np.isfinite(x[i + 1 : horizon])
                & (
                    np.abs(x[i + 1 : horizon] - row["previous_balance"])
                    <= parameters["return_tolerance"] * abs(row["change"])
                )
            )
            days = int(returns[0] + 1) if len(returns) else None
            records.append(
                {
                    "account_id": key[0],
                    "date": row["date"],
                    "candidate_type": "temporary_excursion" if days else "large_change_unresolved",
                    "score": row["score"],
                    "return_after_days": days,
                }
            )
    return pl.DataFrame(records, schema=EVENT_SCHEMA).sort("account_id", "date")


def flag_level_shifts(balances, parameters):
    """Find sustained balance-level changes independently of isolated extreme movements."""
    events = []
    for (account,), group in (
        balances.collect().sort("account_id", "date").group_by("account_id", maintain_order=True)
    ):
        x, dates = group["balance"].to_numpy(), group["date"].to_list()
        # Adjacent windows: previous 28 days and current/following 28 days.
        width = parameters["regime_window"]
        windows = sliding_window_view(np.pad(x, (width, width - 1), constant_values=np.nan), width)
        before, after = windows[: len(x)], windows[width : width + len(x)]
        enough = (np.isfinite(before).sum(axis=1) >= parameters["regime_min_observations"]) & (
            np.isfinite(after).sum(axis=1) >= parameters["regime_min_observations"]
        )
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", RuntimeWarning
            )  # Empty prehistory windows are expected.
            med_before, med_after = np.nanmedian(before, axis=1), np.nanmedian(after, axis=1)
            iqr_before = np.subtract(*np.nanpercentile(before, [75, 25], axis=1))
            iqr_after = np.subtract(*np.nanpercentile(after, [75, 25], axis=1))
        scale = np.maximum(np.maximum(iqr_before, iqr_after), parameters["regime_scale_floor"])
        scores = np.abs(med_after - med_before) / scale
        candidates = enough & (scores > parameters["regime_threshold"])
        # Emit one candidate per contiguous run, at its strongest boundary.
        indices = np.flatnonzero(candidates)
        for run in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1):
            if len(run):
                i = run[np.argmax(scores[run])]
                events.append(
                    {
                        "account_id": account,
                        "date": dates[i],
                        "candidate_type": "level_shift",
                        "score": float(scores[i]),
                        "return_after_days": None,
                    }
                )
    return pl.DataFrame(events, schema=EVENT_SCHEMA).sort("account_id", "date")
