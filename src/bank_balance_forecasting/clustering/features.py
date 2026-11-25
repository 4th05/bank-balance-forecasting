"""Domain features and curated tsfresh calculators, using observed history only."""

import calendar
import warnings

import numpy as np
import polars as pl
from scipy.stats import theilslopes
from tsfresh.feature_extraction import feature_calculators as fc


def account_series(balances, parameters):
    """Yield each account ID, daily dates and balance array in chronological order.

    Reject observations beyond the cutoff, duplicate account/date pairs and
    broken calendar grids. Preserve missing-day entries in the yielded arrays.
    """
    frame = balances.collect().sort("account_id", "date")
    if frame["date"].max().isoformat() > str(parameters["as_of_date"]):
        raise ValueError("Stage 0 inputs extend beyond the Stage 1 cutoff")
    if frame.select(pl.struct("account_id", "date").n_unique()).item() != frame.height:
        raise ValueError("Duplicate account/date observations")
    for (account,), group in frame.group_by("account_id", maintain_order=True):
        dates = group["date"].to_numpy().astype("datetime64[D]")
        if len(dates) > 1 and not np.all(np.diff(dates).astype(int) == 1):
            raise ValueError("Keep the original daily calendar, including missing days")
        yield account, dates, group["balance"].to_numpy()


def changes(x):
    """Return adjacent balance differences, prepending NaN to preserve array length."""
    return np.diff(x, prepend=np.nan)


def mad(x):
    """Return median absolute deviation over finite values, or NaN when none exist."""
    x = x[np.isfinite(x)]
    return float(np.median(np.abs(x - np.median(x)))) if len(x) else np.nan


def iqr(x):
    """Return the 75th-minus-25th percentile of finite values, or NaN if unavailable."""
    x = x[np.isfinite(x)]
    return float(np.subtract(*np.percentile(x, [75, 25]))) if len(x) else np.nan


def windows(x, width, stride):
    """Cover all observed history with fixed-length, complete calendar windows."""
    indices = np.flatnonzero(np.isfinite(x))
    if len(indices) == 0:
        return
    start, end = indices[0], indices[-1] + 1
    if end - start < width:
        return
    starts = sorted(set(range(start, end - width + 1, stride)) | {end - width})
    for i in starts:
        if np.isfinite(x[i : i + width]).all():
            yield x[i : i + width]


def extract_activity(balances, parameters):
    """Describe maturity of observed history and gaps between account movements."""
    rows = []
    for account, _, x in account_series(balances, parameters):
        d = changes(x)
        active = np.flatnonzero(np.abs(d) > parameters["tolerance"])
        gaps = np.diff(active)
        runs, run = [], 0
        for i, value in enumerate(x):
            if not np.isfinite(value):
                if run:
                    runs.append(run)
                run = 0
            elif i and np.isfinite(d[i]) and abs(d[i]) <= parameters["tolerance"]:
                run += 1
            else:
                if run:
                    runs.append(run)
                run = 1
        if run:
            runs.append(run)
        n = int(np.isfinite(x).sum())
        rows.append(
            {
                "account_id": account,
                "meta_history_days": n,
                "activity__history_log": float(np.log1p(n)),
                "activity__movement_rate": len(active) / max(1, np.isfinite(d).sum()),
                "activity__longest_flat_share": max(runs, default=0) / max(1, n),
                "activity__median_flat_run": float(np.median(runs)) if runs else np.nan,
                "activity__movement_gap_cv": float(np.std(gaps) / np.mean(gaps))
                if len(gaps)
                else np.nan,
            }
        )
    return pl.DataFrame(rows).sort("account_id")


def extract_balance_state(balances, parameters):
    """Keep account scale and financial state without unstable mean-based ratios."""
    rows = []
    for account, _, x in account_series(balances, parameters):
        observed, d = x[np.isfinite(x)], changes(x)
        moved = d[np.isfinite(d) & (np.abs(d) > parameters["tolerance"])]
        median = float(np.median(observed)) if len(observed) else np.nan
        scale = iqr(observed)
        crossings = (x[1:] * x[:-1] < 0).sum()
        rows.append(
            {
                "account_id": account,
                "state__signed_log_level": float(np.sign(median) * np.log1p(abs(median))),
                "state__log_balance_iqr": float(np.log1p(scale)),
                "state__negative_share": float(np.mean(observed < 0)) if len(observed) else np.nan,
                "state__zero_crossing_rate": float(crossings / max(1, np.isfinite(d).sum())),
                "state__log_movement_mad": float(np.log1p(mad(moved))),
                "state__relative_variability": mad(moved) / max(scale, 0.01),
            }
        )
    return pl.DataFrame(rows).sort("account_id")


def inflow_regularity(dates, d):
    """Measure consistency in the calendar day of the largest monthly inflow.

    Use complete months with fully observed changes and a positive inflow.
    Return the largest fraction falling within a circular two-day day-of-month
    window, or NaN when fewer than six usable months are available.
    """
    peaks = []
    months = dates.astype("datetime64[M]")
    for month in np.unique(months):
        mask = months == month
        day = month.astype(object)
        if mask.sum() != calendar.monthrange(day.year, day.month)[1]:
            continue
        values = d[mask]
        if not np.isfinite(values).all() or np.max(values) <= 0:
            continue
        peaks.append(dates[mask][np.argmax(values)].astype(object).day)
    if len(peaks) < 6:
        return np.nan
    return max(
        np.mean(np.minimum(abs(np.array(peaks) - day), 31 - abs(np.array(peaks) - day)) <= 2)
        for day in range(1, 32)
    )


def extract_dynamics(balances, calendar_profiles, parameters):
    """Curated tsfresh dependence, entropy and complexity plus calendar regularity."""
    weekdays = (
        calendar_profiles.collect()
        .group_by("account_id", "weekday")
        .agg((pl.col("movement_days").sum() / pl.col("eligible_pairs").sum()).alias("rate"))
    )
    rates = {(r["account_id"], r["weekday"]): r["rate"] for r in weekdays.to_dicts()}
    rows = []
    for account, dates, x in account_series(balances, parameters):
        row = {"account_id": account}
        finite = np.flatnonzero(np.isfinite(x))
        d = np.diff(x[finite[0] : finite[-1] + 1]) if len(finite) else np.array([])
        # Do not join separate observations across internal gaps.
        usable = len(d) > max(parameters["acf_lags"]) and np.isfinite(d).all() and np.std(d) > 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for lag in parameters["acf_lags"]:
                row[f"dynamics__acf_change_{lag}"] = (
                    float(fc.autocorrelation(d, lag)) if usable else np.nan
                )
            for lag in [1, 7]:
                row[f"dynamics__acf_absolute_change_{lag}"] = (
                    float(fc.autocorrelation(abs(d), lag)) if usable else np.nan
                )
            entropy, complexity = [], []
            for window in windows(x, parameters["graph_window"], parameters["graph_stride"]):
                delta = np.diff(window)
                if np.std(delta) > 0:
                    entropy.append(
                        float(fc.fourier_entropy(delta, bins=parameters["entropy_bins"]))
                    )
                    complexity.append(
                        float(fc.cid_ce(delta, normalize=True) / np.sqrt(len(delta) - 1))
                    )
            row["dynamics__spectral_entropy"] = float(np.median(entropy)) if entropy else np.nan
            row["dynamics__normalized_complexity"] = (
                float(np.median(complexity)) if complexity else np.nan
            )
        weekday_rates = np.array([rates.get((account, day), np.nan) for day in range(1, 8)])
        average = float(np.nanmean(weekday_rates)) if np.isfinite(weekday_rates).any() else np.nan
        for weekday, rate in enumerate(weekday_rates, 1):
            row[f"dynamics__weekday_{weekday}_activity"] = (
                float(rate / average) if average > 0 else 0.0 if average == 0 else np.nan
            )
        row["dynamics__inflow_regularity"] = inflow_regularity(dates, changes(x))
        rows.append(row)
    return pl.DataFrame(rows).sort("account_id")


def extract_regime_features(balances, event_flags, parameters):
    """Describe long-term trend, recent behaviour and Stage 0 event frequencies."""
    events = event_flags.collect().partition_by("account_id", as_dict=True)
    rows = []
    for account, dates, x in account_series(balances, parameters):
        observed, d = x[np.isfinite(x)], changes(x)
        scale = max(iqr(observed), 0.01)
        recent, previous = x[-90:], x[-180:-90]
        months = dates.astype("datetime64[M]")
        monthly = [
            (int(month.astype(int)), float(np.nanmedian(x[months == month])))
            for month in np.unique(months)
            if np.isfinite(x[months == month]).sum() >= 20
        ]
        slope = (
            float(theilslopes([v for _, v in monthly], [t for t, _ in monthly]).slope * 12 / scale)
            if len(monthly) >= 3
            else np.nan
        )
        full_moves = np.abs(d[np.isfinite(d)]) > parameters["tolerance"]
        recent_d = d[-90:][np.isfinite(d[-90:])]
        full_d = d[np.isfinite(d)]
        cap = float(np.quantile(abs(full_d), 0.95)) if len(full_d) else np.nan

        def change_scale(values):
            """Average absolute daily changes capped at the full-history 95th percentile.

            The input contains finite changes; return NaN for an empty period.
            """
            return float(np.mean(np.minimum(abs(values), cap))) if len(values) else np.nan

        def median(values):
            """Return the median ignoring missing entries, or NaN if no finite value exists."""
            return float(np.nanmedian(values)) if np.isfinite(values).any() else np.nan

        row = {
            "account_id": account,
            "regime__normalized_trend": slope,
            "regime__recent_level_change": (median(recent) - median(previous)) / scale,
            "regime__recent_activity_change": float(
                np.mean(abs(recent_d) > parameters["tolerance"]) - np.mean(full_moves)
            )
            if len(recent_d) and len(full_moves)
            else np.nan,
            "regime__log_change_scale_ratio": float(
                np.log((change_scale(recent_d) + 0.01) / (change_scale(full_d) + 0.01))
            ),
        }
        account_events = events.get((account,), pl.DataFrame())
        for kind in ["large_change_unresolved", "temporary_excursion", "level_shift"]:
            selected = (
                account_events.filter(pl.col("candidate_type") == kind)
                if account_events.height
                else account_events
            )
            row[f"regime__{kind}_annual_rate"] = selected.height * 365.25 / max(1, len(observed))
            if kind == "level_shift":
                last = np.datetime64(selected["date"].max()) if selected.height else None
                row["regime__shift_recency_share"] = (
                    float((dates[-1] - last).astype(int) / max(1, len(x)))
                    if last is not None
                    else 1.0
                )
        rows.append(row)
    return pl.DataFrame(rows).sort("account_id")


def feature_label(name):
    """Readable labels for business reports; original feature keys remain in the registry."""
    labels = {
        "activity__history_log": "Observed history length",
        "activity__movement_rate": "Balance-change frequency",
        "activity__longest_flat_share": "Longest quiet period relative to history",
        "activity__median_flat_run": "Typical quiet-period length",
        "activity__movement_gap_cv": "Irregularity of time between movements",
        "state__signed_log_level": "Typical balance level",
        "state__log_balance_iqr": "Balance variability",
        "state__negative_share": "Time with a negative balance",
        "state__zero_crossing_rate": "Frequency of crossing zero",
        "state__log_movement_mad": "Typical variability of movements",
        "state__relative_variability": "Movement variability relative to balances",
        "dynamics__inflow_regularity": "Regularity of monthly largest inflows",
        "dynamics__spectral_entropy": "Irregularity of movement frequencies",
        "dynamics__normalized_complexity": "Complexity of daily movements",
        "regime__normalized_trend": "Long-term balance trend",
        "regime__recent_level_change": "Recent change in balance level",
        "regime__recent_activity_change": "Recent change in account activity",
        "regime__log_change_scale_ratio": "Recent change in robust daily movement scale",
        "regime__shift_recency_share": "Time since latest level-shift candidate",
    }
    if name in labels:
        return labels[name]
    if name.startswith("dynamics__weekday_"):
        weekday = int(name.split("_")[-2])
        return f"Relative {calendar.day_name[weekday - 1]} activity"
    return name.replace("__", ": ").replace("_", " ")
