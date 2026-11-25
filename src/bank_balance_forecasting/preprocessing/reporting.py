"""Publish explicit contracts and compact, reproducible audit summaries."""

import json

import polars as pl

from .profiles import summarize_accounts


def quality_report(
    balances, profiles, events, regimes, corrections, shared_dates, manifest, parameters
):
    """Summarize data quality and retained behaviour without modifying observations."""
    frame, accounts = balances.collect(), profiles.collect()
    # Hash the per-account vectors to report identical histories without quadratic comparisons.
    identical = (
        frame.group_by("account_id")
        .agg(pl.col("balance_raw"))
        .group_by("balance_raw")
        .agg(pl.len().alias("count"))
        .filter(pl.col("count") > 1)
    )
    complete = accounts.filter(pl.col("missing_days") == 0).select("account_id")
    cohort = (
        regimes.collect()
        .join(complete, on="account_id")
        .group_by("month")
        .agg(
            pl.col("median_balance").median().alias("median_account_monthly_median"),
            pl.col("movement_rate").mean().alias("mean_account_movement_rate"),
        )
        .sort("month")
    )
    summary = {
        "source": manifest,
        "parameters": parameters,
        "accounts": accounts.height,
        "calendar_days": frame["date"].n_unique(),
        "rows": frame.height,
        "missing_values": frame["balance"].null_count(),
        "usable_observations": frame["is_usable"].sum(),
        "corrected_balances": corrections.collect().height,
        "accounts_with_missing_history": accounts.filter(pl.col("missing_days") > 0).height,
        "duplicate_series_groups": identical.height,
        "exact_zero_days": accounts["zero_days"].sum(),
        "negative_days": accounts["negative_days"].sum(),
        "event_candidates": events.collect().height,
        "shared_dates_for_review": shared_dates.collect().filter(pl.col("review")).to_dicts(),
        "complete_cohort_accounts": complete.height,
        "complete_cohort_monthly": cohort.to_dicts(),
    }
    # Convert native dates and NumPy scalars to portable JSON values.
    summary = json.loads(json.dumps(summary, default=str))
    metrics = [
        "accounts",
        "calendar_days",
        "rows",
        "missing_values",
        "usable_observations",
        "corrected_balances",
        "exact_zero_days",
        "negative_days",
        "event_candidates",
    ]
    lines = ["# Stage 0 quality report", "", "| Metric | Value |", "|---|---:|"]
    lines += [f"| {key} | {summary[key]} |" for key in metrics]
    lines += [
        "",
        "Financial events are retained. Flags are screening candidates, not fraud labels.",
        "COVID history is retained; monthly comparisons do not establish causality.",
        "Movement rates below are descriptive and are not adjusted for weekday composition.",
        "",
        f"Complete-history comparison cohort: {complete.height} accounts.",
        "",
        "| Month | Median account monthly median (€) | Mean movement rate |",
        "|---|---:|---:|",
    ]
    lines += [
        f"| {row['month']} | {row['median_account_monthly_median']} | "
        f"{row['mean_account_movement_rate']} |"
        for row in summary["complete_cohort_monthly"]
    ]
    lines += [
        "",
        "## Source and resolved configuration",
        "",
        "```json",
        json.dumps({"source": manifest, "parameters": parameters}, indent=2),
        "```",
    ]
    return summary, "\n".join(lines) + "\n"


def report_findings(
    balances,
    coverage,
    activity,
    calendar,
    regimes,
    movements,
    shifts,
    diagnostics,
    corrections,
    shared_dates,
    manifest,
    parameters,
):
    """Present all independent checks together, preserving their separate detail tables."""
    events = pl.concat([movements.collect(), shifts.collect()]).sort("account_id", "date")
    profiles = summarize_accounts(coverage, activity, calendar, regimes, events.lazy(), diagnostics)
    summary, markdown = quality_report(
        balances,
        profiles.lazy(),
        events.lazy(),
        regimes,
        corrections,
        shared_dates,
        manifest,
        parameters,
    )
    return profiles, events, summary, markdown
