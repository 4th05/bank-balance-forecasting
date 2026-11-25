"""Descriptive account, calendar and monthly behaviour profiles."""

import numpy as np
import polars as pl


def with_changes(balances):
    """Sort each account chronologically and add adjacent daily balance changes.

    Keep the complete calendar; a missing balance leaves its adjacent change
    unavailable rather than connecting observations across the gap.
    """
    return (
        balances.collect()
        .sort("account_id", "date")
        .with_columns(pl.col("balance").diff().over("account_id").alias("change"))
    )


def profile_calendar(balances, parameters):
    """Summarize movement frequency and mean changes by account and calendar bucket.

    Group by weekday and day of month together, recording the number of usable
    daily pairs and changes exceeding the configured movement tolerance.
    """
    frame = with_changes(balances).with_columns(
        pl.col("date").dt.weekday().alias("weekday"),
        pl.col("date").dt.day().alias("day_of_month"),
        (pl.col("change").abs() > parameters["tolerance"]).alias("moved"),
    )
    return (
        frame.group_by("account_id", "weekday", "day_of_month")
        .agg(
            pl.col("change").count().alias("eligible_pairs"),
            pl.col("moved").sum().alias("movement_days"),
            pl.col("change").mean().alias("mean_change"),
        )
        .sort("account_id", "weekday", "day_of_month")
    )


def profile_regimes(balances, parameters):
    """Describe each account month using observed balances and valid daily changes.

    Return monthly coverage, mean/median balance, movement frequency, population
    change volatility and the share of observed balances that are negative.
    """
    return (
        with_changes(balances)
        .with_columns(
            pl.col("date").dt.truncate("1mo").alias("month"),
            (pl.col("change").abs() > parameters["tolerance"]).alias("moved"),
        )
        .group_by("account_id", "month")
        .agg(
            pl.col("balance").count().alias("observations"),
            pl.col("balance").mean().alias("mean_balance"),
            pl.col("balance").median().alias("median_balance"),
            pl.col("moved").mean().alias("movement_rate"),
            pl.col("change").std(ddof=0).alias("change_volatility"),
            (pl.col("balance") < 0).mean().alias("negative_share"),
        )
        .sort("account_id", "month")
    )


def profile_activity(balances, parameters):
    """Describe inactivity, unchanged runs, balance scale and zero/negative balances."""
    records = []
    for (account,), group in with_changes(balances).group_by("account_id", maintain_order=True):
        x = group["balance"].to_numpy()
        observed = x[np.isfinite(x)]
        delta = group["change"].to_numpy()
        run = longest = 0
        for i, value in enumerate(x):
            if not np.isfinite(value):
                run = 0
            else:
                run = (
                    run + 1
                    if i and np.isfinite(delta[i]) and abs(delta[i]) <= parameters["tolerance"]
                    else 1
                )
                longest = max(longest, run)
        records.append(
            {
                "account_id": account,
                "longest_constant_run": longest,
                "mean_balance": float(observed.mean()) if len(observed) else None,
                "balance_std": float(observed.std()) if len(observed) else None,
                "negative_days": int(np.sum(observed < 0)),
                "zero_days": int(np.sum(np.abs(observed) <= parameters["tolerance"])),
                "abs_below_10_days": int(np.sum(np.abs(observed) < 10)),
                "abs_below_100_days": int(np.sum(np.abs(observed) < 100)),
                "movement_days": int(np.sum(np.abs(delta) > parameters["tolerance"])),
            }
        )
    return pl.DataFrame(records)


def summarize_accounts(coverage, activity, calendar, regimes, events, diagnostics):
    """Combine the independently computed findings into one account summary."""
    movement_rates = (
        calendar.collect()
        .group_by("account_id")
        .agg(
            (pl.col("movement_days").sum() / pl.col("eligible_pairs").sum()).alias("movement_rate")
        )
    )
    recent = (
        regimes.collect()
        .sort("month")
        .group_by("account_id")
        .agg(
            pl.col("median_balance").last().alias("latest_month_median"),
            pl.col("change_volatility").last().alias("latest_month_change_volatility"),
        )
    )
    counts = events.collect().group_by("account_id").agg(pl.len().alias("event_flags"))
    return (
        coverage.collect()
        .join(activity.collect(), on="account_id")
        .join(movement_rates, on="account_id")
        .join(recent, on="account_id")
        .join(counts, on="account_id", how="left")
        .with_columns(pl.col("event_flags").fill_null(0))
        .join(diagnostics.collect(), on="account_id")
    )
