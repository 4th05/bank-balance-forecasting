"""Trace large percentage errors to their source observations."""

from datetime import timedelta

import polars as pl

from bank_balance_forecasting.forecast_pilot import EVALUATION_START, HORIZON

from .threshold import PRIMARY


def diagnose_mape_outlier(balances, events, cases, classical, cap):
    """Explain the largest uncapped persistence error without exporting account IDs."""
    scored = (
        cases.filter(
            (pl.col("study") == PRIMARY) & (pl.col("origin") >= EVALUATION_START)
        )
        .select("case_id", "account_id", "origin", "actual")
        .join(
            classical.filter(pl.col("model") == "Last balance"),
            on="case_id",
            validate="1:1",
        )
        .with_columns(pl.int_ranges(1, HORIZON + 1).alias("day"))
        .explode("day", "actual", "prediction", empty_as_null=True)
        .filter(pl.col("actual").is_finite() & (pl.col("actual") != 0))
        .with_columns(date=pl.col("origin") + pl.duration(days=pl.col("day")))
        .with_columns(
            ape=(pl.col("actual") - pl.col("prediction")).abs()
            / pl.col("actual").abs(),
            capped_ape=(pl.col("actual") - pl.col("prediction").clip(-cap, cap)).abs()
            / pl.col("actual").abs(),
        )
    )
    worst = scored.sort(
        "ape", "date", "account_id", descending=[True, False, False]
    ).row(0, named=True)
    account = worst["account_id"]
    history = (
        balances.filter(pl.col("account_id") == account)
        .sort("date")
        .with_columns(episode=pl.col("balance").rle_id())
    )
    episode_id = history.filter(pl.col("date") == worst["date"])["episode"].item()
    episode = history.filter(pl.col("episode") == episode_id)
    start, end = episode["date"].min(), episode["date"].max()
    segment = scored.filter(
        (pl.col("account_id") == account) & pl.col("date").is_between(start, end)
    )
    flags = events.filter((pl.col("account_id") == account) & (pl.col("date") == start))
    before = history.filter(pl.col("date") <= worst["origin"])["balance"]
    near_zero = scored.filter(pl.col("actual").abs() < 1)
    return pl.DataFrame(
        {
            "Forecast origin": [worst["origin"]],
            "Balance before drop (€)": [
                history.filter(pl.col("date") == start - timedelta(days=1))[
                    "balance"
                ].first()
            ],
            "Forecast at origin (€)": [worst["prediction"]],
            "Smallest absolute balance through origin (€)": [before.abs().min()],
            "Observed balance (€)": [worst["actual"]],
            "First day at observed balance": [start],
            "Last recorded day at observed balance": [end],
            "Days at observed balance": [episode.height],
            "Scored days in episode": [segment.height],
            "Stage 0 event flags": [flags["candidate_type"].unique().sort().to_list()],
            "Source balance unchanged": [
                episode["balance"].equals(episode["balance_raw"])
            ],
            "Uncapped last-balance MAPE (%)": [100 * scored["ape"].mean()],
            "Episode contribution to uncapped MAPE (pp)": [
                100 * segment["ape"].sum() / scored.height
            ],
            "Episode contribution to capped MAPE (pp)": [
                100 * segment["capped_ape"].sum() / scored.height
            ],
            "Scored days": [scored.height],
            "Scored days below €1 in absolute balance": [near_zero.height],
            "Days below €1 contribution to capped MAPE (pp)": [
                100 * near_zero["capped_ape"].sum() / scored.height
            ],
        }
    )
