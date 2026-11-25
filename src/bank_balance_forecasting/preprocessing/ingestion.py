"""Validate raw files, retain the calendar, and apply explicit source corrections."""

import csv
import hashlib
from datetime import date
from pathlib import Path

import polars as pl


def validate_source(source, parameters):
    """Validate the wide CSV schema, daily calendar and unique account IDs.

    Read the source headers and account column, then return a SHA-256 manifest
    with the account count and available date range. Reject invalid input before
    normalization or corrections are applied.
    """
    path = Path(parameters["source_path"])
    with path.open(newline="") as stream:
        headers = next(csv.reader(stream))
    if headers[0] != "ACCOUNT_ID" or len(headers) != len(set(headers)):
        raise ValueError("Expected ACCOUNT_ID and unique daily date columns")
    dates = [date.fromisoformat(value) for value in headers[1:]]
    if not dates or any((b - a).days != 1 for a, b in zip(dates, dates[1:])):
        raise ValueError("Date columns must be ordered and consecutive")
    ids = source.select("ACCOUNT_ID").collect().to_series()
    if ids.null_count() or ids.str.strip_chars().eq("").any() or ids.n_unique() != len(ids):
        raise ValueError("Account IDs must be present and unique")
    with path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "sha256": checksum,
        "source": str(path),
        "accounts": len(ids),
        "first_date": str(dates[0]),
        "last_date": str(dates[-1]),
    }


def normalize(source, manifest, parameters):
    """Convert the wide source into account/date/raw-balance rows through the cutoff.

    Preserve missing values and every source account, parse dates and numeric
    balances, and reject non-finite observed values. Return a sorted DataFrame.
    """
    cutoff = date.fromisoformat(str(parameters["as_of_date"]))
    columns = [c for c in source.collect_schema().names()[1:] if date.fromisoformat(c) <= cutoff]
    if not columns:
        raise ValueError("AS_OF_DATE precedes the source history")
    frame = (
        source.select("ACCOUNT_ID", *columns)
        .unpivot(index="ACCOUNT_ID", variable_name="date", value_name="balance_raw")
        .rename({"ACCOUNT_ID": "account_id"})
        .with_columns(pl.col("date").str.to_date(), pl.col("balance_raw").cast(pl.Float64))
        .sort("account_id", "date")
        .collect()
    )
    if frame["account_id"].n_unique() != manifest["accounts"]:
        raise ValueError("Normalization changed the source account count")
    if frame.filter(
        pl.col("balance_raw").is_not_null() & ~pl.col("balance_raw").is_finite()
    ).height:
        raise ValueError("Balances must be finite numbers or empty missing values")
    return frame


def audit_coverage(balances):
    """Summarize observed history and missing-calendar coverage for each account.

    Return calendar bounds, first/last observations and leading, internal and
    trailing gap counts. Mark accounts without observations; their observation
    bounds and derived gap positions remain unavailable.
    """
    frame = balances.collect()
    return (
        frame.group_by("account_id")
        .agg(
            pl.col("date").min().alias("calendar_start"),
            pl.col("date").max().alias("calendar_end"),
            pl.col("date")
            .filter(pl.col("balance_raw").is_not_null())
            .min()
            .alias("first_observed_date"),
            pl.col("date")
            .filter(pl.col("balance_raw").is_not_null())
            .max()
            .alias("last_observed_date"),
            pl.col("balance_raw").count().alias("history_days"),
            pl.col("balance_raw").null_count().alias("missing_days"),
        )
        .with_columns(
            (pl.col("first_observed_date") - pl.col("calendar_start"))
            .dt.total_days()
            .alias("leading_missing"),
            (pl.col("calendar_end") - pl.col("last_observed_date"))
            .dt.total_days()
            .alias("trailing_missing"),
        )
        .with_columns(
            (pl.col("missing_days") - pl.col("leading_missing") - pl.col("trailing_missing")).alias(
                "internal_missing"
            ),
            (pl.col("history_days") == 0).alias("no_observations"),
        )
        .sort("account_id")
    )


def screen_shared_dates(balances, parameters):
    """Flag dates with doubling patterns shared across many accounts.

    Compare raw balances with adjacent days, count exact double-and-return and
    near-double cases, and flag unusually large affected shares for review.
    This diagnostic does not change balances or approve additional corrections.
    """
    frame = (
        balances.collect()
        .sort("account_id", "date")
        .with_columns(
            pl.col("balance_raw").shift(1).over("account_id").alias("previous"),
            pl.col("balance_raw").shift(-1).over("account_id").alias("following"),
        )
        .with_columns(
            (pl.col("following").abs() > parameters["ratio_min_balance"])
            .fill_null(False)
            .alias("eligible"),
            ((pl.col("balance_raw") - 2 * pl.col("previous")).abs() < parameters["tolerance"])
            .and_((pl.col("following") - pl.col("previous")).abs() < parameters["tolerance"])
            .and_(pl.col("previous").abs() > parameters["tolerance"])
            .fill_null(False)
            .alias("exact_double_return"),
        )
        .with_columns(
            (
                pl.col("eligible")
                & (pl.col("balance_raw") / pl.col("following")).is_between(
                    parameters["ratio_low"], parameters["ratio_high"], closed="none"
                )
            )
            .fill_null(False)
            .alias("near_double")
        )
        .group_by("date")
        .agg(
            pl.col("eligible").sum(),
            pl.col("near_double").sum(),
            pl.col("exact_double_return").sum(),
        )
        .with_columns((pl.col("near_double") / pl.col("eligible").replace(0, None)).alias("share"))
        .with_columns(
            (pl.col("share") > parameters["shared_date_share"]).fill_null(False).alias("review")
        )
    )
    return frame.sort("date")


def classify_missing_history(balances):
    """Identify absent prehistory and genuine gaps; preserve missing values without filling."""
    coverage = audit_coverage(balances)
    frame = (
        balances.collect()
        .join(
            coverage.select("account_id", "first_observed_date", "last_observed_date"),
            on="account_id",
        )
        .with_columns(pl.col("balance_raw").alias("balance"))
    )
    flag = (
        pl.when(pl.col("balance_raw").is_not_null())
        .then(pl.lit("observed"))
        .when(pl.col("first_observed_date").is_null())
        .then(pl.lit("no_observations"))
        .when(pl.col("date") < pl.col("first_observed_date"))
        .then(pl.lit("prehistory_missing"))
        .when(pl.col("date") > pl.col("last_observed_date"))
        .then(pl.lit("trailing_missing"))
        .otherwise(pl.lit("internal_missing"))
    )
    return frame.select(
        "account_id",
        "date",
        "balance_raw",
        "balance",
        pl.col("balance").is_not_null().alias("is_usable"),
        flag.alias("quality_flags"),
    ), coverage


def apply_corrections(balances, parameters):
    """Apply explicit source repairs, preserving raw values and missing-history status."""
    frame = balances.collect()
    rules = parameters["corrections"]
    if len({str(r["date"]) for r in rules}) != len(rules):
        raise ValueError("Configure at most one correction per date")
    logs = []
    for rule in rules:
        day = date.fromisoformat(str(rule["date"]))
        affected = (pl.col("date") == day) & pl.col("balance_raw").is_not_null()
        logs.append(
            frame.filter(affected).select(
                "account_id",
                "date",
                "balance_raw",
                (pl.col("balance_raw") * rule["multiplier"]).alias("balance"),
                pl.lit(rule["multiplier"]).alias("multiplier"),
                pl.lit(rule["reason"]).alias("reason"),
            )
        )
        frame = frame.with_columns(
            pl.when(affected)
            .then(pl.col("balance_raw") * rule["multiplier"])
            .otherwise(pl.col("balance"))
            .alias("balance")
        )
    log = (
        pl.concat(logs)
        if logs
        else frame.head(0).select(
            "account_id",
            "date",
            "balance_raw",
            "balance",
            pl.lit(None, dtype=pl.Float64).alias("multiplier"),
            pl.lit(None, dtype=pl.String).alias("reason"),
        )
    )
    corrected_dates = log.select("date").unique().to_series()
    frame = frame.with_columns(
        pl.when(pl.col("date").is_in(corrected_dates.implode()) & pl.col("is_usable"))
        .then(pl.lit("known_date_scaled"))
        .otherwise(pl.col("quality_flags"))
        .alias("quality_flags")
    )
    return frame, log


def prepare_history(source, parameters):
    """Validate the source and prepare all available history through the cutoff."""
    manifest = validate_source(source, parameters)
    return normalize(source, manifest, parameters), manifest
