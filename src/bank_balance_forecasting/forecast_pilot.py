"""Rolling-origin study of model choice for the three Stage 1 groups."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegression, Ridge

from bank_balance_forecasting.preprocessing.events import (
    classify_reversals,
    flag_extreme_changes,
    flag_level_shifts,
)
from bank_balance_forecasting.preprocessing.profiles import profile_calendar
from bank_balance_forecasting.clustering.features import (
    extract_activity,
    extract_balance_state,
    extract_dynamics,
    extract_regime_features,
)
from bank_balance_forecasting.clustering.graphs import (
    extract_hvg_features,
    extract_nvg_features,
)
from bank_balance_forecasting.clustering.representation import (
    build_representation,
    distance_array,
)
from bank_balance_forecasting.clustering.spectral import spectral_partition

HORIZON = 30
ORIGINS = tuple(date(2022, month, 1) - timedelta(days=1) for month in range(5, 13))
EVALUATION_START = date(2022, 9, 30)
FOUNDATION_MODELS = {
    "Chronos-Bolt Tiny": ("amazon/chronos-bolt-tiny", "a0e552de83495b5c28c14c71c374f3e33280b340"),
    "Chronos-2 Small": ("autogluon/chronos-2-small", "ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a"),
}


def fit_groups(history, settings, origin):
    """Recompute Stage 0 profiles and Stage 1 memberships using history through the origin."""
    stage0 = settings["stage0"]
    stage1 = {**settings["stage1"], "as_of_date": str(origin)}
    series = history.lazy()
    calendar = profile_calendar(series, stage0)
    large, _ = flag_extreme_changes(series, stage0)
    events = pl.concat(
        [classify_reversals(series, large.lazy(), stage0), flag_level_shifts(series, stage0)]
    )
    activity = extract_activity(series, stage1)
    state = extract_balance_state(series, stage1)
    _, matrix, _, distances = build_representation(
        activity.lazy(),
        state.lazy(),
        extract_dynamics(series, calendar.lazy(), stage1).lazy(),
        extract_regime_features(series, events.lazy(), stage1).lazy(),
        extract_nvg_features(series, stage1).lazy(),
        extract_hvg_features(series, stage1).lazy(),
        stage1,
    )
    accounts = matrix.filter("eligible").sort("account_id")["account_id"]
    labels = spectral_partition(distance_array(distances), 3, stage1)["labels"]
    assigned = pl.DataFrame({"account_id": accounts, "cluster": labels})
    profile = (
        assigned.join(activity, on="account_id")
        .join(state, on="account_id")
        .group_by("cluster")
        .agg(
            pl.len().alias("accounts"),
            pl.col("activity__movement_rate").median().alias("movement"),
            pl.col("state__negative_share").median().alias("negative"),
        )
        .sort("negative", descending=True)
    )
    negative = profile["cluster"][0]
    quieter, busier = profile.filter(pl.col("cluster") != negative).sort("movement")["cluster"]
    roles = {negative: "G02", quieter: "G01", busier: "G03"}
    assigned = assigned.with_columns(pl.col("cluster").replace_strict(roles).alias("group"))
    profile = profile.with_columns(
        pl.col("cluster").replace_strict(roles).alias("group"), pl.lit(origin).alias("origin")
    )
    return dict(assigned.select("account_id", "group").iter_rows()), profile.drop("cluster")


def month_peak_day(changes, days, tolerance):
    """Find the account's strongest recurring two-day window around a calendar day."""
    frame = pd.DataFrame({"change": changes}, index=days)
    peaks = []
    for month, values in frame.groupby(days.to_period("M")):
        valid = values["change"]
        if len(valid) == month.days_in_month and valid.notna().all() and valid.max() > tolerance:
            peaks.append(valid.idxmax().day)
    if len(peaks) < 6:
        return None
    candidates = np.arange(1, 32)
    return int(candidates[np.argmax((abs(np.asarray(peaks)[:, None] - candidates) <= 2).sum(0))])


def calendar_design(days, peak_day):
    """Encode known future weekdays and an inflow window fitted from past months."""
    weekday = np.eye(7)[days.dayofweek.to_numpy()]
    if peak_day is None:
        return weekday
    return np.column_stack([weekday, abs(days.day.to_numpy() - peak_day) <= 2])


def local_smoothing(values):
    """Compute fixed-gain local level and damped Holt paths."""
    level = trend_level = values[0]
    trend = 0.0
    damping = 0.9
    for value in values[1:]:
        level = 0.2 * value + 0.8 * level
        updated = 0.2 * value + 0.8 * (trend_level + damping * trend)
        trend = 0.05 * (updated - trend_level) + 0.95 * damping * trend
        trend_level = updated
    return np.full(HORIZON, level), trend_level + trend * np.cumsum(
        damping ** np.arange(1, HORIZON + 1)
    )


def classical_forecasts(history, past_days, tolerance):
    """Produce each candidate's next 30 daily balances from the same account history."""
    filled = pd.Series(history, index=past_days).ffill().dropna().to_numpy()
    last = filled[-1]
    step = np.arange(1, HORIZON + 1)
    ses, holt = local_smoothing(filled[-365:])
    changes = np.diff(history, prepend=np.nan)
    finite = np.isfinite(changes)
    consecutive = finite[1:] & finite[:-1]
    lag, target = changes[:-1][consecutive], changes[1:][consecutive]
    slope = (
        np.clip(np.cov(lag, target, ddof=0)[0, 1] / max(np.var(lag), 1e-12), -0.95, 0.95)
        if len(lag) >= 3
        else 0.0
    )
    center = float(np.mean(changes[finite])) if finite.any() else 0.0
    previous = changes[-1] if finite[-1] else center
    ar_changes = center + slope**step * (previous - center)

    future_days = pd.date_range(past_days[-1] + pd.Timedelta(days=1), periods=HORIZON, freq="D")
    peak = month_peak_day(changes, past_days, tolerance)
    design = calendar_design(past_days, peak)
    future = calendar_design(future_days, peak)
    recent = finite & (np.arange(len(history)) >= len(history) - 365)
    x, y = design[recent], changes[recent]
    forecasts = {
        "Last balance": np.full(HORIZON, last),
        "SMA 7": np.full(HORIZON, filled[-7:].mean()),
        "SMA 14": np.full(HORIZON, filled[-14:].mean()),
        "SMA 28": np.full(HORIZON, filled[-28:].mean()),
        "Seasonal naive 7": np.resize(filled[-7:], HORIZON),
        "Local level": ses,
        "Damped Holt": holt,
        "Drift 90": last + step * (last - filled[-91]) / 90,
        "AR(1) changes": last + np.cumsum(ar_changes),
    }
    moves = abs(y) > tolerance
    if not len(y):
        for name in ("Weekday changes", "Calendar changes", "Two-part calendar"):
            forecasts[name] = forecasts["Last balance"]
        return forecasts, trend_repeats(history)
    weekday_delta = Ridge(alpha=10).fit(x[:, :7], y).predict(future[:, :7])
    calendar_delta = Ridge(alpha=10).fit(x, y).predict(future)
    if moves.any() and not moves.all():
        probability = LogisticRegression(C=1).fit(x, moves).predict_proba(future)[:, 1]
    else:
        probability = np.full(HORIZON, moves.mean())
    size = Ridge(alpha=10).fit(x[moves], y[moves]).predict(future) if moves.any() else 0.0
    forecasts.update(
        {
            "Weekday changes": last + np.cumsum(weekday_delta),
            "Calendar changes": last + np.cumsum(calendar_delta),
            "Two-part calendar": last + np.cumsum(probability * size),
        }
    )
    return forecasts, trend_repeats(history)


def trend_repeats(history):
    """Check direction agreement for material slopes in two 13-week blocks."""
    recent = history[-182:]
    if len(recent) < 182 or not np.isfinite(recent).all():
        return None
    weeks = np.median(recent.reshape(26, 7), axis=1)
    slopes = [12 * np.polyfit(np.arange(13), block, 1)[0] for block in np.split(weeks, 2)]
    threshold = 0.25 * np.ptp(np.quantile(recent, [0.25, 0.75]))
    return bool(
        min(abs(slope) for slope in slopes) > max(threshold, 1e-6)
        and np.sign(slopes[0]) == np.sign(slopes[1])
    )


def predict_chronos(contexts, model, device="cpu", quantile_levels=None):
    """Return median paths by default, or all requested quantiles for each day."""
    import torch
    from chronos import BaseChronosPipeline

    torch.set_num_threads(4)
    checkpoint, revision = FOUNDATION_MODELS[model]
    pipeline = BaseChronosPipeline.from_pretrained(
        checkpoint, revision=revision, device_map=device, local_files_only=True
    )
    forecasts = []
    for start in range(0, len(contexts), 128):
        batch = torch.tensor(contexts[start : start + 128], dtype=torch.float32)
        if model == "Chronos-2 Small":
            batch = batch[:, None, :]
        quantiles, _ = pipeline.predict_quantiles(
            batch, prediction_length=HORIZON, quantile_levels=quantile_levels or [0.5]
        )
        if isinstance(quantiles, list):
            quantiles = torch.cat(quantiles, dim=0)
        forecasts.append(
            quantiles.numpy() if quantile_levels is not None else quantiles[..., 0].numpy()
        )
    return np.concatenate(forecasts)


def run_pilot(balances, settings, origins=ORIGINS):
    """Score matched forecasts and return local contexts for the foundation benchmark."""
    scores, rows, contexts, actuals, profiles, coverage = [], [], [], [], [], []
    by_account = balances.partition_by("account_id", as_dict=True)
    for origin in origins:
        history = balances.filter(pl.col("date") <= origin)
        groups, profile = fit_groups(history, settings, origin)
        profiles.append(profile)
        split = "Selection" if origin < EVALUATION_START else "Evaluation"
        start = len(rows)
        for (account,), account_rows in by_account.items():
            account_rows = account_rows.sort("date")
            end = account_rows.filter(pl.col("date") <= origin).height
            past = account_rows["balance"].to_numpy()[:end]
            actual = account_rows["balance"].to_numpy()[end : end + HORIZON]
            if (
                len(actual) != HORIZON
                or not np.isfinite(actual).all()
                or np.isfinite(past).sum() < 91
                or not np.isfinite(past[-1])
            ):
                continue
            days = pd.DatetimeIndex(account_rows["date"].to_list()[:end])
            assert days[-1].date() == origin
            assert account_rows["date"][end] == origin + timedelta(days=1)
            forecasts, trend = classical_forecasts(past, days, settings["stage1"]["tolerance"])
            key = (origin, split, account, groups.get(account, "Review"))
            rows.append((*key, trend))
            contexts.append(past[-512:])
            actuals.append(actual)
            scores.extend(
                (*key, model, float(np.mean(abs(prediction - actual))))
                for model, prediction in forecasts.items()
            )
        coverage.append((origin, split, len(by_account), len(rows) - start))
        print(f"{origin} · {len(rows) - start} matched accounts", flush=True)
    return (
        pl.DataFrame(
            scores, schema=["origin", "split", "account_id", "group", "model", "mae"], orient="row"
        ),
        pl.DataFrame(
            rows, schema=["origin", "split", "account_id", "group", "trend_repeats"], orient="row"
        ),
        np.asarray(contexts, dtype=np.float32),
        np.asarray(actuals),
        pl.concat(profiles),
        pl.DataFrame(coverage, schema=["origin", "split", "total", "scored"], orient="row"),
    )


def add_foundation_scores(scores, keys, forecasts, actuals, model):
    """Add the foundation model on exactly the same account origins as other candidates."""
    if forecasts.shape != actuals.shape or not np.isfinite(forecasts).all():
        raise ValueError("Expected a finite 30-day forecast for each scored account origin")
    foundation = keys.select("origin", "split", "account_id", "group").with_columns(
        pl.lit(model).alias("model"),
        pl.Series("mae", np.mean(abs(forecasts - actuals), axis=1)),
    )
    return pl.concat([scores, foundation])


def compare_policies(scores):
    """Freeze both model-selection policies before scoring the evaluation origins."""
    select = scores.filter(pl.col("split") == "Selection")
    evaluation = scores.filter(pl.col("split") == "Evaluation")
    global_model = (
        select.group_by("model").agg(pl.col("mae").mean()).sort("mae", "model")["model"][0]
    )
    routed = (
        select.filter(pl.col("group") != "Review")
        .group_by("group", "model")
        .agg(pl.col("mae").mean())
        .sort("group", "mae", "model")
        .unique("group", keep="first", maintain_order=True)
        .select("group", pl.col("model").alias("routed_model"))
    )
    by_account = evaluation.join(routed, on="group", how="left").with_columns(
        pl.col("routed_model").fill_null(global_model)
    )
    common = by_account.filter(pl.col("model") == global_model).select(
        "origin", "account_id", "group", pl.col("mae").alias("common_mae")
    )
    route = by_account.filter(pl.col("model") == pl.col("routed_model")).select(
        "origin", "account_id", "group", "routed_model", pl.col("mae").alias("routed_mae")
    )
    paired = common.join(route, on=["origin", "account_id", "group"], validate="1:1")
    assert paired.height == common.height == route.height
    return (
        global_model,
        routed,
        paired.with_columns((pl.col("routed_mae") - pl.col("common_mae")).alias("delta")),
    )


def random_route_control(scores, repeats=50, seed=37):
    """Compare with random groups of the same sizes using stable shuffled account ranks."""
    keys = scores.select("origin", "account_id", "group").unique()
    eligible = keys.filter(pl.col("group") != "Review").sort("origin", "group", "account_id")
    accounts = eligible["account_id"].unique().sort()
    rng = np.random.default_rng(seed)
    results = []
    for trial in range(repeats):
        ranks = pl.DataFrame({"account_id": accounts, "rank": rng.permutation(len(accounts))})
        shuffled = (
            eligible.join(ranks, on="account_id")
            .sort("origin", "rank")
            .with_columns(eligible["group"].alias("random_group"))
        )
        rerouted = scores.join(
            shuffled.select("origin", "account_id", "random_group"),
            on=["origin", "account_id"],
            how="left",
        ).with_columns(pl.col("random_group").fill_null("Review").alias("group"))
        _, _, paired = compare_policies(rerouted)
        results.append((trial, paired["delta"].mean()))
    return pl.DataFrame(results, schema=["trial", "delta"], orient="row")
