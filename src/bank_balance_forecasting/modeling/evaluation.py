"""Summarize the same capped forecasts by the group at each forecast origin."""

import numpy as np
import polars as pl

from bank_balance_forecasting.forecast_pilot import EVALUATION_START

from .threshold import EARLY_START, PRIMARY, REPLAY, _errors


def score_model_groups(cases, labels, model_paths, cap):
    """Score the groups and measure the cap's effect on the selected forecast."""
    rows, cap_effects = [], []
    for study, start in ((PRIMARY, EVALUATION_START), (REPLAY, EARLY_START)):
        cohort = (
            cases.filter((pl.col("study") == study) & (pl.col("origin") >= start))
            .select("case_id", "account_id", "origin", "actual")
            .join(labels, on=["account_id", "origin"], how="left", validate="m:1")
            .with_columns(pl.col("group").fill_null("Review"))
            .sort("case_id")
        )
        candidates = model_paths.join(
            cohort.select("case_id"), on="case_id", how="semi"
        )
        groups = np.asarray(cohort["group"].to_list())
        actual = np.asarray(cohort["actual"].to_list(), dtype=float)
        for (model, selected), frame in candidates.group_by("model", "Selected"):
            frame = frame.sort("case_id")
            if not frame["case_id"].equals(cohort["case_id"]):
                raise ValueError(
                    f"{model} must forecast all evaluation cases for {study}"
                )
            raw = np.asarray(frame["prediction"].to_list())
            capped = np.clip(raw, -cap, cap)
            if selected:
                for treatment, forecast in (
                    ("Before clipping", raw),
                    ("Shared cap", capped),
                ):
                    cap_effects.append(
                        {
                            "study": study,
                            "model": model,
                            "Treatment": treatment,
                            **_errors(actual, forecast),
                        }
                    )
            for group in np.unique(groups):
                mask = groups == group
                target, forecast = actual[mask], capped[mask]
                observed = np.isfinite(target)
                scored = observed & (target != 0)
                absolute = np.abs(target)
                near_zero = scored & (absolute < 1)
                low = observed & (absolute >= 1) & (absolute < 5)
                above = observed & (absolute >= 1)
                error = np.abs(target - forecast)
                if scored.any():
                    metrics = _errors(target, forecast)
                    mape, wape = metrics["MAPE (%)"], metrics["WAPE (%)"]
                    near_zero_points = (
                        100
                        * np.sum(error[near_zero] / absolute[near_zero])
                        / scored.sum()
                    )
                    low_points = 100 * np.sum(error[low] / absolute[low]) / scored.sum()
                else:
                    mape = wape = near_zero_points = low_points = None
                rows.append(
                    {
                        "study": study,
                        "model": model,
                        "group": group,
                        "Selected": selected,
                        "MAPE (%)": mape,
                        "MAE (€)": float(error[observed].mean())
                        if observed.any()
                        else None,
                        "WAPE (%)": wape,
                        "Observed days": int(observed.sum()),
                        "MAPE days": int(scored.sum()),
                        "Sub-euro days": int(near_zero.sum()),
                        "Days from €1 to under €5": int(low.sum()),
                        "Absolute balance sum (€)": float(absolute[observed].sum()),
                        "Median |balance| (€)": float(np.median(absolute[observed]))
                        if observed.any()
                        else None,
                        "Sub-euro MAPE contribution (pp)": near_zero_points,
                        "€1 to under €5 MAPE contribution (pp)": low_points,
                        "MAPE at €1 or more (%)": 100
                        * float(np.mean(error[above] / absolute[above]))
                        if above.any()
                        else None,
                    }
                )
    return pl.DataFrame(rows).sort("study", "model", "group"), pl.DataFrame(cap_effects)
