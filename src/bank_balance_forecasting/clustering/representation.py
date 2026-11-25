"""A fitted, bounded feature representation and balanced mixed-feature distances."""

import hashlib

import numpy as np
import polars as pl

from .features import feature_label


def transform_features(values, registry):
    """Apply saved feature bounds and clip the resulting values to [0, 1].

    Preserve unavailable entries and never refit scales when transforming
    training replays or new account observations.
    """
    frame = values.collect() if isinstance(values, pl.LazyFrame) else values
    matrix = (
        np.column_stack(
            [
                np.clip(
                    (frame[c].to_numpy() - registry["bounds"][c][0])
                    / (registry["bounds"][c][1] - registry["bounds"][c][0]),
                    0,
                    1,
                )
                for c in registry["features"]
            ]
        )
        if registry["features"]
        else np.empty((frame.height, 0))
    )
    return matrix


def balanced_distances(left, right, registry, families=None):
    """Average within each family, renormalizing over actually comparable families."""
    if not registry["features"]:
        return np.zeros((len(left), len(right)))
    total = np.zeros((len(left), len(right)))
    weights = np.zeros_like(total)
    for family, weight in registry["weights"].items():
        if families is not None and family not in families:
            continue
        columns = [i for i, c in enumerate(registry["features"]) if c.split("__")[0] == family]
        if not columns:
            continue
        sums, counts = np.zeros_like(total), np.zeros_like(total)
        for col in columns:
            difference = abs(left[:, col, None] - right[None, :, col])
            valid = np.isfinite(difference)
            sums += np.where(valid, difference, 0)
            counts += valid
        present = counts > 0
        total += weight * np.divide(sums, counts, out=np.zeros_like(sums), where=present)
        weights += weight * present
    return np.divide(total, weights, out=np.ones_like(total), where=weights > 0)


def pair_table(matrix, accounts):
    """Flatten a square distance matrix into labeled source/target account pairs.

    Keep explicit string/float schemas, including when the matrix is empty.
    """
    return pl.DataFrame(
        {
            "source_account": np.repeat(accounts, len(accounts)),
            "target_account": np.tile(accounts, len(accounts)),
            "distance": matrix.ravel(),
        },
        schema={"source_account": pl.String, "target_account": pl.String, "distance": pl.Float64},
    )


def distance_array(pairs):
    """Restore the square distance matrix from a complete all-pairs table.

    Sort source and target account IDs to recover canonical row/column order.
    Return a (0, 0) array when no eligible account pairs exist.
    """
    frame = pairs.collect() if isinstance(pairs, pl.LazyFrame) else pairs
    if frame.is_empty():
        return np.empty((0, 0))
    return (
        frame.sort("source_account", "target_account")["distance"]
        .to_numpy()
        .reshape(int(np.sqrt(frame.height)), -1)
    )


def fit_feature_space(frame, parameters):
    """Fit redundancy pruning and scales on only the supplied training accounts."""
    historical = frame["meta_history_days"].to_numpy() >= parameters["min_history"]
    fitting = historical if historical.any() else np.ones(frame.height, dtype=bool)
    columns = [
        c for c in frame.columns if "__" in c and parameters["weights"].get(c.split("__")[0], 0) > 0
    ]
    matrix = frame.select(columns).to_numpy().astype(float)
    bounds, retained, removed = {}, [], {}
    for j, col in enumerate(columns):
        sample = matrix[fitting, j]
        finite = sample[np.isfinite(sample)]
        if len(finite) < max(2, parameters["min_feature_coverage"] * fitting.sum()):
            removed[col] = "insufficient_coverage"
            continue
        low, high = np.quantile(finite, [0.05, 0.95])
        if high <= low:
            low, high = np.min(finite), np.max(finite)
        if high <= low:
            removed[col] = "constant"
            continue
        retained.append(col)
        bounds[col] = [float(low), float(high)]
    # Direct activity descriptors take precedence over proxies in other families.
    families = ["activity", "state", "dynamics", "regime", "nvg", "hvg"]
    order = sorted(
        retained,
        key=lambda c: (
            -np.isfinite(frame[c].to_numpy()[fitting]).sum(),
            families.index(c.split("__")[0]),
            c,
        ),
    )
    kept = []
    ranks = frame.select(order).to_pandas().loc[fitting].corr(method="spearman") if order else None
    for col in order:
        if any(abs(ranks.loc[col, other]) > parameters["correlation_threshold"] for other in kept):
            removed[col] = "redundant_across_families"
        else:
            kept.append(col)
    registry = {
        "features": kept,
        "feature_labels": {c: feature_label(c) for c in kept},
        "version": 2,
        "coverage_features": columns,
        "bounds": {c: bounds[c] for c in kept},
        "removed": removed,
        "weights": parameters["weights"],
        "as_of_date": str(parameters["as_of_date"]),
        "definitions": {
            "activity": "Observed history and account movement/flat-run behaviour",
            "state": "Balance level, sign and robust scale",
            "dynamics": "Curated tsfresh, relative weekday activity and inflow regularity",
            "regime": "Full-history/recent changes; 95%-winsorized absolute daily change scale",
            "nvg": "Natural visibility; median/IQR of 90-day-window descriptors",
            "hvg": "Horizontal visibility; median/IQR of 90-day-window descriptors",
        },
        "tsfresh_calculators": {
            "autocorrelation": parameters["acf_lags"],
            "absolute_change_autocorrelation": [1, 7],
            "fourier_entropy": {"bins": parameters["entropy_bins"]},
            "cid_ce": {"normalize": True, "divide_by_sqrt_length": True},
        },
        "parameters": parameters,
    }
    scaled = transform_features(frame, registry)
    eligible = eligible_rows(frame, registry, scaled, parameters)
    return registry, scaled, eligible


def build_representation(activity, state, dynamics, regime, nvg, hvg, parameters):
    """Select nonredundant features, fit robust scales and balance six feature families."""
    tables = [t.collect() for t in [activity, state, dynamics, regime, nvg, hvg]]
    frame = tables[0]
    for table in tables[1:]:
        if table["account_id"].n_unique() != table.height or set(table["account_id"]) != set(
            frame["account_id"]
        ):
            raise ValueError("Feature families must cover the same unique account IDs")
        frame = frame.join(table, on="account_id", how="left")
    frame = frame.sort("account_id")
    registry, scaled, eligible = fit_feature_space(frame, parameters)
    kept = registry["features"]
    registry["feature_values_sha256"] = hashlib.sha256(frame.write_json().encode()).hexdigest()
    scaled_frame = pl.DataFrame(
        {
            "account_id": frame["account_id"],
            "eligible": eligible,
            **{c: scaled[:, i] for i, c in enumerate(kept)},
        }
    )
    indices = np.flatnonzero(eligible)
    distances = balanced_distances(scaled[indices], scaled[indices], registry)
    np.fill_diagonal(distances, 0)
    accounts = frame["account_id"].to_numpy()[indices]
    return frame, scaled_frame, registry, pair_table(distances, accounts)


def eligible_rows(frame, registry, scaled, parameters):
    """Use the same observed-history and feature-coverage policy at fit and prediction."""
    # Redundancy removal must not make an otherwise observed family appear missing.
    columns = registry.get("coverage_features", registry["features"])
    availability = np.zeros(frame.height, int)
    for family_group in [["activity"], ["state"], ["dynamics"], ["regime"], ["nvg", "hvg"]]:
        family_columns = [c for c in columns if c.split("__")[0] in family_group]
        if family_columns:
            availability += np.isfinite(frame.select(family_columns).to_numpy()).mean(axis=1) >= 0.5
    historical = frame["meta_history_days"].to_numpy() >= parameters["min_history"]
    return historical & (
        (availability >= parameters["min_available_blocks"]) if registry["features"] else True
    )
