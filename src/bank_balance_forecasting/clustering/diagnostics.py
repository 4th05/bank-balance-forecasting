"""Refitted feature ablations, graph sensitivity and comparable historical snapshots."""

import copy

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score

from .clustering import assess_stability
from .representation import balanced_distances, distance_array, fit_feature_space
from .spectral import affinity_matrix, spectral_partition


def compare_feature_sets(values, feature_matrix, models, parameters):
    """Refit every view and every candidate K; do not choose a graph view by a preferred count."""
    accounts = feature_matrix.collect().filter("eligible")["account_id"].to_list()
    frame = values.collect().filter(pl.col("account_id").is_in(accounts)).sort("account_id")
    reference = {model["k"]: np.array(model["labels"]) for model in models}
    rows = []
    for name, graphs in [
        ("statistics_only", []),
        ("statistics_plus_nvg", ["nvg"]),
        ("statistics_plus_hvg", ["hvg"]),
        ("combined", ["nvg", "hvg"]),
    ]:
        config = copy.deepcopy(parameters)
        for family in ["nvg", "hvg"]:
            config["weights"][family] = 0.2 / len(graphs) if family in graphs else 0.0
        registry, x, eligible = fit_feature_space(frame, config)
        indices = np.flatnonzero(eligible)
        distance = balanced_distances(x[indices], x[indices], registry)
        np.fill_diagonal(distance, 0)
        affinity = affinity_matrix(distance, config)
        components = (
            connected_components(csr_matrix(affinity), directed=False)[0] if len(distance) else 0
        )
        fitted = {}
        for k in reference:
            if 1 < k < len(distance) and components <= k <= len(np.unique(distance, axis=0)):
                fitted[k] = spectral_partition(distance, k, config, affinity)
        matrix = pl.DataFrame({"account_id": frame["account_id"], "eligible": eligible})
        validation, _, _ = assess_stability(
            frame.lazy(), matrix.lazy(), list(fitted.values()), config
        )
        scores = {row["k"]: row["mean_ari"] for row in validation.to_dicts()}
        for k in reference:
            model = fitted.get(k)
            rows.append(
                {
                    "feature_set": name,
                    "k": k,
                    "feature_count": len(registry["features"]),
                    "agreement_with_primary": float(
                        adjusted_rand_score(reference[k][indices], model["labels"])
                    )
                    if model
                    else None,
                    "mean_resample_ari": scores.get(k),
                    "minimum_group_size": int(np.bincount(model["labels"], minlength=k).min())
                    if model
                    else 0,
                    "common_accounts": len(indices),
                    "valid": model is not None,
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "feature_set": pl.String,
            "k": pl.Int64,
            "feature_count": pl.Int64,
            "agreement_with_primary": pl.Float64,
            "mean_resample_ari": pl.Float64,
            "minimum_group_size": pl.Int64,
            "common_accounts": pl.Int64,
            "valid": pl.Boolean,
        },
    )


def assess_neighborhood_sensitivity(distances, models, parameters):
    """Evaluate neighborhood sensitivity for every candidate partition."""
    distance = distance_array(distances)
    counts = sorted(
        {
            min(max(5, int(np.ceil(len(distance) * f))), len(distance) - 1)
            for f in [0.02, 0.03, 0.05, 0.08, 0.12]
        }
    )
    rows = []
    for count in counts if len(distance) > 1 else []:
        config = {**parameters, "min_neighbors": count, "max_neighbors": count}
        affinity = affinity_matrix(distance, config)
        components = connected_components(csr_matrix(affinity), directed=False)[0]
        for model in models:
            k = model["k"]
            if k <= 1:
                continue
            valid = components <= k
            fit = spectral_partition(distance, k, config, affinity) if valid else None
            rows.append(
                {
                    "neighbors": count,
                    "k": k,
                    "graph_components": components,
                    "valid": valid,
                    "agreement_with_primary": float(
                        adjusted_rand_score(model["labels"], fit["labels"])
                    )
                    if fit
                    else None,
                    "minimum_group_size": int(np.bincount(fit["labels"], minlength=k).min())
                    if fit
                    else 0,
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "neighbors": pl.Int64,
            "k": pl.Int64,
            "graph_components": pl.Int64,
            "valid": pl.Boolean,
            "agreement_with_primary": pl.Float64,
            "minimum_group_size": pl.Int64,
        },
    )
