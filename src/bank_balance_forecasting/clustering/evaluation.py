"""Complementary cluster-count evidence, with explicit geometry and null assumptions."""

import logging

import numpy as np
import polars as pl
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, laplacian
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
)
from threadpoolctl import threadpool_limits

from .representation import balanced_distances, distance_array
from .spectral import affinity_matrix, spectral_partition


def within_dispersion(distance, labels):
    """Pairwise squared-dissimilarity dispersion, sum_g sum_ij D_ij² / (2 n_g)."""
    return float(
        sum(
            np.square(distance[np.ix_(labels == group, labels == group)]).sum()
            / (2 * np.sum(labels == group))
            for group in np.unique(labels)
        )
    )


def evaluate_separation(distances, feature_matrix, registry, models):
    """Evaluate balanced distances, graph partitions and an explicitly Euclidean feature view."""
    matrix = feature_matrix.collect().filter("eligible").sort("account_id")
    accounts = matrix["account_id"].to_list()
    x = matrix.select(registry["features"]).to_numpy()
    weighted = x.copy()
    for family, weight in registry["weights"].items():
        columns = [i for i, c in enumerate(registry["features"]) if c.split("__")[0] == family]
        if columns:
            weighted[:, columns] *= np.sqrt(weight / len(columns))
    distance = distance_array(distances)
    rows, samples = [], []
    for model in models:
        k, labels = model["k"], np.array(model["labels"])
        valid = 1 < k < len(labels)
        silhouette = silhouette_samples(distance, labels, metric="precomputed") if valid else None
        same = labels[:, None] == labels[None, :]
        diameter = float(distance[same].max()) if len(labels) else 0.0
        separation = float(distance[~same].min()) if valid else None
        euclidean_valid = valid and weighted.shape[1] > 0 and np.isfinite(weighted).all()
        rows.append(
            {
                "k": k,
                "within_dispersion": within_dispersion(distance, labels),
                "silhouette": float(np.mean(silhouette)) if valid else None,
                "negative_silhouette_share": float(np.mean(silhouette < 0)) if valid else None,
                "dunn": separation / diameter if valid and diameter > 0 else None,
                "zero_within_diameter": diameter == 0,
                "calinski_harabasz_euclidean": float(calinski_harabasz_score(weighted, labels))
                if euclidean_valid
                else None,
                "davies_bouldin_euclidean": float(davies_bouldin_score(weighted, labels))
                if euclidean_valid
                else None,
                "euclidean_view_available": bool(euclidean_valid),
            }
        )
        if valid:
            samples.extend(
                {"account_id": account, "k": k, "group": int(group), "silhouette": float(score)}
                for account, group, score in zip(accounts, labels, silhouette)
            )
    return (
        pl.DataFrame(
            rows,
            schema={
                "k": pl.Int64,
                "within_dispersion": pl.Float64,
                "silhouette": pl.Float64,
                "negative_silhouette_share": pl.Float64,
                "dunn": pl.Float64,
                "zero_within_diameter": pl.Boolean,
                "calinski_harabasz_euclidean": pl.Float64,
                "davies_bouldin_euclidean": pl.Float64,
                "euclidean_view_available": pl.Boolean,
            },
        ),
        pl.DataFrame(
            samples,
            schema={
                "account_id": pl.String,
                "k": pl.Int64,
                "group": pl.Int64,
                "silhouette": pl.Float64,
            },
        ),
    )


def inspect_spectrum(distances, parameters):
    """Inspect normalized-Laplacian eigengaps, including K=1, over graph neighborhoods."""
    distance = distance_array(distances)
    n = len(distance)
    counts = sorted(
        {min(n - 1, max(5, int(np.ceil(n * f)))) for f in [0.02, 0.03, 0.05, 0.08, 0.12]}
    )
    rows = []
    unique = len(np.unique(distance, axis=0)) if n else 0
    for count in counts if n > 1 and unique > 1 else []:
        config = {**parameters, "min_neighbors": count, "max_neighbors": count}
        affinity = affinity_matrix(distance, config)
        components = connected_components(csr_matrix(affinity), directed=False)[0]
        minimum = max(
            parameters["min_group_size"], int(np.ceil(n * parameters["min_group_fraction"]))
        )
        limit = min(parameters["max_groups"], n - 1, max(1, n // minimum), unique)
        with threadpool_limits(limits=1):
            eigenvalues = eigh(
                laplacian(affinity, normed=True), subset_by_index=[0, limit], eigvals_only=True
            )
        rows.extend(
            {
                "neighbors": count,
                "k": k,
                "eigenvalue": float(eigenvalues[k - 1]),
                "next_eigenvalue": float(eigenvalues[k]),
                "eigengap": float(eigenvalues[k] - eigenvalues[k - 1]),
                "graph_components": int(components),
            }
            for k in range(1, limit + 1)
        )
    return pl.DataFrame(
        rows,
        schema={
            "neighbors": pl.Int64,
            "k": pl.Int64,
            "eigenvalue": pl.Float64,
            "next_eigenvalue": pl.Float64,
            "eigengap": pl.Float64,
            "graph_components": pl.Int64,
        },
    )


def gap_one_se(table):
    """Tibshirani et al.'s first K with Gap(K) >= Gap(K+1) - s(K+1)."""
    rows = table.sort("k").to_dicts()
    if not rows or any(row["gap"] is None for row in rows):
        return None
    for left, right in zip(rows, rows[1:]):
        if right["k"] != left["k"] + 1:
            return None
        if left["gap"] >= right["gap"] - right["gap_se"]:
            return left["k"]
    return rows[-1]["k"]


def estimate_gap(feature_matrix, registry, models, parameters):
    """Conditional gap diagnostic under both axis- and PCA-aligned uniform reference boxes.

    The representation is held fixed. These reference assumptions do not preserve
    financial feature marginals and are not a model of real bank customers.
    """
    frame = feature_matrix.collect().filter("eligible").sort("account_id")
    x = frame.select(registry["features"]).to_numpy()
    schema = {
        "reference": pl.String,
        "k": pl.Int64,
        "log_w": pl.Float64,
        "reference_log_w": pl.Float64,
        "gap": pl.Float64,
        "gap_se": pl.Float64,
        "mc_se_mean": pl.Float64,
        "valid_references": pl.Int64,
        "requested_references": pl.Int64,
    }
    trial_schema = {"reference": pl.String, "repeat": pl.Int64, "k": pl.Int64, "log_w": pl.Float64}
    if len(x) < 3 or not x.shape[1] or not np.isfinite(x).all() or parameters["gap_references"] < 2:
        return pl.DataFrame(schema=schema), pl.DataFrame(schema=trial_schema)
    observed = balanced_distances(x, x, registry)
    np.fill_diagonal(observed, 0)
    dispersion = {
        model["k"]: within_dispersion(observed, np.array(model["labels"])) for model in models
    }
    if any(value <= 0 for value in dispersion.values()):
        return pl.DataFrame(schema=schema), pl.DataFrame(schema=trial_schema)
    with threadpool_limits(limits=1):
        pca = PCA(svd_solver="full").fit(x)
    transformed = pca.transform(x)
    trials = []
    for ref_index, reference in enumerate(["axis_uniform", "pca_uniform"]):
        space = x if reference == "axis_uniform" else transformed
        low, high = space.min(axis=0), space.max(axis=0)
        for repeat in range(parameters["gap_references"]):
            rng = np.random.default_rng(parameters["seed"] + 50000 + ref_index * 10000 + repeat)
            simulated = rng.uniform(low, high, size=space.shape)
            simulated = (
                simulated if reference == "axis_uniform" else pca.inverse_transform(simulated)
            )
            distance = balanced_distances(simulated, simulated, registry)
            np.fill_diagonal(distance, 0)
            affinity = affinity_matrix(distance, parameters)
            components = connected_components(csr_matrix(affinity), directed=False)[0]
            for k in dispersion:
                if k == 1:
                    labels = np.zeros(len(x), int)
                elif components <= k <= len(np.unique(distance, axis=0)):
                    labels = np.array(
                        spectral_partition(distance, k, parameters, affinity)["labels"]
                    )
                else:
                    continue
                value = within_dispersion(distance, labels)
                if value > 0:
                    trials.append(
                        {
                            "reference": reference,
                            "repeat": repeat,
                            "k": k,
                            "log_w": float(np.log(value)),
                        }
                    )
            if (repeat + 1) % 25 == 0:
                logging.getLogger(__name__).info(
                    "Gap reference %s: %s/%s", reference, repeat + 1, parameters["gap_references"]
                )
    trial_frame = pl.DataFrame(trials, schema=trial_schema)
    rows = []
    for reference in ["axis_uniform", "pca_uniform"]:
        for k, value in dispersion.items():
            values = trial_frame.filter((pl.col("reference") == reference) & (pl.col("k") == k))[
                "log_w"
            ].to_numpy()
            valid = len(values) >= max(2, 0.8 * parameters["gap_references"])
            mean = float(np.mean(values)) if valid else None
            sd = float(np.std(values, ddof=1)) if valid else None
            rows.append(
                {
                    "reference": reference,
                    "k": k,
                    "log_w": float(np.log(value)),
                    "reference_log_w": mean,
                    "gap": mean - np.log(value) if valid else None,
                    "gap_se": sd * np.sqrt(1 + 1 / len(values)) if valid else None,
                    "mc_se_mean": sd / np.sqrt(len(values)) if valid else None,
                    "valid_references": len(values),
                    "requested_references": parameters["gap_references"],
                }
            )
    return pl.DataFrame(rows, schema=schema), trial_frame


def assess_cluster_count(metrics, stability, quality, spectrum, gaps, parameters):
    """Retain evidence and stable candidates; do not vote correlated indices or choose a final K."""
    table = (
        metrics.collect()
        .join(stability.collect(), on="k", how="left")
        .join(quality.collect().drop("silhouette"), on="k", how="left")
        .with_columns(
            (
                (pl.col("k") > 1)
                & pl.col("size_ok")
                & (pl.col("mean_ari") >= parameters["minimum_ari"])
                & (pl.col("minimum_mean_jaccard") >= parameters["minimum_jaccard"])
            )
            .fill_null(False)
            .alias("passes_stability_and_size")
        )
    )
    suggestions = {}
    for column, ascending in [
        ("silhouette", False),
        ("dunn", False),
        ("calinski_harabasz_euclidean", False),
        ("davies_bouldin_euclidean", True),
    ]:
        valid = table.filter(pl.col(column).is_finite())
        suggestions[column] = (
            int(valid.sort(column, descending=not ascending)["k"][0]) if valid.height else None
        )
    spectrum = spectrum.collect()
    suggestions["eigengap_by_neighbors"] = {
        str(count): int(group.sort("eigengap", descending=True)["k"][0])
        for (count,), group in spectrum.group_by("neighbors", maintain_order=True)
    }
    gaps = gaps.collect()
    suggestions["gap_1se_by_reference"] = {
        name: gap_one_se(gaps.filter(pl.col("reference") == name))
        for name in gaps["reference"].unique().sort().to_list()
    }
    return {
        "selected_k": None,
        "status": "pending_forecast_validation",
        "stable_candidates": table.filter("passes_stability_and_size")["k"].to_list(),
        "available_candidates": table["k"].to_list(),
        "diagnostic_suggestions": suggestions,
        "decision_rule": "No target count or automatic vote. Report conflicting structural evidence; choose final K using held-out per-client forecast errors.",
        "caveats": [
            "Diagnostics are conditional on the feature representation and similarity graph.",
            "CH/DB use a weighted Euclidean feature view, not the balanced clustering distance.",
            "Gap results are conditional on the stated uniform reference assumptions.",
            "Stability measures reproducibility; it does not establish a true or forecast-optimal K.",
        ],
    }, table
