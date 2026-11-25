"""Candidate spectral partitions, refitted stability and explicit-K saved assignments."""

import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score, silhouette_score

from .representation import (
    balanced_distances,
    distance_array,
    eligible_rows,
    fit_feature_space,
    transform_features,
)
from .spectral import (
    affinity_array,
    affinity_matrix,
    neighbor_count,
    neighborhood,
    spectral_partition,
)

METHOD = "self_tuning_spectral_v2"


def fit_candidates(distances, similarity_graph, parameters):
    """Retain the unsegmented baseline and every feasible K, without a target count."""
    distance, affinity = distance_array(distances), affinity_array(similarity_graph)
    n = len(distance)
    minimum = max(parameters["min_group_size"], int(np.ceil(parameters["min_group_fraction"] * n)))
    unique = len(np.unique(distance, axis=0)) if n else 0
    components = connected_components(csr_matrix(affinity), directed=False)[0] if n > 1 else n
    models, rows = [], []
    for k in range(1, min(parameters["max_groups"], max(1, n // minimum), unique) + 1):
        if k > 1 and k < components:
            continue
        model = (
            spectral_partition(distance, k, parameters, affinity)
            if k > 1
            else {
                "k": 1,
                "labels": [0] * n,
                "medoids": [int(np.argmin(distance.sum(axis=1)))],
                "loss": float(distance.sum(axis=1).min()),
                "graph_components": components,
                "neighbors": neighbor_count(n, parameters),
            }
        )
        labels = np.array(model["labels"])
        sizes = np.bincount(labels, minlength=k)
        models.append(model)
        rows.append(
            {
                "k": k,
                "loss": model["loss"],
                "silhouette": float(silhouette_score(distance, labels, metric="precomputed"))
                if k > 1
                else None,
                "minimum_group_size": int(sizes.min()),
                "size_ok": bool(sizes.min() >= minimum),
                "graph_components": model["graph_components"],
                "neighbors": model["neighbors"],
            }
        )
    schema = {
        "k": pl.Int64,
        "loss": pl.Float64,
        "silhouette": pl.Float64,
        "minimum_group_size": pl.Int64,
        "size_ok": pl.Boolean,
        "graph_components": pl.Int64,
        "neighbors": pl.Int64,
    }
    return models, pl.DataFrame(rows, schema=schema)


def align_labels(reference, labels, k):
    """Match candidate labels to reference groups by maximum account overlap.

    Use a one-to-one Hungarian assignment to remove arbitrary label permutations.
    Return aligned labels and the Jaccard overlap for each reference group.
    """
    intersection = np.zeros((k, k), int)
    np.add.at(intersection, (reference, labels), 1)
    a, b = linear_sum_assignment(-intersection)
    mapping = np.zeros(k, int)
    mapping[b] = a
    aligned = mapping[labels]
    jaccard = []
    for cluster in range(k):
        left, right = reference == cluster, aligned == cluster
        union = (left | right).sum()
        jaccard.append(float((left & right).sum() / union) if union else 1.0)
    return aligned, jaccard


def assess_stability(values, feature_matrix, models, parameters):
    """Refit feature pruning, scales and the customer graph on common account subsamples."""
    accounts = (
        feature_matrix.collect().filter("eligible").sort("account_id")["account_id"].to_list()
    )
    frame = values.collect().filter(pl.col("account_id").is_in(accounts)).sort("account_id")
    n = len(accounts)
    models = [model for model in models if model["k"] > 1]
    results = {m["k"]: {"ari": [], "jaccard": [], "votes": np.zeros(n)} for m in models}
    seen = np.zeros(n)
    trials = []
    if parameters["resamples"] < 1 or not 0 < parameters["sample_fraction"] <= 1:
        raise ValueError("Use at least one resample and a sample_fraction in (0, 1]")
    for repeat in range(parameters["resamples"] if models else 0):
        rng = np.random.default_rng(parameters["seed"] + repeat)
        size = min(n, max(max(results) + 1, int(n * parameters["sample_fraction"])))
        subset = np.sort(rng.choice(n, size, replace=False))
        sample = frame[subset]
        registry, scaled, eligible = fit_feature_space(sample, parameters)
        distance = balanced_distances(scaled, scaled, registry)
        np.fill_diagonal(distance, 0)
        affinity = affinity_matrix(distance, parameters)
        components = connected_components(csr_matrix(affinity), directed=False)[0]
        unique = len(np.unique(distance, axis=0))
        seen[subset] += 1
        for model in models:
            k, state = model["k"], results[model["k"]]
            reference = np.array(model["labels"])[subset]
            valid = bool(eligible.all() and components <= k <= unique)
            if valid:
                fitted = spectral_partition(distance, k, parameters, affinity)
                aligned, jaccard = align_labels(reference, np.array(fitted["labels"]), k)
                ari = float(adjusted_rand_score(reference, aligned))
                state["votes"][subset] += aligned == reference
                minimum_size = int(np.bincount(aligned, minlength=k).min())
            else:
                ari, jaccard, minimum_size = 0.0, [0.0] * k, 0
            state["ari"].append(ari)
            state["jaccard"].append(jaccard)
            trials.append(
                {
                    "k": k,
                    "resample": repeat,
                    "ari": ari,
                    "minimum_jaccard": min(jaccard),
                    "minimum_group_size": minimum_size,
                    "valid": valid,
                }
            )
    rows, account_rows = [], []
    for k, state in results.items():
        scores = state["ari"]
        rows.append(
            {
                "k": k,
                "mean_ari": float(np.mean(scores)),
                "std_ari": float(np.std(scores)),
                "ari_q10": float(np.quantile(scores, 0.1)),
                "ari_q90": float(np.quantile(scores, 0.9)),
                "minimum_mean_jaccard": float(np.mean(state["jaccard"], axis=0).min()),
            }
        )
        agreement = np.divide(state["votes"], seen, out=np.zeros(n), where=seen > 0)
        account_rows.extend(
            {"account_id": a, "k": k, "assignment_stability": float(s)}
            for a, s in zip(accounts, agreement)
        )
    return (
        pl.DataFrame(
            rows,
            schema={
                "k": pl.Int64,
                "mean_ari": pl.Float64,
                "std_ari": pl.Float64,
                "ari_q10": pl.Float64,
                "ari_q90": pl.Float64,
                "minimum_mean_jaccard": pl.Float64,
            },
        ),
        pl.DataFrame(
            account_rows,
            schema={"account_id": pl.String, "k": pl.Int64, "assignment_stability": pl.Float64},
        ),
        pl.DataFrame(
            trials,
            schema={
                "k": pl.Int64,
                "resample": pl.Int64,
                "ari": pl.Float64,
                "minimum_jaccard": pl.Float64,
                "minimum_group_size": pl.Int64,
                "valid": pl.Boolean,
            },
        ),
    )


def predict_groups(values, model, candidate_k=None):
    """Replay exact fitted memberships; use explicitly flagged neighbor transfer for new observations."""
    if model.get("version") != 3:
        raise ValueError("Legacy selected-K model: rerun Stage 1 to export candidate partitions")
    if candidate_k is None:
        raise ValueError(
            "No final K has been selected; pass candidate_k explicitly to inspect a partition"
        )
    if str(candidate_k) not in model["partitions"]:
        raise ValueError("Requested candidate_k is not present in this fitted model")
    partition = model["partitions"][str(candidate_k)]
    frame = (values.collect() if isinstance(values, pl.LazyFrame) else values).sort("account_id")
    registry = model["representation"]
    scaled = transform_features(frame, registry)
    eligible = eligible_rows(frame, registry, scaled, registry["parameters"])
    training = np.array(
        [[np.nan if v is None else v for v in row] for row in model["training_features"]],
        dtype=float,
    ).reshape(len(model["training_accounts"]), len(registry["features"]))
    distances = balanced_distances(scaled, training, registry)
    training_labels = np.array(partition["labels"], int)
    fitted_rows = {account: i for i, account in enumerate(model["training_accounts"])}
    k = candidate_k
    rows = []
    for i, account in enumerate(frame["account_id"]):
        row = {
            "account_id": account,
            "candidate_k": k,
            "group_id": "G00",
            "representative_account": None,
            "distance_to_medoid": None,
            "assignment_margin": None,
            "assignment_status": "insufficient_history_or_features",
        }
        if not eligible[i] or not k:
            rows.append(row)
            continue
        fitted = fitted_rows.get(account)
        unchanged = fitted is not None and np.array_equal(
            scaled[i], training[fitted], equal_nan=True
        )
        order = np.argsort(distances[i], kind="stable")
        if unchanged:
            order = order[order != fitted]
        nearest = order[: max(1, model["neighbors"])]
        local_scale = max(float(distances[i, nearest[-1]]), 1e-12) if len(nearest) else 1.0
        log_weights = -(distances[i, nearest] ** 2) / (
            local_scale * np.array(model["local_scales"])[nearest]
        )
        weights = np.exp(log_weights - log_weights.max()) if len(nearest) else np.array([])
        support = np.bincount(training_labels[nearest], weights=weights, minlength=k)
        cluster = int(training_labels[fitted]) if unchanged else int(np.argmax(support))
        alternative = float(np.max(np.delete(support, cluster))) if k > 1 else 0.0
        margin = (
            float((support[cluster] - alternative) / max(support[cluster], alternative, 1e-12))
            if k > 1
            else None
        )
        row.update(
            {
                "group_id": f"G{cluster + 1:02d}",
                "representative_account": partition["medoid_accounts"][cluster],
                "distance_to_medoid": float(distances[i, partition["medoids"][cluster]]),
                "assignment_margin": margin,
                "assignment_status": "unsegmented_baseline"
                if k == 1
                else "neighbor_transfer"
                if not unchanged
                else "candidate_assigned"
                if partition["passes_stability_and_size"]
                else "unstable_or_small_candidate",
            }
        )
        rows.append(row)
    return pl.DataFrame(
        rows,
        schema={
            "account_id": pl.String,
            "candidate_k": pl.Int64,
            "group_id": pl.String,
            "representative_account": pl.String,
            "distance_to_medoid": pl.Float64,
            "assignment_margin": pl.Float64,
            "assignment_status": pl.String,
        },
    )


def assign_accounts(
    values, feature_matrix, registry, models, stability, stability_metrics, distances, parameters
):
    """Export every candidate, sharing one fitted representation; no default K is selected."""
    eligible = feature_matrix.collect().filter("eligible").sort("account_id")
    x = (
        eligible.select(registry["features"]).to_numpy()
        if registry["features"]
        else np.empty((eligible.height, 0))
    )
    distance = distance_array(distances)
    count = neighbor_count(len(distance), parameters)
    _, scales = neighborhood(distance, count)
    accounts = eligible["account_id"].to_list()
    summaries = {row["k"]: row for row in stability_metrics.collect().to_dicts()}
    minimum = max(
        parameters["min_group_size"], int(np.ceil(parameters["min_group_fraction"] * len(accounts)))
    )
    partitions = {}
    for fitted in models:
        k = fitted["k"]
        validation = summaries.get(k, {})
        supported = (
            k > 1
            and min(np.bincount(fitted["labels"], minlength=k)) >= minimum
            and validation.get("mean_ari", 0) >= parameters["minimum_ari"]
            and validation.get("minimum_mean_jaccard", 0) >= parameters["minimum_jaccard"]
        )
        partitions[str(k)] = {
            **fitted,
            "passes_stability_and_size": bool(supported),
            "medoid_accounts": [accounts[i] for i in fitted["medoids"]],
        }
    if not partitions:
        partitions["0"] = {
            "k": 0,
            "labels": [],
            "medoids": [],
            "medoid_accounts": [],
            "passes_stability_and_size": False,
        }
    model = {
        "version": 3,
        "method": METHOD,
        "selected_k": None,
        "decision_status": "pending_forecast_validation",
        "representation": registry,
        "partitions": partitions,
        "training_accounts": accounts,
        "training_features": [[float(v) if np.isfinite(v) else None for v in row] for row in x],
        "neighbors": count,
        "local_scales": scales.tolist(),
        "out_of_sample_policy": "explicit candidate_k and weighted neighbor-label transfer; transfer always needs review",
    }
    assigned = pl.concat([predict_groups(values, model, int(k)) for k in partitions])
    scores = stability.collect().rename({"k": "candidate_k"})
    assigned = assigned.join(scores, on=["account_id", "candidate_k"], how="left").with_columns(
        (
            (pl.col("assignment_margin") < parameters["low_margin"])
            | (pl.col("assignment_stability") < parameters["low_stability"])
            | (pl.col("assignment_status") != "candidate_assigned")
        )
        .fill_null(True)
        .alias("review_assignment"),
        pl.lit(str(parameters["as_of_date"])).alias("as_of_date"),
        pl.lit(METHOD).alias("segmentation_method"),
    )
    return assigned, model
