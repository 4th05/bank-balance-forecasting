"""A local similarity graph between customers, distinct from each series' visibility graph."""

import warnings

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import SpectralClustering
from threadpoolctl import threadpool_limits

from .representation import distance_array


def neighbor_count(n, parameters):
    """Scale the neighbor count with cohort size and apply configured bounds.

    Never request more neighbors than other accounts or a negative count for
    an empty or single-account cohort.
    """
    return max(
        0,
        min(
            n - 1,
            parameters["max_neighbors"],
            max(parameters["min_neighbors"], int(np.ceil(n * parameters["neighbor_fraction"]))),
        ),
    )


def neighborhood(distance, count):
    """Exclude self explicitly, including when different customers have identical features."""
    if not count:
        return np.empty((len(distance), 0), int), np.ones(len(distance))
    search = distance.copy()
    np.fill_diagonal(search, np.inf)
    neighbors = np.argsort(search, axis=1, kind="stable")[:, :count]
    scales = np.maximum(distance[np.arange(len(distance)), neighbors[:, -1]], 1e-12)
    return neighbors, scales


def affinity_matrix(distance, parameters):
    """Convert local neighbor distances into symmetric, locally scaled affinities.

    Exclude self-edges, weight connections with an RBF kernel and average
    forward/reverse affinities to obtain an undirected customer graph.
    """
    neighbors, scales = neighborhood(distance, neighbor_count(len(distance), parameters))
    affinity = np.zeros_like(distance)
    for i, indices in enumerate(neighbors):
        affinity[i, indices] = np.exp(-(distance[i, indices] ** 2) / (scales[i] * scales[indices]))
    return (affinity + affinity.T) / 2


def build_customer_similarity_graph(distances, parameters):
    """Connect each client to nearby clients with locally scaled, symmetric affinities."""
    pairs = distances.collect()
    accounts = pairs["source_account"].unique().sort().to_numpy()
    affinity = affinity_matrix(distance_array(pairs), parameters)
    left, right = np.nonzero(affinity)
    return pl.DataFrame(
        {
            "source_account": accounts[left],
            "target_account": accounts[right],
            "affinity": affinity[left, right],
        },
        schema={"source_account": pl.String, "target_account": pl.String, "affinity": pl.Float64},
    )


def affinity_array(edges):
    """Restore a dense affinity matrix from edge records in sorted account order.

    Unlisted connections have zero affinity; an empty table returns no nodes.
    """
    frame = edges.collect() if isinstance(edges, pl.LazyFrame) else edges
    accounts = frame["source_account"].unique().sort().to_list()
    indices = {account: i for i, account in enumerate(accounts)}
    affinity = np.zeros((len(accounts), len(accounts)))
    if frame.height:
        left = [indices[a] for a in frame["source_account"]]
        right = [indices[a] for a in frame["target_account"]]
        affinity[left, right] = frame["affinity"].to_numpy()
    return affinity


def spectral_partition(distance, k, parameters, affinity=None):
    """Fit one supplied candidate K; this function does not estimate the final K.

    Check graph feasibility, obtain spectral memberships and canonicalize labels
    using within-group medoid indices. Return memberships, representatives and
    graph diagnostics; representative distance is not the clustering objective.
    """
    affinity = affinity_matrix(distance, parameters) if affinity is None else affinity
    components = connected_components(csr_matrix(affinity), directed=False)[0]
    if components > k or k > len(np.unique(distance, axis=0)):
        raise ValueError("Requested group count is incompatible with the similarity graph")
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.filterwarnings("ignore", message="Graph is not fully connected")
        labels = SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            assign_labels="cluster_qr",
            eigen_solver="arpack",
            random_state=parameters["seed"],
        ).fit_predict(csr_matrix(affinity))
    medoids = []
    for cluster in range(k):
        members = np.flatnonzero(labels == cluster)
        medoids.append(int(members[np.argmin(distance[np.ix_(members, members)].sum(axis=1))]))
    order = np.argsort(medoids)
    mapping = np.empty(k, int)
    mapping[order] = np.arange(k)
    labels = mapping[labels]
    medoids = np.array(medoids)[order]
    return {
        "k": k,
        "labels": labels.tolist(),
        "medoids": medoids.tolist(),
        "loss": float(distance[np.arange(len(distance)), medoids[labels]].sum()),
        "graph_components": int(components),
        "neighbors": neighbor_count(len(distance), parameters),
    }
