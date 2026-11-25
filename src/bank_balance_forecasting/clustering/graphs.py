"""Strict visibility topology over fixed-size windows covering full observed history."""

import numpy as np
import polars as pl
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from ts2vg import HorizontalVG, NaturalVG

from .features import account_series, iqr, windows

METRICS = [
    "mean_degree",
    "max_degree_share",
    "degree_entropy",
    "transitivity",
    "temporal_span",
    "time_asymmetry",
]


def graph_descriptors(values, kind):
    """Build a time-directed NVG/HVG and return descriptors in METRICS order.

    Center and rescale the signal, retain strict visibility ties, and summarize
    degrees, degree entropy, transitivity, temporal span and time asymmetry.
    """
    normalized = (values - np.median(values)) / max(iqr(values), 0.01)
    graph = (NaturalVG if kind == "nvg" else HorizontalVG)(directed="left_to_right").build(
        normalized
    )
    n = len(values)
    edges = graph.edges
    degree = graph.degrees
    neighbors = [set() for _ in range(n)]
    for u, v in edges:
        neighbors[u].add(v)
        neighbors[v].add(u)
    triplets = np.sum(degree.astype(float) * (degree - 1) / 2)
    common = sum(len(neighbors[u] & neighbors[v]) for u, v in edges)
    counts = np.bincount(degree)
    incoming = np.bincount(graph.degrees_in, minlength=n)
    outgoing = np.bincount(graph.degrees_out, minlength=n)
    return [
        float(np.mean(degree)),
        float(np.max(degree) / max(1, n - 1)),
        float(entropy(counts) / np.log(n)),
        float(common / triplets) if triplets else 0.0,
        float(np.mean([v - u for u, v in edges]) / max(1, n - 1)),
        float(jensenshannon(incoming, outgoing, base=2) ** 2),
    ]


def extract_graph_features(balances, parameters, kind):
    """Summarize rolling visibility topology across each account's full history.

    Build the requested graph for levels and daily changes in complete calendar
    windows. Return descriptor medians, interquartile ranges and window counts;
    leave descriptors unavailable when no complete window exists.
    """
    rows = []
    for account, _, x in account_series(balances, parameters):
        row = {"account_id": account}
        for signal in ["levels", "changes"]:
            descriptors = [
                graph_descriptors(w if signal == "levels" else np.diff(w), kind)
                for w in windows(x, parameters["graph_window"], parameters["graph_stride"])
            ]
            row[f"meta_{kind}_{signal}_windows"] = len(descriptors)
            values = np.array(descriptors) if descriptors else np.full((1, len(METRICS)), np.nan)
            for j, metric in enumerate(METRICS):
                col = values[:, j]
                for suffix, value in [("median", np.median(col)), ("iqr", iqr(col))]:
                    row[f"{kind}__{signal}_{metric}_{suffix}"] = float(value)
        rows.append(row)
    return pl.DataFrame(rows).sort("account_id")


def extract_nvg_features(balances, parameters):
    """Describe geometric visibility of balances and changes across all observed history."""
    return extract_graph_features(balances, parameters, "nvg")


def extract_hvg_features(balances, parameters):
    """Describe order-based visibility, keeping repeated balances and strict ties."""
    return extract_graph_features(balances, parameters, "hvg")
