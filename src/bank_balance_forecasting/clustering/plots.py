"""Interactive diagnostic figures exposed through native Plotly Kedro-Viz previews."""


import numpy as np
import plotly.graph_objects as go
import polars as pl
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from plotly.subplots import make_subplots
from threadpoolctl import threadpool_limits

from .features import feature_label
from .representation import distance_array


def plot_selection(comparison, selected, parameters):
    """Compare candidate separation, refitted stability and sizes without choosing K."""
    table = comparison.collect().sort("k")
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[
            "Silhouette: one separation diagnostic",
            "Stability after refitting features, scales and graph",
            "Weakest cluster's mean Jaccard overlap",
            "Smallest group",
        ],
    )
    for row, column in enumerate(
        ["silhouette", "mean_ari", "minimum_mean_jaccard", "minimum_group_size"], 1
    ):
        fig.add_trace(
            go.Scatter(
                x=table["k"].to_list(), y=table[column].to_list(), mode="lines+markers", name=column
            ),
            row=row,
            col=1,
        )
    for column, fill in [("ari_q10", None), ("ari_q90", "tonexty")]:
        fig.add_trace(
            go.Scatter(
                x=table["k"].to_list(),
                y=table[column].to_list(),
                mode="lines",
                line={"width": 0},
                fill=fill,
                fillcolor="rgba(42,157,143,0.15)",
                hovertemplate=column + ": %{y:.3f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
    fig.add_hline(y=parameters["minimum_ari"], line_dash="dot", row=2, col=1)
    fig.add_hline(y=parameters["minimum_jaccard"], line_dash="dot", row=3, col=1)
    fig.update_layout(
        title="Candidate count diagnostics: no final K selected",
        height=900,
        showlegend=False,
    )
    fig.add_annotation(
        text="Stability shading: 10th–90th resample percentiles. K=1 is the unsegmented control",
        xref="paper",
        yref="paper",
        x=0,
        y=-0.12,
        showarrow=False,
    )
    fig.update_xaxes(title_text="Candidate number of groups", row=4, col=1)
    return fig


def _color(group):
    """Return a consistent tab20 color for a one-based group number within a K."""
    return to_hex(colormaps["tab20"]((group - 1) % 20))


def plot_clusters(distances, groups):
    """Show every candidate in the same 2D projection, with no preferred/default partition."""
    pairs = distances.collect()
    distance = distance_array(pairs)
    accounts = pairs["source_account"].unique().sort().to_list()
    indices = {account: i for i, account in enumerate(accounts)}
    n = len(distance)
    coordinates = np.zeros((n, 2))
    if n:
        center = np.eye(n) - np.ones((n, n)) / n
        with threadpool_limits(limits=1):
            eigenvalues, eigenvectors = np.linalg.eigh(-0.5 * center @ distance**2 @ center)
        order = np.argsort(eigenvalues)[::-1][:2]
        coordinates[:, : len(order)] = eigenvectors[:, order] * np.sqrt(
            np.maximum(eigenvalues[order], 0)
        )
    frame = groups.collect()
    ks = sorted(frame["candidate_k"].unique().to_list())
    rows = max(1, int(np.ceil(len(ks) / 2)))
    scatter = make_subplots(
        rows=rows, cols=2, subplot_titles=[f"K={k}" + (": baseline" if k == 1 else "") for k in ks]
    )
    figure = Figure(figsize=(13, 3.4 * rows), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(rows, 2, squeeze=False)
    for panel, k in enumerate(ks):
        row, col = panel // 2 + 1, panel % 2 + 1
        ax = axes[row - 1, col - 1]
        members = frame.filter((frame["candidate_k"] == k) & (frame["group_id"] != "G00"))
        for (group,), part in members.group_by("group_id", maintain_order=True):
            positions = [indices[a] for a in part["account_id"]]
            color = "#adb5bd" if k == 1 else _color(int(group[1:]))
            scatter.add_trace(
                go.Scatter(
                    x=coordinates[positions, 0].tolist(),
                    y=coordinates[positions, 1].tolist(),
                    mode="markers",
                    marker={"size": 5, "color": color, "opacity": 0.7},
                    text=part["account_id"].to_list(),
                    name=f"K={k}/{group}",
                    hovertemplate="Account %{text}<extra>%{fullData.name}</extra>",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            ax.scatter(
                coordinates[positions, 0], coordinates[positions, 1], s=9, color=color, alpha=0.7
            )
        ax.set_title(f"K={k}" + (": baseline" if k == 1 else ""), fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    for panel in range(len(ks), rows * 2):
        axes.ravel()[panel].set_visible(False)
    figure.suptitle("Candidate partitions: same approximate projection. No selected K")
    scatter.update_layout(
        title="All candidate partitions: same approximate 2D distance projection",
        height=rows * 340,
        showlegend=False,
    )
    heatmap = go.Figure(
        go.Heatmap(z=distance.tolist(), x=accounts, y=accounts, colorscale="Viridis")
    )
    heatmap.update_layout(
        title="Actual pairwise dissimilarities: account order, no chosen partition", height=650
    )
    sizes = frame.group_by("candidate_k", "group_id").len()
    size_plot = go.Figure()
    for group in sorted(sizes["group_id"].unique()):
        part = sizes.filter(sizes["group_id"] == group).sort("candidate_k")
        size_plot.add_trace(
            go.Bar(x=part["candidate_k"].to_list(), y=part["len"].to_list(), name=group)
        )
    size_plot.update_layout(
        title="Group sizes for all candidates: group IDs are specific to each K",
        barmode="stack",
        xaxis_title="Candidate K",
        height=550,
    )
    return scatter, heatmap, size_plot, figure


def explain_groups(values, feature_matrix, groups, balances, assessment, registry, parameters):
    """Describe each candidate separately. No group count is promoted to the final result."""
    raw = values.collect()
    assignments = groups.collect()
    frame = raw.join(assignments, on="account_id")
    features = registry["features"]
    scaled = feature_matrix.collect().join(
        assignments.select("account_id", "candidate_k", "group_id"), on="account_id"
    )
    centers = (
        scaled.group_by("candidate_k", "group_id")
        .agg([pl.col(c).median() for c in features])
        .sort("candidate_k", "group_id")
    )
    baseline = (
        np.nanmedian(feature_matrix.collect().select(features).to_numpy(), axis=0)
        if features
        else np.array([])
    )
    shifts = (
        centers.select(features).to_numpy() - baseline if features else np.empty((len(centers), 0))
    )
    important = np.argsort(-np.nanmax(abs(shifts), axis=0))[:18] if features else np.array([], int)
    cards = []
    for (k, group), members in frame.group_by("candidate_k", "group_id", maintain_order=True):
        movement = members["activity__movement_rate"].median()
        negative = members["state__negative_share"].median()
        activity = (
            "Higher activity"
            if movement is not None and movement > raw["activity__movement_rate"].median()
            else "Lower activity"
        )
        sign = (
            "frequently negative"
            if negative is not None and negative > 0.5
            else "mostly nonnegative"
            if negative is not None and negative < 0.05
            else "mixed signs"
        )
        label = (
            "Unsegmented baseline"
            if k == 1
            else "Insufficient observed history/features"
            if group == "G00"
            else f"{activity}, {sign}"
        )
        cards.append(
            {
                "candidate_k": k,
                "group_id": group,
                "accounts": members.height,
                "representative_account": members["representative_account"][0],
                "review_assignments": int(members["review_assignment"].sum()),
                "median_movement_rate": movement,
                "median_negative_share": negative,
                "group_label": label,
            }
        )
    cards = pl.DataFrame(cards)
    ks = sorted(assignments["candidate_k"].unique().to_list())
    profiles = go.Figure()
    for k in ks:
        part = centers.filter(centers["candidate_k"] == k)
        z = part.select(features).to_numpy()[:, important] if features else np.empty((len(part), 0))
        profiles.add_trace(
            go.Heatmap(
                z=z.tolist(),
                x=[feature_label(features[i]) for i in important],
                y=part["group_id"].to_list(),
                zmin=0,
                zmax=1,
                colorscale="Viridis",
                visible=k == ks[0],
                name=f"K={k}",
            )
        )
    profiles.update_layout(
        title="Candidate profiles: choose a candidate K",
        height=750,
        margin={"b": 230},
        updatemenus=[
            {
                "buttons": [
                    {
                        "label": f"K={k}",
                        "method": "update",
                        "args": [{"visible": [j == i for j in range(len(ks))]}],
                    }
                    for i, k in enumerate(ks)
                ]
            }
        ],
    )
    profiles.update_xaxes(tickangle=-40, tickfont_size=11)
    series = balances.collect().partition_by("account_id", as_dict=True)
    histories = go.Figure()
    trace_ks = []
    for card in cards.to_dicts():
        account = card["representative_account"]
        if account is None:
            continue
        data = series[(account,)].sort("date")
        x = data["balance"].to_numpy()
        finite = x[np.isfinite(x)]
        if not len(finite):
            continue
        scale = max(float(np.percentile(finite, 75) - np.percentile(finite, 25)), 0.01)
        k = card["candidate_k"]
        trace_ks.append(k)
        histories.add_trace(
            go.Scatter(
                x=data["date"].to_list(),
                y=((x - np.median(finite)) / scale).tolist(),
                name=f"K={k}/{card['group_id']} / {account}",
                visible=k == ks[0],
            )
        )
    histories.update_layout(
        title="Full-history representatives: choose candidate K",
        height=650,
        updatemenus=[
            {
                "buttons": [
                    {
                        "label": f"K={k}",
                        "method": "update",
                        "args": [{"visible": [value == k for value in trace_ks]}],
                    }
                    for k in ks
                ]
            }
        ],
    )
    summary = {
        **assessment,
        "as_of_date": str(parameters["as_of_date"]),
        "accounts": raw.height,
        "retained_features": len(features),
        "segmentation_method": assignments["segmentation_method"][0],
        "groups": cards.to_dicts(),
    }
    return cards, summary, profiles, histories


def plot_ablation(table, neighborhoods):
    """Plot feature-view agreement, refitted stability and neighborhood sensitivity.

    Show every available candidate K so changes in representation or graph
    settings remain visible instead of being evaluated only at a chosen count.
    """
    frame, neighbors = table.collect(), neighborhoods.collect()
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=[
            "View agreement by K",
            "Refitted stability by K",
            "Neighborhood sensitivity by K",
        ],
    )
    for index, ((name,), part) in enumerate(frame.group_by("feature_set", maintain_order=True)):
        part = part.filter(part["k"] > 1).sort("k")
        for col, metric in [(1, "agreement_with_primary"), (2, "mean_resample_ari")]:
            fig.add_trace(
                go.Scatter(
                    x=part["k"].to_list(),
                    y=part[metric].to_list(),
                    mode="lines+markers",
                    name=name,
                    legendgroup=name,
                    line_color=_color(2 * index + 1),
                    showlegend=col == 1,
                ),
                row=col,
                col=1,
            )
    for index, ((count,), part) in enumerate(neighbors.group_by("neighbors", maintain_order=True)):
        part = part.sort("k")
        fig.add_trace(
            go.Scatter(
                x=part["k"].to_list(),
                y=part["agreement_with_primary"].to_list(),
                mode="lines+markers",
                name=f"{count} neighbors",
                line_color=_color(2 * index + 9),
            ),
            row=3,
            col=1,
        )
    fig.update_layout(
        title="Feature-view and neighborhood sensitivity: all candidate counts", height=1200
    )
    return fig
