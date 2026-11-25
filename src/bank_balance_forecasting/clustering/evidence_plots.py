"""Figures for every cluster-count diagnostic, without a highlighted preferred K."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .evaluation import gap_one_se


def plot_count_tests(quality, samples, spectrum, gaps, stability):
    """Create interactive count diagnostics and a standalone summary figure.

    Return separation scores, per-client silhouettes, spectral evidence, gap and
    dispersion plots, plus a Matplotlib summary. Keep geometry and reference
    assumptions visible without highlighting a preferred cluster count.
    """
    quality, samples, spectrum, gaps = [
        table.collect() for table in [quality, samples, spectrum, gaps]
    ]
    scores = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=[
            "Dunn index ↑: balanced dissimilarities",
            "Negative silhouettes ↓: overlapping assignments",
            "Calinski–Harabasz ↑: Euclidean feature view",
            "Davies–Bouldin ↓: Euclidean feature view",
        ],
    )
    for row, col, metric in [
        (1, 1, "dunn"),
        (2, 1, "negative_silhouette_share"),
        (3, 1, "calinski_harabasz_euclidean"),
        (4, 1, "davies_bouldin_euclidean"),
    ]:
        scores.add_trace(
            go.Scatter(
                x=quality["k"].to_list(),
                y=quality[metric].to_list(),
                mode="lines+markers",
                name=metric,
            ),
            row=row,
            col=col,
        )
    scores.update_layout(
        title="Complementary internal diagnostics: not independent votes",
        height=1300,
        showlegend=False,
    )

    ks = sorted(samples["k"].unique().to_list())
    silhouettes = make_subplots(
        rows=max(1, int(np.ceil(len(ks) / 2))), cols=2, subplot_titles=[f"K={k}" for k in ks]
    )
    for panel, k in enumerate(ks):
        part = samples.filter(samples["k"] == k).sort("group", "silhouette")
        row, col = panel // 2 + 1, panel % 2 + 1
        for group, block in part.group_by("group", maintain_order=True):
            offset = part.filter(part["group"] < group[0]).height
            silhouettes.add_trace(
                go.Bar(
                    x=block["silhouette"].to_list(),
                    y=list(range(offset, offset + len(block))),
                    orientation="h",
                    showlegend=False,
                    marker_color=f"hsl({group[0] * 137.5 % 360},55%,50%)",
                    name=f"K={k}/G{group[0] + 1:02}",
                    hovertemplate="%{x:.3f}<extra>%{fullData.name}</extra>",
                ),
                row=row,
                col=col,
            )
        silhouettes.add_vline(
            x=float(part["silhouette"].mean()),
            line_dash="dash",
            line_color="black",
            row=row,
            col=col,
        )
        silhouettes.update_xaxes(range=[-1, 1], row=row, col=col)
        silhouettes.update_yaxes(showticklabels=False, row=row, col=col)
    silhouettes.update_layout(
        title="Silhouette distributions for every candidate: dashed lines are means",
        height=max(450, 340 * int(np.ceil(len(ks) / 2))),
        bargap=0,
        showlegend=False,
    )

    spectral = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=["Normalized-Laplacian eigenvalues", "Eigengaps λ(K+1) − λ(K)"],
    )
    for (neighbors,), part in spectrum.group_by("neighbors", maintain_order=True):
        part = part.sort("k")
        for col, metric in [(1, "eigenvalue"), (2, "eigengap")]:
            spectral.add_trace(
                go.Scatter(
                    x=part["k"].to_list(),
                    y=part[metric].to_list(),
                    mode="lines+markers",
                    name=f"{neighbors} neighbors",
                    legendgroup=str(neighbors),
                    showlegend=col == 1,
                ),
                row=col,
                col=1,
            )
    spectral.update_layout(
        title="Spectral structure and graph-setting sensitivity: K=1 included", height=900
    )

    gap_plot = go.Figure()
    for (reference,), part in gaps.group_by("reference", maintain_order=True):
        part = part.sort("k")
        suggestion = gap_one_se(part)
        gap_plot.add_trace(
            go.Scatter(
                x=part["k"].to_list(),
                y=part["gap"].to_list(),
                error_y={"type": "data", "array": part["gap_se"].to_list()},
                mode="lines+markers",
                name=f"{reference}, 1-SE suggestion={suggestion}",
                customdata=part.select("valid_references", "requested_references", "mc_se_mean")
                .to_numpy()
                .tolist(),
                hovertemplate="K=%{x}<br>Gap=%{y:.3f}<br>References %{customdata[0]}/%{customdata[1]}<br>MC SE(mean) %{customdata[2]:.4f}<extra>%{fullData.name}</extra>",
            )
        )
    gap_plot.update_layout(
        title="Conditional gap statistic: two uniform-reference assumptions",
        xaxis_title="K (including no-segmentation control)",
        yaxis_title="Gap ± s(K)",
        height=550,
    )

    dispersion = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=[
            "Observed within-group dispersion (elbow diagnostic)",
            "Observed versus reference log dispersion",
        ],
    )
    dispersion.add_trace(
        go.Scatter(
            x=quality["k"].to_list(),
            y=quality["within_dispersion"].to_list(),
            mode="lines+markers",
            name="Observed W",
        ),
        row=1,
        col=1,
    )
    valid = quality.filter(quality["within_dispersion"] > 0)
    dispersion.add_trace(
        go.Scatter(
            x=valid["k"].to_list(),
            y=np.log(valid["within_dispersion"].to_numpy()).tolist(),
            mode="lines+markers",
            name="Observed log W",
        ),
        row=2,
        col=1,
    )
    for (reference,), part in gaps.group_by("reference", maintain_order=True):
        dispersion.add_trace(
            go.Scatter(
                x=part["k"].to_list(),
                y=part["reference_log_w"].to_list(),
                mode="lines+markers",
                name=f"{reference}: E(log W)",
            ),
            row=2,
            col=1,
        )
    dispersion.update_layout(title="Dispersion curves: no manually chosen elbow", height=900)
    return (
        scores,
        silhouettes,
        spectral,
        gap_plot,
        dispersion,
        count_evidence_png(quality, spectrum, gaps, stability.collect()),
    )


def count_evidence_png(quality, spectrum, gaps, stability):
    """A standalone summary. Full distributions remain in the native interactive figures."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(12, 19), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(6, 2).ravel()
    for ax, column, title in zip(
        axes[:4],
        ["silhouette", "dunn", "calinski_harabasz_euclidean", "davies_bouldin_euclidean"],
        [
            "Mean silhouette ↑",
            "Dunn ↑",
            "Calinski–Harabasz ↑ (Euclidean view)",
            "Davies–Bouldin ↓ (Euclidean view)",
        ],
    ):
        ax.plot(quality["k"], quality[column], "o-")
        ax.set_title(title)
    for ax, column, title in zip(
        axes[4:6],
        ["mean_ari", "minimum_mean_jaccard"],
        ["Refitted resampling ARI ↑", "Weakest group's mean Jaccard ↑"],
    ):
        ax.plot(stability["k"], stability[column], "o-")
        ax.set_title(title)
    axes[4].fill_between(
        stability["k"].to_numpy(),
        stability["ari_q10"].to_numpy(),
        stability["ari_q90"].to_numpy(),
        alpha=0.15,
    )
    for (reference,), part in gaps.group_by("reference", maintain_order=True):
        axes[6].errorbar(part["k"], part["gap"], yerr=part["gap_se"], fmt="o-", label=reference)
        axes[11].plot(part["k"], part["reference_log_w"], "o-", label=reference)
    axes[6].set_title("Conditional Gap ± s(K)")
    axes[6].legend(fontsize=8)
    for (neighbors,), part in spectrum.group_by("neighbors", maintain_order=True):
        axes[7].plot(part["k"], part["eigengap"], "o-", label=f"neighbors={neighbors}")
        axes[10].plot(part["k"], part["eigenvalue"], "o-", label=f"neighbors={neighbors}")
    axes[7].set_title("Eigengap versus K")
    axes[7].legend(fontsize=7)
    axes[8].plot(quality["k"], quality["within_dispersion"], "o-")
    axes[8].set_title("Within-group dispersion: elbow diagnostic")
    axes[9].plot(quality["k"], quality["negative_silhouette_share"], "o-")
    axes[9].set_title("Negative-silhouette share ↓")
    axes[10].set_title("Normalized-Laplacian eigenvalues")
    part = quality.filter(quality["within_dispersion"] > 0)
    axes[11].plot(part["k"], np.log(part["within_dispersion"]), "o-", label="observed")
    axes[11].set_title("Observed/reference log dispersion")
    axes[11].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("Candidate K")
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Cluster-count evidence: no preferred or final K", fontsize=17)
    return figure
