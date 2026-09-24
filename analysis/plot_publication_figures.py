"""Generate publication-quality experiment figures without rerunning training.

The numerical values below are the existing fold summaries recorded in
实验数据.md. This script only redraws them with a consistent visual language.
By default it writes PDF and SVG files to a separate directory:

    python analysis/plot_publication_figures.py

To replace the figures referenced by the paper after inspecting the output:

    python analysis/plot_publication_figures.py \
        --output-dir /path/to/iclr2027/支撑材料 --formats pdf svg
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SUPPORTED_FORMATS = {"pdf", "svg", "png"}

# Okabe-Ito inspired, color-blind-friendly palette.
BLUE = "#0072B2"
VERMILLION = "#D55E00"
TEAL = "#009E73"
GOLD = "#E69F00"
SLATE = "#667085"
LIGHT_GRID = "#D9DEE7"
TEXT = "#20242A"

METRICS = ("AUC", "AUPR", "F1", "ACC")
METRIC_COLORS = {
    "AUC": BLUE,
    "AUPR": VERMILLION,
    "F1": TEAL,
    "ACC": SLATE,
}
METRIC_MARKERS = {"AUC": "o", "AUPR": "s", "F1": "D", "ACC": "^"}

ABLATION_ORDER = ("Full model", "w/o CCG", "w/o GSA", "w/o GSK")
ABLATION_COLORS = (TEXT, "#4C7899", "#D9A441", "#C96A88")
ABLATION_DATA = {
    "ONEIL": {
        "Full model": ([0.9692, 0.9690, 0.9160, 0.9163], [0.0034, 0.0054, 0.0058, 0.0060]),
        "w/o CCG": ([0.9679, 0.9666, 0.9138, 0.9114], [0.0056, 0.0066, 0.0100, 0.0105]),
        "w/o GSA": ([0.9665, 0.9641, 0.9094, 0.9069], [0.0032, 0.0034, 0.0080, 0.0082]),
        "w/o GSK": ([0.9164, 0.9224, 0.8405, 0.8398], [0.0061, 0.0076, 0.0063, 0.0056]),
    },
    "ALMANAC": {
        "Full model": ([0.9232, 0.9365, 0.8565, 0.8592], [0.0037, 0.0027, 0.0062, 0.0051]),
        "w/o CCG": ([0.9235, 0.9364, 0.8602, 0.8616], [0.0042, 0.0018, 0.0065, 0.0080]),
        "w/o GSA": ([0.9213, 0.9349, 0.8575, 0.8601], [0.0016, 0.0013, 0.0060, 0.0056]),
        "w/o GSK": ([0.8446, 0.8546, 0.7695, 0.7515], [0.0093, 0.0107, 0.0101, 0.0265]),
    },
}

FRACTIONS = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
LEARNING_FULL = {
    "mean": np.array([0.8970, 0.9416, 0.9549, 0.9616, 0.9677]),
    "std": np.array([0.0121, 0.0053, 0.0043, 0.0051, 0.0041]),
}
LEARNING_WO_GSK = {
    "mean": np.array([0.8659, 0.8927, 0.9099, 0.9121, 0.9170]),
    "std": np.array([0.0061, 0.0057, 0.0044, 0.0071, 0.0042]),
}

MOTIF_COMBOS = ("0,2,5", "0,1,5", "0,1,3,5", "0,2,4,5", "0,1,2,3,4,5", "0,2,3,4,5")
MOTIF_DATA = {
    "AUC": (
        np.array([0.9707, 0.9697, 0.9701, 0.9718, 0.9702, 0.9714]),
        np.array([0.0077, 0.0089, 0.0071, 0.0075, 0.0073, 0.0072]),
    ),
    "AUPR": (
        np.array([0.9689, 0.9685, 0.9689, 0.9709, 0.9697, 0.9712]),
        np.array([0.0108, 0.0119, 0.0105, 0.0113, 0.0096, 0.0091]),
    ),
}

DROPOUTS = np.array([0.0, 0.1, 0.2])
DROPOUT_DATA = {
    "AUC": (
        np.array([0.9682, 0.9678, 0.9692]),
        np.array([0.0043, 0.0041, 0.0034]),
    ),
    "AUPR": (
        np.array([0.9674, 0.9658, 0.9690]),
        np.array([0.0059, 0.0071, 0.0054]),
    ),
}


def apply_publication_style() -> None:
    """Apply a compact style that remains legible after paper scaling."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.4,
            "lines.markersize": 4.5,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def _validate_formats(formats: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(str(fmt).lower().lstrip(".") for fmt in formats)
    unsupported = sorted(set(normalized) - SUPPORTED_FORMATS)
    if unsupported:
        raise ValueError(f"Unsupported format(s): {', '.join(unsupported)}")
    if not normalized:
        raise ValueError("At least one output format is required")
    return normalized


def save_figure(
    fig: plt.Figure,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Save one figure in each requested format and close it."""
    normalized = _validate_formats(formats)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for fmt in normalized:
        path = output_base.with_suffix(f".{fmt}")
        save_kwargs = {"dpi": 300} if fmt == "png" else {}
        fig.savefig(path, format=fmt, **save_kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def _clean_axis(ax: plt.Axes, grid_axis: str = "x") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, color=LIGHT_GRID, linewidth=0.6, alpha=0.75)
    ax.set_axisbelow(True)


def make_ablation_figure(dataset: str) -> plt.Figure:
    """Render GeneSyn-HGNN ablations for a half-width paper slot."""
    data = ABLATION_DATA[dataset]
    display_metrics = ("AUC", "AUPR", "ACC", "F1")
    metric_indices = (0, 1, 3, 2)
    x = np.arange(len(display_metrics), dtype=float)
    bar_width = 0.19
    offsets = (np.arange(len(ABLATION_ORDER)) - 1.5) * bar_width
    fig, ax = plt.subplots(figsize=(3.35, 2.35))

    all_lower: list[float] = []
    all_upper: list[float] = []
    for variant_index, variant in enumerate(ABLATION_ORDER):
        means = np.array([data[variant][0][idx] for idx in metric_indices])
        stds = np.array([data[variant][1][idx] for idx in metric_indices])
        all_lower.extend(means - stds)
        all_upper.extend(means + stds)
        ax.bar(
            x + offsets[variant_index],
            means,
            width=bar_width,
            yerr=stds,
            color=ABLATION_COLORS[variant_index],
            edgecolor="white",
            linewidth=0.35,
            capsize=1.7,
            error_kw={"ecolor": SLATE, "elinewidth": 0.7, "capthick": 0.7},
            label=variant,
            zorder=3,
        )

    lower = max(0.0, np.floor((min(all_lower) - 0.01) * 20.0) / 20.0)
    upper = min(1.0, np.ceil((max(all_upper) + 0.01) * 20.0) / 20.0)
    ax.set_ylim(lower, upper)
    ax.set_xlim(-0.55, len(display_metrics) - 0.45)
    ax.set_xticks(x)
    ax.set_xticklabels(display_metrics)
    ax.set_ylabel("Performance")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        handlelength=0.9,
        handletextpad=0.4,
        columnspacing=0.9,
    )
    _clean_axis(ax, grid_axis="y")
    fig.subplots_adjust(left=0.16, right=0.99, top=0.75, bottom=0.16)
    return fig


def _interpolate_fraction(target: float, fractions: np.ndarray, values: np.ndarray) -> float:
    for left in range(len(values) - 1):
        y0, y1 = values[left], values[left + 1]
        if y0 <= target <= y1:
            weight = (target - y0) / (y1 - y0)
            return float(fractions[left] + weight * (fractions[left + 1] - fractions[left]))
    raise ValueError(f"Target {target} is outside the learning curve")


def make_learning_curve_figure() -> plt.Figure:
    fig, (ax_curve, ax_delta) = plt.subplots(
        1, 2, figsize=(6.7, 2.35), gridspec_kw={"width_ratios": [2.25, 1.0]}
    )

    ax_curve.plot(
        FRACTIONS,
        LEARNING_FULL["mean"],
        marker="o",
        color=BLUE,
        label="Full model",
    )
    ax_curve.fill_between(
        FRACTIONS,
        LEARNING_FULL["mean"] - LEARNING_FULL["std"],
        LEARNING_FULL["mean"] + LEARNING_FULL["std"],
        color=BLUE,
        alpha=0.14,
        linewidth=0,
    )
    ax_curve.plot(
        FRACTIONS,
        LEARNING_WO_GSK["mean"],
        marker="s",
        color=VERMILLION,
        label="w/o GSK",
    )
    ax_curve.fill_between(
        FRACTIONS,
        LEARNING_WO_GSK["mean"] - LEARNING_WO_GSK["std"],
        LEARNING_WO_GSK["mean"] + LEARNING_WO_GSK["std"],
        color=VERMILLION,
        alpha=0.13,
        linewidth=0,
    )
    target = 0.90
    full_fraction = _interpolate_fraction(target, FRACTIONS, LEARNING_FULL["mean"])
    ablated_fraction = _interpolate_fraction(target, FRACTIONS, LEARNING_WO_GSK["mean"])
    ax_curve.axhline(target, color=SLATE, linestyle=(0, (3, 2)), linewidth=0.85)
    ax_curve.scatter(
        [full_fraction, ablated_fraction],
        [target, target],
        s=20,
        facecolor="white",
        edgecolor=[BLUE, VERMILLION],
        linewidth=1.0,
        zorder=5,
    )
    ax_curve.annotate(
        f"AUC 0.90: {full_fraction:.0%} vs {ablated_fraction:.0%}",
        xy=(ablated_fraction, target),
        xytext=(0.53, 0.878),
        textcoords="data",
        ha="center",
        fontsize=7.0,
        color=SLATE,
        arrowprops={"arrowstyle": "-", "color": SLATE, "lw": 0.65},
    )
    ax_curve.set_xlabel("Training data retained")
    ax_curve.set_ylabel("Test AUC")
    ax_curve.set_xticks(FRACTIONS)
    ax_curve.set_xticklabels([f"{fraction:.0%}" for fraction in FRACTIONS])
    ax_curve.legend(loc="lower right", frameon=False, handlelength=1.6)
    ax_curve.text(-0.13, 1.03, "(a)", transform=ax_curve.transAxes, fontweight="bold")
    _clean_axis(ax_curve, grid_axis="both")

    delta = LEARNING_FULL["mean"] - LEARNING_WO_GSK["mean"]
    y_positions = np.arange(len(FRACTIONS))
    ax_delta.hlines(y_positions, 0, delta, color=TEAL, linewidth=2.2, alpha=0.72)
    ax_delta.scatter(delta, y_positions, color=TEAL, edgecolor="white", linewidth=0.5, zorder=3)
    for position, value in zip(y_positions, delta):
        ax_delta.annotate(
            f"{value:.3f}",
            (value, position),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=6.8,
            color=TEXT,
        )
    ax_delta.set_xlabel(r"$\Delta$AUC")
    ax_delta.set_ylabel("Training data retained")
    ax_delta.set_yticks(y_positions)
    ax_delta.set_yticklabels([f"{fraction:.0%}" for fraction in FRACTIONS])
    ax_delta.set_xlim(0, max(delta) + 0.012)
    ax_delta.invert_yaxis()
    ax_delta.text(-0.13, 1.03, "(b)", transform=ax_delta.transAxes, fontweight="bold")
    ax_delta.set_title("Gain from GSK", pad=4, color=SLATE)
    _clean_axis(ax_delta, grid_axis="x")

    fig.tight_layout(w_pad=1.0)
    return fig


def make_motif_figure() -> plt.Figure:
    x = np.arange(len(MOTIF_COMBOS))
    fig, ax = plt.subplots(figsize=(6.55, 2.15))
    offsets = {"AUC": -0.055, "AUPR": 0.055}
    for metric in ("AUC", "AUPR"):
        means, stds = MOTIF_DATA[metric]
        ax.errorbar(
            x + offsets[metric],
            means,
            yerr=stds,
            fmt=METRIC_MARKERS[metric] + "-",
            color=METRIC_COLORS[metric],
            ecolor=METRIC_COLORS[metric],
            capsize=2.5,
            elinewidth=0.9,
            markeredgecolor="white",
            markeredgewidth=0.45,
            label=metric,
        )
        best = int(np.argmax(means))
        ax.scatter(
            [best + offsets[metric]],
            [means[best]],
            s=42,
            facecolor="none",
            edgecolor=GOLD,
            linewidth=1.0,
            zorder=4,
        )
        label_offset = 5 if metric == "AUC" else -8
        for x_value, mean in zip(x + offsets[metric], means):
            ax.annotate(
                f"{mean:.3f}",
                (x_value, mean),
                xytext=(0, label_offset),
                textcoords="offset points",
                ha="center",
                va="bottom" if label_offset > 0 else "top",
                fontsize=6.1,
                color=METRIC_COLORS[metric],
            )
    all_lower = np.concatenate(
        [MOTIF_DATA[metric][0] - MOTIF_DATA[metric][1] for metric in ("AUC", "AUPR")]
    )
    all_upper = np.concatenate(
        [MOTIF_DATA[metric][0] + MOTIF_DATA[metric][1] for metric in ("AUC", "AUPR")]
    )
    ax.set_ylim(float(all_lower.min()) - 0.002, float(all_upper.max()) + 0.002)
    ax.set_ylabel("Score (mean ± std)")
    ax.set_xticks(x)
    ax.set_xticklabels(MOTIF_COMBOS)
    ax.set_xlabel("Hyperedge-type combination")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
    _clean_axis(ax, grid_axis="y")
    fig.tight_layout()
    return fig


def make_dropout_figure() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(3.35, 2.15))
    offsets = {"AUC": -0.008, "AUPR": 0.008}
    for metric in ("AUC", "AUPR"):
        means, stds = DROPOUT_DATA[metric]
        color = METRIC_COLORS[metric]
        ax.errorbar(
            DROPOUTS + offsets[metric],
            means,
            yerr=stds,
            marker=METRIC_MARKERS[metric],
            color=color,
            ecolor=color,
            capsize=3.0,
            elinewidth=0.9,
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=metric,
        )
        label_offset = 5 if metric == "AUC" else -8
        for value_x, value_y in zip(DROPOUTS + offsets[metric], means):
            ax.annotate(
                f"{value_y:.4f}",
                (value_x, value_y),
                xytext=(0, label_offset),
                textcoords="offset points",
                ha="center",
                va="bottom" if label_offset > 0 else "top",
                fontsize=6.1,
                color=color,
            )
    all_lower = np.concatenate(
        [DROPOUT_DATA[metric][0] - DROPOUT_DATA[metric][1] for metric in ("AUC", "AUPR")]
    )
    all_upper = np.concatenate(
        [DROPOUT_DATA[metric][0] + DROPOUT_DATA[metric][1] for metric in ("AUC", "AUPR")]
    )
    ax.set_ylim(float(all_lower.min()) - 0.001, float(all_upper.max()) + 0.001)
    ax.set_ylabel("Score (mean ± std)")
    ax.set_xlabel("RHGNN2 dropout")
    ax.set_xticks(DROPOUTS)
    ax.set_xticklabels([f"{value:.1f}" for value in DROPOUTS])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
    _clean_axis(ax, grid_axis="y")
    fig.tight_layout()
    return fig


def generate_all(output_dir: Path, formats: Sequence[str] = ("pdf", "svg")) -> list[Path]:
    """Generate all five static paper figures and return their paths."""
    normalized = _validate_formats(formats)
    apply_publication_style()
    output_dir = Path(output_dir)
    figures = [
        ("ablation_ONEIL", make_ablation_figure("ONEIL")),
        ("ablation_ALMANAC", make_ablation_figure("ALMANAC")),
        ("learning_curve_ONEIL", make_learning_curve_figure()),
        ("motif_combos_ONEIL", make_motif_figure()),
        ("rhgnn2_dropout_ONEIL", make_dropout_figure()),
    ]
    outputs: list[Path] = []
    for name, fig in figures:
        outputs.extend(save_figure(fig, output_dir / name, normalized))
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "publication_figures",
        help="Directory for generated figures",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "svg"],
        help="Output formats: pdf, svg, and/or png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = generate_all(args.output_dir, formats=args.formats)
    for output in outputs:
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
