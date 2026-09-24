"""Motif（超边组合）消融结果绘图（SVG）。数据来自实验控制台输出，无需重跑实验。

x 轴为 6 种超边组合，每个组合一组 4 指标柱（AUC/AUPR/F1/ACC），带误差棒。

用法:
  python analysis/plot_motif_combos.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_here = os.path.dirname(os.path.abspath(__file__))

COMBOS = ["0,2,5", "0,1,5", "0,1,3,5", "0,2,4,5", "0,1,2,3,4,5", "0,2,3,4,5"]
METRICS = ["AUC", "AUPR", "F1", "ACC"]

# 组合 -> (mean[4], std[4])，顺序对应 AUC/AUPR/F1/ACC
DATA = {
    "0,2,5":       ([0.9707, 0.9689, 0.9215, 0.9204], [0.0077, 0.0108, 0.0090, 0.0092]),
    "0,1,5":       ([0.9697, 0.9685, 0.9192, 0.9190], [0.0089, 0.0119, 0.0130, 0.0128]),
    "0,1,3,5":     ([0.9701, 0.9689, 0.9246, 0.9233], [0.0071, 0.0105, 0.0114, 0.0117]),
    "0,2,4,5":     ([0.9718, 0.9709, 0.9235, 0.9227], [0.0075, 0.0113, 0.0117, 0.0094]),
    "0,1,2,3,4,5": ([0.9702, 0.9697, 0.9236, 0.9230], [0.0073, 0.0096, 0.0107, 0.0113]),
    "0,2,3,4,5":   ([0.9714, 0.9712, 0.9243, 0.9241], [0.0072, 0.0091, 0.0121, 0.0165]),
}

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def main():
    n_combos = len(COMBOS)
    n_metrics = len(METRICS)
    x = np.arange(n_combos)
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(10, 5.4))

    for m, (name, color) in enumerate(zip(METRICS, COLORS)):
        means = [DATA[c][0][m] for c in COMBOS]
        stds = [DATA[c][1][m] for c in COMBOS]
        offset = (m - n_metrics / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, label=name, color=color,
               edgecolor="#333333", linewidth=0.5, capsize=2.5,
               error_kw={"elinewidth": 0.8})

    ax.set_xticks(x)
    ax.set_xticklabels(COMBOS, fontsize=9)
    ax.set_xlabel("edge-type combination")
    ax.set_ylabel("score")
    ax.set_ylim(0.88, 0.99)
    ax.set_title("Motif combination ablation (ONEIL)", fontsize=12)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="lower right", fontsize=9, ncol=2)

    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    out = os.path.join(_here, "motif_combos_ONEIL.svg")
    fig.savefig(out, format="svg")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
