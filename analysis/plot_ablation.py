"""消融实验结果绘图（SVG）。数据来自实验控制台输出，无需重跑实验。

x 轴为 4 个配置（w all 及各 w/o），每个配置一组 4 指标柱（AUC/AUPR/F1/ACC），带误差棒。

用法:
  python analysis/plot_ablation.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_here = os.path.dirname(os.path.abspath(__file__))

CONFIGS = ["w all", "w/o CCG", "w/o GSA", "w/o GSK"]
METRICS = ["AUC", "AUPR", "F1", "ACC"]

# 配置 -> (mean[4], std[4])，顺序对应 AUC/AUPR/F1/ACC
# DATA = {
#     "w all":            ([0.9232, 0.9365, 0.8565, 0.8592], [0.0037, 0.0027, 0.0062, 0.0051]),
#     "w/o cell_gate":    ([0.9235, 0.9364, 0.8602, 0.8616], [0.0042, 0.0018, 0.0065, 0.0080]),
#     "w/o gene_attn":    ([0.9213, 0.9349, 0.8575, 0.8601], [0.0016, 0.0013, 0.0060, 0.0056]),
#     "w/o gene_synergy": ([0.8446, 0.8546, 0.7695, 0.7515], [0.0093, 0.0107, 0.0101, 0.0265]),
# }

DATA = {
    "w all":            ([0.9692, 0.9690, 0.9160, 0.9163], [0.0034, 0.0054, 0.0058, 0.0060]),
    "w/o cell_gate":    ([0.9679, 0.9666, 0.9138, 0.9114], [0.0056, 0.0066, 0.01003, 0.0105]),
    "w/o gene_attn":    ([0.9665, 0.9641, 0.9094, 0.9069], [0.0032, 0.0034, 0.0080, 0.0082]),
    "w/o gene_synergy": ([0.9164, 0.9224, 0.8405, 0.8398 ], [0.0061, 0.0076, 0.0063, 0.0056]),
}
ORDER = ["w all", "w/o cell_gate", "w/o gene_attn", "w/o gene_synergy"]

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def main():
    n_configs = len(ORDER)
    n_metrics = len(METRICS)
    x = np.arange(n_configs)
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(9, 5.2))

    for m, (name, color) in enumerate(zip(METRICS, COLORS)):
        means = [DATA[c][0][m] for c in ORDER]
        stds = [DATA[c][1][m] for c in ORDER]
        offset = (m - n_metrics / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, label=name, color=color,
               edgecolor="#333333", linewidth=0.5, capsize=2.5, error_kw={"elinewidth": 0.8})

    ax.set_xticks(x)
    ax.set_xticklabels(CONFIGS, fontsize=9)
    ax.set_ylabel("score")
    ax.set_ylim(0.80, 0.99)
    ax.set_title("Ablation study (ONEIL)", fontsize=12)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="lower right", fontsize=9, ncol=2)

    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    out = os.path.join(_here, "ablation_ONEIL.svg")
    fig.savefig(out, format="svg")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
