"""RHGNN2 dropout 消融结果绘图（SVG）。数据来自实验控制台输出，无需重跑实验。

用法:
  python analysis/plot_rhgnn2_dropout.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_here = os.path.dirname(os.path.abspath(__file__))

DROP_OUTS = [0.0, 0.1, 0.2]

# 指标名 -> (mean, std)，与实验输出一致
METRICS = {
    "AUC":  ([0.9682, 0.9678, 0.9692], [0.0043, 0.0041, 0.0034]),
    "AUPR": ([0.9674, 0.9658, 0.9690], [0.0059, 0.0071, 0.0054]),
    "F1":   ([0.9137, 0.9155, 0.9160], [0.0109, 0.0108, 0.0058]),
    "ACC":  ([0.9129, 0.9149, 0.9155], [0.0105, 0.0111, 0.0060]),
}

# 各指标 y 轴范围（放大局部差异）
Y_LIMS = {
    "AUC":  (0.962, 0.972),
    "AUPR": (0.960, 0.972),
    "F1":   (0.906, 0.920),
    "ACC":  (0.905, 0.919),
}


def main():
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    fig.suptitle("RHGNN2 dropout ablation (ONEIL)", fontsize=13, y=0.98)

    for ax, (name, (mean, std)) in zip(axes.flat, METRICS.items()):
        ax.errorbar(DROP_OUTS, mean, yerr=std, marker="o", capsize=4, lw=1.6,
                    color="#1f77b4", ecolor="#888888", label=name)
        ax.set_xlabel("RHGNN2 dropout")
        ax.set_ylabel(name)
        ax.set_ylim(*Y_LIMS[name])
        ax.set_xticks(DROP_OUTS)
        ax.set_xticklabels([f"{d:.1f}" for d in DROP_OUTS])
        ax.grid(alpha=0.3)
        for x, y in zip(DROP_OUTS, mean):
            ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)
        ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(_here, "rhgnn2_dropout_ONEIL.svg")
    fig.savefig(out, format="svg")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
