"""学习曲线结果绘图（SVG）。数据来自实验控制台输出，无需重跑实验。

左图：full（w/ gene_synergy）vs w/o gene_synergy 的测试 AUC 学习曲线；
右图：ΔAUC = full − w/o，随训练比例的变化。

用法:
  python analysis/plot_learning_curve.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_here = os.path.dirname(os.path.abspath(__file__))

FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]

FULL = {
    "mean": [0.8970, 0.9416, 0.9549, 0.9616, 0.9677],
    "std":  [0.0121, 0.0053, 0.0043, 0.0051, 0.0041],
}
W_O = {
    "mean": [0.8659, 0.8927, 0.9099, 0.9121, 0.9170],
    "std":  [0.0061, 0.0057, 0.0044, 0.0071, 0.0042],
}
DELTA = [f - w for f, w in zip(FULL["mean"], W_O["mean"])]


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

    # ---- 左图：学习曲线 ----
    ax1.errorbar(FRACTIONS, FULL["mean"], yerr=FULL["std"], marker="o", capsize=4,
                 lw=1.6, color="#1f77b4", label="full (w/ GSK)")
    ax1.errorbar(FRACTIONS, W_O["mean"], yerr=W_O["std"], marker="s", capsize=4,
                 lw=1.6, color="#d62728", label="w/o GSK")
    # 水平参考线：AUC 0.90，展示水平样本效率差距
    ax1.axhline(0.90, ls="--", lw=0.9, color="#999999")
    ax1.annotate("AUC 0.90: full≈0.21 vs w/o≈0.49 (samples ~halved)",
                 xy=(0.35, 0.90), xytext=(0.28, 0.885),
                 fontsize=8.5, color="#555555")
    ax1.set_xlabel("train fraction")
    ax1.set_ylabel("test AUC")
    ax1.set_xticks(FRACTIONS)
    ax1.set_xticklabels([f"{x:.1f}" for x in FRACTIONS])
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower right", fontsize=9)

    # ---- 右图：ΔAUC ----
    bars = ax2.bar([str(f) for f in FRACTIONS], DELTA, color="#2ca02c",
                   width=0.55, edgecolor="#333333", linewidth=0.6)
    for x, d in zip(range(len(FRACTIONS)), DELTA):
        ax2.text(x, d + 0.0015, f"{d:+.4f}", ha="center", fontsize=8.5)
    ax2.set_xlabel("train fraction")
    ax2.set_ylabel(r"$\Delta$AUC (full $-$ w/o)")
    ax2.set_ylim(0, 0.06)
    ax2.grid(alpha=0.3, axis="y")
    ax2.axhline(0, color="#333333", lw=0.8)

    fig.suptitle("Learning curve: Gene-Synergy Kernel sample-efficiency (ONEIL)", fontsize=12)
    fig.text(0.5, 0.005, "GSK: Gene-Synergy Kernel", ha="center", fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    out = os.path.join(_here, "learning_curve_ONEIL.svg")
    fig.savefig(out, format="svg")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
