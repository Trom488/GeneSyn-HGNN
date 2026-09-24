"""Paper Table 1: expressiveness and sample efficiency of the three components
(SVG + PDF + PNG).

Mirrors the "results comparison table" placed in the Introduction of the ICLR
sample paper (Local GD, Table 1): baseline family vs ours, parameter count, and
the theoretical/empirical guarantee.

Usage:
  python analysis/plot_table1_theory.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

_here = os.path.dirname(os.path.abspath(__file__))

HEAD = ["Component", "Baseline function family", "Ours",
        "Parameters", "Guarantee"]

ROWS = [
    ["CCG", r"$F_{\mathrm{fixed}}$ (shared/fixed gate)",
     r"$F_{\mathrm{gate}}=\{\sigma(W x_{\mathrm{cell}})\}\supsetneq F_{\mathrm{fixed}}$",
     r"$O(d\,d_c)$, non-increasing", "strictly more expressive"],
    ["GSK", r"full Drug$\times$Drug$\times$Gene tensor ($O(d^2 G)$)",
     r"diagonal low-rank 3-way kernel ($O(d\,d_c)$)", r"$O(d\,d_c)$",
     "exact equivalence (simultaneously\ndiagonalizable) + generalization bound"],
    ["GSA", r"uniform gene weights", r"drug-pair-conditioned\nattention",
     r"$O(d)$", "drug-pair-selective\ngene selection"],
]

NOTE = ("Sample efficiency (ONEIL): the full model reaches AUC ≈ 0.90 at ≈20% of the "
        "training data, while the variant without GSK requires ≈ 50%.")


def main():
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axis("off")

    n_rows, n_cols = len(ROWS), len(HEAD)
    col_w = [0.10, 0.24, 0.24, 0.16, 0.26]
    row_h = 0.16

    x, y = 0.0, 0.90  # start top-left, drawing downward

    # header
    for c, (label, w) in enumerate(zip(HEAD, col_w)):
        ax.add_patch(Rectangle((x, y - row_h), w, row_h, fc="#2f6fbf", ec="white",
                               lw=1.0))
        ax.text(x + w / 2, y - row_h / 2, label, ha="center", va="center",
                fontsize=10.5, color="white", fontweight="bold")
        x += w

    x = 0.0
    y -= row_h
    for r, row in enumerate(ROWS):
        fc = "#f5f8fc" if r % 2 == 0 else "white"
        for c, (cell, w) in enumerate(zip(row, col_w)):
            ax.add_patch(Rectangle((x, y - row_h), w, row_h, fc=fc, ec="#b9c6d8",
                                   lw=0.8))
            if c == 0:
                ax.text(x + w / 2, y - row_h / 2, cell, ha="center", va="center",
                        fontsize=10.5, fontweight="bold")
            else:
                ax.text(x + w / 2, y - row_h / 2, cell, ha="center", va="center",
                        fontsize=10)
            x += w
        x = 0.0
        y -= row_h

    ax.text(0.0, y - 0.09, NOTE, fontsize=9, color="#444444", ha="left", va="top")

    fig.tight_layout(pad=0.3)
    for ext in ("svg", "pdf", "png"):
        out = os.path.join(_here, f"table1_theory.{ext}")
        fig.savefig(out, format=ext, bbox_inches="tight", dpi=150)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
