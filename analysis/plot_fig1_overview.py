"""Paper Figure 1: overview contrast diagram (SVG + PDF + PNG).

Left panel: prior methods fuse cell-line info once at the input and score drug
pairs with a cell-free bilinear function. Right panel: the GeneSyn-HGNN decoder
brings cell state and gene dimensions into the scoring structure via
CCG / GSK / GSA.

Usage:
  python analysis/plot_fig1_overview.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

_here = os.path.dirname(os.path.abspath(__file__))

GRAY = "#555555"
DARK = "#1a1a1a"
LGRAY = "#f5f6f8"
BLUE = "#2f6fbf"
GREEN = "#2e8b57"
RED = "#c0392b"
PURPLE = "#7d4bb5"
ORANGE = "#d9822b"

SMALL = 8.5
NORMAL = 9.5
TITLE = 12


def box(ax, x0, y0, x1, y1, text, fc="white", ec=DARK, fs=NORMAL,
        weight="normal", dashed=False, lw=1.3, tc=DARK, pad=0.3,
        rounding=0.7, z=3, align="center"):
    """Draw a rounded rectangle with centered text. (x0,y0,x1,y1) in canvas units."""
    style = f"round,pad={pad},rounding_size={rounding}"
    ls = (0, (3, 2)) if dashed else "-"
    p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=style, fc=fc,
                       ec=ec, lw=lw, linestyle=ls, zorder=z)
    ax.add_patch(p)
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, fontweight=weight, zorder=z + 1)
    if align != "center":
        ax.text(x0 + 1.2, (y0 + y1) / 2, "", fontsize=fs)  # keep api symmetric


def arrow(ax, p0, p1, color=DARK, lw=1.4, ls="-", style="-|>",
          connection=None, ms=11, z=4):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms,
                        color=color, lw=lw, linestyle=ls, zorder=z,
                        connectionstyle=connection, shrinkA=1, shrinkB=1)
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_aspect("equal")

    # ---------------- shared upstream ----------------
    box(ax, 6, 88, 26, 97, "Drug features\n(drug fingerprints)", fs=NORMAL)
    box(ax, 74, 88, 94, 97, "Cell-line features\n(expression profiles)", fs=NORMAL)
    box(ax, 30, 84, 70, 97, "Encoder: BioEncoder + hypergraph\n(RHGNN1 / RHGNN2)",
        fc="#eaf1fb", ec=BLUE, fs=NORMAL)
    arrow(ax, (24, 92.5), (34, 90.8))
    arrow(ax, (80, 92.5), (66, 90.8))

    box(ax, 38, 72, 62, 81, "drug embeddings\n(DrugA, DrugB)", fc="#eaf1fb",
        ec=BLUE, fs=NORMAL)
    arrow(ax, (50, 84), (50, 81))

    box(ax, 76, 72, 94, 81, "cell expression\nx_cell", fc="#f0f0f0", ec=GRAY,
        fs=NORMAL)
    arrow(ax, (84, 88), (84, 81), ls=(0, (3, 2)))

    # ---------------- left panel: prior ----------------
    p = FancyBboxPatch((3, 4), 43, 64, boxstyle="round,pad=0.3,rounding_size=1.5",
                       fc="#fbfbfc", ec=RED, lw=1.6, zorder=1)
    ax.add_patch(p)
    ax.text(24.5, 64.8, "Prior methods", ha="center", va="center", fontsize=TITLE,
            color=RED, fontweight="bold", zorder=2)

    box(ax, 6, 54, 22, 61, "cell info fused\nonce at input", fc="#fdf3f1",
        ec=RED, dashed=True, fs=SMALL)
    box(ax, 6, 36, 30, 49, "score =\nDrugAᵀ W DrugB\n(cell-free bilinear)",
        fc="#fdf3f1", ec=RED, fs=NORMAL)

    for i, s in enumerate(["cell condition absent\nfrom scoring",
                           "no explicit\nDrug×Drug×Gene term",
                           "uniform gene\nweighting"]):
        ax.text(8, 28 - i * 7.5, "✗  " + s, fontsize=SMALL, color=RED,
                va="center", ha="left", linespacing=1.25)

    arrow(ax, (38, 76), (20, 49), color=RED)
    arrow(ax, (74, 76), (14, 61), color=RED, ls=(0, (3, 2)),
          connection="angle,angleA=0,angleB=90,rad=0")
    arrow(ax, (14, 54), (16, 49), color=RED, ls=(0, (3, 2)))
    ax.text(58, 77.5, "x_cell", fontsize=SMALL, color=GRAY)

    # ---------------- right panel: ours ----------------
    p = FancyBboxPatch((54, 4), 43, 64, boxstyle="round,pad=0.3,rounding_size=1.5",
                       fc="#fbfdfa", ec=GREEN, lw=1.6, zorder=1)
    ax.add_patch(p)
    ax.text(75.5, 64.8, "GeneSyn-HGNN (ours)", ha="center", va="center",
            fontsize=TITLE, color=GREEN, fontweight="bold", zorder=2)

    box(ax, 58, 56, 92, 62, "BilinearDecoder", fc="#eef7ef", ec=GREEN, fs=NORMAL,
        weight="bold")
    box(ax, 58, 44, 92, 52, "①  CCG: gate = σ(W·x_cell)\nper-dimension "
        "modulation", fc="#eef7ef", ec=GREEN, fs=NORMAL)
    box(ax, 58, 33, 92, 41, "②  GSK: (DrugA⊙DrugB)·(x_cellᵀ W_gs)\n"
        "diagonal low-rank 3-way kernel", fc="#eef5fb", ec=BLUE, fs=NORMAL)
    box(ax, 58, 22, 92, 30, "③  GSA: drug-pair-query\ngene cross-attention",
        fc="#f5eefb", ec=PURPLE, fs=NORMAL)
    box(ax, 58, 8, 92, 16, "score = synergy pattern\n× cell sensitivity",
        fc="#fef6ea", ec=ORANGE, fs=NORMAL, weight="bold")

    ax.text(93.2, 48.0, "✓", fontsize=13, color=GREEN, ha="center", va="center")
    ax.text(93.2, 37.0, "✓", fontsize=13, color=GREEN, ha="center", va="center")
    ax.text(93.2, 26.0, "✓", fontsize=13, color=GREEN, ha="center", va="center")

    arrow(ax, (62, 76), (75, 52), color=GREEN)
    arrow(ax, (84, 72), (78, 52), color=GREEN, ls=(0, (3, 2)))
    arrow(ax, (75, 44), (75, 41), color=GRAY, lw=1.1)
    arrow(ax, (75, 33), (75, 30), color=GRAY, lw=1.1)
    arrow(ax, (75, 22), (75, 16), color=GRAY, lw=1.1)

    fig.tight_layout(pad=0.2)
    for ext in ("svg", "pdf", "png"):
        out = os.path.join(_here, f"fig1_overview.{ext}")
        fig.savefig(out, format=ext, bbox_inches="tight", dpi=150)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
