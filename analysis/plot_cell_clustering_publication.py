"""Generate a publication-quality GSK cell-line clustering figure.

This script loads an existing GeneSyn-HGNN checkpoint; it does not train a
model.  PCA is the default because its axes have an interpretable variance
summary.  The previous t-SNE view remains available through a CLI flag.

Example:

    python analysis/plot_cell_clustering_publication.py \
        --dataset ONEIL \
        --k-fold 10 \
        --checkpoint checkpoints/ONEIL_pretrained.pt \
        --output-dir analysis/publication_figures \
        --formats pdf svg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from plot_publication_figures import (
    BLUE,
    GOLD,
    LIGHT_GRID,
    SLATE,
    TEAL,
    TEXT,
    VERMILLION,
    _validate_formats,
    apply_publication_style,
    save_figure,
)


CLUSTER_COLORS = (BLUE, TEAL, VERMILLION, SLATE, GOLD)
DEFAULT_HIGHLIGHTS = ("A2058", "A2780", "A375", "A427", "COLO320DM")


def project_embedding(
    embeddings: np.ndarray,
    method: str = "pca",
    random_state: int = 42,
) -> tuple[np.ndarray, str, str]:
    """Project an N-by-D embedding matrix to two display coordinates."""
    embeddings = np.asarray(embeddings, dtype=float)
    if embeddings.ndim != 2 or embeddings.shape[0] < 3 or embeddings.shape[1] < 2:
        raise ValueError("embeddings must have shape (n_samples >= 3, n_features >= 2)")

    method = method.lower()
    if method == "pca":
        projector = PCA(n_components=2)
        coordinates = projector.fit_transform(embeddings)
        explained = projector.explained_variance_ratio_ * 100.0
        return (
            coordinates,
            f"PC1 ({explained[0]:.1f}% variance)",
            f"PC2 ({explained[1]:.1f}% variance)",
        )
    if method == "tsne":
        perplexity = min(15, embeddings.shape[0] - 1)
        projector = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=random_state,
        )
        coordinates = projector.fit_transform(embeddings)
        return coordinates, "t-SNE 1", "t-SNE 2"
    raise ValueError("method must be either 'pca' or 'tsne'")


def cluster_embedding(
    embeddings: np.ndarray,
    cluster_count: int = 5,
) -> tuple[np.ndarray, float]:
    """Perform Ward clustering and evaluate labels in the original GSK space."""
    embeddings = np.asarray(embeddings, dtype=float)
    if not 2 <= cluster_count < embeddings.shape[0]:
        raise ValueError("cluster_count must be between 2 and n_samples - 1")
    hierarchy = linkage(embeddings, method="ward")
    labels = fcluster(hierarchy, cluster_count, criterion="maxclust")
    silhouette = float(silhouette_score(embeddings, labels))
    return labels, silhouette


def _representative_indices(
    coords: np.ndarray,
    labels: np.ndarray,
    names: Sequence[str],
    highlights: Sequence[str],
    max_labels: int,
) -> list[int]:
    """Choose a small deterministic label set instead of annotating every point."""
    chosen = {idx for idx, name in enumerate(names) if name in set(highlights)}
    for cluster in sorted(np.unique(labels)):
        indices = np.flatnonzero(labels == cluster)
        cluster_coords = coords[indices]
        centroid = cluster_coords.mean(axis=0)
        distances = np.linalg.norm(cluster_coords - centroid, axis=1)
        chosen.add(int(indices[int(np.argmin(distances))]))
        if len(indices) > 1:
            chosen.add(int(indices[int(np.argmax(distances))]))

    if len(chosen) > max_labels:
        highlighted = [idx for idx in sorted(chosen) if names[idx] in set(highlights)]
        remaining = [idx for idx in sorted(chosen) if idx not in highlighted]
        chosen = set((highlighted + remaining)[:max_labels])
    return sorted(chosen)


def plot_clustering(
    coords: np.ndarray,
    cluster_labels: np.ndarray,
    cell_names: Sequence[str],
    silhouette: float,
    x_label: str,
    y_label: str,
    output_base: Path,
    formats: Sequence[str] = ("pdf", "svg"),
    highlight_names: Sequence[str] = DEFAULT_HIGHLIGHTS,
    max_labels: int = 14,
) -> list[Path]:
    """Plot projected cells with restrained labels and save vector output."""
    coords = np.asarray(coords, dtype=float)
    cluster_labels = np.asarray(cluster_labels)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (n_samples, 2)")
    if len(coords) != len(cluster_labels) or len(coords) != len(cell_names):
        raise ValueError("coords, cluster_labels, and cell_names must have equal length")
    if max_labels < 1:
        raise ValueError("max_labels must be positive")
    _validate_formats(formats)
    apply_publication_style()

    fig, ax = plt.subplots(figsize=(5.8, 4.05))
    clusters = sorted(np.unique(cluster_labels))
    for position, cluster in enumerate(clusters):
        mask = cluster_labels == cluster
        color = CLUSTER_COLORS[position % len(CLUSTER_COLORS)]
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=31,
            color=color,
            edgecolor="white",
            linewidth=0.55,
            alpha=0.92,
            label=f"Cluster {cluster}",
            zorder=3,
        )

    selected = _representative_indices(
        coords,
        cluster_labels,
        list(cell_names),
        highlight_names,
        max_labels=max_labels,
    )
    x_mid = float(np.median(coords[:, 0]))
    y_mid = float(np.median(coords[:, 1]))
    for idx in selected:
        x_value, y_value = coords[idx]
        x_offset = 4 if x_value <= x_mid else -4
        y_offset = 4 if y_value <= y_mid else -5
        horizontal_alignment = "left" if x_offset > 0 else "right"
        is_highlight = cell_names[idx] in set(highlight_names)
        ax.annotate(
            str(cell_names[idx]),
            (x_value, y_value),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            ha=horizontal_alignment,
            va="bottom" if y_offset > 0 else "top",
            fontsize=6.8,
            fontweight="bold" if is_highlight else "normal",
            color=TEXT if is_highlight else SLATE,
            bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.72},
            zorder=4,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.text(
        0.015,
        0.02,
        f"Silhouette = {silhouette:.3f} (original GSK space)",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.0,
        color=SLATE,
    )
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=min(5, len(clusters)),
        frameon=False,
        handletextpad=0.35,
        columnspacing=0.8,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color=LIGHT_GRID, linewidth=0.55, alpha=0.65)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return save_figure(fig, Path(output_base), formats)


def _import_model_modules(
    repo_root: Path,
    dataset: str,
    cuda: str,
    threshold: int,
    k_fold: int,
):
    """Import repository modules with only their recognized CLI arguments."""
    repo_root = repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    original_argv = sys.argv[:]
    sys.argv = [
        original_argv[0],
        "--dataset",
        dataset,
        "--cuda",
        str(cuda),
        "--threshold",
        str(threshold),
        "--k_fold",
        str(k_fold),
    ]
    try:
        import Config
        import Data_Process
        import Synergy_Models

        model_args = Config.parse()
    finally:
        sys.argv = original_argv
    return model_args, Data_Process, Synergy_Models


def extract_gsk_embeddings(
    repo_root: Path,
    dataset: str,
    checkpoint: Path,
    cuda: str,
    threshold: int,
    k_fold: int,
) -> tuple[np.ndarray, list[str]]:
    """Load an existing checkpoint and return x_cell @ W_gs for each cell line."""
    import torch

    model_args, data_process, models = _import_model_modules(
        repo_root=repo_root,
        dataset=dataset,
        cuda=cuda,
        threshold=threshold,
        k_fold=k_fold,
    )
    device = model_args.device
    (
        real_folds,
        _,
        _,
        _,
        num_nodes_view1,
        drug_features,
        cell_features,
        _,
        _,
        cell_line_to_id,
        vertices_view1,
        edges_view1,
        edge_count_view1,
        edge_length_view1,
        degree_view1,
        _,
    ) = data_process.process_data(dataset)

    drug_features = torch.tensor(drug_features).float().to(device)
    cell_features = torch.tensor(cell_features).float().to(device)
    vertices_view1 = torch.tensor(vertices_view1).long().to(device)
    edges_view1 = torch.tensor(edges_view1).long().to(device)
    for idx in range(len(degree_view1)):
        degree_view1[idx] = torch.tensor(degree_view1[idx]).float().to(device)

    fold_zero = data_process.torch_from_numpy(real_folds[0], device)
    num_drugs = int(fold_zero.numDrug)
    num_cells = int(cell_features.shape[0])
    num_proteins = int(num_nodes_view1 - num_drugs)
    degree_view2 = {
        idx: fold_zero.degV_dict[idx].float().to(device)
        for idx in range(len(fold_zero.degV_dict))
    }
    model = models.Synergy(
        num_drugs,
        models.BioEncoder(
            drug_features.shape[1],
            cell_features.shape[1],
            num_cells,
            num_proteins,
            512,
            device,
        ),
        models.RHGNN(
            vertices_view1,
            edges_view1,
            edge_count_view1,
            edge_length_view1,
            degree_view1,
            512,
            256,
            256,
            num_edge_types=6,
            dropout=0.2,
        ),
        models.RHGNN(
            fold_zero.V,
            fold_zero.E,
            fold_zero.hypergraph_edge_num,
            fold_zero.edge_length,
            degree_view2,
            512,
            128,
            256,
            num_edge_types=2,
            dropout=0.0,
        ),
        models.BilinearDecoder(
            feature_dim=256,
            numDrug=num_drugs,
            cellscount=num_cells,
            cell_dim=512,
            gene_dim=cell_features.shape[1],
            use_cell_gate=True,
            use_gene_synergy=True,
            use_gene_attn=True,
        ),
    ).to(device)

    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    state = {str(key).removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()

    with torch.no_grad():
        _, encoded_cells, _ = model.BioEncoder(drug_features, cell_features)
        gsk_embeddings = torch.matmul(encoded_cells, model.decoder.W_gs).cpu().numpy()

    name_column = "Cell_Line" if "Cell_Line" in cell_line_to_id.columns else cell_line_to_id.columns[0]
    cell_names = [str(name) for name in cell_line_to_id[name_column].tolist()]
    if len(cell_names) != len(gsk_embeddings):
        raise ValueError("Cell-line metadata and extracted embeddings have different lengths")
    return gsk_embeddings, cell_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    analysis_dir = Path(__file__).resolve().parent
    repo_root = analysis_dir.parent
    parser.add_argument("--dataset", choices=("ONEIL", "ALMANAC"), default="ONEIL")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--output-dir", type=Path, default=analysis_dir / "publication_figures")
    parser.add_argument("--projection", choices=("pca", "tsne"), default="pca")
    parser.add_argument("--cluster-count", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--threshold", type=int, default=None)
    parser.add_argument("--k-fold", type=int, default=10)
    parser.add_argument("--formats", nargs="+", default=["pdf", "svg"])
    parser.add_argument("--max-labels", type=int, default=14)
    parser.add_argument("--highlight-cells", nargs="*", default=list(DEFAULT_HIGHLIGHTS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    threshold = args.threshold
    if threshold is None:
        threshold = 10 if args.dataset == "ALMANAC" else 30
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = args.repo_root / "checkpoints" / f"{args.dataset}_pretrained.pt"

    embeddings, cell_names = extract_gsk_embeddings(
        repo_root=args.repo_root,
        dataset=args.dataset,
        checkpoint=checkpoint,
        cuda=args.cuda,
        threshold=threshold,
        k_fold=args.k_fold,
    )
    cluster_labels, silhouette = cluster_embedding(embeddings, args.cluster_count)
    coordinates, x_label, y_label = project_embedding(
        embeddings,
        method=args.projection,
        random_state=args.random_state,
    )

    print(
        f"GSK clustering: dataset={args.dataset}, k={args.cluster_count}, "
        f"silhouette={silhouette:.3f}, projection={args.projection}"
    )
    for cluster in sorted(np.unique(cluster_labels)):
        members = [
            cell_names[idx]
            for idx in range(len(cell_names))
            if cluster_labels[idx] == cluster
        ]
        print(f"  Cluster {cluster} ({len(members)} cells): {', '.join(members)}")

    outputs = plot_clustering(
        coords=coordinates,
        cluster_labels=cluster_labels,
        cell_names=cell_names,
        silhouette=silhouette,
        x_label=x_label,
        y_label=y_label,
        output_base=args.output_dir / f"cell_clustering_{args.dataset}",
        formats=args.formats,
        highlight_names=args.highlight_cells,
        max_labels=args.max_labels,
    )
    for output in outputs:
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
