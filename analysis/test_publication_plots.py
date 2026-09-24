"""Behavior tests for the publication-quality plotting scripts."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ANALYSIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYSIS_DIR))

import plot_cell_clustering_publication as clustering  # noqa: E402
import plot_publication_figures as publication  # noqa: E402


class PublicationFigureTests(unittest.TestCase):
    def test_generate_all_writes_the_five_paper_figure_pdfs(self):
        """Catches missing generators, wrong filenames, and empty output files."""
        expected_names = {
            "ablation_ONEIL.pdf",
            "ablation_ALMANAC.pdf",
            "learning_curve_ONEIL.pdf",
            "motif_combos_ONEIL.pdf",
            "rhgnn2_dropout_ONEIL.pdf",
        }

        with tempfile.TemporaryDirectory() as tmp:
            outputs = publication.generate_all(Path(tmp), formats=("pdf",))

            self.assertEqual({path.name for path in outputs}, expected_names)
            for path in outputs:
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 1_000)

    def test_generate_all_rejects_unsupported_formats(self):
        """Catches silently accepting a format the save pipeline cannot produce."""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Unsupported format"):
                publication.generate_all(Path(tmp), formats=("jpeg",))

    def test_ablation_uses_grouped_bars_for_four_metrics_and_four_variants(self):
        """Catches regressions in the GeneSyn-HGNN grouped comparison."""
        fig = publication.make_ablation_figure("ONEIL")
        try:
            self.assertEqual(len(fig.axes), 1)
            self.assertLessEqual(fig.get_figwidth(), 3.6)
            ax = fig.axes[0]
            self.assertEqual(len(ax.patches), 16)
            self.assertEqual(
                [tick.get_text() for tick in ax.get_xticklabels()],
                ["AUC", "AUPR", "ACC", "F1"],
            )
            self.assertEqual(
                [text.get_text() for text in ax.get_legend().get_texts()],
                ["Full model", "w/o CCG", "w/o GSA", "w/o GSK"],
            )
        finally:
            publication.plt.close(fig)

    def test_motif_and_dropout_put_both_metrics_on_one_dense_axis(self):
        """Catches duplicated panels that spread six or three x-values over excess space."""
        motif = publication.make_motif_figure()
        dropout = publication.make_dropout_figure()
        try:
            self.assertEqual(len(motif.axes), 1)
            self.assertEqual(len(dropout.axes), 1)
            self.assertLessEqual(dropout.get_figwidth(), 3.6)
        finally:
            publication.plt.close(motif)
            publication.plt.close(dropout)

    def test_learning_curve_keeps_the_gain_panel_narrow_and_height_compact(self):
        """Catches equal-size panels that make a five-value gain summary look empty."""
        fig = publication.make_learning_curve_figure()
        try:
            self.assertEqual(len(fig.axes), 2)
            main_width = fig.axes[0].get_position().width
            gain_width = fig.axes[1].get_position().width
            self.assertGreater(main_width / gain_width, 1.8)
            self.assertLessEqual(fig.get_figheight(), 2.5)
        finally:
            publication.plt.close(fig)


class CellClusteringFigureTests(unittest.TestCase):
    def test_pca_projection_returns_two_coordinates_and_axis_labels(self):
        """Catches a projection that loses samples or omits interpretable axes."""
        embeddings = np.array(
            [
                [0.0, 0.1, 0.2],
                [0.1, 0.2, 0.3],
                [1.0, 1.1, 1.2],
                [1.1, 1.2, 1.3],
                [2.0, 2.1, 2.2],
                [2.1, 2.2, 2.3],
            ],
            dtype=float,
        )

        coords, x_label, y_label = clustering.project_embedding(
            embeddings, method="pca", random_state=42
        )

        self.assertEqual(coords.shape, (6, 2))
        self.assertRegex(x_label, r"PC1 \(\d+\.\d% variance\)")
        self.assertRegex(y_label, r"PC2 \(\d+\.\d% variance\)")

    def test_plot_clustering_writes_a_nonempty_vector_pdf(self):
        """Catches a plotting path that drops labels or fails to save vector output."""
        coords = np.array(
            [[-1.0, -0.8], [-0.7, -1.1], [0.8, 0.9], [1.1, 0.7]],
            dtype=float,
        )
        labels = np.array([1, 1, 2, 2])
        names = ["A2058", "A2780", "HT29", "RKO"]

        with tempfile.TemporaryDirectory() as tmp:
            outputs = clustering.plot_clustering(
                coords=coords,
                cluster_labels=labels,
                cell_names=names,
                silhouette=0.25,
                x_label="PC1 (60.0% variance)",
                y_label="PC2 (30.0% variance)",
                output_base=Path(tmp) / "cell_clustering_ONEIL",
                formats=("pdf",),
            )

            self.assertEqual([path.name for path in outputs], ["cell_clustering_ONEIL.pdf"])
            self.assertGreater(outputs[0].stat().st_size, 1_000)


if __name__ == "__main__":
    unittest.main()
