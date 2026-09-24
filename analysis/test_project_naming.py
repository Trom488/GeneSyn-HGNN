"""Regression checks for public-facing GeneSyn-HGNN project names."""

from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class ProjectNamingTest(unittest.TestCase):
    def read(self, relative_path: str) -> str:
        return (REPO_ROOT / relative_path).read_text(encoding="utf-8")

    def test_runtime_names_use_genesyn_hgnn(self):
        config = self.read("Config.py")
        entrypoint = self.read("GeneSyn-HGNN.py")

        self.assertIn("GeneSyn-HGNN: Gene-aware Drug Synergy Hypergraph Neural Network", config)
        self.assertNotIn('set_experiment("MRHGNN', entrypoint)
        self.assertIn('set_experiment("GeneSyn-HGNN-', entrypoint)
        self.assertNotIn("output_MRHGNN_", entrypoint)
        self.assertIn("output_GeneSyn-HGNN_", entrypoint)

    def test_user_facing_commands_use_current_entrypoint(self):
        for path in ("gene_interpret.py", "causal_intervention.py"):
            content = self.read(path)
            self.assertNotIn("python MRHGNN.py", content)
            self.assertIn("python GeneSyn-HGNN.py", content)

    def test_publication_helpers_use_current_project_name(self):
        for path in (
            "analysis/plot_publication_figures.py",
            "analysis/test_publication_plots.py",
        ):
            self.assertNotIn("MRHGNN-style", self.read(path))

    def test_readme_documents_current_output_name_and_upstream_credit(self):
        readme = self.read("README.md")
        self.assertIn("output_GeneSyn-HGNN_<DATASET>.txt", readme)
        self.assertNotIn("output_MRHGNN_<DATASET>.txt", readme)
        self.assertIn("extends the MRHGNN research code", readme)


if __name__ == "__main__":
    unittest.main()
