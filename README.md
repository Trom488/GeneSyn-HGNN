# GeneSyn-HGNN

**Gene-aware Drug Synergy Hypergraph Neural Network** for cell-line-specific anticancer drug synergy prediction.

GeneSyn-HGNN couples motif-guided multi-relational hypergraph encoding with a structured, cell-conditioned decoder. The encoder learns complementary drug representations from drug-target motifs and labeled drug-cell interactions. The decoder then models three distinct roles of cellular state through Cell-Conditioned Gating (CCG), the Gene-Synergy Kernel (GSK), and Gene-Selective Attention (GSA).

> **Status:** Research code accompanying a manuscript under review. Interfaces and file layouts may change before the archival release.

## Highlights

- **Motif-guided hypergraph construction:** six drug-target motif relations, M1-M6, define typed higher-order relations for structural drug encoding.
- **Cell-Conditioned Gating (CCG):** modulates both drug representations using the encoded cell state.
- **Gene-Synergy Kernel (GSK):** models a parameter-efficient drug-drug-latent-cell interaction.
- **Gene-Selective Attention (GSA):** uses the drug pair as a query over expression-scaled gene embeddings to produce a pair-specific additive term.
- **Cell-wise evaluation:** supports random ten-fold evaluation and cold-start split modes for cells, drugs, or drug pairs.
- **Reproducibility utilities:** includes component ablations, learning-curve analysis, motif sensitivity, RHGNN2 dropout analysis, cell-line clustering, and publication-figure generation.

## Model Overview

<p align="center">
  <img src="assets/overview.png" alt="Overview of the GeneSyn-HGNN architecture" width="100%">
</p>

RHGNN1 encodes motif-defined drug-protein relations. RHGNN2 encodes the drug-cell hypergraph constructed from training-fold synergy and antagonism labels. The two structural drug views are averaged before decoding. CCG is applied first, followed by a base bilinear score and the parallel GSK and GSA terms.

## Repository Structure

```text
GeneSyn-HGNN/
├── GeneSyn-HGNN.py        # Training and ten-fold evaluation entry point
├── Synergy_Models.py      # BioEncoder, RHGNN, CCG, GSK, and GSA
├── Data_Process.py        # Data loading, labeling, splitting, and hypergraph construction
├── Config.py              # Command-line arguments and defaults
├── gene_interpret.py      # Gene-attention analysis
├── causal_intervention.py # Gene-level intervention analysis
├── analysis/              # Ablations, learning curves, clustering, and plotting scripts
├── DATA/                  # Expected input data directory
└── requirements.txt       # Core Python dependencies
```

## Installation

The current environment was developed with Python 3.10 and PyTorch 2.0.1. GPU training is recommended. The code automatically falls back to CPU when CUDA is unavailable.

```bash
conda create -n genesyn-hgnn python=3.10
conda activate genesyn-hgnn
pip install -r requirements.txt
```

The pinned PyTorch Geometric packages must match the installed PyTorch and CUDA versions. If the direct installation fails, install `torch`, `torch-geometric`, `torch-scatter`, and `torch-sparse` from the wheels recommended by the [PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html), then install the remaining requirements.

The publication plotting scripts additionally require Matplotlib:

```bash
pip install matplotlib
```

## Data Layout

Place the input files under `DATA/` using the following structure:

```text
DATA/
├── Chemical_Target_Interaction.csv
├── Drug_Drug_Interaction.csv
├── Protein_Protein_Interaction.csv
├── ONEIL/
│   ├── ONEIL_DRUG.csv
│   ├── ONEIL_SCORE.csv
│   └── ONEIL_CELL_LINE_EXPRESSION.csv
└── ALMANAC/
    ├── ALMANAC_DRUG.csv
    ├── ALMANAC_SCORE.csv
    └── ALMANAC_CELL_LINE_EXPRESSION.csv
```

Expected columns include:

| File | Required columns |
| --- | --- |
| `<DATASET>_DRUG.csv` | `Name`, `PubChem_CID` |
| `<DATASET>_SCORE.csv` | `Drug_A`, `Drug_B`, `Cell_Line`, `Score` |
| `<DATASET>_CELL_LINE_EXPRESSION.csv` | `Cell_Line` followed by gene-expression features |
| `Chemical_Target_Interaction.csv` | `PubChem_CID`, `Entry_ID` |
| `Protein_Protein_Interaction.csv` | `Protein_A`, `Protein_B` |
| `Drug_Drug_Interaction.csv` | `DrugA`, `DrugB` |

The binary labels follow the manuscript protocol:

- O'Neil scores greater than or equal to 30 are positive.
- NCI-ALMANAC scores greater than or equal to 10 are positive.
- Scores below 0 are negative.
- Scores between 0 and the dataset-specific positive threshold are excluded.

Make sure that you have permission to use and redistribute the source datasets. This repository does not change their original licenses or terms of use.

## Training and Evaluation

Run all commands from the repository root.

### O'Neil

```bash
python GeneSyn-HGNN.py \
  --dataset ONEIL \
  --threshold 30 \
  --k_fold 10 \
  --split_mode random \
  --epochs 1500 \
  --learning_rate 1e-3 \
  --weight_decay 1e-6 \
  --cuda 0
```

### NCI-ALMANAC

```bash
python GeneSyn-HGNN.py \
  --dataset ALMANAC \
  --threshold 10 \
  --k_fold 10 \
  --split_mode random \
  --epochs 1500 \
  --learning_rate 1e-3 \
  --weight_decay 1e-6 \
  --cuda 0
```

The default random protocol performs cell-wise ten-fold evaluation. Within every cell line, positive and negative examples are split separately. AUPR is computed from continuous prediction scores with `average_precision_score`.

### Alternative Split Modes

The data loader also supports:

| Mode | Meaning |
| --- | --- |
| `random` | Cell-wise random ten-fold split |
| `lco` | Leave-cell-out evaluation |
| `ldo` | Leave-drug-out evaluation |
| `lbo` | Leave-both-drugs-out evaluation |

Example:

```bash
python GeneSyn-HGNN.py \
  --dataset ONEIL \
  --threshold 30 \
  --k_fold 10 \
  --split_mode ldo
```

These cold-start modes answer different generalization questions from the random protocol and should be reported separately.

## Component Ablations

Disable one decoder component at a time while keeping the remaining configuration fixed:

```bash
# Without CCG
python GeneSyn-HGNN.py --dataset ONEIL --threshold 30 --k_fold 10 --no_cell_gate

# Without GSK
python GeneSyn-HGNN.py --dataset ONEIL --threshold 30 --k_fold 10 --no_gene_synergy

# Without GSA
python GeneSyn-HGNN.py --dataset ONEIL --threshold 30 --k_fold 10 --no_gene_attn
```

Select a subset of the six RHGNN1 motif-defined hyperedge types with `--edge_types`:

```bash
python GeneSyn-HGNN.py \
  --dataset ONEIL \
  --threshold 30 \
  --k_fold 10 \
  --edge_types 0,2,4,5
```

## Additional Analyses

### Learning Curve

```bash
python analysis/learning_curve.py \
  --dataset ONEIL \
  --threshold 30 \
  --k_fold 10 \
  --epochs 1500 \
  --fractions 0.2,0.4,0.6,0.8,1.0
```

### RHGNN2 Dropout Sensitivity

```bash
python analysis/rhgnn2_dropout.py \
  --dataset ONEIL \
  --threshold 30 \
  --k_fold 10 \
  --epochs 1500 \
  --dropouts 0,0.1,0.2
```

### Publication Figures

The following command redraws the recorded fold summaries without rerunning training:

```bash
python analysis/plot_publication_figures.py
```

By default, the figures are written to `analysis/publication_figures/` in PDF and SVG formats.

### Cell-Line Clustering

This analysis loads an existing checkpoint and projects the learned GSK cell representations:

```bash
python analysis/plot_cell_clustering_publication.py \
  --dataset ONEIL \
  --checkpoint checkpoints/ONEIL_pretrained.pt \
  --k-fold 10 \
  --projection pca \
  --output-dir analysis/publication_figures \
  --formats pdf svg
```

## Outputs

Training produces:

- fold-level metrics and aggregate means and standard deviations in the console
- local MLflow records under `mlruns/`
- a first-fold checkpoint at `checkpoints/<DATASET>_pretrained.pt`
- a fold-summary text file named `output_MRHGNN_<DATASET>.txt` for backward compatibility with the original codebase

The reported metrics are ordered as AUC, AUPR, F1, and ACC. The F1 threshold is selected from the precision-recall curve within each evaluated split, so ACC and F1 should not be interpreted as fixed-threshold deployment performance.

## Reported Results

Under the reported cell-wise ten-fold random protocol:

| Dataset | AUC | AUPR |
| --- | ---: | ---: |
| O'Neil | 0.97 +/- 0.01 | 0.97 +/- 0.01 |
| NCI-ALMANAC | 0.93 +/- 0.01 | 0.94 +/- 0.01 |

These values summarize the recorded ten-fold runs. Results under cold-start or cross-dataset settings are not directly comparable to the random protocol.

## Tests

Run the plotting and projection behavior tests with:

```bash
python -m unittest analysis/test_publication_plots.py
```

You can inspect every command-line option with:

```bash
python GeneSyn-HGNN.py --help
```

## Citation

The GeneSyn-HGNN manuscript is currently under review. Citation information will be added after the archival publication is available.

## Acknowledgements

This implementation extends the MRHGNN research code and retains portions of its data-processing and relational hypergraph infrastructure. Please also consult and cite the original MRHGNN work when appropriate:

```bibtex
@article{chen2025mrhgnn,
  title   = {MRHGNN: Enhanced Multimodal Relational Hypergraph Neural Network for Synergistic Drug Combination Forecasting},
  author  = {Chen, Mengjie and Zhang, Ming and Yan, Guiying and Wang, Guanghui and Qu, Cunquan},
  journal = {IEEE Transactions on Neural Networks and Learning Systems},
  year    = {2025},
  doi     = {10.1109/TNNLS.2025.3553385}
}
```

## License

No license has been specified for this repository yet. Add a `LICENSE` file before public redistribution or reuse.
