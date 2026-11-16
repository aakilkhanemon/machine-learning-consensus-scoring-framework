# Consensus-Driven Multi-Tiered Machine Learning Framework for Marine Natural Product Lead Discovery

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aakilkhanemon/machine-learning-consensus-scoring-framework/blob/main/consensus_lead_scoring_colab.ipynb)]
[![DOI](https://img.shields.io/badge/Manuscript-In_Preparation-orange.svg)](#citation)

<p align="center">
  <img src="assets/workflow_architecture.png" alt="Consensus Machine Learning Scoring Architecture" width="850">
</p>

Official implementation of the multi-criteria machine learning consensus scoring and lead prioritization architecture from:  
**"Consensus-Driven Multi-Tiered Machine Learning Framework for Marine Natural Product Lead Discovery Across Dual-Family Viral Ensembles"**  
*Mohammed Emon et al.*

---

## Algorithmic Architecture & Multi-Criteria Decision Logic

High-throughput virtual screening of structurally complex marine natural products (MNPs) frequently suffers from single-estimator bias—where static docking scores over-reward high-molecular-weight macrocycles despite pharmacokinetic or toxicological liabilities. This framework reconciles six orthogonal evaluation tracks across dual-family viral target ensembles (**SARS-CoV-2**; **Influenza A H5N1**`) into a unified, dimensionless coordinate landscape ($S \in [0.00, 100.00]$):

1. **Tier 0: Ingestion, Directional Sign-Inversion & Feature Scaling (`src/normalizer.py`)**
   * **Spatial & Thermodynamic Tracks:** GNINA empirical Vina affinity ($\text{kcal/mol}$), GNINA 3D Convolutional Neural Network (CNN) pose probability ($0.00–1.00$), and MOE induced-fit GBVI/WSA $\Delta G$ ($\text{kcal/mol}$).
   * **Pharmacokinetic & Phenotypic Tracks:** SwissADME/pkCSM composite ADME score, multi-organ toxicological safety hard-stop index (Ames, hERG I/II, GHS $\text{LD}_{50}$), and Way2Drug PASS antiviral bioactivity probability ($P_a > P_i$).
   * **Transformation:** Lower-is-better energetic metrics undergo directional sign inversion ($X'_i = -X_i$) prior to localized receptor-subgroup Min-Max normalization.

2. **Tier 1: Unsupervised Consensus Method — UCM (`src/consensus_engine.py`)**
   * **Formulation:** Evaluates an unweighted arithmetic mean across all $N = 6$ standardized descriptor tracks ($w_k = 16.67\%$ per domain) to eliminate single-predictor noise and establish an unbiased baseline stability index:
     $$S_{\text{UCM}} = \frac{1}{N} \sum_{k=1}^{N} S_k$$

3. **Tier 2: Supervised Consensus Method — SCM (`src/consensus_engine.py`)**
   * **Formulation:** Applies a biologically constrained linear combination allocating **65% weight to 3D structural engagement** and **35% weight to 2D ADMET/safety profiles**, followed by non-parametric hierarchical tie-breaking (`Receptor_ID` $\rightarrow$ `Supervised` $\rightarrow$ `Unsupervised`):
     $$S_{\text{SCM}} = 0.25 S_{\text{Vina}} + 0.20 S_{\text{CNN}} + 0.20 S_{\text{Induced}} + 0.12 S_{\text{ADME}} + 0.13 S_{\text{Tox}} + 0.10 S_{\text{PASS}}$$

4. **Tier 3: Perturbation Sensitivity Analysis (`src/consensus_engine.py`)**
   * Validates ranking robustness against heuristic weight drift by executing Monte Carlo / systematic $\pm 10\%$ weight perturbations and reporting Kendall's $\tau$ rank correlation stability.

---

## Directory Layout

```text
├── assets/                 # Workflow and consensus architecture diagrams
├── data/
│   ├── raw/                # Input multi-track screening matrices (CSV)
│   └── processed/          # Normalized & ranked consensus lead tables
├── notebooks/
│   └── consensus_lead_scoring_colab.ipynb  # Interactive Google Colab notebook
├── src/
│   ├── __init__.py
│   ├── normalizer.py       # Directional sign inversion & Min-Max scaling
│   ├── consensus_engine.py # UCM/SCM scoring, tie-breaking & sensitivity audit
│   └── visualizer.py       # 600 DPI publication figures (Pareto & sensitivity plots)
├── results/
│   ├── figures/            # Exported 600 DPI PNG/TIFF figures
│   └── tables/             # Exported ranked lead tables (Table 1 & Table S2)
├── CITATION.cff            # Academic citation metadata
├── environment.yml         # Reproducible Conda environment
├── LICENSE                 # MIT License
└── README.md               # Repository documentation
