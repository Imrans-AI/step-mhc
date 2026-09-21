# STEP-MHC: Paired-Chain TCR–Peptide–HLA Recognition

![License](https://img.shields.io/badge/License-MIT-lightgrey)
![python](https://img.shields.io/badge/python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)

Official implementation of **STEP-MHC**, a convolutional model that predicts T-cell
receptor recognition of peptide–HLA class I complexes from sequence alone, using all
six complementarity-determining region (CDR) loops of the paired αβ receptor.

> **STEP-MHC: paired-chain physicochemical modelling of T-cell receptor–peptide–HLA
> recognition, with evaluation-regime controls and structural energetic
> cross-comparison.** Muhammad Imran, [author list to be finalised], Dongqing Wei.
> *Manuscript in preparation, 2026.*

---

## Highlights

- **0.740 ROC-AUC** and **0.357 average precision** on a frozen neoantigen benchmark
  (684 records, 114 positive, prevalence 1/6), against **0.486–0.512** for six released
  predictors evaluated on identical records
- **Permutation significance P < 0.001** (1,000 label permutations); none of the six
  comparators reached significance (P = 0.35–0.70)
- **0.37 AUROC shift from negative construction alone** — one released model and its own
  positives, changing only how negatives are generated (0.966 → 0.593)
- **Regime-dependent ranking**: NetTCR-2.2 leads on its viral, peptide-rich split
  (0.961 vs 0.815); the ordering reverses on neoantigens (0.508 vs 0.670)
- **Clinical ranking** across four retrospective cohorts (154 records): pooled median
  percentile **0.760** against an empirical null of 0.578 (P ≈ 0.002)
- **Orthogonal structural check**: 233 alanine substitutions across nine crystal
  structures, five paired FoldX replicates each

---

## Model Architecture

STEP-MHC integrates:

1. **Five-channel physicochemical difference maps** — TCR × peptide (5 × 73 × 12) and
   peptide × HLA (5 × 12 × 34), encoding pairwise absolute differences in
   hydrophobicity, charge, flexibility, refractivity and accessible surface area
2. **All six CDR loops** of the paired receptor (A1 7, A2 8, A3 22, B1 6, B2 7, B3 23
   residues) on a 73-position axis
3. **Explicit HLA context** as a 34-residue pseudosequence
4. **Two independent ResNet encoders** (3 × 3 convolutions, 32 → 64 → 128 channels,
   stride 1, adaptive average pooling)
5. **Seven global biophysical descriptors** (Δ TCR–peptide), layer-normalised with a
   single-token residual transform
6. **Fused 263-feature head** with dropout 0.4 → sigmoid recognition score

**633,071 trainable parameters.**

---

## Installation

```bash
git clone https://github.com/Imrans-AI/step-mhc.git
cd step-mhc
conda env create -f environment.yml
conda activate stepmhc
```

### Requirements

- Python 3.10+
- PyTorch 2.x (STEP-MHC was trained with CUDA 12.8)
- NumPy 1.26.4, pandas, SciPy, scikit-learn, matplotlib
- `peptides` 0.5.0 (global descriptors)
- R 4.3+ with ggplot2, dplyr, tidyr, patchwork, pROC (figures only)

> The HLA pseudosequence lookup is not redistributed here. Supply it via
> `export STEPMHC_HLA_CSV=/path/to/common_hla_sequence.csv` (two columns,
> `HLA_type` and `HLA_sequence`, 34 residues per allele).

---

## Quick Start

### Run Inference

```python
import torch
from src.model_stepmhc_small import STEPMHCModelSmall
from src.dataset_stepmhc import StepMHCDataset
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = STEPMHCModelSmall(use_structure_priors=False).to(device)
model.load_state_dict(torch.load(
    'models/stepmhc_small_fullpool_seed42.pt', map_location=device))
model.eval()

# Load data
dataset = StepMHCDataset('your_pairs.csv')
loader = DataLoader(dataset, batch_size=256, shuffle=False)

# Predict
predictions = []
with torch.no_grad():
    for te, ph, g, _ in loader:
        logits = model(te.to(device), ph.to(device), g.to(device))
        predictions.extend(torch.sigmoid(logits).cpu().numpy())
```

### Train from Scratch

```bash
python src/bench_train_step.py
```

---

## Data

### Input Format

CSV file with columns:

| Column | Description |
| --- | --- |
| `A1` `A2` `A3` | CDR1α, CDR2α, CDR3α amino acid sequences |
| `B1` `B2` `B3` | CDR1β, CDR2β, CDR3β amino acid sequences |
| `peptide` | Peptide sequence (MHC class I, 8–12 aa) |
| `allele` | HLA allele (e.g. `HLA-A*02:01`) |
| `label` | 1 = binding, 0 = non-binding |

### Data Sources

| Database | Retained records | Reference |
| --- | --- | --- |
| IEDB | 7,046 | Vita et al., 2019 |
| VDJdb | 2,373 | Shugay et al., 2018 |
| McPAS-TCR | 1,244 | Tickotsky et al., 2017 |

After harmonisation, HLA resolution, deduplication and benchmark exclusion:
**10,663 positive records** spanning **1,149 peptide sequences**. Negatives are
peptide-swapped within peptide at up to 5 per positive, giving **63,566 rows** at a
positive fraction of 0.168.

> **Note:** Training data are not redistributed here. Source databases are available
> from their own repositories under their respective terms.

---

## Reproducing Paper Results

### Benchmark and permutation tests

```bash
python src/build_common_testset_v5.py
python src/bench_metrics.py
```

### Clinical cohorts and NeoTCR

```bash
python src/caushi_score_v2.py
python src/neotcr_score.py
```

### Structural cohort and FoldX comparison

```bash
cd src/structure_figures_v1
python 10_cohort_eligibility_screen.py
python 11_freeze_cohort.py
python 15_prepare_and_repair.py       # requires local FoldX 5.1
python 02_residue_perturbation.py
```

### Figures

```bash
Rscript src/figures/fig_bench_v9_stepmhc.R     # Figure 2: benchmark
Rscript src/figures/fig3_v2_stepmhc.R          # Figure 3: breakdowns and regimes
Rscript src/figures/fig_clinical_upper.R       # Figure 4: clinical cohorts
Rscript src/figures/fig_immrep_roc.R           # Supplementary S4: IMMREP
```

---

## Repository Structure

---

## Performance

### Frozen neoantigen benchmark (n = 684, 114 positive, prevalence 1/6)

| Model | ROC-AUC | Average precision | Permutation P |
| --- | --- | --- | --- |
| **STEP-MHC (paired)** | **0.740** | **0.357** | **< 0.001** |
| **STEP-MHC (β-only)** | **0.669** | **0.315** | **< 0.001** |
| PanPep | 0.512 | 0.175 | 0.35 |
| TEIM | 0.509 | 0.175 | 0.37 |
| NetTCR-2.2 | 0.508 | 0.189 | 0.39 |
| UniPMT | 0.491 | 0.166 | 0.64 |
| pMTnet | 0.490 | 0.159 | 0.62 |
| ImRex | 0.486 | 0.170 | 0.70 |

Average precision expected under random ranking is the prevalence, 0.167. TEIM and
ImRex scored 678 of 684 records under their native input constraints. Non-significance
is a failure to detect association, not evidence of equivalence to chance.

### Evaluation regime

| Condition | NetTCR-2.2 | STEP-MHC |
| --- | --- | --- |
| Viral, peptide-rich (NetTCR-2.2 split) | **0.961** | 0.815 |
| Neoantigen (BNT221 benchmark) | 0.508 | **0.670** |

Neither model dominates. Separately, holding one released model and its positives fixed
and changing only negative generation moved AUROC from **0.966 to 0.593**.

### Retrospective clinical cohorts

| Cohort | n | Median percentile |
| --- | --- | --- |
| BNT221 | 114 | 0.797 |
| HBV+ HCC | 24 | 0.716 |
| Caushi (lung) | 9 | 0.658 |
| NSCLC TIL | 7 | 0.515 |
| **Pooled** | **154** | **0.760** |

Empirical permutation null 0.578; add-one Monte Carlo P ≈ 0.002 (0 of 500 draws).
A pre-specified analysis excluding BNT221 gives P ≈ 0.060 and **is not significant**.
Reference panel is matched on peptide and HLA only, not cohort, tumour or patient.

### Candidate recovery

| Selection | Top-k | Fold over random ranking |
| --- | --- | --- |
| Neoantigen (32-candidate panel) | 1 | 3.6× |
| Neoantigen | 5 | 2.9× |
| Receptor (601 candidates per antigen) | 5 | 6.2× |
| Receptor | 25 | 3.3× |

Retrospective recovery, not prospectively observed clinical benefit.

### External and structural

IMMREP pooled ROC-AUC **0.776** (macro-average 0.699); per-epitope 0.480–0.904,
correlated with training depth (Spearman ρ = 0.750, P = 0.005, n = 12).

FoldX on nine sequence-matched structures: strongest stable effects 2BNR peptide W5
**6.93 kcal mol⁻¹** and TCRα Y31 5.20; 3QDJ TCRβ F100 2.52 and peptide I4 1.57.
Within-structure rank correspondence with model sensitivity is weak (median Spearman
+0.024 peptide, +0.135 TCR, stable subset) — the two quantify complementary properties.

---

## Reproducibility

Seeds are fixed where they govern data: **42** for negative sampling, assembly and
train/validation splitting; **1** for the R bootstrap; **20260831** for the structure
bootstrap. Permutation analyses use a NumPy generator seeded at 42.

> **Retraining reproduces the protocol, not the exact weights.**
> `bench_train_step.py` uses an unseeded data-loader shuffle and PyTorch's default
> non-deterministic kernels. The trained checkpoint is included so that every reported
> score can be reproduced exactly.

FoldX is licensed software and is not redistributed; `src/structure_figures_v1/`
expects a local FoldX 5.1 installation. Comparator models are obtained from their own
repositories; the adapters here convert inputs and collect outputs only.

---

## Data and Model Availability

The trained checkpoint (`models/stepmhc_small_fullpool_seed42.pt`, 2.5 MB) and source
tables for every published figure (`data/figure_source/`) are included in this
repository. Figure source data are released under CC BY 4.0.

*A Zenodo archive DOI will be added on publication.*

---

## Citation

*Citation details will be added on publication.*

---

## License

Code is licensed under the MIT License — see [LICENSE](LICENSE).
Figure source data are licensed under CC BY 4.0.

---

## Contact

Muhammad Imran — Shanghai Jiao Tong University.
For questions, please open an issue.
