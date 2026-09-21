# STEP-MHC

Paired-chain physicochemical modelling of T-cell receptor–peptide–HLA recognition.

STEP-MHC encodes all six complementarity-determining region loops of the paired αβ
receptor, the peptide and a 34-residue HLA pseudosequence as physicochemical
difference maps, and scores receptor–peptide–HLA recognition from sequence alone.

This repository accompanies the manuscript. Code is released under MIT; figure source
data under CC BY 4.0.

## Contents

## Requirements

```bash
conda env create -f environment.yml
conda activate stepmhc
```

Python 3.10, PyTorch, NumPy, pandas, SciPy, scikit-learn, matplotlib, peptides.
Figures additionally require R with ggplot2, dplyr, tidyr, patchwork and pROC.

## Configuration

Two paths are supplied by the user rather than bundled:

```bash
export STEPMHC_ROOT=/path/to/working/directory
export STEPMHC_HLA_CSV=/path/to/common_hla_sequence.csv
```

`STEPMHC_HLA_CSV` is the HLA pseudosequence lookup (two columns, `HLA_type` and
`HLA_sequence`, 34 residues per allele). It is distributed with PISTE and is not
redistributed here.

## Scoring with the trained model

```python
import torch
from src.model_stepmhc_small import STEPMHCModelSmall
from src.dataset_stepmhc import StepMHCDataset
from torch.utils.data import DataLoader

model = STEPMHCModelSmall(use_structure_priors=False)
model.load_state_dict(torch.load("models/stepmhc_small_fullpool_seed42.pt"))
model.eval()

ds = StepMHCDataset("your_pairs.csv")          # columns: A1 A2 A3 B1 B2 B3, peptide, allele
with torch.no_grad():
    for te, ph, g, _ in DataLoader(ds, batch_size=256):
        logits = model(te, ph, g)
        scores = torch.sigmoid(logits)
```

The output is a recognition score for ranking candidates. It is not a calibrated
probability and not a binding affinity.

## Reproducibility

Random seeds are fixed where they govern data: 42 for negative sampling, dataset
assembly and train/validation splitting; 1 for the R bootstrap; 20260831 for the
structure bootstrap. Permutation analyses use a NumPy generator seeded at 42.

**Retraining reproduces the protocol but not the exact weights.** `bench_train_step.py`
uses an unseeded data-loader shuffle and PyTorch's default non-deterministic kernels,
so a retrained model will differ numerically from the released checkpoint. The
checkpoint is therefore included so that every reported score can be reproduced
exactly.

Training data are not redistributed here. The positive pool derives from VDJdb,
McPAS-TCR and IEDB under those resources' own terms; clinical cohorts are cited in
the manuscript with their accessions.

## Not included

FoldX is licensed software and is not redistributed; `src/structure_figures_v1/`
expects a local FoldX 5.1 installation. Comparator models (NetTCR-2.2, UniPMT,
pMTnet, TEIM, PanPep, ImRex) are obtained from their own repositories; the adapters
here convert inputs and collect outputs only.

## Citation

Citation details will be added on publication.
