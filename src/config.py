"""Central config for STEP-MHC (Module 2). Import constants/paths from here, not hardcoded."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA, SRC, SCRIPTS, RESULTS = ROOT/"data", ROOT/"src", ROOT/"scripts", ROOT/"results"
SPLITS = DATA/"splits"
PRIORS = {"mhc": DATA/"priors/structure_priors_mhc.npz",
          "triplet": DATA/"priors/structure_priors_triplet.npz"}
STRUCTURES = DATA/"structures"

# HLA pseudo-sequence lookup (generic NetMHCpan 34-mer table; reference only, not PISTE train/test)
HLA_CSV = Path("<EXTERNAL_DATA>/previous_data/Sequence_datasets/"
               "Previous_Studies/PISTE/data/raw_data/common_hla_sequence.csv")

# geometry (locked, verified against NetTCR-2.2)
CAPS  = {"A1":7, "A2":8, "A3":22, "B1":6, "B2":7, "B3":23}
LOOPS = list(CAPS); L_TCR = sum(CAPS.values())  # 73
L_PEP, L_HLA = 12, 34
TCR_EPI_SHAPE, PEP_HLA_SHAPE = (5, L_TCR, L_PEP), (5, L_PEP, L_HLA)

# training defaults
SEED, BS, LR, WD, MAX_EPOCHS, PATIENCE = 42, 32, 5e-4, 1e-4, 40, 6
PRIOR_MIN_POS = 10  # per-epitope n_pos floor for macro metrics
