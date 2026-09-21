"""Stage 3: internal dedup + wall-off vs PISTE test I/II + dbPepNeo.
Internal dedup on 7-tuple (A1,A2,A3,B1,B2,B3,peptide).
Wall-off key = (CDR3beta, peptide) from held-out benchmark sets (all labels).
NeoTCR: ABSENT from disk -> NOT walled off (documented gap). Writes positives_clean.csv."""
import os
import pandas as pd
from pathlib import Path

ROOT = Path(os.environ.get("STEPMHC_ROOT", "."))
IN   = ROOT/"data/broadened/positives_pseudo.csv"
OUT  = ROOT/"data/broadened/positives_clean.csv"
PD   = Path("<EXTERNAL_DATA>/previous_data/Sequence_datasets/Previous_Studies/PISTE")

WALLOFF = [
    PD/"data/raw_data/pos_test_1.csv",          # PISTE test set I (489)
    PD/"data/raw_data/pos_test_2.csv",          # PISTE test set II (425)
    PD/"data/reftcr/dbpepneo_data.csv",         # dbPepNeo (neoantigen)
    PD/"data/random/dbpepneo_data.csv",
    PD/"data/unipep/dbpepneo_data.csv",
    PD/"data/reftcr/test_data.csv",             # reftcr-scheme test (incl negatives)
    PD/"data/random/test_data.csv",
    PD/"data/unipep/test_data.csv",
    PD/"data/raw_data/ref_tcr_test1.csv",
    PD/"data/raw_data/ref_tcr_test2.csv",
]

df = pd.read_csv(IN)
n0 = len(df)

# 1. internal dedup on full complex + peptide
key7 = ["A1","A2","A3","B1","B2","B3","peptide"]
df = df.drop_duplicates(subset=key7).reset_index(drop=True)
print(f"internal dedup: {n0} -> {len(df)} ({n0-len(df)} dup complexes removed)")

# 2. build wall-off key set = (CDR3beta upper, peptide upper) from all held-out files, all labels
wall = set()
for f in WALLOFF:
    if not f.exists():
        print(f"   [skip, missing] {f.name}"); continue
    w = pd.read_csv(f)
    if "CDR3" not in w.columns or "MT_pep" not in w.columns:
        print(f"   [skip, no CDR3/MT_pep cols] {f.name}: {list(w.columns)}"); continue
    pairs = set(zip(w.CDR3.astype(str).str.upper().str.strip(),
                    w.MT_pep.astype(str).str.upper().str.strip()))
    wall |= pairs
    print(f"   walloff source {f.name}: +{len(pairs)} pairs (cum {len(wall)})")

print(f"\nNeoTCR: NOT walled off (absent from disk) -- documented gap; close before any neoantigen claim.")

# 3. drop broadened rows whose (B3,peptide) is in the wall set
bkey = list(zip(df.B3.astype(str).str.upper().str.strip(),
                df.peptide.astype(str).str.upper().str.strip()))
leak = pd.Series([k in wall for k in bkey], index=df.index)
print(f"\nwall-off: {int(leak.sum())} rows removed (their CDR3b+peptide appears in a held-out set)")
df = df[~leak].reset_index(drop=True)

vc = df.peptide.value_counts()
df.to_csv(OUT, index=False)
print(f"\n=== CLEAN broadened positives ===")
print(f"rows={len(df)} epitopes={df.peptide.nunique()} alleles={df.hla_key.nunique()}")
print(f"usable epitopes (>=10 pos): {(vc>=10).sum()}  (>=20): {(vc>=20).sum()}")
print(f"by origin: {df.origin.value_counts().to_dict()}")
print(f"saved -> {OUT}")
