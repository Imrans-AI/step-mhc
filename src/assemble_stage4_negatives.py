"""Stage 4: reference-TCR negatives (EPACT/PISTE scheme), up to 1:5.
Each negative tagged with neg_per_pos = (its rank within epitope)/(epitope's positives),
so a 1:N subset = neg_per_pos <= N, uniform across epitopes of any size.
Guard: a sampled (TCR,epitope) negative is rejected if it's a real positive anywhere.
Writes broadened_train_ready.csv with binder in {0,1}; positives have neg_per_pos=0."""
import os
import pandas as pd, numpy as np
from pathlib import Path

ROOT = Path(os.environ.get("STEPMHC_ROOT", "."))
IN   = ROOT/"data/broadened/positives_clean.csv"
OUT  = ROOT/"data/broadened/broadened_train_ready.csv"
RATIO, SEED = 5, 42
rng = np.random.RandomState(SEED)

pos = pd.read_csv(IN)
TCRCOLS = ["A1","A2","A3","B1","B2","B3"]
pos_pairs = set(zip(pos[TCRCOLS].astype(str).agg("|".join, axis=1), pos.peptide))
tcr_pool = pos.drop_duplicates(subset=TCRCOLS)[TCRCOLS].reset_index(drop=True)
tcr_keys = tcr_pool.astype(str).agg("|".join, axis=1).values
epi_meta = pos.groupby("peptide").agg(hla_key=("hla_key","first"),
                                      hla_pseudo=("hla_pseudo","first"),
                                      allele=("allele","first")).to_dict("index")

neg_rows = []
for pep, grp in pos.groupby("peptide"):
    npos = len(grp); want = npos * RATIO
    cand = [i for i,k in enumerate(tcr_keys) if (k, pep) not in pos_pairs]
    rng.shuffle(cand)
    meta = epi_meta[pep]
    for rank, ci in enumerate(cand[:want], start=1):
        neg_rows.append({**tcr_pool.iloc[ci].to_dict(), "peptide":pep,
                         "allele":meta["allele"], "hla_key":meta["hla_key"],
                         "hla_pseudo":meta["hla_pseudo"], "binder":0,
                         "origin":"ref_neg", "neg_per_pos": rank/npos})

neg = pd.DataFrame(neg_rows)
pos2 = pos.copy(); pos2["neg_per_pos"] = 0.0
full = pd.concat([pos2, neg], ignore_index=True)
full.to_csv(OUT, index=False)

vc = full[full.binder==1].peptide.value_counts()
def frac_at(n): 
    s = full[(full.binder==1) | (full.neg_per_pos<=n) & (full.binder==0)]
    return s.binder.mean()
print(f"positives={int((full.binder==1).sum())} negatives={int((full.binder==0).sum())}")
print(f"pos fraction full(<=5): {full.binder.mean():.3f}")
print(f"pos fraction at 1:1 (neg_per_pos<=1): {frac_at(1):.3f}")
print(f"pos fraction at 1:3 (neg_per_pos<=3): {frac_at(3):.3f}")
print(f"epitopes={full.peptide.nunique()} usable(>=10 pos)={(vc>=10).sum()}")
neg_pairs = set(zip(neg[TCRCOLS].astype(str).agg('|'.join,axis=1), neg.peptide))
print(f"INTEGRITY collisions (MUST be 0): {len(neg_pairs & pos_pairs)}")
print(f"saved -> {OUT}")
