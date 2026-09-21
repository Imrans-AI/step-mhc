"""Stage 2: attach 34-mer HLA pseudoseq to broadened positives.
Normalizes messy allele strings -> PISTE common_hla_sequence.csv keys (HLA-A02:01 form).
Drops rows whose allele can't resolve. Writes positives_pseudo.csv + coverage report."""
import os
import pandas as pd, re
from pathlib import Path

ROOT = Path(os.environ.get("STEPMHC_ROOT", "."))
IN   = ROOT/"data/broadened/positives_raw.csv"
OUT  = ROOT/"data/broadened/positives_pseudo.csv"
HLA  = os.environ["STEPMHC_HLA_CSV"]

lut = pd.read_csv(HLA)
PSEUDO = dict(zip(lut.HLA_type, lut.HLA_sequence))
KEYS = set(PSEUDO)

def norm_allele(a):
    """Map source allele -> PISTE key 'HLA-A02:01'. Returns key or None."""
    if pd.isna(a): return None
    a = str(a).strip().upper().replace("\xa0","").replace(" ","")
    a = a.replace("HLA-","")                 # A*02:01:48 -> work on body
    a = a.replace("*","")                    # A02:01
    m = re.match(r"([ABC])(\d+)(?::(\d+))?", a)
    if not m: return None
    gene, grp, prot = m.group(1), m.group(2), m.group(3)
    grp = grp.zfill(2)                       # 2 -> 02
    # try with given protein field (2-field), else default :01
    cands = []
    if prot:
        cands.append(f"HLA-{gene}{grp}:{prot.zfill(2)}")
    cands.append(f"HLA-{gene}{grp}:01")      # serotype default
    for c in cands:
        if c in KEYS: return c
    # last resort: first PISTE key with this gene+group
    pref = f"HLA-{gene}{grp}:"
    hits = sorted(k for k in KEYS if k.startswith(pref))
    return hits[0] if hits else None

df = pd.read_csv(IN)
df["hla_key"] = df.allele.map(norm_allele)
resolved = df.hla_key.notna()
print(f"rows: {len(df)} -> allele-resolved: {int(resolved.sum())} ({resolved.mean()*100:.0f}%)")
print(f"unresolved allele examples: {sorted(df.allele[~resolved].dropna().unique())[:12]}")

df = df[resolved].copy()
df["hla_pseudo"] = df.hla_key.map(PSEUDO)
plen = df.hla_pseudo.str.len()
print(f"pseudoseq lengths present: {sorted(plen.unique())}")   # expect [34]
df.to_csv(OUT, index=False)

vc = df.peptide.value_counts()
print(f"\n=== after allele resolution ===")
print(f"rows={len(df)} epitopes={df.peptide.nunique()} alleles(canonical)={df.hla_key.nunique()}")
print(f"usable epitopes (>=10 pos): {(vc>=10).sum()}  (>=20): {(vc>=20).sum()}")
print(f"top canonical alleles: {df.hla_key.value_counts().head(8).to_dict()}")
print(f"saved -> {OUT}")
