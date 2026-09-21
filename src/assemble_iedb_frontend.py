"""IEDB front-end: parse receptor_full_v3 -> VDJdb/McPAS-style rows for the assembler.
Chain-typed by Type field (alpha/beta), full C...F CDR3 convention only (no fabrication),
paired + linear epitope (8-12) + HLA-I. Emits cdr3.alpha/beta, v.alpha/beta, peptide, allele."""
import pandas as pd, re, warnings; warnings.filterwarnings("ignore")
from pathlib import Path

SRC = "<EXTERNAL_DATA>/previous_data/Sequence_datasets/Databases/IEDB/receptor_full_v3/tcr_full_v3.csv"
OUT = Path("data/broadened/iedb_harmonized.csv")

df = pd.read_csv(SRC, header=[0,1], low_memory=False)
df.columns = [f"{a}|{b}" for a,b in df.columns]
n0 = len(df)

# chain-type guard: Chain 1 must be alpha, Chain 2 must be beta (drop gamma/delta)
c1t, c2t = df["Chain 1|Type"], df["Chain 2|Type"]
df = df[(c1t=="alpha") & (c2t=="beta")].copy()

a3 = df["Chain 1|CDR3 Curated"].fillna(df["Chain 1|CDR3 Calculated"]).astype(str).str.strip().str.upper()
b3 = df["Chain 2|CDR3 Curated"].fillna(df["Chain 2|CDR3 Calculated"]).astype(str).str.strip().str.upper()
va = df["Chain 1|Curated V Gene"].fillna(df["Chain 1|Calculated V Gene"])
vb = df["Chain 2|Curated V Gene"].fillna(df["Chain 2|Calculated V Gene"])
epi = df["Epitope|Name"].astype(str).str.strip().str.upper()
mhc = df["Assay|MHC Allele Names"].astype(str)

# full C...F convention only (no fabrication of anchors)
full_cdr3 = lambda s: s.str.match(r"^C[ACDEFGHIKLMNPQRSTVWY]{3,}[FW]$")
keep = (full_cdr3(a3) & full_cdr3(b3)
        & va.notna() & vb.notna()
        & epi.str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]{8,12}")
        & mhc.str.contains(r"HLA-[ABC]", case=False, na=False))

out = pd.DataFrame({
    "cdr3.alpha": a3[keep], "cdr3.beta": b3[keep],
    "v.alpha": va[keep], "v.beta": vb[keep],
    "antigen.epitope": epi[keep],
    # IEDB MHC names are verbose ("HLA-A*02:01" or "HLA class I... A*02:01") -> extract allele
    "mhc.a": mhc[keep].str.extract(r"(HLA-[ABC]\*?\d{2}:?\d{0,2})")[0],
})
out = out.dropna(subset=["mhc.a"]).reset_index(drop=True)
out.to_csv(OUT, index=False)

print(f"IEDB: {n0} rows -> {len(out)} clean paired (full C..F, HLA-I, linear epitope)")
print(f"  distinct epitopes: {out['antigen.epitope'].nunique()}")
vc = out["antigen.epitope"].value_counts()
print(f"  epitopes >=10 paired TCRs: {(vc>=10).sum()}")
print(f"  V-gene samples: alpha {out['v.alpha'].dropna().head(3).tolist()}, beta {out['v.beta'].dropna().head(3).tolist()}")
print(f"  allele samples: {out['mhc.a'].dropna().unique()[:6].tolist()}")
print(f"saved -> {OUT}")
