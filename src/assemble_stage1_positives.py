"""Stage 1: harmonize VDJdb + McPAS -> positives in NetTCR-2.2 schema.
Loops A1/A2/B1/B2 from V-genes via ANARCI IMGT germlines (validated == NetTCR 20/20).
A3/B3 = CDR3 directly. Writes data/broadened/positives_raw.csv. Positives only."""
import os
import pandas as pd, re
from pathlib import Path
from anarci import germlines as G

ROOT = Path(os.environ.get("STEPMHC_ROOT", "."))
OUT  = ROOT/"data/broadened/positives_raw.csv"
VDJ  = "<EXTERNAL_DATA>/previous_data/Sequence_datasets/Databases/VDJdb/vdjdb-2025-07-30/vdjdb_full_filtered.txt"
MCP  = "<EXTERNAL_DATA>/previous_data/Sequence_datasets/Databases/McPAS/original_McPAS-TCR.csv"

Aseq = G.all_germlines['V']['A']['human']; Akeys = sorted(Aseq)
Bseq = G.all_germlines['V']['B']['human']; Bkeys = sorted(Bseq)

def stem_of(k): return k.split("*")[0].split("/")[0]

def norm(g, keys):
    if pd.isna(g): return None
    g = str(g).strip().upper().replace("\xa0","").replace(" ","").replace(":","*")
    g = re.sub(r"-(\d{2})$", r"*\1", g)
    g = re.sub(r"(TRA[VBDJ])0(\d)", r"\1\2", g)
    if g in keys: return g
    stem = g.split("*")[0]
    c = [k for k in keys if stem_of(k)==stem]
    if c: return sorted(c)[0]
    c = [k for k in keys if stem_of(k).startswith(stem+"-")]
    if c: return sorted(c, key=lambda k:(len(stem_of(k)),k))[0]
    return None

def loops(key, seqd):
    s = seqd[key]
    return s[26:38].replace("-",""), s[55:65].replace("-","")

def clean_cdr3(x):
    x = str(x).strip().upper()
    return x if re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]{4,}", x) else None

def harmonize(name, df, ca, cb, va, vb, ep, allele):
    n0 = len(df); rows = []; drops = {"cdr3":0,"vgene":0,"loop":0,"pep":0}
    for _, r in df.iterrows():
        a3, b3 = clean_cdr3(r[ca]), clean_cdr3(r[cb])
        if not a3 or not b3: drops["cdr3"]+=1; continue
        ak, bk = norm(r[va], Akeys), norm(r[vb], Bkeys)
        if ak is None or bk is None: drops["vgene"]+=1; continue
        try:
            a1,a2 = loops(ak, Aseq); b1,b2 = loops(bk, Bseq)
        except Exception:
            drops["loop"]+=1; continue
        if not (a1 and a2 and b1 and b2): drops["loop"]+=1; continue
        pep = str(r[ep]).strip().upper()
        if not re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]{8,12}", pep): drops["pep"]+=1; continue
        rows.append(dict(A1=a1,A2=a2,A3=a3,B1=b1,B2=b2,B3=b3,peptide=pep,
                         allele=str(r[allele]).strip(), binder=1, origin=name))
    out = pd.DataFrame(rows)
    print(f"\n[{name}] {n0} filtered -> {len(out)} positives")
    print(f"   drops: cdr3={drops['cdr3']} vgene={drops['vgene']} loop={drops['loop']} pep={drops['pep']}")
    print(f"   epitopes={out.peptide.nunique()} alleles={out.allele.nunique()}")
    return out

v = pd.read_csv(VDJ, sep="\t", low_memory=False)
v = v[(v.species=="HomoSapiens")&(v["mhc.class"]=="MHCI")&(v["vdjdb.score"]>=1)]
v = v[v["cdr3.alpha"].notna()&v["cdr3.beta"].notna()]
vh = harmonize("VDJdb", v, "cdr3.alpha","cdr3.beta","v.alpha","v.beta","antigen.epitope","mhc.a")

m = pd.read_csv(MCP, encoding="latin-1", low_memory=False)
m = m[(m.Species=="Human")&m["MHC"].astype(str).str.match(r"HLA-[ABC]")]
m = m[m["CDR3.alpha.aa"].notna()&m["CDR3.beta.aa"].notna()]
mh = harmonize("McPAS", m, "CDR3.alpha.aa","CDR3.beta.aa","TRAV","TRBV","Epitope.peptide","MHC")

# --- IEDB (pre-harmonized by assemble_iedb_frontend.py) ---
ied = pd.read_csv(ROOT/"data/broadened/iedb_harmonized.csv")
ih = harmonize("IEDB", ied, "cdr3.alpha","cdr3.beta","v.alpha","v.beta","antigen.epitope","mhc.a")

both = pd.concat([vh, mh, ih], ignore_index=True)
both.to_csv(OUT, index=False)
print("\n=== COMBINED (pre-dedup) ===")
print(f"rows={len(both)} epitopes={both.peptide.nunique()} alleles={both.allele.nunique()}")
print(f"by origin: {both.origin.value_counts().to_dict()}")
print(f"top epitopes: {both.peptide.value_counts().head(8).to_dict()}")
print(f"saved -> {OUT}")
