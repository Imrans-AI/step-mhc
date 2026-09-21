"""Caushi 2021 — PRIMARY, CORRECTED NEGATIVES.

WHY THIS SUPERSEDES THE FIRST RUN
  The first run used BNT221 healthy-donor reference TCRs as negatives. That was the wrong
  control: the permutation null came out at 0.666, not 0.500, because ANY tumour-infiltrating
  expanded clone outscores healthy-donor repertoires on these peptides regardless of which
  peptide it is. The test was measuring "TIL vs healthy donor", not "cognate vs non-cognate".

FIX: negatives are NON-REACTIVE TUMOUR-INFILTRATING clonotypes (GSE255549, 23,560 available,
paired alpha/beta, 8 NSCLC patients). Positives and negatives are then matched on BOTH
  (a) the peptide + HLA they are scored against, and
  (b) repertoire origin (tumour-infiltrating, expanded).
The only remaining difference is measured reactivity. This is the construction that made
BNT221 interpretable and the control that exposed the MAGE-A4 artifact.

ALSO FIXED
  - C*02:02 recovered from the full pseudo-sequence table (was dropped for want of a pseudo
    in the training pool alone), restoring FAAQAGAWKI -> 9 clonotypes instead of 8.
  - Influenza rows are a PIPELINE CHECK ONLY, not a result: both chains of all three are
    already in the training pool (public GILGFVFTL motifs TRBV19/CASSIRSSYEQYF), so their
    AUROC 0.972 is substantially memorisation. Reported as such, never as evidence.

PRE-SPECIFIED: permutation null with 500 shuffles; permutation p is the reportable value.
A null is reported as a null.
"""
import pandas as pd, numpy as np, torch, sys, glob, re
from sklearn.metrics import roc_auc_score
sys.path.insert(0, '.')
from src.model_stepmhc_small import STEPMHCModelSmall
from src.dataset_stepmhc import _imap, _globals, CAPS, LOOPS, L_TCR, L_PEP, L_HLA
import src.dataset_stepmhc as DS
from anarci import germlines as G

N_NEG = 600          # non-reactive TIL clonotypes per peptide
rng_global = np.random.default_rng(42)

# ---------------------------------------------------------------- germline loops
Aseq = G.all_germlines['V']['A']['human']; Akeys = sorted(Aseq)
Bseq = G.all_germlines['V']['B']['human']; Bkeys = sorted(Bseq)
stem = lambda k: k.split("*")[0].split("/")[0]

def norm_v(g, keys):
    if pd.isna(g): return None
    s = str(g).strip().upper().replace("\xa0", "")
    s = s.replace("TCRBV", "TRBV").replace("TCRBJ", "TRBJ").replace("TCRAV", "TRAV")
    s = re.sub(r"\*\d+$", "", s)
    s = re.sub(r"(TR[AB]V)0(\d)", r"\1\2", s)
    s = re.sub(r"-0(\d)", r"-\1", s)
    for cand in (s, s.split("/")[0]):
        if cand in keys: return cand
        hit = [k for k in keys if stem(k) == cand]
        if hit: return sorted(hit)[0]
        hit = [k for k in keys if stem(k).startswith(cand + "-")]
        if hit: return sorted(hit, key=lambda k: (len(stem(k)), k))[0]
    return None

def loops(key, seqd):
    s = seqd[key]
    return s[26:38].replace("-", ""), s[55:65].replace("-", "")

# ---------------------------------------------------------------- HLA pseudo-sequences
# training pool first, then the FULL NetMHCpan pseudo table used by dataset_stepmhc
nrm = lambda a: str(a).replace("HLA-", "").replace("*", "").replace(":", "").strip().upper()
PSEUDO = {}
pool = pd.concat([pd.read_csv(f) for f in
                  glob.glob("data/splits/unseen_epitope_broadened/fold*/train.csv")],
                 ignore_index=True)
for _, r in pool.drop_duplicates("allele").iterrows():
    PSEUDO.setdefault(nrm(r.allele), r.hla_pseudo)
n_pool = len(PSEUDO)
try:
    h = pd.read_csv(DS.HLA_CSV)
    for k, v in zip(h.iloc[:, 0], h.iloc[:, 1]):
        PSEUDO.setdefault(nrm(k), v)
    print(f"pseudo-seqs: {n_pool} from training pool + {len(PSEUDO)-n_pool} from {DS.HLA_CSV}")
except Exception as e:
    print("could not load full HLA table:", e)

def pseudo_for(allele):
    k = nrm(allele)
    for kk in (k, k[:5], k[:4], k[:3]):
        if kk in PSEUDO: return PSEUDO[kk], (kk == k)
    return None, False

# ---------------------------------------------------------------- Caushi positives
c = pd.read_csv("data/caushi/caushi_screened.csv")
c.columns = [str(x).strip() for x in c.columns]
rows = []
for _, d in c.iterrows():
    va, vb = norm_v(d["TRAV"], Akeys), norm_v(d["TRBV"], Bkeys)
    if not va or not vb:
        print(f"  !! V unmapped {d['TRAV']}/{d['TRBV']}"); continue
    a1, a2 = loops(va, Aseq); b1, b2 = loops(vb, Bseq)
    rows.append(dict(patient=d["Patient"], A1=a1, A2=a2, A3=d["CDR3α"],
                     B1=b1, B2=b2, B3=d["CDR3β"],
                     peptide=d["Peptide recognized"], allele=d["HLA restriction"],
                     antigen=d["Antigen"], mpr=d["MPR status"]))
C = pd.DataFrame(rows)
FLU = C[C.peptide.str.contains("Influenza", case=False, na=False)].copy()
NEO = C[~C.peptide.str.contains("Influenza", case=False, na=False)].copy()
FLU["peptide"], FLU["allele"] = "GILGFVFTL", "HLA-A*02:01"
print(f"\nCaushi: {len(NEO)} neoantigen clonotypes, {len(FLU)} influenza (pipeline check only)")
for (p, al), g in NEO.groupby(["peptide", "allele"]):
    ps, exact = pseudo_for(al)
    print(f"  {p:12} {al:12} n={len(g)}  pseudo={'exact' if exact else ('fallback' if ps else 'MISSING')}")

# ---------------------------------------------------------------- cohort-matched negatives
til = pd.read_csv("data/nsclc_til_sc/paired_6cdr.csv")
reactive = set(pd.read_csv("data/nsclc_til_sc/fest_clonotype_antigen.csv").cdr3b)
til = til[~til.cdr3_TRB.isin(reactive)]
til = til.dropna(subset=["cdr1_TRA","cdr2_TRA","cdr3_TRA","cdr1_TRB","cdr2_TRB","cdr3_TRB"])
til = til.drop_duplicates(["patient","cdr3_TRA","cdr3_TRB"])
print(f"\nnon-reactive tumour-infiltrating clonotypes available: {len(til)}")
neg = til.sample(min(N_NEG, len(til)), random_state=42)
NEGD = [dict(A1=r.cdr1_TRA, A2=r.cdr2_TRA, A3=r.cdr3_TRA,
             B1=r.cdr1_TRB, B2=r.cdr2_TRB, B3=r.cdr3_TRB) for r in neg.itertuples()]
print(f"negatives used: {len(NEGD)} (cohort-matched: TIL, expanded, paired chains)")

# ---------------------------------------------------------------- model
assert np.abs(_globals("CASSLGQAYEQYF", "GILGFVFTL")).sum() > 0
ck = torch.load("models/stepmhc_small_fullpool_seed42.pt")
m = STEPMHCModelSmall(use_structure_priors=False)
m.load_state_dict(ck["state_dict"]); m.cuda().eval()

tstr = lambda d: "".join(str(d[k])[:CAPS[k]].ljust(CAPS[k], "-") for k in LOOPS)

def score(ds, pep, hp):
    out, buf = [], []
    def flush():
        nonlocal buf
        if not buf: return
        te = torch.from_numpy(np.stack([b[0] for b in buf])).cuda()
        ph = torch.from_numpy(np.stack([b[1] for b in buf])).cuda()
        gg = torch.from_numpy(np.stack([b[2] for b in buf])).float().cuda()
        with torch.no_grad():
            out.extend(torch.sigmoid(m(te, ph, gg)).cpu().numpy().tolist())
        buf = []
    for d in ds:
        ts = tstr(d)
        buf.append((_imap(ts, pep, L_TCR, L_PEP), _imap(pep, hp, L_PEP, L_HLA), _globals(ts, pep)))
        if len(buf) >= 512: flush()
    flush(); return np.array(out)

def run(df, label, note=""):
    print(f"\n{'='*76}\n{label}\n{'='*76}")
    if note: print(f"  {note}\n")
    recs, cache = [], {}
    for (pep, al), g in df.groupby(["peptide", "allele"]):
        hp, exact = pseudo_for(al)
        if hp is None:
            print(f"  {pep}: no pseudo for {al} — SKIPPED"); continue
        s_neg = score(NEGD, pep, hp); cache[pep] = s_neg
        s_pos = score(g.to_dict("records"), pep, hp)
        pct = [(s_neg < v).mean() for v in s_pos]
        auc = np.nan
        if len(s_pos) >= 2:
            y = np.r_[np.ones(len(s_pos)), np.zeros(len(s_neg))]
            auc = roc_auc_score(y, np.r_[s_pos, s_neg])
        tag = "" if exact else "  [HLA fallback]"
        print(f"  {pep:12} {al:12} n={len(g)}{tag}")
        print(f"     cognate {np.round(s_pos,3)} | TIL-negative mean {s_neg.mean():.3f} "
              f"sd {s_neg.std():.3f}")
        print(f"     percentiles {np.round(pct,3)}  median {np.median(pct):.3f}"
              + (f"  AUROC {auc:.3f}" if not np.isnan(auc) else "  (n=1)"))
        for v, p in zip(s_pos, pct):
            recs.append(dict(peptide=pep, allele=al, score=float(v), pct=float(p),
                             n_pos=len(g), auroc=auc))
    R = pd.DataFrame(recs)
    if len(R):
        print(f"\n  POOLED n={len(R)}  median percentile {R.pct.median():.3f}  (chance 0.500)")
        print(f"         >0.5: {int((R.pct>0.5).sum())}/{len(R)}   >0.9: {int((R.pct>0.9).sum())}/{len(R)}")
    return R, cache

# 1. pipeline check
Rflu, _ = run(FLU, "PIPELINE CHECK — influenza GILGFVFTL",
              "NOT A RESULT: all 3 have both chains in the training pool (public motifs). "
              "This only confirms the machinery runs.")

# 2. primary
Rneo, cache = run(NEO, "PRIMARY — neoantigens vs COHORT-MATCHED non-reactive TIL clonotypes")

# 3. permutation null
print(f"\n{'='*76}\nNEGATIVE CONTROL — 500 shuffles of cognate peptide assignment\n{'='*76}")
allc = NEO.to_dict("records")
pos_by_pep = {}
for pep in cache:
    pos_by_pep[pep] = score(allc, pep, pseudo_for(NEO[NEO.peptide==pep].allele.iloc[0])[0])
rng = np.random.default_rng(7); null = []
peps = list(cache)
for _ in range(500):
    perm = rng.permutation(len(allc))
    pv = []
    for i in range(len(allc)):
        pep = allc[perm[i]]["peptide"]
        if pep not in cache: continue
        pv.append((cache[pep] < pos_by_pep[pep][i]).mean())
    if pv: null.append(np.median(pv))
null = np.array(null); obs = Rneo.pct.median()
print(f"  null: mean {null.mean():.3f}, 5-95% [{np.percentile(null,5):.3f}, {np.percentile(null,95):.3f}]")
print(f"  observed {obs:.3f}")
print(f"  PERMUTATION p = {(null >= obs).mean():.4f}    <-- reportable")

Rneo.to_csv("results/caushi/primary_tilneg.csv", index=False)
print("\nwrote results/caushi/primary_tilneg.csv")
