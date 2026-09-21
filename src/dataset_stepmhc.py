"""STEP-MHC Dataset: paired 6-CDR TCR x epitope map + epitope x HLA map + 7 global feats."""
import os
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset
from pathlib import Path
try:
    import peptides
except ImportError:
    peptides = None

HYD={'A':.62,'R':0,'N':.16,'D':.11,'C':.68,'Q':.19,'E':.14,'G':.48,'H':.32,'I':1,'L':.98,'K':.05,'M':.85,'F':.97,'P':.51,'S':.35,'T':.39,'W':.89,'Y':.63,'V':.86}
CHG={'A':.5,'R':1,'N':.5,'D':0,'C':.5,'Q':.5,'E':0,'G':.5,'H':.75,'I':.5,'L':.5,'K':1,'M':.5,'F':.5,'P':.5,'S':.5,'T':.5,'W':.5,'Y':.5,'V':.5}
FLX={'A':.35,'R':.53,'N':.46,'D':.51,'C':.35,'Q':.49,'E':.5,'G':.54,'H':.32,'I':.46,'L':.37,'K':.47,'M':.3,'F':.31,'P':.51,'S':.51,'T':.44,'W':.31,'Y':.42,'V':.39}
REF={'A':.2,'R':.65,'N':.33,'D':.32,'C':.35,'Q':.46,'E':.44,'G':0,'H':.58,'I':.57,'L':.57,'K':.52,'M':.59,'F':.76,'P':.36,'S':.23,'T':.33,'W':1,'Y':.78,'V':.48}
ARE={'A':.31,'R':.64,'N':.43,'D':.41,'C':.41,'Q':.51,'E':.49,'G':0,'H':.59,'I':.68,'L':.68,'K':.6,'M':.67,'F':.76,'P':.51,'S':.33,'T':.44,'W':1,'Y':.82,'V':.61}
PROPS = [HYD,CHG,FLX,REF,ARE]

CAPS = {"A1":7,"A2":8,"A3":22,"B1":6,"B2":7,"B3":23}
LOOPS = list(CAPS); L_TCR = sum(CAPS.values()); L_PEP = 12; L_HLA = 34
HLA_CSV = os.environ["STEPMHC_HLA_CSV"]

def _vec(seq, L, d):
    v = np.full(L, np.nan, np.float32)
    for i, c in enumerate(str(seq)[:L]):
        if c in d: v[i] = d[c]
    return v

def _imap(sa, sb, La, Lb):
    out = np.zeros((5, La, Lb), np.float32)
    for k, d in enumerate(PROPS):
        va, vb = _vec(sa, La, d), _vec(sb, Lb, d)
        out[k] = np.nan_to_num(np.abs(va[:, None] - vb[None, :]))
    return out

def _globals(tcr, epi):
    if peptides is None: return np.zeros(7, np.float32)
    tbl = peptides.tables.HYDROPHOBICITY["KyteDoolittle"]
    try:
        t, e = peptides.Peptide(tcr), peptides.Peptide(epi)
        f = lambda p: np.array([p.isoelectric_point(), p.instability_index(), p.aliphatic_index(),
                                p.boman(), p.hydrophobic_moment(), p.molecular_weight(),
                                p.auto_correlation(table=tbl, lag=1)])
        return (f(t) - f(e)).astype(np.float32)
    except Exception:
        return np.zeros(7, np.float32)

class StepMHCDataset(Dataset):
    def __init__(self, csv, hla_csv=HLA_CSV, ternary=False):
        self.ternary = ternary
        self.df = pd.read_csv(csv).reset_index(drop=True)
        hla = pd.read_csv(hla_csv)
        norm = lambda a: str(a).replace("HLA-", "").replace("*", "").replace(":", "").strip()
        self.hmap = {norm(k): v for k, v in zip(hla.iloc[:, 0], hla.iloc[:, 1])}
        self.tcr_fixed, self.tcr_cat, self.glob = [], [], []
        for _, r in self.df.iterrows():
            self.tcr_fixed.append("".join(str(r[c])[:CAPS[c]].ljust(CAPS[c], "-") for c in LOOPS))
            cat = "".join(str(r[c]) for c in LOOPS)
            self.tcr_cat.append(cat)
            self.glob.append(_globals(cat, str(r.peptide)))
        self.has_pseudo = 'hla_pseudo' in self.df.columns
        if not self.has_pseudo:
            miss = sorted({norm(a) for a in self.df.allele.unique()} - set(self.hmap))
            assert not miss, f"alleles missing pseudo-seq: {miss}"

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        norm = lambda a: str(a).replace("HLA-", "").replace("*", "").replace(":", "").strip()
        te = _imap(self.tcr_fixed[i], r.peptide, L_TCR, L_PEP)
        pseudo = r.hla_pseudo if self.has_pseudo else self.hmap[norm(r.allele)]
        ph = _imap(r.peptide, pseudo, L_PEP, L_HLA)
        y = torch.tensor(int(r.binder), dtype=torch.long)
        if getattr(self, "ternary", False):
            # TCR x HLA face: the third surface of the ternary complex (E2)
            th = _imap(self.tcr_fixed[i], pseudo, L_TCR, L_HLA)
            return (torch.from_numpy(te), torch.from_numpy(ph),
                    torch.from_numpy(self.glob[i]).float(),
                    torch.from_numpy(th), y)
        return (torch.from_numpy(te), torch.from_numpy(ph),
                torch.from_numpy(self.glob[i]).float(), y)

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    ds = StepMHCDataset("data/splits/unseen_epitope/fold0/train.csv")
    te, ph, g, y = next(iter(DataLoader(ds, batch_size=8, shuffle=True)))
    print("dataset size:", len(ds))
    print("TCRxEpi batch:", tuple(te.shape), "| PepxHLA:", tuple(ph.shape),
          "| global:", tuple(g.shape), "| labels:", y.tolist())
    print(f"map value range: [{te.min():.2f}, {te.max():.2f}]  NaNs: {torch.isnan(te).any().item() or torch.isnan(ph).any().item()}")
    print(f"global NaNs: {torch.isnan(g).any().item()}  pos rate (full): {ds.df.binder.mean():.3f}")
