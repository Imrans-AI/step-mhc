"""Rigorous unseen-epitope CV: per-fold train/val/test, early-stop on val, test once, aggregate."""
import os
import sys, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, str(Path(__file__).parent))
from dataset_stepmhc import StepMHCDataset
from model_stepmhc import STEPMHCModel

MODE = sys.argv[1] if len(sys.argv) > 1 else "uniform"
ROOT = Path(os.environ.get("STEPMHC_ROOT", "."))
SPLITS, PRIOR = ROOT/"data/splits/unseen_epitope", ROOT/"data/priors/structure_priors_mhc.npz"
OUTD = ROOT/"results"/f"cv_{MODE}"; OUTD.mkdir(parents=True, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.backends.cudnn.benchmark = True
MAXEP, BS, LR, WD, PAT, N_VAL = 40, 32, 5e-4, 1e-4, 6, 3

def macro(df, prob):
    df = df.copy(); df["p"] = prob; v = []
    for _, s in df.groupby("peptide"):
        if s.binder.sum() >= 10 and s.binder.nunique() > 1:
            v.append((average_precision_score(s.binder, s.p), roc_auc_score(s.binder, s.p)))
    a = np.array(v); return a[:, 0].mean(), a[:, 1].mean()

def pred(model, loader):
    model.eval(); ps, ys = [], []
    with torch.no_grad():
        for a, b, g, y in loader:
            ps.append(torch.sigmoid(model(a.to(dev), b.to(dev), g.to(dev))).cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ys)

rows = []
for fold in sorted(p.name for p in SPLITS.iterdir() if p.is_dir() and p.name.startswith("fold")):
    torch.manual_seed(42); fdir = SPLITS/fold
    full = pd.read_csv(fdir/"train.csv")
    cnt = full.groupby("peptide").binder.sum(); elig = cnt[cnt >= 20].index.tolist()
    val_epi = set(np.random.RandomState(0).choice(elig, min(N_VAL, len(elig)), replace=False))
    trdf, vadf = full[~full.peptide.isin(val_epi)], full[full.peptide.isin(val_epi)]
    trdf.to_csv(fdir/"_tr.csv", index=False); vadf.to_csv(fdir/"_va.csv", index=False)
    tr, va, te = StepMHCDataset(fdir/"_tr.csv"), StepMHCDataset(fdir/"_va.csv"), StepMHCDataset(fdir/"test.csv")
    pw = torch.tensor(float((1 - trdf.binder.mean())/trdf.binder.mean()), device=dev)
    trl = DataLoader(tr, BS, shuffle=True, num_workers=4, drop_last=True)
    vl, tel = DataLoader(va, 256, num_workers=4), DataLoader(te, 256, num_workers=4)
    model = STEPMHCModel(use_structure_priors=(MODE=="prior"), prior_path=str(PRIOR) if MODE=="prior" else None).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD); lossfn = nn.BCEWithLogitsLoss(pos_weight=pw)
    best, bstate, wait, stop = -1, None, 0, 0
    for e in range(1, MAXEP+1):
        model.train()
        for a, b, g, y in trl:
            loss = lossfn(model(a.to(dev), b.to(dev), g.to(dev)), y.float().to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        vaupr, _ = macro(vadf, pred(model, vl)[0]); stop = e
        if vaupr > best: best, wait = vaupr, 0; bstate = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PAT: break
    model.load_state_dict(bstate)
    taupr, tauc = macro(te.df, pred(model, tel)[0])
    print(f"{fold}: val_best={best:.3f}  TEST aupr={taupr:.3f} auc={tauc:.3f}  (ep{stop})")
    rows.append((fold, taupr, tauc)); (fdir/"_tr.csv").unlink(); (fdir/"_va.csv").unlink()
r = pd.DataFrame(rows, columns=["fold","test_aupr","test_auc"]); r.to_csv(OUTD/"cv_results.csv", index=False)
print(f"\n[{MODE}] 5-fold TEST macro-AUPR {r.test_aupr.mean():.3f} ± {r.test_aupr.std():.3f} | AUC {r.test_auc.mean():.3f} ± {r.test_auc.std():.3f}")
