"""Benchmark: train STEP-MHC (small, proper val early-stop, NO test peeking) on a
broadened unseen-epitope fold, score test ONCE, write per-row test scores for head-to-head.
Usage: python bench_train_step.py fold0 [seed]"""
import os
import sys, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from dataset_stepmhc import StepMHCDataset
from model_stepmhc_small import STEPMHCModelSmall

FOLD = sys.argv[1] if len(sys.argv)>1 else "fold0"
SEED = int(sys.argv[2]) if len(sys.argv)>2 else 42
ROOT = Path(os.environ.get("STEPMHC_ROOT", "."))
SPL  = ROOT/"data/splits/unseen_epitope_broadened"/FOLD
OUT  = ROOT/"results/bench"; OUT.mkdir(parents=True, exist_ok=True)
dev  = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED)
MAXEP, BS, LR, WD, PAT = 40, 32, 5e-4, 1e-4, 6

full = pd.read_csv(SPL/"train.csv")
vm = np.random.RandomState(SEED).rand(len(full)) < 0.10      # val carved from TRAIN (no test peeking)
vadf, trdf = full[vm], full[~vm]
trdf.to_csv(SPL/"_btr.csv", index=False); vadf.to_csv(SPL/"_bva.csv", index=False)
tr, va, te = StepMHCDataset(SPL/"_btr.csv"), StepMHCDataset(SPL/"_bva.csv"), StepMHCDataset(SPL/"test.csv")
pw = torch.tensor(float((1-trdf.binder.mean())/max(trdf.binder.mean(),1e-6)), device=dev)
trl = DataLoader(tr, BS, shuffle=True, num_workers=4, drop_last=True)
vl  = DataLoader(va, 256, num_workers=4)
tel = DataLoader(te, 256, num_workers=4)
model = STEPMHCModelSmall(use_structure_priors=False, prior_path=None).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
lossfn = nn.BCEWithLogitsLoss(pos_weight=pw)

best, bstate, wait = float("inf"), None, 0
for e in range(1, MAXEP+1):
    model.train()
    for a,b,g,y in trl:
        loss = lossfn(model(a.to(dev),b.to(dev),g.to(dev)), y.float().to(dev))
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); vl_loss=[]
    with torch.no_grad():
        for a,b,g,y in vl:
            vl_loss.append(lossfn(model(a.to(dev),b.to(dev),g.to(dev)), y.float().to(dev)).item())
    vloss=float(np.mean(vl_loss))
    if vloss < best-1e-4: best,wait,bstate = vloss,0,{k:v.cpu().clone() for k,v in model.state_dict().items()}
    else:
        wait+=1
        if wait>=PAT: break
model.load_state_dict(bstate)
torch.save(bstate, OUT/f"step_{FOLD}_seed{SEED}.pt")

# score test ONCE, write per-row
model.eval(); ps=[]
with torch.no_grad():
    for a,b,g,y in tel:
        ps.append(torch.sigmoid(model(a.to(dev),b.to(dev),g.to(dev))).cpu().numpy())
te.df["step_score"] = np.concatenate(ps)
te.df.to_csv(OUT/f"step_scores_{FOLD}_seed{SEED}.csv", index=False)
(SPL/"_btr.csv").unlink(); (SPL/"_bva.csv").unlink()
print(f"trained on {len(trdf)}, val {len(vadf)}, test {len(te.df)} ({te.df.peptide.nunique()} epitopes)")
print(f"saved step_scores_{FOLD}_seed{SEED}.csv with step_score column")
