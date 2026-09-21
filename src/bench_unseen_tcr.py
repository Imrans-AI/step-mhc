"""Unseen-TCR head-to-head: STEP-MHC vs EPACT on broadened leave-TCR-out folds.
Both get full C..F CDR3 (EPACT-compatible). Score on EPACT-overlap-free subset, macro-per-epitope.
STEP retrained per fold (no test peeking); EPACT released weights."""
import os
import sys, subprocess, json, numpy as np, pandas as pd
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, str(Path(__file__).parent))
from dataset_stepmhc import StepMHCDataset
from model_stepmhc_small import STEPMHCModelSmall

ROOT=Path(os.environ.get("STEPMHC_ROOT", "."))
EP=ROOT/"baselines/EPACT"; SPLITS=ROOT/"data/splits/unseen_tcr_broadened"
WORK=ROOT/"results/bench_utcr"; WORK.mkdir(parents=True,exist_ok=True)
dev="cuda" if torch.cuda.is_available() else "cpu"
SEED=42; MAXEP,BS,LR,WD,PAT=40,32,5e-4,1e-4,6

lib=json.load(open(EP/"hla_library.json"))
from EPACT.utils.encoding import hla_allele_to_seq
def allele_ok(a): return hla_allele_to_seq(str(a),lib) is not None
EP_TRAIN=pd.read_csv(EP/"binding/Full-TCR/train_full_tcr_pmhc_data.csv")
ep_keys=set(zip(EP_TRAIN["CDR3.alpha.aa"].astype(str),EP_TRAIN["CDR3.beta.aa"].astype(str),EP_TRAIN["Epitope.peptide"].astype(str)))

def train_step(fold):
    spl=SPLITS/fold
    full=pd.read_csv(spl/"train.csv")
    vm=np.random.RandomState(SEED).rand(len(full))<0.10
    full[~vm].to_csv(spl/"_tr.csv",index=False); full[vm].to_csv(spl/"_va.csv",index=False)
    tr,va,te=StepMHCDataset(spl/"_tr.csv"),StepMHCDataset(spl/"_va.csv"),StepMHCDataset(spl/"test.csv")
    pw=torch.tensor(float((1-full[~vm].binder.mean())/max(full[~vm].binder.mean(),1e-6)),device=dev)
    trl=DataLoader(tr,BS,shuffle=True,num_workers=4,drop_last=True); vl=DataLoader(va,256,num_workers=4); tel=DataLoader(te,256,num_workers=4)
    model=STEPMHCModelSmall(use_structure_priors=False,prior_path=None).to(dev)
    opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=WD); lf=nn.BCEWithLogitsLoss(pos_weight=pw)
    best,bstate,wait=float("inf"),None,0
    for e in range(MAXEP):
        model.train()
        for a,b,g,y in trl:
            l=lf(model(a.to(dev),b.to(dev),g.to(dev)),y.float().to(dev)); opt.zero_grad(); l.backward(); opt.step()
        model.eval(); vlz=[]
        with torch.no_grad():
            for a,b,g,y in vl: vlz.append(lf(model(a.to(dev),b.to(dev),g.to(dev)),y.float().to(dev)).item())
        v=float(np.mean(vlz))
        if v<best-1e-4: best,wait,bstate=v,0,{k:val.cpu().clone() for k,val in model.state_dict().items()}
        else:
            wait+=1
            if wait>=PAT: break
    model.load_state_dict(bstate); model.eval(); ps=[]
    with torch.no_grad():
        for a,b,g,y in tel: ps.append(torch.sigmoid(model(a.to(dev),b.to(dev),g.to(dev))).cpu().numpy())
    d=te.df.copy(); d["step_score"]=np.concatenate(ps)
    (spl/"_tr.csv").unlink(); (spl/"_va.csv").unlink()
    return d

def run_epact(fold,k,testdf):
    proj=testdf[testdf.allele.map(allele_ok)].copy()
    out=pd.DataFrame({"CDR1.alpha.aa":proj.A1,"CDR2.alpha.aa":proj.A2,"CDR3.alpha.aa":proj.A3,
        "CDR1.beta.aa":proj.B1,"CDR2.beta.aa":proj.B2,"CDR3.beta.aa":proj.B3,
        "Epitope.peptide":proj.peptide,"MHC":proj.allele.str.replace("HLA-","",regex=False),"Target":proj.binder})
    inp=WORK/f"epin_{fold}.csv"; out.to_csv(inp,index=False); logd=WORK/f"epout_{fold}"
    r=subprocess.run(["python","scripts/predict/predict_tcr_pmhc_binding.py","--config",
        "configs/config-paired-cdr123-pmhc-binding.yml","--input_data_path",str(inp),
        "--model_location",str(EP/f"paired-cdr123-pmhc-binding/paired-cdr123-pmhc-binding-model-fold-{k+1}.pt"),
        "--log_dir",str(logd)],cwd=str(EP),capture_output=True,text=True,timeout=3600)
    pf=logd/"predictions.csv"
    if not pf.exists(): print(f"{fold} EPACT fail: {r.stderr[-300:]}"); return None
    proj=proj.reset_index(drop=True); proj["epact"]=pd.read_csv(pf)["Pred"].values
    return proj[["A3","B3","peptide","epact"]]

def macro(df,col):
    v=[(roc_auc_score(s.binder,s[col]),average_precision_score(s.binder,s[col]))
       for _,s in df.groupby("peptide") if s.binder.sum()>=10 and s.binder.nunique()>1]
    a=np.array(v); return (a[:,0].mean(),a[:,1].mean(),len(v)) if len(v) else (np.nan,np.nan,0)

rows=[]
for k in range(5):
    fold=f"fold{k}"
    sd=train_step(fold)
    ep=run_epact(fold,k,sd)
    if ep is None: continue
    m=sd.merge(ep,on=["A3","B3","peptide"],how="inner")
    m["_seen"]=[(a,b,p) in ep_keys for a,b,p in zip(m.A3.astype(str),m.B3.astype(str),m.peptide.astype(str))]
    mc=m[~m._seen]   # EPACT-clean only
    sa,sap,ne=macro(mc,"step_score"); ea,eap,_=macro(mc,"epact")
    print(f"{fold}: clean n={len(mc)} epi={ne}  STEP {sa:.3f}/{sap:.3f}  EPACT {ea:.3f}/{eap:.3f}",flush=True)
    rows.append([fold,ne,sa,sap,ea,eap])
df=pd.DataFrame(rows,columns=["fold","n_epi","step_auc","step_aupr","epact_auc","epact_aupr"])
df.to_csv(ROOT/"results/bench_unseen_tcr.csv",index=False)
print("\n===== UNSEEN-TCR: STEP vs EPACT (EPACT-clean subset) =====")
print(f"STEP  AUROC {df.step_auc.mean():.3f}+/-{df.step_auc.std():.3f}  AUPR {df.step_aupr.mean():.3f}")
print(f"EPACT AUROC {df.epact_auc.mean():.3f}+/-{df.epact_auc.std():.3f}  AUPR {df.epact_aupr.mean():.3f}")
print(f"delta AUROC {df.step_auc.mean()-df.epact_auc.mean():+.3f}")
