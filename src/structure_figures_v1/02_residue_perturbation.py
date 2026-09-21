"""Genuine single-residue perturbation through the CANONICAL deterministic STEP-MHC model.
METHOD (declared before any result is seen):
  - substitute one residue at a time with alanine, re-encode through the exact pipeline,
    re-score;  dscore = original - substituted  (positive => residue supported the score)
  - positions already alanine are MISSING, never 0
  - HLA and all other loops preserved
  - MIN_SUPPORT = 5 unique complexes per heatmap cell (fixed in advance)
  - bootstrap at the COMPLEX level, 2000 resamples, seed 0
Outputs (new files only):
  results/structure_figures_v1/perturb_peptide.csv   per residue x complex
  results/structure_figures_v1/perturb_loops.csv     per loop position x complex
"""
OUT='results/structure_figures_v1/'          # SET FIRST - HANDOFF_9 S8
MIN_SUPPORT=5; NBOOT=2000; BSEED=0
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys, numpy as np, pandas as pd, torch
sys.path.insert(0,'.')
torch.use_deterministic_algorithms(True, warn_only=True)
from src.model_stepmhc_small import STEPMHCModelSmall
from src.dataset_stepmhc import _imap,_globals,CAPS,LOOPS,L_TCR,L_PEP,L_HLA
AAS='ACDEFGHIKLMNPQRSTVWY'
dev='cuda' if torch.cuda.is_available() else 'cpu'
ck=torch.load('models/stepmhc_small_fullpool_seed42.pt',map_location='cpu',weights_only=False)
m=STEPMHCModelSmall(use_structure_priors=False); m.load_state_dict(ck['state_dict']); m.to(dev).eval()

def tcr_str(r): return ''.join(str(r[c])[:CAPS[c]].ljust(CAPS[c],'-') for c in LOOPS)
@torch.no_grad()
def score(loops,pep,pseudo):
    ts=''.join(str(loops[c])[:CAPS[c]].ljust(CAPS[c],'-') for c in LOOPS)
    te=torch.from_numpy(_imap(ts,pep,L_TCR,L_PEP)).unsqueeze(0).to(dev)
    ph=torch.from_numpy(_imap(pep,pseudo,L_PEP,L_HLA)).unsqueeze(0).to(dev)
    gg=torch.from_numpy(_globals(''.join(str(loops[c]) for c in LOOPS),pep)).float().unsqueeze(0).to(dev)
    return float(torch.sigmoid(m(te,ph,gg)).item())

ex=pd.read_csv(OUT+'exemplars_final.csv')
prow,lrow=[],[]
for _,r in ex.iterrows():
    loops={c:r[c] for c in LOOPS}; pep=r.peptide; ps=r.hla_pseudo
    base=score(loops,pep,ps)
    for i,aa in enumerate(pep):                                   # peptide scan
        if aa=='A': prow.append(dict(pdb=r.pdb,peptide=pep,pos=i+1,wt=aa,base=base,
                                     mut=np.nan,dscore=np.nan)); continue
        mp=pep[:i]+'A'+pep[i+1:]
        s=score(loops,mp,ps)
        prow.append(dict(pdb=r.pdb,peptide=pep,pos=i+1,wt=aa,base=base,mut=s,dscore=base-s))
    for c in LOOPS:                                               # loop scan
        seq=str(r[c])
        for i,aa in enumerate(seq):
            if aa=='A': lrow.append(dict(pdb=r.pdb,loop=c,pos=i+1,wt=aa,base=base,
                                         mut=np.nan,dscore=np.nan)); continue
            l2=dict(loops); l2[c]=seq[:i]+'A'+seq[i+1:]
            s=score(l2,pep,ps)
            lrow.append(dict(pdb=r.pdb,loop=c,pos=i+1,wt=aa,base=base,mut=s,dscore=base-s))
P=pd.DataFrame(prow); L=pd.DataFrame(lrow)
P.to_csv(OUT+'perturb_peptide.csv',index=False); L.to_csv(OUT+'perturb_loops.csv',index=False)
print(f'peptide rows {len(P)} ({P.dscore.notna().sum()} scored, {P.dscore.isna().sum()} already-Ala)')
print(f'loop rows    {len(L)} ({L.dscore.notna().sum()} scored, {L.dscore.isna().sum()} already-Ala)')
print(f'\nbase-score range across {ex.pdb.nunique()} complexes: '
      f'{P.base.min():.4f} - {P.base.max():.4f}')
print('\nmedian dscore by peptide position (support >= %d):'%MIN_SUPPORT)
g=P.dropna(subset=['dscore']).groupby('pos').dscore.agg(['median','count'])
print(g[g['count']>=MIN_SUPPORT].round(4).to_string())
print('\nmedian dscore by loop:')
print(L.dropna(subset=['dscore']).groupby('loop').dscore.agg(['median','count']).round(4).to_string())
