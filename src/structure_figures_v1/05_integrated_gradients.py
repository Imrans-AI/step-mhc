"""Integrated gradients over the canonical STEP-MHC TCR x peptide input map.
Attribution is over MAP CELLS, whose axes ARE residue positions (73 TCR x 12 peptide),
so aggregating the 5 physicochemical channels gives a residue-pair attribution.
This is NOT attention and NOT saliency.
VALIDATION: convergence over step counts, two baselines, completeness residual.
Exemplar: pre-declared rule (results/structure_figures_v1/exemplars_final.csv, row 0).
Writes ig_*.csv/npz + ATTRIBUTION_VALIDATION.md"""
OUT='results/structure_figures_v1/'
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys, numpy as np, pandas as pd, torch
sys.path.insert(0,'.')
torch.use_deterministic_algorithms(True, warn_only=True)
from src.model_stepmhc_small import STEPMHCModelSmall
from src.dataset_stepmhc import _imap,_globals,CAPS,LOOPS,L_TCR,L_PEP,L_HLA
dev='cuda' if torch.cuda.is_available() else 'cpu'
ck=torch.load('models/stepmhc_small_fullpool_seed42.pt',map_location='cpu',weights_only=False)
m=STEPMHCModelSmall(use_structure_priors=False); m.load_state_dict(ck['state_dict']); m.to(dev).eval()

ex=pd.read_csv(OUT+'exemplars_final.csv').iloc[0]
ts=''.join(str(ex[c])[:CAPS[c]].ljust(CAPS[c],'-') for c in LOOPS)
te=torch.from_numpy(_imap(ts,ex.peptide,L_TCR,L_PEP)).unsqueeze(0).to(dev)
ph=torch.from_numpy(_imap(ex.peptide,ex.hla_pseudo,L_PEP,L_HLA)).unsqueeze(0).to(dev)
gg=torch.from_numpy(_globals(''.join(str(ex[c]) for c in LOOPS),ex.peptide)).float().unsqueeze(0).to(dev)
print(f'exemplar {ex.pdb.upper()}  {ex.peptide}  {ex.allele}')

def f(x):                                  # scalar logit output
    return m(x,ph,gg).squeeze()
def ig(base, steps):
    g=torch.zeros_like(te)
    for a in torch.linspace(1.0/steps,1.0,steps,device=dev):
        x=(base+a*(te-base)).clone().requires_grad_(True)
        f(x).backward()
        g+=x.grad
    return (te-base)*g/steps

BASE={'zeros':torch.zeros_like(te), 'mean':te.mean()*torch.ones_like(te)}
V=[]; v=lambda s='':(V.append(s),print(s))
v('# ATTRIBUTION VALIDATION — integrated gradients')
v(f'\nexemplar: {ex.pdb.upper()} | {ex.peptide} | {ex.allele} | in_clean39={ex.in_clean39}')
v(f'checkpoint: models/stepmhc_small_fullpool_seed42.pt, {sum(p.numel() for p in m.parameters()):,} params')
v('\n## Convergence and completeness')
v('| baseline | steps | sum(IG) | f(x)-f(base) | residual | rel.err |')
v('|---|---|---|---|---|---|')
store={}
for bn,b in BASE.items():
    fb=float(f(b)); fx=float(f(te)); target=fx-fb
    for st in (32,64,128,256,512):
        A=ig(b,st); s=float(A.sum()); r=s-target
        v(f'| {bn} | {st} | {s:+.4f} | {target:+.4f} | {r:+.4f} | {abs(r)/max(abs(target),1e-9):.3%} |')
        if st==512: store[bn]=A.squeeze(0).cpu().numpy()
agg={k:np.abs(a).sum(0) for k,a in store.items()}          # (73,12), |.| over 5 channels
from scipy.stats import spearmanr
rho=spearmanr(agg['zeros'].ravel(),agg['mean'].ravel()).statistic
v(f'\n## Baseline sensitivity\nSpearman(zeros, mean) over the 73x12 aggregate = {rho:.4f}')
v('STABLE across baselines' if rho>0.8 else 'NOT STABLE — report both baselines')
np.savez(OUT+'ig_attribution.npz', zeros=store['zeros'], mean=store['mean'],
         agg_zeros=agg['zeros'], agg_mean=agg['mean'])
pd.DataFrame(agg['zeros']).to_csv(OUT+'ig_agg_73x12_zeros.csv',index=False)
o=0
v('\n## Aggregated attribution by loop (baseline=zeros)')
for k in LOOPS:
    s=agg['zeros'][o:o+CAPS[k]]; v(f'  {k}: mean {s.mean():.4f}  max {s.max():.4f}'); o+=CAPS[k]
open(OUT+'ATTRIBUTION_VALIDATION.md','w').write('\n'.join(V))
print('\nwrote ig_attribution.npz, ig_agg_73x12_zeros.csv, ATTRIBUTION_VALIDATION.md')
