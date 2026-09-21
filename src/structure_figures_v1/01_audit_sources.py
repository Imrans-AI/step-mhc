"""STAGE A audit. Recomputes every summary statistic from source; trusts no prose.
Writes results/structure_figures_v1/STAGE_A_AUDIT.md. Creates no other artefact."""
OUT = 'results/structure_figures_v1/STAGE_A_AUDIT.md'   # SET FIRST - HANDOFF_9 S8
import sys, json, glob, hashlib
import numpy as np, pandas as pd, torch
sys.path.insert(0, '.')
L = []
def w(s=''): L.append(s); print(s)

w('# STAGE A AUDIT — structure figures v1')
w(f'Generated independently of CLAUDE.md prose. numpy {np.__version__}, torch {torch.__version__}')
w()

w('## 1. Clean-prior ablation (recomputed from res_*.csv)')
rs = sorted(glob.glob('results/struct_val/clean_prior/res_*.csv'))
d = pd.concat([pd.read_csv(f) for f in rs])
w(f'files: {len(rs)}')
w('```'); w(d.sort_values(["mode","seed"]).to_string(index=False)); w('```')
g = d.groupby('mode').auc.agg(['mean','std','count'])
w('```'); w(g.round(4).to_string()); w('```')
piv = d.pivot(index='seed', columns='mode', values='auc')
piv['paired_delta'] = piv['prior'] - piv['none']
w('paired seed-level differences:')
w('```'); w(piv.round(4).to_string()); w('```')
delta = g.loc['prior','mean'] - g.loc['none','mean']
w(f'**mean delta = {delta:+.4f}**  (prereg success threshold: > +0.02)')
w(f'**THRESHOLD REACHED: {"YES" if delta > 0.02 else "NO"}**')
w(f'paired deltas: {piv.paired_delta.round(4).tolist()} — sign consistent: '
  f'{bool((piv.paired_delta>0).all() or (piv.paired_delta<0).all())}')
w()

w('## 2. Contact-alignment test (recomputed)')
pc = pd.read_csv('results/struct_val/per_complex_rho.csv')
nl = np.load('results/struct_val/null_rho.npy')
sm = pd.read_csv('results/struct_val/summary.csv')
obs = pc.rho.median()
p_emp = (np.sum(nl >= obs) + 1) / (len(nl) + 1)
w(f'complexes: {len(pc)}   (summary.csv says {int(sm.n_complexes[0])})')
w(f'observed median rho recomputed: {obs:.4f}   (summary.csv: {sm.obs_median[0]:.4f})')
w(f'null draws: {len(nl)}, median {np.median(nl):.4f}, '
  f'2.5–97.5% [{np.percentile(nl,2.5):.4f}, {np.percentile(nl,97.5):.4f}]')
w(f'empirical p recomputed: {p_emp:.4f}   (summary.csv: {sm.p[0]:.4f})')
ok = abs(obs - sm.obs_median[0]) < 1e-6
w(f'**MATCH: {"YES" if ok else "NO — DISCREPANCY, STOP"}**')
w()

w('## 3. Structure counts')
scr = pd.read_csv('results/struct_val/prior_pdb_screen.csv')
cln = pd.read_csv('results/struct_val/clean_prior_pdbs.csv')
ca  = json.load(open('results/struct_val/_chain_assignment_clean39.json'))
prov = json.load(open('data/priors/provenance_clean39.json'))
w(f'screened PDBs: {len(scr)}   contaminated {int((scr.status=="CONTAMINATED").sum())}   '
  f'clean {int((scr.status=="clean").sum())}')
w(f'clean set after Hamming>=3 filter: {len(cln)}')
w(f'chain assignments for clean set: {len(ca)}')
w(f'numbered successfully (provenance ok=True): {sum(1 for x in prov if x.get("ok"))}')
w(f'failed: {[x["pdb"] for x in prov if not x.get("ok")]}')
w()

w('## 4. Structural matrices')
z = np.load('data/priors/structure_priors_clean39.npz')
for k in z.files:
    a = z[k]
    if a.ndim == 0: w(f'{k}: scalar = {a}'); continue
    w(f'{k}: shape {a.shape}, nonzero {int((a!=0).sum())}/{a.size}, '
      f'range {a.min():.4f}–{a.max():.4f}, std {a.std():.4f}')
from src.dataset_stepmhc import CAPS, LOOPS
cw = z['cw_te']; o = 0
w('per-loop (recomputed):')
for k in LOOPS:
    s = cw[o:o+CAPS[k]]
    w(f'  {k}: rows {o}-{o+CAPS[k]-1}  mean {s.mean():.4f}  max {s.max():.4f}  '
      f'nonzero {int((s>0).sum())}/{s.size}')
    o += CAPS[k]
w()

w('## 5. Canonical model reproducibility')
ck = torch.load('models/stepmhc_small_fullpool_seed42.pt', map_location='cpu', weights_only=False)
n = sum(v.numel() for v in ck['state_dict'].values())
w(f'checkpoint keys: {list(ck.keys())}')
w(f'parameters: {n:,}   seed {ck.get("seed")}   val_bce {ck.get("val_bce")}')
h = hashlib.sha256(open('models/stepmhc_small_fullpool_seed42.pt','rb').read()).hexdigest()
w(f'sha256: {h}')
from src.model_stepmhc_small import STEPMHCModelSmall
m = STEPMHCModelSmall(use_structure_priors=False)
miss, unexp = m.load_state_dict(ck['state_dict'], strict=False)
w(f'load_state_dict: missing {len(miss)}, unexpected {len(unexp)}')
w(f'**LOADS REPRODUCIBLY: {"YES" if not miss and not unexp else "NO"}**')
w()

w('## 6. Gradient availability through the canonical model')
from src.dataset_stepmhc import _imap, _globals, L_TCR, L_PEP, L_HLA
ts = 'DSASNY ' .replace(' ','') + 'IRSNVGE' + 'CAASGGGSQGNLIF' + 'LNHDA' + 'SQIVND' + 'CASSIRSQETQYF'
te = torch.from_numpy(_imap(ts.ljust(L_TCR,'-'), 'GILGFVFTL', L_TCR, L_PEP)).unsqueeze(0)
ph = torch.from_numpy(_imap('GILGFVFTL', 'Y'*34, L_PEP, L_HLA)).unsqueeze(0)
gg = torch.from_numpy(_globals(ts, 'GILGFVFTL')).float().unsqueeze(0)
te.requires_grad_(True)
m.eval(); out = m(te, ph, gg); out.sum().backward()
gr = te.grad is not None and torch.isfinite(te.grad).all().item()
w(f'gradient wrt input map: {"AVAILABLE" if gr else "NOT AVAILABLE"}')
w(f'  grad shape {tuple(te.grad.shape)}, absmax {te.grad.abs().max():.3e}' if gr else '')
w('⚠ NOTE: `_imap` builds the map in numpy from sequence, so gradients exist wrt the MAP')
w('  but NOT wrt residue identity. Integrated gradients would attribute over map cells;')
w('  projecting map cells back to residue PAIRS needs a defensible definition (prompt §D).')
w()

open(OUT,'w').write('\n'.join(L))
print(f'\nwrote {OUT}')
