"""CORRECTED 2026-08-26: contact rows are SEQUENCE-indexed to match the model input
and dp_interaction.npz. The first version used IMGT offsets and compared different residues.
Does the double-perturbation interaction matrix track real crystallographic contacts?
Same complex, same numbering. Permutation PRESERVES loop/position structure by shuffling
TCR rows WITHIN each loop (contacts and interactions both have strong loop-level structure,
so a free shuffle would be anti-conservative). Reports the result whether or not it is null.
Sensitivity: with/without native-alanine cells. Writes dp_contact_validation.md + csv"""
OUT='results/structure_figures_v1/'
import sys, json, numpy as np, pandas as pd
sys.path.insert(0,'.')
from scipy.stats import spearmanr, mannwhitneyu
from src.extract_priors import atoms_by_chain, contact_matrix, AA3
from src.dataset_stepmhc import CAPS, LOOPS
from anarci import run_anarci
IMGT={'1':(27,38),'2':(56,65),'3':(105,117)}
z=np.load(OUT+'dp_interaction.npz', allow_pickle=True)
I=z['I']; NA=z['native_ala']; PDB=str(z['pdb']); PEP=str(z['peptide'])
meta=json.load(open(OUT+'dp_meta.json'))
ca={c['pdb']:c for c in json.load(open('data/priors/_chain_assignment.json'))}[PDB]
ch=atoms_by_chain(open(f'data/structures/{PDB.lower()}.pdb').read())
off={}; o=0
for k in LOOPS: off[k]=o; o+=CAPS[k]

def rows_for(cid, tag):
    res=list(ch[cid].items()); seq=''.join(AA3[r[1][0]] for r in res)
    numb=run_anarci([('x',seq)],scheme='imgt')[1][0][0][0]
    si, per = 0, {}
    for (pos,ins),aa in numb:
        if aa=='-': continue
        while si<len(res) and AA3[res[si][1][0]]!=aa: si+=1
        if si>=len(res): break
        for cdr,(lo,hi) in IMGT.items():
            if lo<=pos<=hi: per.setdefault(tag+cdr,[]).append(res[si][1][1])
        si+=1
    out={}
    for key,coords in per.items():          # SEQUENCE index, matching the model and 08
        for i,cc in enumerate(coords):
            if i<CAPS[key]: out[off[key]+i]=cc
    return out
rows={}; rows.update(rows_for(ca['tra'],'A')); rows.update(rows_for(ca['trb'],'B'))
pep=[v[1] for k,v in ch[ca['pep']].items()]
idx=sorted(rows)
D=contact_matrix([rows[i] for i in idx], pep)          # boolean <=5 A
dist=np.full(I.shape, np.nan); cont=np.zeros(I.shape, bool)
for a,i in enumerate(idx):
    for j in range(min(len(pep), I.shape[1])): cont[i,j]=D[a,j]
mask=~np.isnan(I) & np.array([[ (r in rows) and (c<len(pep)) for c in range(I.shape[1])]
                              for r in range(I.shape[0])])
L=[]; w=lambda s='':(L.append(s),print(s))
w(f'# Double-perturbation vs crystallographic contact — {PDB.upper()} {PEP}')
w(f'\ncomplex chosen by the pre-declared rule (not by contact agreement). '
  f'in_clean39={meta["in_clean39"]}')
w(f'cells compared: {int(mask.sum())} | contacts <=5 A: {int(cont[mask].sum())}')
for tag, mm in [('all cells', mask), ('excluding native-Ala', mask & ~NA)]:
    x=I[mm]; c=cont[mm]
    if c.sum()<3 or (~c).sum()<3: w(f'\n{tag}: too few in one class'); continue
    rho=spearmanr(x, c.astype(float)).statistic
    u=mannwhitneyu(x[c], x[~c], alternative='two-sided')
    rng=np.random.default_rng(0); null=[]
    occ={k:[r for r in range(off[k],off[k]+CAPS[k]) if r in rows] for k in LOOPS}
    for _ in range(2000):                    # permute OCCUPIED rows within each loop only
        pm=I.copy()
        for k in LOOPS:
            r=occ[k]
            if len(r)>1:
                perm=list(rng.permutation(r))
                pm[r,:]=I[perm,:]
        xx=pm[mm]
        if np.isfinite(xx).all(): null.append(spearmanr(xx, c.astype(float)).statistic)
    null=np.array(null)
    if len(null)==0: w(f'\n{tag}: permutation produced 0 valid draws — STOP'); continue; pv=(np.sum(np.abs(null)>=abs(rho))+1)/(len(null)+1)
    w(f'\n## {tag}  (n={int(mm.sum())})')
    w(f'Spearman(I, contact) = {rho:+.4f}')
    w(f'median I contacted {np.median(x[c]):+.4f} vs non-contacted {np.median(x[~c]):+.4f}, '
      f'MWU p = {u.pvalue:.4f}')
    w(f'within-loop permutation null: median {np.median(null):+.4f}, '
      f'2.5-97.5% [{np.percentile(null,2.5):+.4f}, {np.percentile(null,97.5):+.4f}], '
      f'empirical p = {pv:.4f}')
    w(f'**{"EXCEEDS null" if pv<0.05 else "DOES NOT exceed null — reported as null"}**')
np.savez(OUT+'dp_contact.npz', contact=cont, mask=mask)
open(OUT+'dp_contact_validation.md','w').write('\n'.join(L))
print('\nwrote dp_contact_validation.md, dp_contact.npz')
