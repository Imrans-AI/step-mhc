"""Double-perturbation interaction, sequence-valid: every mutant is re-encoded from
MUTATED SEQUENCES through the canonical preprocessing, so all five channels regenerate.
  I_ij = f(x) - f(x_i) - f(x_j) + f(x_ij)     on the LOGIT scale
SUBSTITUTION POLICY (declared before results): alanine, EXCEPT native alanine -> glycine.
Native-alanine cells are flagged, never zeroed. Padding never mutated.
This is model epistasis, NOT binding energy, attention, contact or causal importance.
Writes dp_interaction.npz, dp_singles.csv, dp_meta.json"""
OUT='results/structure_figures_v1/'
import os; os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys, json, time, numpy as np, pandas as pd, torch
sys.path.insert(0,'.')
torch.use_deterministic_algorithms(True, warn_only=True)
from src.model_stepmhc_small import STEPMHCModelSmall
from src.dataset_stepmhc import _imap,_globals,CAPS,LOOPS,L_TCR,L_PEP,L_HLA
dev='cuda' if torch.cuda.is_available() else 'cpu'
CKPT='models/stepmhc_small_fullpool_seed42.pt'
ck=torch.load(CKPT,map_location='cpu',weights_only=False)
m=STEPMHCModelSmall(use_structure_priors=False); m.load_state_dict(ck['state_dict']); m.to(dev).eval()
SUB=lambda aa: 'G' if aa=='A' else 'A'

ex=pd.read_csv(OUT+'exemplars_final.csv').iloc[0]
loops={c:str(ex[c]) for c in LOOPS}; PEP=ex.peptide; PS=ex.hla_pseudo
tpos=[(c,i) for c in LOOPS for i in range(len(loops[c]))]        # real residues only
print(f'{ex.pdb.upper()} {PEP}: {len(tpos)} TCR x {len(PEP)} peptide = {len(tpos)*len(PEP)} pairs')

@torch.no_grad()
def logit(lp, pep):
    ts=''.join(lp[c][:CAPS[c]].ljust(CAPS[c],'-') for c in LOOPS)
    te=torch.from_numpy(_imap(ts,pep,L_TCR,L_PEP)).unsqueeze(0).to(dev)
    ph=torch.from_numpy(_imap(pep,PS,L_PEP,L_HLA)).unsqueeze(0).to(dev)
    gg=torch.from_numpy(_globals(''.join(lp[c] for c in LOOPS),pep)).float().unsqueeze(0).to(dev)
    return float(m(te,ph,gg).item())
def mut_t(k):
    c,i=tpos[k]; lp=dict(loops); s=lp[c]; lp[c]=s[:i]+SUB(s[i])+s[i+1:]; return lp
def mut_p(j): return PEP[:j]+SUB(PEP[j])+PEP[j+1:]

t0=time.time(); n_eval=0
f0=logit(loops,PEP); n_eval+=1
fi=np.array([logit(mut_t(k),PEP) for k in range(len(tpos))]); n_eval+=len(tpos)
fj=np.array([logit(loops,mut_p(j)) for j in range(len(PEP))]); n_eval+=len(PEP)
I=np.full((L_TCR,L_PEP),np.nan); NA=np.zeros((L_TCR,L_PEP),bool); PAD=np.ones((L_TCR,L_PEP),bool)
off={}; o=0
for c in LOOPS: off[c]=o; o+=CAPS[c]
for k,(c,i) in enumerate(tpos):
    row=off[c]+i
    for j in range(len(PEP)):
        fij=logit(mut_t(k), mut_p(j)); n_eval+=1
        I[row,j]=f0-fi[k]-fj[j]+fij
        NA[row,j]= (loops[c][i]=='A') or (PEP[j]=='A')
        PAD[row,j]=False
el=time.time()-t0
print(f'{n_eval} model evaluations in {el:.1f}s ({el/n_eval*1000:.1f} ms each)')
print(f'I range {np.nanmin(I):+.4f} to {np.nanmax(I):+.4f} | native-Ala cells {NA.sum()}')
np.savez(OUT+'dp_interaction.npz', I=I, native_ala=NA, padding=PAD,
         f0=f0, fi=fi, fj=fj, peptide=PEP, pdb=ex.pdb)
pd.DataFrame(dict(kind=['tcr']*len(tpos)+['pep']*len(PEP),
                  loop=[c for c,_ in tpos]+['-']*len(PEP),
                  pos=[i+1 for _,i in tpos]+list(range(1,len(PEP)+1)),
                  wt=[loops[c][i] for c,i in tpos]+list(PEP),
                  sub=[SUB(loops[c][i]) for c,i in tpos]+[SUB(a) for a in PEP],
                  logit=list(fi)+list(fj))).to_csv(OUT+'dp_singles.csv',index=False)
json.dump(dict(pdb=ex.pdb, peptide=PEP, allele=ex.allele, checkpoint=CKPT,
               policy='A->G if native A else X->A', n_eval=n_eval, seconds=round(el,1),
               f0=f0, in_clean39=bool(ex.in_clean39)), open(OUT+'dp_meta.json','w'), indent=1)
print('wrote dp_interaction.npz, dp_singles.csv, dp_meta.json')
