"""STAGE D — per-biological-copy mapping + exact overlap audit. READ-ONLY, no inference.
Pairs chains by PHYSICAL CONTACT, never alphabetic adjacency.
Writes CANCER_EXTENSION_CHAIN_MAP.csv and CANCER_EXTENSION_OVERLAP_AUDIT.csv"""
P='results/structure_figures_v1/foldx_energy_v1/cancer_extension_preflight/'
import os
import pandas as pd, numpy as np, sys, re, collections
sys.path.insert(0,'.')
from src.extract_priors import atoms_by_chain, AA3
from src.dataset_stepmhc import CAPS, LOOPS
from anarci import run_anarci
IMGT={'1':(27,38),'2':(56,65),'3':(105,117)}
ASSIGN={'4EUP':'HLA-A*02:01','6RSY':'HLA-A*02:01','6VM7':'HLA-A*02:01','6VM8':'HLA-A*02:01',
 '6VM9':'HLA-A*02:01','6VMA':'HLA-A*02:01','6VMC':'HLA-A*02:01','7RM4':'HLA-A*02:01',
 '8ES8':'HLA-A*02:01','8ES9':'HLA-A*02:01','6VQO':'HLA-A*02:01','6P64':'HLA-A*02:06',
 '6UK4':'HLA-A*02:06','8I5C':'HLA-A*11:01','8I5D':'HLA-A*11:01','8WTE':'HLA-A*11:01',
 '8WUL':'HLA-A*11:01','9IKY':'HLA-A*11:01','7PB2':'HLA-A*11:01','6ULN':'HLA-C*08:02',
 '6ULR':'HLA-C*08:02','6UON':'HLA-C*08:02'}
h=pd.read_csv(os.environ["STEPMHC_HLA_CSV"])
norm=lambda a: str(a).replace("HLA-","").replace("*","").replace(":","").strip()
HMAP={norm(h.iloc[i,0]): h.iloc[i,1] for i in range(len(h))}
d=pd.read_csv('results/structure_figures_v1/pdb_discovery_20260828/NEW_CANCER_TCR_PMHC_CANDIDATES.csv')
pos=pd.read_csv('data/broadened/broadened_train_ready.csv').query('binder==1')
cen=lambda coords: np.mean(np.vstack(coords),axis=0)
mind=lambda A,B: float(np.sqrt(((np.vstack(A)[:,None,:]-np.vstack(B)[None,:,:])**2).sum(-1)).min())

rows=[]
for pid,allele in ASSIGN.items():
    raw=open(d[d.pdb_id.str.upper()==pid].path.iloc[0]).read(); ch=atoms_by_chain(raw)
    L={c:len(v) for c,v in ch.items()}
    peps=[c for c in ch if 7<=L[c]<=15]
    mhcs=[c for c in ch if 240<=L[c]<=300 and run_anarci([('x',''.join(AA3[v[0]] for v in ch[c].values()))],scheme='imgt')[1][0] is None]
    tcrs={}
    for c in ch:
        if not (150<=L[c]<=300): continue
        seq=''.join(AA3[v[0]] for v in ch[c].values())
        o=run_anarci([('x',seq)],scheme='imgt')[1][0]
        if o is None: continue
        numb=o[0][0]
        tcrs[c]={k:''.join(a for (p,i),a in numb if lo<=p<=hi and a!='-') for k,(lo,hi) in IMGT.items()}
    coords={c:[v[1] for v in ch[c].values()] for c in ch}
    for i,pc in enumerate(peps):                       # one copy per peptide chain
        mh=min(mhcs,key=lambda m: mind(coords[pc],coords[m])) if mhcs else None
        near=sorted(tcrs, key=lambda t: mind(coords[pc],coords[t]))[:2]
        if len(near)<2: continue
        # alpha vs beta by CDR3 length heuristic is unsafe; use ANARCI chain type via loops
        a,b=near
        rows.append(dict(pdb=pid, copy=i+1, pep_chain=pc,
            peptide=''.join(AA3[v[0]] for v in ch[pc].values()),
            mhc_chain=mh, allele=allele, hla_pseudo=HMAP[norm(allele)],
            tcr_c1=a, tcr_c2=b,
            c1_CDR1=tcrs[a]['1'], c1_CDR2=tcrs[a]['2'], c1_CDR3=tcrs[a]['3'],
            c2_CDR1=tcrs[b]['1'], c2_CDR2=tcrs[b]['2'], c2_CDR3=tcrs[b]['3'],
            d_pep_c1=round(mind(coords[pc],coords[a]),2),
            d_pep_c2=round(mind(coords[pc],coords[b]),2)))
M=pd.DataFrame(rows); M.to_csv(P+'CANCER_EXTENSION_CHAIN_MAP.csv',index=False)
print(f'{len(M)} biological copies across {M.pdb.nunique()} PDBs')
print(M.groupby('pdb').copy.count().to_string())

# exact overlap: all six loops (anchors restored) + peptide + pseudo
R=lambda s:'C'+s+'F'
ov=[]
for _,r in M.iterrows():
    lp={r.c1_CDR1,r.c1_CDR2,R(r.c1_CDR3),r.c2_CDR1,r.c2_CDR2,R(r.c2_CDR3)}
    same_pep=pos[pos.peptide==r.peptide]
    exact=same_pep[same_pep.apply(lambda x:{x.A1,x.A2,x.A3,x.B1,x.B2,x.B3}==lp,axis=1)]
    pep_hla=same_pep[same_pep.hla_pseudo==r.hla_pseudo]
    tcr_only=pos[pos.apply(lambda x:{x.A1,x.A2,x.A3,x.B1,x.B2,x.B3}==lp,axis=1)]
    lvl=('A_exact_record' if len(exact) else
         'B_peptide_HLA' if len(pep_hla) else
         'C_peptide_only' if len(same_pep) else
         'D_TCR_only' if len(tcr_only) else 'F_no_detected_overlap')
    ov.append(dict(pdb=r.pdb, copy=r.copy, peptide=r.peptide, allele=r.allele,
                   n_exact=len(exact), n_pep_hla=len(pep_hla), n_pep=len(same_pep),
                   n_tcr_only=len(tcr_only), overlap_level=lvl))
O=pd.DataFrame(ov); O.to_csv(P+'CANCER_EXTENSION_OVERLAP_AUDIT.csv',index=False)
print('\nOVERLAP LEVELS:'); print(O.overlap_level.value_counts().to_string())
print(O.groupby(['pdb','overlap_level']).size().to_string())
