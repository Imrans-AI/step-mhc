"""Repair-geometry and contact-retention audit for the 18-structure cohort.
Acceptance (locked prereg S9): unchanged sequences; global and peptide C-alpha RMSD < 0.1 A;
100% TCR-peptide contact retention; >= 85% whole TCR-pMHC retention.
Also repairs the broken repair_total_energy column by reading the .fxout files.
Writes REPAIR_ACCEPTANCE_AUDIT.csv and PREPARED_CHECKSUMS_v2.csv. READ-ONLY otherwise."""
OUT='results/structure_figures_v1/foldx_cohort_v1/'
import pandas as pd, numpy as np, sys, os, re
sys.path.insert(0,'.')
from src.extract_priors import atoms_by_chain, AA3
T=pd.read_csv(OUT+'FOLDX_CHAIN_TABLE.csv'); C=pd.read_csv(OUT+'PREPARED_CHECKSUMS.csv')

def load(p):
    d={}
    for l in open(p):
        if l.startswith('ATOM'):
            d[(l[21], l[22:27].strip(), l[12:16].strip())]=np.array(
                [float(l[30:38]),float(l[38:46]),float(l[46:54])])
    return d
def seqs(p):
    ch=atoms_by_chain(open(p).read())
    return {c:''.join(AA3[v[0]] for v in ch[c].values()) for c in ch}, ch
def contacts(ch, g1, g2, cut=5.0):
    A={(c,k):np.asarray(v[1]) for c in g1 if c in ch for k,v in ch[c].items()}
    B={(c,k):np.asarray(v[1]) for c in g2 if c in ch for k,v in ch[c].items()}
    out=set()
    for ka,va in A.items():
        for kb,vb in B.items():
            if np.sqrt(((va[:,None,:]-vb[None,:,:])**2).sum(-1)).min()<=cut: out.add((ka,kb))
    return out

rows=[]
for _,r in T.iterrows():
    d=OUT+f'prepared/{r.pdb.upper()}'
    pp=f'{d}/{r.pdb.lower()}_prepared.pdb'; rp=f'{d}/{r.pdb.lower()}_prepared_Repair.pdb'
    fx=f'{d}/{r.pdb.lower()}_prepared_Repair.fxout'
    e=None
    if os.path.exists(fx):
        for l in open(fx):
            m=re.match(r'^Total\s+=?\s*(-?\d+\.?\d*)', l.strip())
            if m: e=float(m.group(1))
    so,cho=seqs(pp); sr,chr_=seqs(rp)
    seq_ok=all(so.get(c)==sr.get(c) for c in so)
    ao,ar=load(pp),load(rp)
    ca=[k for k in ao if k[2]=='CA' and k in ar]
    g=np.array([np.linalg.norm(ao[k]-ar[k]) for k in ca])
    pca=[k for k in ca if k[0]==r.chain_pep]
    pg=np.array([np.linalg.norm(ao[k]-ar[k]) for k in pca])
    g1,g2=list(r.foldx_group1),list(r.foldx_group2)
    co=contacts(cho,g1,g2); cr=contacts(chr_,g1,g2)
    tp_o={x for x in co if x[1][0]==r.chain_pep}; tp_r={x for x in cr if x[1][0]==r.chain_pep}
    rows.append(dict(pdb=r.pdb, repair_energy=e, seq_unchanged=seq_ok,
        ca_rmsd=round(float(np.sqrt((g**2).mean())),4),
        pep_ca_rmsd=round(float(np.sqrt((pg**2).mean())),4),
        n_contacts_orig=len(co), n_contacts_repair=len(cr),
        whole_retention=round(len(co&cr)/max(len(co),1)*100,1),
        tcr_pep_orig=len(tp_o), tcr_pep_retention=round(len(tp_o&tp_r)/max(len(tp_o),1)*100,1)))
A=pd.DataFrame(rows)
A['ACCEPT']=(A.seq_unchanged & (A.ca_rmsd<0.1) & (A.pep_ca_rmsd<0.1)
             & (A.tcr_pep_retention>=100.0) & (A.whole_retention>=85.0))
A.to_csv(OUT+'REPAIR_ACCEPTANCE_AUDIT.csv',index=False)
C.drop(columns=['repair_total_energy']).merge(A[['pdb','repair_energy']],on='pdb')\
 .to_csv(OUT+'PREPARED_CHECKSUMS_v2.csv',index=False)
print(A.to_string(index=False))
print(f'\nACCEPTED {A.ACCEPT.sum()}/{len(A)}')
if not A.ACCEPT.all(): print('FAILURES:\n', A[~A.ACCEPT].to_string(index=False))
