"""STAGE 5 — frozen per-PDB FoldX chain table with sequence verification, plus expected
mutation/model counts. READ-ONLY. Writes FOLDX_CHAIN_TABLE.csv and MUTATION_FORECAST.csv."""
OUT='results/structure_figures_v1/foldx_cohort_v1/'
import json, sys, collections
import pandas as pd, numpy as np
sys.path.insert(0,'.')
from src.extract_priors import atoms_by_chain, AA3
from src.dataset_stepmhc import CAPS, LOOPS
from anarci import run_anarci
IMGT={'1':(27,38),'2':(56,65),'3':(105,117)}
coh=pd.read_csv(OUT+'COHORT_FROZEN_v2.csv')
ex=pd.read_csv('results/structure_figures_v1/exemplars_final.csv')
ca={c['pdb'].lower():c for c in json.load(open('data/priors/_chain_assignment.json'))}
dl=pd.read_csv('results/structure_figures_v1/perturb_loops_clustered_logit.csv')
dp=pd.read_csv('results/structure_figures_v1/perturb_peptide_clustered_logit.csv')

def loops_of(res):
    seq=''.join(AA3[x[1][0]] for x in res)
    o=run_anarci([('x',seq)],scheme='imgt')[1][0]
    if o is None: return {}
    numb=o[0][0]; si=0; out=collections.defaultdict(list)
    for (pos,ins),aa in numb:
        if aa=='-': continue
        while si<len(res) and AA3[res[si][1][0]]!=aa: si+=1
        if si>=len(res): break
        for cdr,(lo,hi) in IMGT.items():
            if lo<=pos<=hi: out[cdr].append(aa)
        si+=1
    return {k:''.join(v) for k,v in out.items()}

rows=[]; fore=[]
for _,r in coh.iterrows():
    c=ca[r.pdb.lower()]; ch=atoms_by_chain(open(f'data/structures/{r.pdb.lower()}.pdb').read())
    b2m=[cid for cid in ch if cid not in (c['mhc'],c['pep'],c['tra'],c['trb'])]
    pep=''.join(AA3[v[0]] for v in ch[c['pep']].values())
    la=loops_of(list(ch[c['tra']].items())); lb=loops_of(list(ch[c['trb']].items()))
    e=ex[ex.pdb==r.pdb].iloc[0]
    core=lambda s: s[1:-1] if s.startswith('C') and s[-1] in 'FW' else s
    ver=(pep==r.peptide and la.get('3','')==core(e.A3) and lb.get('3','')==core(e.B3)
         and la.get('1','')==e.A1 and la.get('2','')==e.A2
         and lb.get('1','')==e.B1 and lb.get('2','')==e.B2)
    g1=c['tra']+c['trb']
    g2=''.join(sorted(set([c['mhc'],c['pep']]+b2m)))
    rows.append(dict(pdb=r.pdb, peptide=r.peptide, antigen=r.antigen,
                     antigen_class=r.antigen_class, resolution=r.resolution,
                     chain_mhc=c['mhc'], chain_b2m=';'.join(b2m) or 'none',
                     chain_pep=c['pep'], chain_tra=c['tra'], chain_trb=c['trb'],
                     foldx_group1=g1, foldx_group2=g2,
                     source='data/priors/_chain_assignment.json',
                     peptide_observed=pep, seq_verified=ver))
    npep=sum(1 for a in r.peptide if a!='A')
    nloop=sum(1 for L in LOOPS for a in str(e[L]) if a!='A')
    fore.append(dict(pdb=r.pdb, peptide=r.peptide, antigen_class=r.antigen_class,
                     n_peptide_nonAla=npep, n_loop_nonAla_upper_bound=nloop,
                     note='TCR count is an UPPER BOUND; final set needs FoldX interface membership'))
T=pd.DataFrame(rows); T.to_csv(OUT+'FOLDX_CHAIN_TABLE.csv',index=False)
F=pd.DataFrame(fore); F.to_csv(OUT+'MUTATION_FORECAST.csv',index=False)
print(f'chain table -> FOLDX_CHAIN_TABLE.csv ({len(T)} structures)')
print(T[['pdb','chain_mhc','chain_b2m','chain_pep','chain_tra','chain_trb',
         'foldx_group1','foldx_group2','seq_verified']].to_string(index=False))
print(f'\nALL SEQUENCES VERIFIED: {T.seq_verified.all()}')
print(f'distinct Group1/Group2 combinations: {T.groupby(["foldx_group1","foldx_group2"]).size().to_dict()}')
print(f'\npeptide mutations (exact)       : {F.n_peptide_nonAla.sum()}')
print(f'TCR-loop mutations (upper bound): {F.n_loop_nonAla_upper_bound.sum()}')
tot=F.n_peptide_nonAla.sum()+F.n_loop_nonAla_upper_bound.sum()
print(f'total upper bound               : {tot} mutations')
print(f'BuildModel PDBs (x5 runs, +WT)  : {tot*10} | AnalyseComplex calls: {tot*10}')
print(f'est. runtime: RepairPDB {len(T)*200/60:.0f} min + AnalyseComplex {tot*10*1.7/60:.0f} min')
