"""Establish the independent experimental unit before any CI is computed.
Writes results/structure_figures_v1/DEPENDENCE_AUDIT.md. Creates nothing else."""
OUT='results/structure_figures_v1/DEPENDENCE_AUDIT.md'
import sys, pandas as pd, numpy as np
sys.path.insert(0,'.')
L=[]; w=lambda s='': (L.append(s), print(s))
ex=pd.read_csv('results/structure_figures_v1/exemplars_final.csv')
P=pd.read_csv('results/structure_figures_v1/perturb_peptide.csv')
Lo=pd.read_csv('results/structure_figures_v1/perturb_loops.csv')

w('# DEPENDENCE AUDIT — what is the independent unit?')
w()
w('## Counts')
w(f'verified PDB complexes            : {ex.pdb.nunique()}')
w(f'distinct peptides                 : {ex.peptide.nunique()}')
w(f'distinct CDR3a                    : {ex.A3.nunique()}')
w(f'distinct CDR3b                    : {ex.B3.nunique()}')
w(f'distinct complete paired TCRs (6 loops): {ex[["A1","A2","A3","B1","B2","B3"]].drop_duplicates().shape[0]}')
w(f'distinct (peptide, paired TCR) pairs   : {ex[["peptide","A1","A2","A3","B1","B2","B3"]].drop_duplicates().shape[0]}')
w(f'distinct (peptide, CDR3b) pairs        : {ex[["peptide","B3"]].drop_duplicates().shape[0]}')
w(f'distinct alleles                  : {ex.allele.nunique()}')
w()
w('## Redundancy beyond identical peptide strings')
dup=ex.groupby('peptide').pdb.agg(list)
for pep,pl in dup[dup.map(len)>1].items():
    sub=ex[ex.peptide==pep]
    ntcr=sub[['A3','B3']].drop_duplicates().shape[0]
    w(f'  {pep:12s} {len(pl)} PDBs, {ntcr} distinct CDR3a/b pairs: {pl}')
w()
same=ex.groupby(['peptide','A3','B3']).pdb.agg(list)
rep=same[same.map(len)>1]
w(f'IDENTICAL (peptide, CDR3a, CDR3b) appearing in >1 PDB: {len(rep)}')
for k,v in rep.items(): w(f'  {k[0]} / {k[1]} / {k[2]}  -> {v}')
w()
w('## Decision')
w('Peptide-position analyses: cluster unit = DISTINCT PEPTIDE '
  f'({ex.peptide.nunique()} clusters over {ex.pdb.nunique()} complexes).')
w('Loop analyses: cluster unit = DISTINCT (peptide, paired TCR) if it exceeds the peptide count, '
  'else peptide. Chosen value printed above.')
w('Cluster bootstrap: resample CLUSTERS with replacement; all complexes in a cluster move together.')
w('Report as "50 verified complexes / 25 distinct peptide clusters". '
  'Never "n = 50 independent complexes".')
open(OUT,'w').write('\n'.join(L)); print(f'\nwrote {OUT}')
