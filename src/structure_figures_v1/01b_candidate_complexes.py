"""Which clean-39 complexes have a verified, model-compatible record?
Writes results/structure_figures_v1/candidate_complexes.csv. Creates nothing else."""
OUT='results/structure_figures_v1/candidate_complexes.csv'
import json, sys, pandas as pd
sys.path.insert(0,'.')
from src.extract_priors import atoms_by_chain, AA3
from src.dataset_stepmhc import CAPS, LOOPS

pool = pd.read_csv('data/broadened/broadened_train_ready.csv')
pos  = pool[pool.binder==1]
rows=[]
for c in json.load(open('results/struct_val/_chain_assignment_clean39.json')):
    pdb=c['pdb']
    txt=open(f'data/structures/{pdb.lower()}.pdb').read()
    ch=atoms_by_chain(txt)
    pep=''.join(AA3[v[0]] for k,v in ch[c['pep']].items())
    m=pos[pos.peptide==pep]
    rows.append(dict(pdb=pdb, peptide=pep, pep_len=len(pep),
                     pool_positives=len(m),
                     alleles=';'.join(sorted(m.allele.unique()))[:60] if len(m) else '',
                     model_compatible=bool(len(m) and 8<=len(pep)<=12)))
d=pd.DataFrame(rows).sort_values(['model_compatible','pool_positives'], ascending=False)
d.to_csv(OUT,index=False)
print(f'{d.model_compatible.sum()} of {len(d)} complexes have >=1 model-compatible positive')
print(d.head(15).to_string(index=False))
