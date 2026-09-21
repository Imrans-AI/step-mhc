"""Objective exemplar selection, criteria declared BEFORE inspection:
   (1) peptide length 9  (2) complete chain assignment  (3) the PDB's own CDR3a+CDR3b
   match a training-pool POSITIVE with that peptide  (4) tie-break: most pool positives.
Writes results/structure_figures_v1/exemplar_selection.csv"""
OUT='results/structure_figures_v1/exemplar_selection.csv'
import json, sys, pandas as pd
sys.path.insert(0,'.')
from src.extract_priors import atoms_by_chain, AA3
from anarci import run_anarci
IMGT3=(105,117)

def cdr3(seq):
    o=run_anarci([('x',seq)],scheme='imgt')[1][0]
    if o is None: return None
    return ''.join(a for (p,i),a in o[0][0] if IMGT3[0]<=p<=IMGT3[1] and a!='-')

pos=pd.read_csv('data/broadened/broadened_train_ready.csv').query('binder==1')
rows=[]
for c in json.load(open('data/priors/_chain_assignment.json')):
    pdb=c['pdb']
    try: ch=atoms_by_chain(open(f'data/structures/{pdb.lower()}.pdb').read())
    except Exception: continue
    if not all(c[k] in ch for k in ('pep','tra','trb')): continue
    pep=''.join(AA3[v[0]] for k,v in ch[c['pep']].items())
    if len(pep)!=9: continue
    a3=cdr3(''.join(AA3[v[0]] for k,v in ch[c['tra']].items()))
    b3=cdr3(''.join(AA3[v[0]] for k,v in ch[c['trb']].items()))
    if not a3 or not b3: continue
    ex=pos[(pos.peptide==pep)&(pos.A3==a3)&(pos.B3==b3)]
    rows.append(dict(pdb=pdb, peptide=pep, pdb_A3=a3, pdb_B3=b3,
                     exact_pool_match=len(ex),
                     pool_pos_for_peptide=int((pos.peptide==pep).sum()),
                     allele=ex.allele.iloc[0] if len(ex) else ''))
d=pd.DataFrame(rows).sort_values(['exact_pool_match','pool_pos_for_peptide'],ascending=False)
d.to_csv(OUT,index=False)
print(f'{(d.exact_pool_match>0).sum()} of {len(d)} 9-mer complexes have an EXACT (A3,B3,peptide) pool positive')
print(d.head(12).to_string(index=False))
