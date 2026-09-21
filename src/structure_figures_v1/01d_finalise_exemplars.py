"""Apply the pre-declared exemplar criteria and freeze the choice.
CRITERIA (declared before inspection, prompt Fig1-D / Fig2-D):
  1. peptide length 9
  2. complete chain assignment (mhc/pep/tra/trb)
  3. PDB's own CDR3a+CDR3b, anchors restored, match a training-pool POSITIVE with that peptide
  4. tie-break: most pool positives for the peptide
Writes results/structure_figures_v1/exemplars_final.csv"""
OUT='results/structure_figures_v1/exemplars_final.csv'
import pandas as pd
d=pd.read_csv('results/structure_figures_v1/exemplar_selection.csv')
pos=pd.read_csv('data/broadened/broadened_train_ready.csv').query('binder==1')
clean={x.upper() for x in pd.read_csv('results/struct_val/clean_prior_pdbs.csv').pdb}
R=lambda s:'C'+str(s)+'F'
rows=[]
for _,r in d.iterrows():
    a,b=R(r.pdb_A3),R(r.pdb_B3)
    ex=pos[(pos.peptide==r.peptide)&(pos.A3==a)&(pos.B3==b)]
    if not len(ex): continue
    e=ex.iloc[0]
    rows.append(dict(pdb=r.pdb, peptide=r.peptide, A1=e.A1, A2=e.A2, A3=a,
                     B1=e.B1, B2=e.B2, B3=b, allele=e.allele, hla_pseudo=e.hla_pseudo,
                     n_pool_records=len(ex),
                     pool_pos_for_peptide=int(r.pool_pos_for_peptide),
                     in_clean39=r.pdb.upper() in clean))
f=pd.DataFrame(rows).sort_values('pool_pos_for_peptide',ascending=False).reset_index(drop=True)
f.to_csv(OUT,index=False)
print(f'{len(f)} verified exemplars')
print(f.head(6)[['pdb','peptide','A3','B3','allele','pool_pos_for_peptide','in_clean39']].to_string(index=False))
print(f'\nPRIMARY EXEMPLAR (criteria applied): {f.iloc[0].pdb.upper()}  {f.iloc[0].peptide}  {f.iloc[0].allele}')
print(f'  in clean-39 prior set: {f.iloc[0].in_clean39}  <-- must be stated in the caption')
