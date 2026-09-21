"""STAGE 2 — deterministic one-structure-per-peptide selection + frozen protocol.
Rule declared BEFORE selection, applied mechanically, no energies consulted:
  ELIGIBLE = peptide sequence exact AND all six loops match the model record.
  RANK within peptide (first non-tie wins):
    1. fewest interface-chain altLoc atoms
    2. no non-water heteroatoms preferred
    3. no internal chain gaps preferred
    4. best (lowest) resolution
    5. PDB ID alphabetical  [deterministic tie-break]
Writes ONLY  results/structure_figures_v1/foldx_cohort_v1/COHORT_FROZEN.csv"""
OUT='results/structure_figures_v1/foldx_cohort_v1/COHORT_FROZEN.csv'
import pandas as pd
d=pd.read_csv('results/structure_figures_v1/foldx_cohort_v1/COHORT_ELIGIBILITY_RAW.csv')
TUM={'SLLMWITQC','SLLMWITQV','AAGIGILTV'}
ANTIGEN={'SLLMWITQC':'NY-ESO-1','SLLMWITQV':'NY-ESO-1','AAGIGILTV':'MART-1'}
d['antigen_class']=d.peptide.map(lambda p:'tumour' if p in TUM else 'viral')
d['antigen']=d.peptide.map(ANTIGEN).fillna(d.peptide)
e=d[d.peptide_exact & d.all_six_loops_match].copy()
e['n_iface_alt']=e.apply(lambda r: r.n_altloc_atoms if r.altloc_in_iface_chains else 0, axis=1)
e=e.sort_values(['peptide','n_iface_alt','has_nonwater_het','has_internal_gap',
                 'resolution','pdb'],
                ascending=[True,True,True,True,True,True])
sel=e.groupby('peptide',as_index=False).first()
sel=sel.sort_values(['antigen_class','resolution'])
cols=['pdb','peptide','antigen','antigen_class','allele','resolution','n_iface_alt',
      'has_nonwater_het','has_internal_gap','in_clean39_prior','pool_pos_for_peptide',
      'chain_mhc','chain_pep','chain_tra','chain_trb']
sel[cols].to_csv(OUT,index=False)
print(f'ELIGIBLE {len(e)} structures over {e.peptide.nunique()} peptides')
print(f'SELECTED {len(sel)} structures, one per peptide\n')
print(sel[cols[:8]].to_string(index=False))
print('\nby class :', sel.groupby('antigen_class').size().to_dict())
print('by antigen (clustering unit):', sel.groupby('antigen').size().to_dict())
print(f'\nDROPPED peptides (no loop-matching structure): '
      f'{sorted(set(d.peptide)-set(e.peptide))}')
