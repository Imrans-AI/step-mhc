"""STAGES F+G — grouping, deterministic primary selection, and the seven report files.
READ-ONLY on all prior results. Writes only into cancer_extension_preflight/.
NO FoldX. NO model inference. NO preregistration locked."""
P='results/structure_figures_v1/foldx_energy_v1/cancer_extension_preflight/'
import pandas as pd, numpy as np
M=pd.read_csv(P+'CANCER_EXTENSION_CHAIN_MAP_v2.csv')
M=M[M.valid].copy(); M['tcr_key']=M.c1_CDR3+'|'+M.c2_CDR3
O=pd.read_csv(P+'CANCER_EXTENSION_OVERLAP_AUDIT_v2.csv')
M=M.merge(O[['pdb','copy_id','overlap_level']],on=['pdb','copy_id'],how='left')

FAM={'ALGIGILTV':('MART-1','altered_peptide_ligand','tumour_associated_self'),
 'KQWLVWLFL':('HHAT','native','mutation_derived_neoantigen'),
 'RMFPNAPYL':('WT1','native','tumour_associated_self'),
 'HMTEVVRHC':('p53-R175H','native','mutation_derived_neoantigen'),
 'ITDQVPFSV':('gp100','native','tumour_associated_self'),
 'IMDQVPFSV':('gp100','altered_peptide_ligand_T2M','tumour_associated_self'),
 'ILDQVPFSV':('gp100','altered_peptide_ligand_T2L','tumour_associated_self'),
 'GADGVGKSA':('KRAS-G12D','native','mutation_derived_neoantigen'),
 'GADGVGKSAL':('KRAS-G12D','native','mutation_derived_neoantigen'),
 'VVVGADGVGK':('KRAS-G12D','native','mutation_derived_neoantigen'),
 'VVGAVGVGK':('KRAS-G12V','native','mutation_derived_neoantigen'),
 'GVYDGREHTV':('MAGE-A4','native','cancer_testis_antigen')}
TCRSTAT={'8I5C':'engineered_constant_domain','8I5D':'engineered_constant_domain',
 '8WUL':'status_not_verifiable','9IKY':'status_not_verifiable',
 '8WTE':'status_not_verifiable','6RSY':'status_not_verifiable_possible_hybrid'}
METH={'8ES8':'CRYO-EM','8ES9':'CRYO-EM'}
RES=dict(zip(['4EUP','6P64','6RSY','6UK4','6ULN','6ULR','6UON','6VM7','6VM8','6VM9','6VMA',
 '6VMC','6VQO','7PB2','7RM4','8ES8','8ES9','8I5C','8I5D','8WTE','8WUL','9IKY'],
 [2.88,3.05,2.95,2.70,2.01,3.20,3.50,2.41,2.41,2.90,2.75,2.85,3.00,3.41,3.33,2.65,3.25,
  3.34,3.30,2.17,2.36,3.45]))
M['antigen_family']=M.peptide.map(lambda p:FAM[p][0])
M['peptide_status']=M.peptide.map(lambda p:FAM[p][1])
M['antigen_class']=M.peptide.map(lambda p:FAM[p][2])
M['tcr_status']=M.pdb.map(TCRSTAT).fillna('native_no_contrary_evidence')
M['method']=M.pdb.map(METH).fillna('X-RAY')
M['resolution']=M.pdb.map(RES)

# primary copy within each (pdb, TCR) unit: closest peptide-MHC contact, then lowest chain id
M=M.sort_values(['pdb','tcr_key','d_pep_mhc','pep_chain'])
prim=M.groupby(['pdb','tcr_key'],as_index=False).first()
prim['unit']=prim.pdb+'_'+prim.tcr_key.str[:8]
M.to_csv(P+'CANCER_EXTENSION_ELIGIBILITY_RAW.csv',index=False)

exc=pd.DataFrame([
 dict(pdb='6AVG',reason='peptide length 13 > L_PEP 12; no truncation permitted',stage='C'),
 dict(pdb='9IKY',reason='copy 7 only: peptide 41.98 A from nearest MHC — incomplete copy',stage='D'),
])
exc.to_csv(P+'CANCER_EXTENSION_EXCLUSIONS.csv',index=False)
prim.to_csv(P+'CANCER_EXTENSION_CANDIDATES_PASS.csv',index=False)

# three proposed sets
xray=prim[(prim.method=='X-RAY')]
primary=xray[(xray.peptide_status=='native') & (xray.tcr_status=='native_no_contrary_evidence')]
variant=prim[~prim.index.isin(primary.index) & (prim.method=='X-RAY')]
cryo=prim[prim.method=='CRYO-EM']
for nm,s in [('PRIMARY_XRAY',primary),('VARIANT_SENSITIVITY',variant),('CRYOEM_SENSITIVITY',cryo)]:
    s.assign(proposed_set=nm).to_csv(P+f'PROPOSED_{nm}.csv',index=False)
pd.concat([primary.assign(set='PRIMARY_XRAY'),variant.assign(set='VARIANT_SENSITIVITY'),
           cryo.assign(set='CRYOEM_SENSITIVITY')]).to_csv(P+'CANCER_EXTENSION_PROPOSED_COHORT.csv',index=False)
prim.groupby(['antigen_family','peptide','peptide_status','antigen_class']).size()\
    .reset_index(name='n_units').to_csv(P+'CANCER_EXTENSION_ANTIGEN_FAMILIES.csv',index=False)

print(f'=== 25 (PDB,TCR) units from 52 valid copies ===\n')
for nm,s in [('PRIMARY_XRAY',primary),('VARIANT_SENSITIVITY',variant),('CRYOEM_SENSITIVITY',cryo)]:
    print(f'--- {nm}: {len(s)} units ---')
    print(s[['pdb','peptide','antigen_family','peptide_status','tcr_status','resolution',
             'overlap_level']].to_string(index=False)); print()
print('antigen families:'); print(prim.groupby('antigen_family').size().to_string())
print('\noverlap:'); print(prim.overlap_level.value_counts().to_string())
# workload
prim['n_pep_mut']=prim.peptide.map(lambda p:sum(1 for a in p if a!='A'))
prim['n_tcr_upper']=prim.apply(lambda r:sum(1 for L in [r.c1_CDR1,r.c1_CDR2,r.c1_CDR3,
        r.c2_CDR1,r.c2_CDR2,r.c2_CDR3] for a in L if a!='A'),axis=1)
tot=prim.n_pep_mut.sum()+prim.n_tcr_upper.sum()
print(f'\nWORKLOAD (all 25 units): peptide {prim.n_pep_mut.sum()} + TCR upper {prim.n_tcr_upper.sum()} = {tot}')
print(f'  x10 models = {tot*10} BuildModel PDBs and AnalyseComplex calls')
print(f'  ~40% interface filter -> ~{int(prim.n_pep_mut.sum()+prim.n_tcr_upper.sum()*0.4)} mutations')
print(f'  RepairPDB {len(prim)*200/60:.0f} min + AnalyseComplex ~{tot*10*1.7/60:.0f} min')
