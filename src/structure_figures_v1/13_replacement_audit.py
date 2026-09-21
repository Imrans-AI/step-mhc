"""STAGE 4 — replacement audit for peptides whose selected structure has a TCR internal gap
with flanks within 5 A of pMHC. READ-ONLY on prior results.
Writes ONLY results/structure_figures_v1/foldx_cohort_v1/REPLACEMENT_AUDIT.csv
and COHORT_FROZEN_v2.csv.  COHORT_FROZEN.csv is PRESERVED, not overwritten."""
OUT='results/structure_figures_v1/foldx_cohort_v1/'
import json, sys, collections
import numpy as np, pandas as pd
sys.path.insert(0,'.')
from src.extract_priors import atoms_by_chain, AA3
from src.dataset_stepmhc import CAPS
from anarci import run_anarci
IMGT={'1':(27,38),'2':(56,65),'3':(105,117)}
GAPTHRESH=5.0
raw=pd.read_csv(OUT+'COHORT_ELIGIBILITY_RAW.csv')
coh=pd.read_csv(OUT+'COHORT_FROZEN.csv')
ca={c['pdb'].lower():c for c in json.load(open('data/priors/_chain_assignment.json'))}

def gap_report(pdb):
    """-> (worst_min_dist_TCR_gap, n_tcr_gaps, detail)  None if no TCR gap."""
    c=ca[pdb.lower()]; txt=open(f'data/structures/{pdb.lower()}.pdb').read()
    ch=atoms_by_chain(txt)
    pm=[np.asarray(v[1]) for cid in (c['mhc'],c['pep']) if cid in ch for v in ch[cid].values()]
    if not pm: return (np.nan,0,'no pMHC atoms')
    pmhc=np.vstack(pm)
    worst=np.inf; n=0; det=[]
    for cid in (c['tra'],c['trb']):
        if cid not in ch: continue
        nums=[]
        for k in ch[cid]:
            kk=k.strip()
            while kk and not kk[-1].isdigit(): kk=kk[:-1]
            if kk: nums.append(int(kk))
        for a,b in zip(sorted(set(nums)),sorted(set(nums))[1:]):
            if b-a<=1: continue
            n+=1
            fl=[]
            for x in (a,b):
                ks=[k for k in ch[cid] if k.strip().rstrip('ABCDEFGHIJKLMNOPQRSTUVWXYZ')==str(x)]
                if ks: fl.append(np.asarray(ch[cid][ks[0]][1]))
            if fl:
                f=np.vstack(fl)
                d=float(np.sqrt(((f[:,None,:]-pmhc[None,:,:])**2).sum(-1)).min())
                worst=min(worst,d); det.append(f'{cid}:{a}-{b}({b-a-1}res,{d:.2f}A)')
    return (worst if worst<np.inf else np.nan, n, ';'.join(det))

# structures whose selection is disqualified
bad=[]
for _,r in coh.iterrows():
    w,n,det=gap_report(r.pdb)
    if n and w==w and w<GAPTHRESH: bad.append((r.pdb,r.peptide,w,det))
print('DISQUALIFIED selections (TCR gap flank < 5 A from pMHC):')
for p,pep,w,det in bad: print(f'  {p} {pep}: min {w:.2f} A | {det}')

rows=[]; repl={}
for pdb,pep,w,det in bad:
    cands=raw[(raw.peptide==pep) & raw.peptide_exact & raw.all_six_loops_match]
    print(f'\n--- {pep}: {len(cands)} eligible structure(s) -> {sorted(cands.pdb)}')
    ok=[]
    for _,c2 in cands.iterrows():
        w2,n2,det2=gap_report(c2.pdb)
        safe=(n2==0) or (w2!=w2) or (w2>=GAPTHRESH)
        rows.append(dict(peptide=pep, candidate=c2.pdb, resolution=c2.resolution,
                         n_iface_alt=c2.n_altloc_atoms if c2.altloc_in_iface_chains else 0,
                         has_nonwater_het=c2.has_nonwater_het,
                         n_tcr_gaps=n2, min_gap_dist=None if w2!=w2 else round(w2,2),
                         gap_detail=det2, gap_safe=safe,
                         was_selected=(c2.pdb==pdb)))
        print(f'    {c2.pdb} res={c2.resolution} alt={rows[-1]["n_iface_alt"]} '
              f'het={c2.has_nonwater_het} gaps={n2} mindist={rows[-1]["min_gap_dist"]} '
              f'-> {"SAFE" if safe else "DISQUALIFIED"}')
        if safe: ok.append(c2)
    if ok:
        o=pd.DataFrame(ok).sort_values(['n_altloc_atoms','has_nonwater_het','resolution','pdb'])
        repl[pep]=(pdb,o.iloc[0].pdb)
        print(f'    => REPLACEMENT: {pdb} -> {o.iloc[0].pdb}')
    else:
        repl[pep]=(pdb,None); print(f'    => NO SAFE REPLACEMENT — peptide EXCLUDED from primary cohort')
pd.DataFrame(rows).to_csv(OUT+'REPLACEMENT_AUDIT.csv',index=False)

v2=coh.copy(); drop=[]
for pep,(old,new) in repl.items():
    if new is None: drop.append(pep)
    else:
        src=raw[raw.pdb==new].iloc[0]
        i=v2.index[v2.peptide==pep][0]
        for col in ['pdb','resolution','chain_mhc','chain_pep','chain_tra','chain_trb']:
            if col in src: v2.loc[i,col]=src[col]
v2=v2[~v2.peptide.isin(drop)]
v2.to_csv(OUT+'COHORT_FROZEN_v2.csv',index=False)
print(f'\n=== REVISED COHORT: {len(v2)} structures ({len(coh)} - {len(drop)} excluded) ===')
print('excluded peptides:', drop if drop else 'none')
print(v2[['pdb','peptide','antigen','antigen_class','resolution']].to_string(index=False))
