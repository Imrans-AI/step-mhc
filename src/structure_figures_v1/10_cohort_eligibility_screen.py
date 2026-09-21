"""STAGE 1 — pre-energy eligibility screen for the expanded FoldX cohort.
READ-ONLY on all existing results. Writes ONLY:
  results/structure_figures_v1/foldx_cohort_v1/COHORT_ELIGIBILITY_RAW.csv
No FoldX is run. No selection is made. No energies are computed or read."""
OUT='results/structure_figures_v1/foldx_cohort_v1/COHORT_ELIGIBILITY_RAW.csv'
import json, os, sys, collections
import pandas as pd, numpy as np
sys.path.insert(0,'.')
from src.extract_priors import atoms_by_chain, AA3
from src.dataset_stepmhc import CAPS, LOOPS
from anarci import run_anarci
IMGT={'1':(27,38),'2':(56,65),'3':(105,117)}

ex=pd.read_csv('results/structure_figures_v1/exemplars_final.csv')
ca={c['pdb'].lower():c for c in json.load(open('data/priors/_chain_assignment.json'))}
prior39={x.lower() for x in pd.read_csv('results/struct_val/clean_prior_pdbs.csv').pdb}

def loops_from_chain(res):
    seq=''.join(AA3[r[1][0]] for r in res)
    o=run_anarci([('x',seq)],scheme='imgt')[1][0]
    if o is None: return None
    numb=o[0][0]; si=0; per=collections.defaultdict(list)
    for (pos,ins),aa in numb:
        if aa=='-': continue
        while si<len(res) and AA3[res[si][1][0]]!=aa: si+=1
        if si>=len(res): break
        for cdr,(lo,hi) in IMGT.items():
            if lo<=pos<=hi: per[cdr].append(aa)
        si+=1
    return {k:''.join(v) for k,v in per.items()}

rows=[]
for _,e in ex.iterrows():
    pdb=e.pdb.lower(); path=f'data/structures/{pdb}.pdb'
    r=dict(pdb=e.pdb, peptide=e.peptide, allele=e.allele,
           pool_pos_for_peptide=e.pool_pos_for_peptide,
           in_clean39_prior=pdb in prior39)
    if not os.path.exists(path):
        rows.append({**r, 'eligible':False, 'reason':'no PDB file'}); continue
    raw=open(path).read(); txt=raw.split('\n')
    c=ca.get(pdb)
    if c is None:
        rows.append({**r, 'eligible':False, 'reason':'no chain assignment'}); continue
    r.update(chain_mhc=c['mhc'], chain_pep=c['pep'], chain_tra=c['tra'], chain_trb=c['trb'])
    # resolution
    res_a=[l for l in txt if l.startswith('REMARK   2 RESOLUTION')]
    try: r['resolution']=float(res_a[0].split()[3])
    except Exception: r['resolution']=np.nan
    r['method']='XRAY' if any(l.startswith('EXPDTA') and 'DIFFRACTION' in l for l in txt) else 'other/unknown'
    ch=atoms_by_chain(raw)
    # peptide
    if c['pep'] not in ch:
        rows.append({**r,'eligible':False,'reason':'peptide chain absent'}); continue
    pep=''.join(AA3[v[0]] for k,v in ch[c['pep']].items())
    r['peptide_observed']=pep; r['peptide_exact']=(pep==e.peptide)
    # altLoc / hetero
    alt=[l for l in txt if l.startswith('ATOM') and l[16].strip()]
    r['n_altloc_atoms']=len(alt)
    r['altloc_chains']=','.join(sorted({l[21] for l in alt})) if alt else ''
    r['altloc_in_iface_chains']=any(l[21] in (c['pep'],c['tra'],c['trb']) for l in alt)
    het=collections.Counter(l[17:20].strip() for l in txt if l.startswith('HETATM'))
    r['n_water']=het.pop('HOH',0); r['nonwater_het']=';'.join(f'{k}:{v}' for k,v in het.items())
    r['has_nonwater_het']=bool(het)
    # loops
    ok_loops=True
    for cid,tag in ((c['tra'],'A'),(c['trb'],'B')):
        if cid not in ch: ok_loops=False; break
        lp=loops_from_chain(list(ch[cid].items()))
        if lp is None: ok_loops=False; break
        for cdr in ('1','2','3'):
            key=tag+cdr
            obs=lp.get(cdr,'')
            exp=str(e[key])
            exp_core=exp[1:-1] if key in ('A3','B3') and exp.startswith('C') and exp[-1] in 'FW' else exp
            r[f'{key}_obs']=obs; r[f'{key}_match']=(obs==exp_core)
            if obs=='' or obs!=exp_core: ok_loops=False
    r['all_six_loops_match']=ok_loops
    # internal gaps in TCR chains
    gaps={}
    for cid in (c['tra'],c['trb'],c['pep']):
        if cid not in ch: continue
        nums=[]
        for k in ch[cid]:
            k=k.strip()
            while k and not k[-1].isdigit(): k=k[:-1]   # strip insertion code
            if k: nums.append(int(k))
        nums=sorted(set(nums))
        gaps[cid]=sum(1 for a,b in zip(nums,nums[1:]) if b-a>1)
    r['internal_gaps']=';'.join(f'{k}:{v}' for k,v in gaps.items())
    r['has_internal_gap']=any(v>0 for v in gaps.values())
    rows.append(r)
d=pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUT),exist_ok=True)
d.to_csv(OUT,index=False)
print(f'screened {len(d)} complexes -> {OUT}\n')
for col in ['peptide_exact','all_six_loops_match','has_nonwater_het','has_internal_gap','altloc_in_iface_chains']:
    if col in d: print(f'  {col:26s} {d[col].sum()} of {d[col].notna().sum()}')
print('\nresolution:', d.resolution.describe()[['min','50%','max']].round(2).to_dict())
print('methods:', d.method.value_counts().to_dict())
