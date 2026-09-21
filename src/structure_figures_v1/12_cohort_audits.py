"""STAGE 3 — antigen-family, altLoc and internal-gap audits for the 21 frozen structures.
READ-ONLY. Writes only into results/structure_figures_v1/foldx_cohort_v1/:
  ANTIGEN_FAMILY_AUDIT.csv · ALTLOC_AUDIT.csv · GAP_AUDIT.csv · CHAIN_MAP.csv
No FoldX. No energies. No preparation yet."""
OUT='results/structure_figures_v1/foldx_cohort_v1/'
import json, sys, collections, re
import numpy as np, pandas as pd
sys.path.insert(0,'.')
from src.extract_priors import atoms_by_chain, AA3
from src.dataset_stepmhc import CAPS, LOOPS
from anarci import run_anarci
IMGT={'1':(27,38),'2':(56,65),'3':(105,117)}
coh=pd.read_csv(OUT+'COHORT_FROZEN.csv')
ca={c['pdb'].lower():c for c in json.load(open('data/priors/_chain_assignment.json'))}

# ---------- 1. antigen family + verified engineered status ----------
fam=[]
for _,r in coh.iterrows():
    txt=open(f'data/structures/{r.pdb.lower()}.pdb').read().split('\n')
    hdr=' '.join(l[10:].strip() for l in txt
                 if l.startswith(('TITLE','COMPND','KEYWDS','REMARK 999')))
    eng=[l[10:].strip() for l in txt if l.startswith('COMPND') and 'ENGINEERED' in l.upper()]
    mut=[l[10:].strip() for l in txt if l.startswith('COMPND') and 'MUTATION' in l.upper()]
    seqadv=[l.rstrip() for l in txt if l.startswith('SEQADV')]
    pepadv=[l for l in seqadv if len(l)>16 and l[16]==r.chain_pep]
    fam.append(dict(pdb=r.pdb, peptide=r.peptide, antigen=r.antigen,
                    antigen_class=r.antigen_class, allele=r.allele,
                    title=(' '.join(l[10:].strip() for l in txt if l.startswith('TITLE')))[:150],
                    compnd_engineered='; '.join(eng)[:120],
                    compnd_mutation='; '.join(mut)[:120],
                    n_seqadv_total=len(seqadv), n_seqadv_peptide_chain=len(pepadv),
                    seqadv_peptide=' | '.join(x[7:60] for x in pepadv)[:200]))
F=pd.DataFrame(fam); F.to_csv(OUT+'ANTIGEN_FAMILY_AUDIT.csv',index=False)
print(f'[1] antigen-family audit -> ANTIGEN_FAMILY_AUDIT.csv ({len(F)} rows)')
print('    structures with SEQADV on the PEPTIDE chain (engineered/variant evidence):')
for _,x in F[F.n_seqadv_peptide_chain>0].iterrows():
    print(f'      {x.pdb} {x.peptide}: {x.n_seqadv_peptide_chain} SEQADV -> {x.seqadv_peptide[:90]}')

# ---------- 2. altLoc audit ----------
alt=[]
for _,r in coh.iterrows():
    for l in open(f'data/structures/{r.pdb.lower()}.pdb'):
        if l.startswith('ATOM') and l[16].strip():
            alt.append(dict(pdb=r.pdb, chain=l[21], resnum=l[22:27].strip(),
                            resname=l[17:20].strip(), atom=l[12:16].strip(),
                            altloc=l[16], occupancy=float(l[54:60]),
                            in_iface_chain=l[21] in (r.chain_pep,r.chain_tra,r.chain_trb)))
A=pd.DataFrame(alt)
if len(A): A.to_csv(OUT+'ALTLOC_AUDIT.csv',index=False)
print(f'\n[2] altLoc audit -> ALTLOC_AUDIT.csv ({len(A)} atom records)')
if len(A):
    for pdb,g in A.groupby('pdb'):
        sites=g.groupby(['chain','resnum','resname']).agg(
            n_atoms=('atom','size'), altlocs=('altloc',lambda s:''.join(sorted(set(s)))),
            occ=('occupancy',lambda s:','.join(f'{v:.2f}' for v in sorted(set(s),reverse=True))),
            iface=('in_iface_chain','first')).reset_index()
        print(f'    {pdb}: {len(g)} atoms at {len(sites)} sites')
        for _,s in sites.iterrows():
            print(f'      chain {s.chain} {s.resname}{s.resnum} altLocs={s.altlocs} '
                  f'occ={s.occ} atoms={s.n_atoms} iface_chain={s.iface}')

# ---------- 3. internal gap audit ----------
def loops_of(res):
    seq=''.join(AA3[x[1][0]] for x in res)
    o=run_anarci([('x',seq)],scheme='imgt')[1][0]
    if o is None: return {}
    numb=o[0][0]; si=0; out={}
    for (pos,ins),aa in numb:
        if aa=='-': continue
        while si<len(res) and AA3[res[si][1][0]]!=aa: si+=1
        if si>=len(res): break
        for cdr,(lo,hi) in IMGT.items():
            if lo<=pos<=hi: out.setdefault(cdr,[]).append(res[si][0])
        si+=1
    return out

gap=[]
for _,r in coh.iterrows():
    raw=open(f'data/structures/{r.pdb.lower()}.pdb').read()
    ch=atoms_by_chain(raw); c=ca[r.pdb.lower()]
    role={c['mhc']:'MHC', c['pep']:'peptide', c['tra']:'TCRa', c['trb']:'TCRb'}
    loopres={}
    for cid,tag in ((c['tra'],'A'),(c['trb'],'B')):
        if cid in ch:
            for k,v in loops_of(list(ch[cid].items())).items(): loopres[(cid,tag+k)]=set(v)
    # interface atoms: TCR within 5 A of pMHC
    tcr=[(cid,k,np.asarray(v[1])) for cid in (c['tra'],c['trb']) if cid in ch for k,v in ch[cid].items()]
    pmhc=np.vstack([np.asarray(v[1]) for cid in (c['mhc'],c['pep']) if cid in ch for k,v in ch[cid].items()])
    for cid,cont in ch.items():
        nums=[]
        for k in cont:
            kk=k.strip()
            while kk and not kk[-1].isdigit(): kk=kk[:-1]
            if kk: nums.append(int(kk))
        nums=sorted(set(nums))
        for a,b in zip(nums,nums[1:]):
            if b-a<=1: continue
            flank=[]
            for x in (a,b):
                key=[k for k in cont if k.strip().rstrip('ABCDEFGHIJKLMNOPQRSTUVWXYZ')==str(x)]
                if key: flank.append(np.asarray(cont[key[0]][1]))
            dmin=np.nan
            if flank:
                fl=np.vstack(flank)
                dmin=float(np.sqrt(((fl[:,None,:]-pmhc[None,:,:])**2).sum(-1)).min())
            hits=[t for (ccid,t),s in loopres.items()
                  if ccid==cid and any(a<int(str(z).rstrip('ABCDEFGHIJKLMNOPQRSTUVWXYZ') or 0)<b for z in s)]
            gap.append(dict(pdb=r.pdb, chain=cid, role=role.get(cid,'other'),
                            flank_lo=a, flank_hi=b, n_missing=b-a-1,
                            intersects_loops=';'.join(hits),
                            min_dist_flank_to_pMHC=round(dmin,2) if dmin==dmin else None))
G=pd.DataFrame(gap)
if len(G): G.to_csv(OUT+'GAP_AUDIT.csv',index=False)
print(f'\n[3] internal-gap audit -> GAP_AUDIT.csv ({len(G)} gaps)')
if len(G):
    print(G.groupby('role').agg(n_gaps=('pdb','size'), n_structures=('pdb','nunique'),
                                total_missing=('n_missing','sum')).to_string())
    crit=G[(G.role=='peptide') | (G.intersects_loops!='')]
    print(f'\n    ⚠ CRITICAL gaps (peptide chain or intersecting a modelled loop): {len(crit)}')
    if len(crit): print(crit.to_string(index=False))
    near=G[(G.min_dist_flank_to_pMHC<8) & (G.role.isin(['TCRa','TCRb']))]
    print(f'    TCR gaps with flanks within 8 A of pMHC: {len(near)}')
    if len(near): print(near.to_string(index=False))

# ---------- 4. chain map ----------
coh[['pdb','peptide','antigen','antigen_class','allele','resolution',
     'chain_mhc','chain_pep','chain_tra','chain_trb']].assign(
     group1='D/E->'+coh.chain_tra+coh.chain_trb,
     group2='A/B/C->'+coh.chain_mhc+coh.chain_pep).to_csv(OUT+'CHAIN_MAP.csv',index=False)
print(f'\n[4] chain map -> CHAIN_MAP.csv')
print(coh[['pdb','chain_mhc','chain_pep','chain_tra','chain_trb']].head(21).to_string(index=False))
