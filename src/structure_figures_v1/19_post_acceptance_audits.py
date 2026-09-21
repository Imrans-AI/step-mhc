"""Post-acceptance audits: 1OGA terminal residues, lost/gained contacts for the seven
repair-sensitive structures, and the repair-energy parser correction.
READ-ONLY on existing outputs. Creates new files only."""
OUT='results/structure_figures_v1/foldx_cohort_v1/'
import pandas as pd, numpy as np, sys, os, re, hashlib
sys.path.insert(0,'.')
from src.extract_priors import atoms_by_chain, AA3
from src.dataset_stepmhc import CAPS
from anarci import run_anarci
IMGT={'1':(27,38),'2':(56,65),'3':(105,117)}
T=pd.read_csv(OUT+'FOLDX_CHAIN_TABLE.csv')
PRIMARY=['2BNR','3QDJ','7DZM','7N2O','7N1F','7N2P','3O4L','3GSN','6MTM']
SENS   =['2BNQ','7NME','7N1E','7N2S','7NMG','5MEN','5HHO']

def ch_of(pid,which):
    d=OUT+f'prepared/{pid.upper()}/{pid.lower()}_prepared'
    return atoms_by_chain(open(d+('_Repair.pdb' if which=='r' else '.pdb')).read())
def contacts(ch,g1,g2,cut=5.0):
    A={(c,k):np.asarray(v[1]) for c in g1 if c in ch for k,v in ch[c].items()}
    B={(c,k):np.asarray(v[1]) for c in g2 if c in ch for k,v in ch[c].items()}
    return {(ka,kb) for ka,va in A.items() for kb,vb in B.items()
            if np.sqrt(((va[:,None,:]-vb[None,:,:])**2).sum(-1)).min()<=cut}

# ---- 1. 1OGA terminal residues ----
print('=== 1OGA terminal-truncation audit ===')
r=T[T.pdb.str.upper()=='1OGA'].iloc[0]
o,rr=ch_of('1oga','o'), ch_of('1oga','r')
pm=np.vstack([np.asarray(v[1]) for c in list(r.foldx_group2) if c in o for v in o[c].values()])
pep=np.vstack([np.asarray(v[1]) for v in o[r.chain_pep].values()])
rows1=[]
for c in list(r.foldx_group1):
    lost=[k for k in o[c] if k not in rr.get(c,{})]
    seq=''.join(AA3[v[0]] for v in o[c].values()); keys=list(o[c].keys())
    a=run_anarci([('x',seq)],scheme='imgt')[1][0]
    loopkeys=set()
    if a:
        si=0
        for (p,i),aa in a[0][0]:
            if aa=='-': continue
            while si<len(keys) and AA3[o[c][keys[si]][0]]!=aa: si+=1
            if si>=len(keys): break
            if any(lo<=p<=hi for lo,hi in IMGT.values()): loopkeys.add(keys[si])
            si+=1
    for k in lost:
        xyz=np.asarray(o[c][k][1])
        rows1.append(dict(chain=c, resnum=k.strip(), resname=o[c][k][0],
            in_loop=k in loopkeys,
            d_to_peptide=round(float(np.sqrt(((xyz[:,None,:]-pep[None,:,:])**2).sum(-1)).min()),2),
            d_to_pMHC=round(float(np.sqrt(((xyz[:,None,:]-pm[None,:,:])**2).sum(-1)).min()),2),
            is_terminal=(k==keys[0] or k==keys[-1])))
L=pd.DataFrame(rows1); L.to_csv(OUT+'1OGA_TERMINAL_TRUNCATION_AUDIT.csv',index=False)
print(L.to_string(index=False))
ok=(not L.in_loop.any()) and (L.d_to_pMHC.min()>8) and L.is_terminal.all()
print(f'\nqualifies for TERMINAL_TRUNCATION_SENSITIVITY: {ok}')

# ---- 2. lost/gained contacts for the seven ----
print('\n=== repair-sensitive contact detail ===')
rows2=[]
for pid in SENS:
    r=T[T.pdb.str.upper()==pid].iloc[0]
    o,rr=ch_of(pid,'o'), ch_of(pid,'r')
    g1,g2=list(r.foldx_group1),list(r.foldx_group2)
    co,cr=contacts(o,g1,g2),contacts(rr,g1,g2)
    tpo={x for x in co if x[1][0]==r.chain_pep}; tpr={x for x in cr if x[1][0]==r.chain_pep}
    for x in sorted(tpo-tpr):
        rows2.append(dict(pdb=pid, event='lost', tcr_chain=x[0][0], tcr_res=x[0][1].strip(),
                          pep_res=x[1][1].strip()))
    for x in sorted(tpr-tpo):
        rows2.append(dict(pdb=pid, event='gained', tcr_chain=x[0][0], tcr_res=x[0][1].strip(),
                          pep_res=x[1][1].strip()))
    print(f'  {pid}: TCR-pep orig {len(tpo)} retained {len(tpo&tpr)} '
          f'lost {len(tpo-tpr)} gained {len(tpr-tpo)} -> {len(tpo&tpr)/len(tpo)*100:.1f}%')
pd.DataFrame(rows2).to_csv(OUT+'REPAIR_SENSITIVE_CONTACT_DETAIL.csv',index=False)

# ---- 3. energy parser correction ----
print('\n=== repair-energy parser correction ===')
log=open(OUT+'prepare_repair.log').read().split('\n')
old=pd.read_csv(OUT+'PREPARED_CHECKSUMS.csv')
en={}
cur=None
for l in log:
    m=re.match(r'^(\w{4})\s+atoms=', l)
    if m: cur=m.group(1)
lg=[l for l in log if 'Total' in l]
print(f'  lines containing "Total" in prepare_repair.log: {len(lg)}')
for l in lg[:3]: print('   ', l.strip()[:80])
