"""Clean-39 prior, SEQUENCE-INDEXED to match StepMHCDataset's ljust packing.
⚠ Supersedes data/priors/structure_priors_clean39.npz, which was IMGT-indexed and therefore
misaligned with the model input for 47% of residues. ANARCI is still used to DEFINE the loop
boundaries; rows are then assigned by position within the observed loop sequence.
Writes data/priors/structure_priors_clean39_seqidx.npz (NEW FILE)."""
OUT='data/priors/structure_priors_clean39_seqidx.npz'
PROV='data/priors/provenance_clean39_seqidx.json'
CHAIN='results/struct_val/_chain_assignment_clean39.json'
import json, sys, numpy as np
sys.path.insert(0,'.')
from anarci import run_anarci
from src.extract_priors import atoms_by_chain, contact_matrix, AA3
from src.dataset_stepmhc import CAPS, LOOPS, L_PEP
IMGT={'1':(27,38),'2':(56,65),'3':(105,117)}
OFF,o={},0
for k in LOOPS: OFF[k]=o; o+=CAPS[k]

def loop_rows(res, tag):
    seq=''.join(AA3[r[1][0]] for r in res)
    out=run_anarci([('x',seq)],scheme='imgt')[1][0]
    if out is None: return None
    numb=out[0][0]; si=0; per={}
    for (pos,ins),aa in numb:                      # collect residues per CDR, IN SEQUENCE ORDER
        if aa=='-': continue
        while si<len(res) and AA3[res[si][1][0]]!=aa: si+=1
        if si>=len(res): break
        for cdr,(lo,hi) in IMGT.items():
            if lo<=pos<=hi: per.setdefault(tag+cdr,[]).append(res[si][1][1])
        si+=1
    rows={}
    for key,coords in per.items():                 # SEQUENCE index, not IMGT offset
        for i,cc in enumerate(coords):
            if i<CAPS[key]: rows[OFF[key]+i]=cc
    return rows

acc=np.zeros((73,L_PEP)); n_ok=0; prov=[]
for c in json.load(open(CHAIN)):
    ch=atoms_by_chain(open(f"data/structures/{c['pdb'].lower()}.pdb").read())
    pep=[v for k,v in ch[c['pep']].items()]
    if not (1<=len(pep)<=L_PEP): prov.append(dict(pdb=c['pdb'],ok=False,why='pep len')); continue
    rows={}
    for tag,cid in (('A',c['tra']),('B',c['trb'])):
        r=loop_rows(list(ch[cid].items()),tag)
        if r is None: rows=None; break
        rows.update(r)
    if rows is None: prov.append(dict(pdb=c['pdb'],ok=False,why='anarci')); continue
    idx=sorted(rows)
    M=contact_matrix([rows[i] for i in idx],[p[1] for p in pep])
    for a,i in enumerate(idx): acc[i,:len(pep)]+=M[a]
    n_ok+=1; prov.append(dict(pdb=c['pdb'],ok=True,n_contacts=int(M.sum())))
cw=acc/max(n_ok,1)
np.savez(OUT, cw_te=cw.astype(np.float32), n_structures=n_ok)
json.dump(prov, open(PROV,'w'), indent=1)
print(f'{n_ok}/39 structures | nonzero {(cw>0).sum()}/{cw.size} | std {cw.std():.4f}')
o=0
for k in LOOPS:
    s=cw[o:o+CAPS[k]]
    print(f'  {k}: occupied rows {[r for r in range(CAPS[k]) if (s[r]>0).any()]}')
    o+=CAPS[k]
print('\n-> rows must now be CONTIGUOUS from 0 in each loop (matching mhc/triplet)')
