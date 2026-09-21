"""STAGE 6 — deterministic preparation + RepairPDB for the 18 primary-cohort structures.
Policy is read from the LOCKED prereg: strip waters and non-water HETATM, resolve altLocs
(highest occupancy, ties -> A), retain beta-2-m, do not model missing terminal residues.
Runtime assertion: Group1/Group2 chains must reproduce the verified sequences BEFORE repair.
Writes ONLY into results/structure_figures_v1/foldx_cohort_v1/prepared/<PDB>/ and
PREPARED_CHECKSUMS.csv. Nothing existing is touched."""
OUT='results/structure_figures_v1/foldx_cohort_v1/'
import os, sys, json, hashlib, subprocess, time, collections
import pandas as pd
sys.path.insert(0,'.')
from src.extract_priors import atoms_by_chain, AA3
T=pd.read_csv(OUT+'FOLDX_CHAIN_TABLE.csv')
os.makedirs(OUT+'prepared',exist_ok=True)
sha=lambda p: hashlib.sha256(open(p,'rb').read()).hexdigest()

def prepare(pdb, keep):
    src=f'data/structures/{pdb.lower()}.pdb'
    best={}
    for l in open(src):
        if not l.startswith('ATOM'): continue
        if l[21] not in keep: continue
        k=(l[21], l[22:27], l[12:16])
        occ=float(l[54:60]) if l[54:60].strip() else 1.0
        alt=l[16]
        if k not in best: best[k]=(occ, alt, l)
        else:
            o0,a0,_=best[k]
            if occ>o0 or (occ==o0 and alt<a0): best[k]=(occ,alt,l)   # highest occ, tie -> A
    out=[]
    for k,(o,a,l) in best.items():
        out.append(l[:16]+' '+l[17:])           # blank the altLoc column
    return sorted(out, key=lambda x:(x[21], int(''.join(c for c in x[22:26] if c.isdigit()) or 0), x[12:16]))

rows=[]
for _,r in T.iterrows():
    keep=set(r.foldx_group1)|set(r.foldx_group2)
    lines=prepare(r.pdb, keep)
    d=OUT+f'prepared/{r.pdb.upper()}'; os.makedirs(d,exist_ok=True)
    p=f'{d}/{r.pdb.lower()}_prepared.pdb'
    open(p,'w').write(''.join(lines)+'END\n')
    # RUNTIME ASSERTION — sequences must survive preparation
    ch=atoms_by_chain(open(p).read())
    pep=''.join(AA3[v[0]] for v in ch[r.chain_pep].values())
    assert pep==r.peptide, f'{r.pdb}: peptide {pep} != {r.peptide}'
    for c in r.foldx_group1:
        assert c in ch and len(ch[c])>50, f'{r.pdb}: TCR chain {c} missing/short'
    for c in r.foldx_group2:
        assert c in ch, f'{r.pdb}: Group2 chain {c} missing'
    t0=time.time()
    res=subprocess.run(['foldx','--command=RepairPDB',f'--pdb={os.path.basename(p)}',
                        f'--pdb-dir={d}',f'--output-dir={d}'],capture_output=True,text=True)
    el=time.time()-t0
    rp=f'{d}/{r.pdb.lower()}_prepared_Repair.pdb'
    ok=os.path.exists(rp)
    tot=[l for l in res.stdout.split('\n') if l.strip().startswith('Total')]
    rows.append(dict(pdb=r.pdb, peptide=r.peptide, group1=r.foldx_group1, group2=r.foldx_group2,
                     n_atoms=len(lines), prepared_sha256=sha(p),
                     repaired_sha256=sha(rp) if ok else None,
                     repair_total_energy=tot[-1].split()[-1] if tot else None,
                     seconds=round(el,1), repair_ok=ok))
    print(f'{r.pdb}  atoms={len(lines):5d}  {"OK" if ok else "FAIL"}  {el:.0f}s  '
          f'E={rows[-1]["repair_total_energy"]}', flush=True)
pd.DataFrame(rows).to_csv(OUT+'PREPARED_CHECKSUMS.csv',index=False)
print(f'\n{sum(r["repair_ok"] for r in rows)}/{len(rows)} repaired -> PREPARED_CHECKSUMS.csv')
