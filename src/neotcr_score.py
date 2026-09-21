"""NeoTCR external neoantigen validation. Protocol identical to src/caushi_score_v2.py:
cognate TCR vs 600 non-reactive TIL clonotypes, same peptide, same HLA, TIL repertoire origin.
Prereg: results/neotcr/PREREG_neotcr.md (written before any score)."""
import os
import pandas as pd, numpy as np, torch, sys, re
from sklearn.metrics import roc_auc_score
sys.path.insert(0,'.')
from anarci import germlines as G
from src.model_stepmhc_small import STEPMHCModelSmall
from src.dataset_stepmhc import _imap, _globals, CAPS, LOOPS, L_TCR, L_PEP, L_HLA

N_NEG, SEED = 600, 42
Aseq=G.all_germlines['V']['A']['human']; Akeys=sorted(Aseq)
Bseq=G.all_germlines['V']['B']['human']; Bkeys=sorted(Bseq)
def stem_of(k): return k.split("*")[0].split("/")[0]
def norm(g,keys):
    if pd.isna(g): return None
    g=str(g).strip().upper().replace("\xa0","").replace(" ","").replace(":","*")
    g=re.sub(r"-(\d{2})$",r"*\1",g); g=re.sub(r"(TRA[VBDJ])0(\d)",r"\1\2",g)
    if g in keys: return g
    stem=g.split("*")[0]
    c=[k for k in keys if stem_of(k)==stem]
    if c: return sorted(c)[0]
    c=[k for k in keys if stem_of(k).startswith(stem+"-")]
    if c: return sorted(c,key=lambda k:(len(stem_of(k)),k))[0]
    return None
def loops(key,seqd):
    s=seqd[key]; return s[26:38].replace("-",""), s[55:65].replace("-","")

H=pd.read_csv(os.environ["STEPMHC_HLA_CSV"])
HM=dict(zip(H.HLA_type,H.HLA_sequence))
def pseudo_for(al):
    k='HLA-'+str(al).replace('HLA-','').replace('*','')
    if k in HM: return HM[k],True
    b=k[:len(k)-3]
    c=[v for kk,v in HM.items() if kk.startswith(b)]
    return (c[0],False) if c else (None,False)

n=pd.read_csv('data/neotcr_external_beta.csv')
n=n[n.Source.isin(['Patient','TIL'])].copy()
rows=[]; drop=0
for r in n.itertuples():
    ak,bk=norm(r.TRAV,Akeys),norm(r.TRBV,Bkeys)
    if ak is None or bk is None: drop+=1; continue
    a1,a2=loops(ak,Aseq); b1,b2=loops(bk,Bseq)
    if not(a1 and a2 and b1 and b2): drop+=1; continue
    rows.append(dict(A1=a1,A2=a2,A3=str(r.TRA_CDR3),B1=b1,B2=b2,B3=str(r.b3),
                     peptide=str(r.ep),allele=str(r.hla),tumor=str(r.Tumor)))
D=pd.DataFrame(rows)
print(f'NeoTCR usable: {len(D)}/{len(n)} (dropped {drop} on V-gene) | epitopes {D.peptide.nunique()}',flush=True)

til=pd.read_csv('data/nsclc_til_sc/paired_6cdr.csv')
reactive=set(pd.read_csv('data/nsclc_til_sc/fest_clonotype_antigen.csv').cdr3b)
til=til[~til.cdr3_TRB.isin(reactive)].dropna(subset=['cdr1_TRA','cdr2_TRA','cdr3_TRA','cdr1_TRB','cdr2_TRB','cdr3_TRB'])
til=til.drop_duplicates(['patient','cdr3_TRA','cdr3_TRB'])
neg=til.sample(min(N_NEG,len(til)),random_state=SEED)
NEGD=[dict(A1=r.cdr1_TRA,A2=r.cdr2_TRA,A3=r.cdr3_TRA,B1=r.cdr1_TRB,B2=r.cdr2_TRB,B3=r.cdr3_TRB) for r in neg.itertuples()]
print(f'negatives: {len(NEGD)} non-reactive TIL clonotypes',flush=True)

ck=torch.load('models/stepmhc_small_fullpool_seed42.pt')
m=STEPMHCModelSmall(use_structure_priors=False); m.load_state_dict(ck['state_dict']); m.cuda().eval()
tstr=lambda d:"".join(str(d[k])[:CAPS[k]].ljust(CAPS[k],"-") for k in LOOPS)
def score(ds,pep,hp):
    out,buf=[],[]
    def flush():
        nonlocal buf
        if not buf: return
        te=torch.from_numpy(np.stack([b[0] for b in buf])).cuda()
        ph=torch.from_numpy(np.stack([b[1] for b in buf])).cuda()
        gg=torch.from_numpy(np.stack([b[2] for b in buf])).float().cuda()
        with torch.no_grad(): out.extend(torch.sigmoid(m(te,ph,gg)).cpu().numpy().tolist())
        buf=[]
    for d in ds:
        ts=tstr(d)
        buf.append((_imap(ts,pep,L_TCR,L_PEP),_imap(pep,hp,L_PEP,L_HLA),_globals(ts,pep)))
        if len(buf)>=512: flush()
    flush(); return np.array(out)

recs=[]
for (pep,al),g in D.groupby(['peptide','allele']):
    hp,exact=pseudo_for(al)
    if hp is None: print(f'  {pep}: no pseudo for {al} — SKIPPED',flush=True); continue
    s_neg=score(NEGD,pep,hp); s_pos=score(g.to_dict('records'),pep,hp)
    pct=[(s_neg<v).mean() for v in s_pos]
    # non-cognate control: TCRs from OTHER epitopes scored on this peptide
    oth=D[D.peptide!=pep].sample(min(50,len(D[D.peptide!=pep])),random_state=SEED)
    s_oth=score(oth.to_dict('records'),pep,hp)
    pct_oth=np.median([(s_neg<v).mean() for v in s_oth])
    for v,p,t in zip(s_pos,pct,g.tumor):
        recs.append(dict(peptide=pep,allele=al,score=float(v),pct=float(p),n_pos=len(g),
                         tumor=t,pct_noncognate=float(pct_oth),hla_exact=exact))
R=pd.DataFrame(recs); R.to_csv('results/neotcr/neotcr_percentiles.csv',index=False)

def rep(x,lab):
    if not len(x): print(f'{lab}: empty'); return
    print(f'{lab}: n={len(x)} median pct {x.pct.median():.3f}')
print(f'\n{"="*66}\nRESULT\n{"="*66}',flush=True)
rep(R,'POOLED')
rep(R[R.peptide!='AMFWSVPTV'],'EXCL AMFWSVPTV')
rep(R[R.tumor.str.contains("lung|Lung",na=False)],'NSCLC/lung (tumour-matched negatives)')
rep(R[~R.tumor.str.contains("lung|Lung",na=False)],'non-lung')
print(f'\nNON-COGNATE CONTROL median: {R.pct_noncognate.median():.3f}  (cognate {R.pct.median():.3f})')
# PROPER NULL (src/pooled_permutation.py protocol): full clonotype x antigen percentile
# matrix; observed reads the diagonal, null permutes the cognate assignment and re-reads.
print('\nbuilding permutation null (full matrix)...', flush=True)
peps = sorted(D.peptide.unique())
NEGC = {}
for pep in peps:
    hp,_ = pseudo_for(D[D.peptide==pep].allele.iloc[0])
    if hp is not None: NEGC[pep] = (score(NEGD, pep, hp), hp)
M = np.full((len(D), len(peps)), np.nan)
recsD = D.to_dict('records')
for j,pep in enumerate(peps):
    if pep not in NEGC: continue
    sn,hp = NEGC[pep]
    sp = score(recsD, pep, hp)
    M[:,j] = [(sn < v).mean() for v in sp]
    print(f'  matrix col {j+1}/{len(peps)}', flush=True) if (j+1)%10==0 else None
pi = {p:j for j,p in enumerate(peps)}
truth = np.array([pi[r['peptide']] for r in recsD])
ok = ~np.isnan(M[np.arange(len(D)), truth])
obs = np.nanmedian(M[np.arange(len(D)), truth][ok])
rng = np.random.default_rng(SEED); null = []
for _ in range(500):
    perm = rng.permutation(truth)
    v = M[np.arange(len(D)), perm]
    null.append(np.nanmedian(v[~np.isnan(v)]))
null = np.array(null)
pperm = (np.sum(null >= obs) + 1) / (len(null) + 1)
print(f'\nOBSERVED median {obs:.3f} | PERMUTATION NULL {np.median(null):.3f} '
      f'[{np.percentile(null,2.5):.3f},{np.percentile(null,97.5):.3f}] | p_perm {pperm:.4f}')
np.save('results/neotcr/null_dist.npy', null)
print('\nper-epitope (n>=5):')
print(R[R.n_pos>=5].groupby('peptide').agg(n=('pct','size'),median_pct=('pct','median')).to_string())
