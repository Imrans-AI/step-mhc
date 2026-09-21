RESULTS = 'results/bench_common/'   # SET FIRST - HANDOFF_9 section 8
import pandas as pd, numpy as np, glob, os, hashlib
d = pd.read_csv(RESULTS+'common_testset_v2.csv')
P = d[d.label==1].reset_index(drop=True)
meta = P.groupby('peptide').agg(allele=('allele','first'), hla_pseudo=('hla_pseudo','first'),
                                neoantigen=('neoantigen','first'), patient=('patient','first'))
LOOPS=['A1','A2','A3','B1','B2','B3']
npos = P.peptide.value_counts().to_dict()
cap0 = {p: 5*k for p,k in npos.items()}          # negatives each peptide must receive
assert sum(cap0.values()) == 5*len(P)

def assign(seed):
    rng = np.random.default_rng(seed)
    cap = dict(cap0); out = {}
    order = rng.permutation(len(P))
    for i in order:
        t = P.iloc[i]; cog = t.peptide
        # prefer peptides with most remaining capacity -> avoids dead ends
        opts = [(c, rng.random(), p) for p,c in cap.items() if c>0 and p!=cog]
        if len(opts) < 5: return None
        opts.sort(key=lambda x: (-x[0], x[1]))
        pick = [p for _,_,p in opts[:5]]
        for p in pick: cap[p] -= 1
        out[i] = pick
    return out if all(v==0 for v in cap.values()) else None

for seed in range(200):
    A = assign(seed)
    if A: print('assignment found, seed', seed); break
assert A, 'no feasible assignment in 200 tries'

rows=[]
for i,t in P.iterrows():
    rows.append({**{c:t[c] for c in LOOPS},'peptide':t.peptide,'allele':t.allele,
                 'hla_pseudo':t.hla_pseudo,'neoantigen':t.neoantigen,'patient':t.patient,
                 'label':1,'src':'cognate'})
    for q in A[i]:
        m = meta.loc[q]
        rows.append({**{c:t[c] for c in LOOPS},'peptide':q,'allele':m.allele,
                     'hla_pseudo':m.hla_pseudo,'neoantigen':m.neoantigen,'patient':m.patient,
                     'label':0,'src':'peptide_swapped'})
df = pd.DataFrame(rows)
print('built',len(df),'|',int(df.label.sum()),'pos |',int((df.label==0).sum()),'neg')
print('dup (B3,peptide):', df.duplicated(['B3','peptide']).sum())
tc = df.groupby('B3').label.agg(['sum','count'])
pc = df.groupby('peptide').label.agg(['sum','count'])
print('per-TCR     neg count: min %d max %d' % ((tc['count']-tc['sum']).min(),(tc['count']-tc['sum']).max()))
print('per-peptide neg:pos  : min %.2f max %.2f' % (((pc['count']-pc['sum'])/pc['sum']).min(),
                                                    ((pc['count']-pc['sum'])/pc['sum']).max()))
trains={os.path.basename(f)[:-4]:pd.read_csv(f) for f in glob.glob(RESULTS+'train_keys/*.csv')}
trim=lambda s:(str(s)[1:] if str(s).startswith('C') else str(s))[:-1] if (str(s)[1:] if str(s).startswith('C') else str(s))[-1:] in 'FW' else (str(s)[1:] if str(s).startswith('C') else str(s))
bad=pd.Series(False,index=df.index)
for m_,t_ in sorted(trains.items()):
    pair=set(t_.b3.astype(str)+'|'+t_.ep.astype(str))
    bad |= (df.B3.astype(str)+'|'+df.peptide.astype(str)).isin(pair)
    bad |= (df.B3.map(trim)+'|'+df.peptide.astype(str)).isin(pair)
print('contaminated dropped:', bad.sum())
df=df[~bad].reset_index(drop=True)
out=RESULTS+'common_testset_v5.csv'; df.to_csv(out,index=False)
h=hashlib.sha256(open(out,'rb').read()).hexdigest()
print(f'\nFROZEN {out}\n  rows {len(df)} | pos {int(df.label.sum())} | neg {int((df.label==0).sum())}')
print(f'  peptides {df.peptide.nunique()} | TCRs {df.B3.nunique()} | alleles {df.allele.nunique()}')
print(f'  sha256 {h}')
open(RESULTS+'common_testset_v5.sha256','w').write(h+'\n')
