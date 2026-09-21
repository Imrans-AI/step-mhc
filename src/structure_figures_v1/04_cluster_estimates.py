"""Collapse exact-duplicate inputs, then cluster-bootstrap at the declared unit.
DECLARED BEFORE RESULTS: peptide-position CIs cluster on PEPTIDE (25);
loop CIs cluster on (peptide, paired TCR) (40). 2000 resamples, seed 0.
Writes perturb_peptide_clustered.csv, perturb_loops_clustered.csv, boot_*.csv"""
OUT='results/structure_figures_v1/'
import numpy as np, pandas as pd
NB, SEED = 2000, 0
ex=pd.read_csv(OUT+'exemplars_final.csv')
ex['unit']=ex.peptide+'|'+ex.A1+ex.A2+ex.A3+ex.B1+ex.B2+ex.B3
u=ex.drop_duplicates('unit')[['pdb','peptide','unit']]
print(f'{len(ex)} complexes -> {ex.unit.nunique()} distinct (peptide, paired TCR) units '
      f'-> {ex.peptide.nunique()} peptide clusters')

def prep(f, keys):
    d=pd.read_csv(OUT+f).merge(ex[['pdb','unit']],on='pdb')
    d=d[d.pdb.isin(u.pdb)]                       # keep ONE pdb per identical input
    d['pep']=d.unit.str.split('|').str[0]
    return d
P=prep('perturb_peptide.csv',None); Lo=prep('perturb_loops.csv',None)
P.to_csv(OUT+'perturb_peptide_clustered.csv',index=False)
Lo.to_csv(OUT+'perturb_loops_clustered.csv',index=False)
print(f'after dedup: peptide rows {len(P)}, loop rows {len(Lo)}')

def cboot(d, group, cluster, val='dscore'):
    rng=np.random.default_rng(SEED); out=[]
    cl=d[cluster].unique()
    for k,g in d.dropna(subset=[val]).groupby(group):
        med=g[val].median(); bs=[]
        for _ in range(NB):
            pick=rng.choice(cl,len(cl),replace=True)
            s=pd.concat([g[g[cluster]==c] for c in pick if (g[cluster]==c).any()])
            if len(s): bs.append(s[val].median())
        out.append(dict(key=k, median=med, lo=np.percentile(bs,2.5), hi=np.percentile(bs,97.5),
                        n_sub=len(g), n_clusters=g[cluster].nunique()))
    return pd.DataFrame(out)

pb=cboot(P,'pos','pep');  pb.to_csv(OUT+'boot_peptide_pos.csv',index=False)
lb=cboot(Lo,'loop','unit');lb.to_csv(OUT+'boot_loops.csv',index=False)
print('\npeptide position (cluster = peptide):'); print(pb.round(4).to_string(index=False))
print('\nloops (cluster = peptide|TCR):');        print(lb.round(4).to_string(index=False))
r=pb['median'].abs().median()/lb['median'].abs().median()
print(f'\npeptide/loop median |dscore| ratio = {r:.1f}x  (defined statistic, not an estimate)')
