"""Cluster bootstrap on the PRIMARY (logit) scale. Reads the *_logit.csv files produced
earlier; does NOT regenerate them. Declared units: peptide-position CIs cluster on PEPTIDE,
loop CIs cluster on (peptide, paired TCR). 2000 resamples, seed 0.
See results/structure_figures_v1/SCALE_DECISION.md for why logit is primary."""
OUT='results/structure_figures_v1/'
import numpy as np, pandas as pd
NB, SEED = 2000, 0
P =pd.read_csv(OUT+'perturb_peptide_clustered_logit.csv')
Lo=pd.read_csv(OUT+'perturb_loops_clustered_logit.csv')
assert 'dlogit' in P.columns and 'dlogit' in Lo.columns, 'dlogit missing - regenerate first'
print(f'peptide rows {len(P)} ({P.dlogit.notna().sum()} scored) | '
      f'loop rows {len(Lo)} ({Lo.dlogit.notna().sum()} scored)')

def cboot(d, group, cluster, val='dlogit'):
    rng=np.random.default_rng(SEED); out=[]; cl=d[cluster].unique()
    for k,g in d.dropna(subset=[val]).groupby(group):
        idx={c:g[g[cluster]==c] for c in cl}
        bs=[]
        for _ in range(NB):
            s=pd.concat([idx[c] for c in rng.choice(cl,len(cl),replace=True) if len(idx[c])])
            if len(s): bs.append(s[val].median())
        out.append(dict(key=k, median=g[val].median(),
                        lo=np.percentile(bs,2.5), hi=np.percentile(bs,97.5),
                        n_sub=len(g), n_clusters=g[cluster].nunique()))
    return pd.DataFrame(out)

pb=cboot(P,'pos','pep');   pb.to_csv(OUT+'boot_peptide_pos_logit.csv',index=False)
lb=cboot(Lo,'loop','unit');lb.to_csv(OUT+'boot_loops_logit.csv',index=False)
print('\npeptide position — LOGIT (cluster = peptide):'); print(pb.round(4).to_string(index=False))
print('\nloops — LOGIT (cluster = peptide|TCR):');         print(lb.round(4).to_string(index=False))
print(f"\npeptide/loop median |dlogit| ratio = "
      f"{pb['median'].abs().median()/lb['median'].abs().median():.1f}x")
