RESULTS = 'results/bench_common/'   # SET FIRST - HANDOFF_9 section 8
import pandas as pd, numpy as np, glob, os
from sklearn.metrics import roc_auc_score, average_precision_score
rng = np.random.default_rng(42)
base = pd.read_csv(RESULTS+'common_testset_v5.csv'); base['row_id']=base.index

rows=[]
for f in sorted(glob.glob(RESULTS+'scores_*.csv')):
    m = os.path.basename(f)[7:-4]
    d = pd.read_csv(f).merge(base[['row_id','allele','patient','peptide']], on='row_id')
    roc = roc_auc_score(d.label, d.score); pr = average_precision_score(d.label, d.score)
    # permutation null: shuffle labels within the file
    null=[roc_auc_score(rng.permutation(d.label.values), d.score) for _ in range(1000)]
    p = (np.sum(np.array(null) >= roc)+1)/(len(null)+1)
    rows.append(dict(model=m, n=len(d), pos=int(d.label.sum()),
                     roc=roc, pr=pr, null_med=np.median(null), p=p,
                     cov=100*len(d)/len(base)))
r = pd.DataFrame(rows).sort_values('roc', ascending=False)
r.to_csv(RESULTS+'metrics_main.csv', index=False)
print(r.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
