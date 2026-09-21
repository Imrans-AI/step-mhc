RESULTS = 'results/bench_common/'   # SET FIRST - HANDOFF_9 section 8
EXPECT  = '3c2e6dfdbdc92746c8978109b9156089f6c7b1a7bc3a63551267a6b10699b7f9'
import pandas as pd, hashlib, os
F = RESULTS + 'common_testset_v5.csv'
assert hashlib.sha256(open(F,'rb').read()).hexdigest() == EXPECT, 'HASH MISMATCH'
d = pd.read_csv(F); d['row_id'] = d.index
o = pd.DataFrame({'cdr3': d.B3.astype(str), 'epitope': d.peptide.astype(str)})  # UNTRIMMED
os.makedirs(RESULTS+'teim', exist_ok=True)
o.to_csv(RESULTS+'teim/input.csv', index=False)
d[['row_id','label','B3','peptide']].to_csv(RESULTS+'teim/rowmap.csv', index=False)
print('wrote', len(o), 'rows | uniq pairs', o.drop_duplicates().shape[0],
      '| cdr3 len', o.cdr3.str.len().min(), '-', o.cdr3.str.len().max(),
      '| epi len', o.epitope.str.len().min(), '-', o.epitope.str.len().max())
print(o.head(2).to_string())
