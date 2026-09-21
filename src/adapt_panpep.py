RESULTS = 'results/bench_common/'   # SET FIRST - HANDOFF_9 section 8
EXPECT  = '3c2e6dfdbdc92746c8978109b9156089f6c7b1a7bc3a63551267a6b10699b7f9'
import pandas as pd, hashlib, os
F = RESULTS + 'common_testset_v5.csv'
assert hashlib.sha256(open(F,'rb').read()).hexdigest() == EXPECT, 'HASH MISMATCH'
d = pd.read_csv(F); d['row_id'] = d.index
s = d.sort_values(['peptide','B3'], kind='mergesort')          # README: sort by peptide
os.makedirs(RESULTS+'panpep', exist_ok=True)
pd.DataFrame({'Peptide': s.peptide.astype(str), 'CDR3': s.B3.astype(str)}) \
  .to_csv(RESULTS+'panpep/input.csv', index=False)
s[['row_id','label','B3','peptide']].to_csv(RESULTS+'panpep/rowmap.csv', index=False)
print('wrote', len(s), 'rows | peptides', s.peptide.nunique(),
      '| cdr3 len', s.B3.str.len().min(), '-', s.B3.str.len().max())
