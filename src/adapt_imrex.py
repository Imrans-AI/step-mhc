RESULTS = 'results/bench_common/'   # SET FIRST - HANDOFF_9 section 8
EXPECT  = '3c2e6dfdbdc92746c8978109b9156089f6c7b1a7bc3a63551267a6b10699b7f9'
import pandas as pd, hashlib, os
F = RESULTS + 'common_testset_v5.csv'
assert hashlib.sha256(open(F,'rb').read()).hexdigest() == EXPECT, 'HASH MISMATCH'
d = pd.read_csv(F); d['row_id'] = d.index
os.makedirs(RESULTS+'imrex', exist_ok=True)
o = pd.DataFrame({'cdr3': d.B3.astype(str), 'antigen.epitope': d.peptide.astype(str)})
o.to_csv(RESULTS+'imrex/input.csv', sep=';', index=False)
d[['row_id','label','B3','peptide']].to_csv(RESULTS+'imrex/rowmap.csv', index=False)
L = d.B3.str.len(); P = d.peptide.str.len()
print('rows', len(o))
print('  cdr3 outside 10-20 :', ((L<10)|(L>20)).sum())
print('  epitope outside 8-11:', ((P<8)|(P>11)).sum())
