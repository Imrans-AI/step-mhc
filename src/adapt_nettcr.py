RESULTS = 'results/bench_common/'   # SET FIRST - HANDOFF_9 section 8
EXPECT  = '3c2e6dfdbdc92746c8978109b9156089f6c7b1a7bc3a63551267a6b10699b7f9'
import pandas as pd, hashlib, os
F = RESULTS + 'common_testset_v5.csv'
assert hashlib.sha256(open(F,'rb').read()).hexdigest() == EXPECT, 'HASH MISMATCH'
d = pd.read_csv(F); d['row_id'] = d.index

def trim(s):
    s = str(s)
    if s.startswith('C'): s = s[1:]
    if s and s[-1] in 'FW':  s = s[:-1]
    return s

o = d[['A1','A2','B1','B2','peptide']].copy()
o['A3'] = d.A3.map(trim)          # trimmed - NetTCR convention
o['B3'] = d.B3.map(trim)
o['binder'] = d.label
o = o[['A1','A2','A3','B1','B2','B3','peptide','binder']]

MAX = dict(A1=7, A2=8, A3=22, B1=6, B2=7, B3=23, peptide=12)
for c, m in MAX.items():
    mx = o[c].astype(str).str.len().max()
    assert mx <= m, f'{c} max len {mx} > NetTCR pad {m} - would truncate'
    print(f'  {c:8s} max {mx:2d} / pad {m}')
os.makedirs(RESULTS+'nettcr', exist_ok=True)
o.to_csv(RESULTS+'nettcr/input.csv', index=False)
d[['row_id','label']].assign(A3=o.A3, B3=o.B3, peptide=o.peptide) \
 .to_csv(RESULTS+'nettcr/rowmap.csv', index=False)
print('\nwrote', len(o), 'rows |', d.A3.iloc[0], '->', o.A3.iloc[0], '|', d.B3.iloc[0], '->', o.B3.iloc[0])
