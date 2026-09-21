RESULTS = 'results/bench_common/'   # SET FIRST - HANDOFF_9 section 8
EXPECT  = '3c2e6dfdbdc92746c8978109b9156089f6c7b1a7bc3a63551267a6b10699b7f9'
import pandas as pd, hashlib, re, os
F = RESULTS + 'common_testset_v5.csv'
assert hashlib.sha256(open(F,'rb').read()).hexdigest() == EXPECT, 'HASH MISMATCH'
d = pd.read_csv(F); d['row_id'] = d.index
conv = lambda a: re.sub(r'^HLA-([ABC])(\d+:\d+)$', r'\1*\2', str(a))
d['HLA'] = d.allele.map(conv)
assert d.HLA.str.match(r'^[ABC]\*\d+:\d+$').all(), 'allele conversion failed'
os.makedirs(RESULTS+'pmtnet', exist_ok=True)
d[['B3','peptide','HLA']].rename(columns={'B3':'CDR3','peptide':'Antigen'}) \
 .to_csv(RESULTS+'pmtnet/input.csv', index=False)
d[['row_id','label','B3','peptide','HLA']].to_csv(RESULTS+'pmtnet/rowmap.csv', index=False)
print('wrote', len(d), 'rows | alleles', d.HLA.nunique(),
      '| dup triplets', d.duplicated(['B3','peptide','HLA']).sum())
