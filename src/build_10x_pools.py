#!/usr/bin/env python3
"""
Build three experimental training pools from broadened + 10x:

  A: broadened + 10x new-peptides only        (12,545 pos, 119 usable)
  B: broadened + 10x capped@100               (11,802 pos, 119 usable)
  C: broadened + 10x uncapped                 (27,668 pos, 119 usable)

All pools:
  - inherit hla_pseudo from the existing pool by allele
  - regenerate 1:5 reference-TCR negatives for the 10x rows
  - preserve the broadened pool's own negatives untouched
  - write to data/pools/{A,B,C}_train_ready.csv
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import random
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
log = logging.getLogger(__name__)

ROOT = Path(os.environ.get('STEPMHC_ROOT', '.'))
DATA = ROOT / 'data'
POOLS = DATA / 'pools'
POOLS.mkdir(exist_ok=True)

BROADENED = DATA / 'broadened' / 'broadened_train_ready.csv'
TENX = DATA / '10x_unique_positives.csv'

SCHEMA = ['A1','A2','A3','B1','B2','B3','peptide','allele',
          'binder','origin','hla_key','hla_pseudo','neg_per_pos']
CDRS = ['A1','A2','A3','B1','B2','B3']

CAP_B = 100
SEED = 42


def load_broadened():
    df = pd.read_csv(BROADENED)
    pos = df[df.binder == 1].copy()
    neg = df[df.binder == 0].copy()
    log.info(f"broadened: {len(pos)} pos / {len(neg)} neg, "
             f"{pos.peptide.nunique()} peps, {(pos.peptide.value_counts()>=10).sum()} usable")
    return df, pos, neg


def hla_pseudo_lookup(broadened_df):
    lut = (broadened_df[['allele','hla_key','hla_pseudo']]
           .drop_duplicates(subset=['allele'])
           .set_index('allele'))
    log.info(f"hla_pseudo lookup: {len(lut)} alleles")
    return lut


def attach_hla_pseudo(df, lut):
    df = df.copy()
    df['hla_pseudo'] = df['allele'].map(lut['hla_pseudo'])
    df['hla_key']    = df['allele'].map(lut['hla_key'])
    missing = df['hla_pseudo'].isna().sum()
    if missing:
        log.warning(f"  {missing} rows have no hla_pseudo — dropping")
        df = df[df['hla_pseudo'].notna()]
    return df


def make_negatives(pos_df, ref_tcrs, all_peptides_by_allele, seed=SEED):
    """
    Reference-TCR negative protocol: for each positive, sample 5 peptides
    presented by the SAME allele elsewhere in the pool, excluding the true one.
    Falls back to global peptide pool if the allele has <5 alternatives.
    """
    rng = random.Random(seed)
    global_peps = sorted({p for ps in all_peptides_by_allele.values() for p in ps})
    negs = []
    for r in pos_df.itertuples(index=False):
        cands = [p for p in all_peptides_by_allele.get(r.allele, []) if p != r.peptide]
        if len(cands) < 5:
            cands = [p for p in global_peps if p != r.peptide]
        for np_ in rng.sample(cands, min(5, len(cands))):
            negs.append({
                'A1': r.A1, 'A2': r.A2, 'A3': r.A3,
                'B1': r.B1, 'B2': r.B2, 'B3': r.B3,
                'peptide': np_, 'allele': r.allele, 'binder': 0,
                'origin': '10x_ref_neg', 'hla_key': r.hla_key,
                'hla_pseudo': r.hla_pseudo, 'neg_per_pos': 5,
            })
    return pd.DataFrame(negs)


def summarize(name, pool):
    pos = pool[pool.binder == 1]
    cnt = pos.peptide.value_counts()
    log.info(f"  [{name}] {len(pool)} rows | {len(pos)} pos / {(pool.binder==0).sum()} neg")
    log.info(f"        peptides {pos.peptide.nunique()} | usable(>=10) {(cnt>=10).sum()} "
             f"| alleles {pos.allele.nunique()}")
    log.info(f"        max peptide count: {cnt.max()} ({cnt.idxmax()})")
    log.info(f"        origins: {pos.origin.value_counts().to_dict()}")


def main():
    log.info("="*72)
    log.info("BUILD 10x EXPERIMENT POOLS  A / B / C")
    log.info("="*72)

    broadened_all, b_pos, b_neg = load_broadened()
    lut = hla_pseudo_lookup(broadened_all)

    tenx = pd.read_csv(TENX)
    tenx = attach_hla_pseudo(tenx, lut)
    tenx['origin'] = '10x_dextramer'
    tenx['binder'] = 1
    tenx['neg_per_pos'] = 5
    log.info(f"10x unique: {len(tenx)} pos, {tenx.peptide.nunique()} peps")

    # --- SAFETY: 10x rows must not duplicate any broadened positive ---
    b_keys = set(map(tuple, b_pos[CDRS + ['peptide']].values))
    before = len(tenx)
    tenx = tenx[~tenx[CDRS + ['peptide']].apply(tuple, axis=1).isin(b_keys)]
    log.info(f"  removed {before - len(tenx)} 10x rows already present in broadened")

    existing_peps = set(b_pos.peptide.unique())
    tenx_new  = tenx[~tenx.peptide.isin(existing_peps)].copy()
    tenx_seen = tenx[ tenx.peptide.isin(existing_peps)].copy()
    log.info(f"  10x split: {len(tenx_new)} pairs on {tenx_new.peptide.nunique()} NEW peps | "
             f"{len(tenx_seen)} pairs on {tenx_seen.peptide.nunique()} seen peps")

    # ---- three 10x positive subsets ----
    subsets = {
        'A': tenx_new,
        'B': (tenx.groupby('peptide', group_keys=False)
                  .apply(lambda g: g.head(CAP_B), include_groups=False)
                  .reset_index(drop=True)
              if False else
              tenx.groupby('peptide', group_keys=False).head(CAP_B)),
        'C': tenx,
    }

    for name, tenx_pos in subsets.items():
        log.info("\n" + "-"*72)
        log.info(f"POOL {name}")
        log.info("-"*72)
        tenx_pos = tenx_pos.copy()
        log.info(f"  10x contribution: {len(tenx_pos)} pos, {tenx_pos.peptide.nunique()} peps")

        merged_pos = pd.concat([b_pos[SCHEMA], tenx_pos[SCHEMA]], ignore_index=True)

        # peptides available per allele, from the MERGED positives (for negative sampling)
        peps_by_allele = merged_pos.groupby('allele')['peptide'].unique().apply(list).to_dict()

        tenx_neg = make_negatives(tenx_pos, None, peps_by_allele, seed=SEED)
        log.info(f"  generated {len(tenx_neg)} negatives for the 10x rows")

        pool = pd.concat([broadened_all[SCHEMA], tenx_pos[SCHEMA], tenx_neg[SCHEMA]],
                         ignore_index=True)

        # final dedup: no exact duplicate (6CDR, peptide, binder) rows
        before = len(pool)
        pool = pool.drop_duplicates(subset=CDRS + ['peptide','binder']).reset_index(drop=True)
        log.info(f"  final dedup removed {before - len(pool)} rows")

        summarize(name, pool)

        out = POOLS / f'{name}_train_ready.csv'
        pool.to_csv(out, index=False)
        log.info(f"  -> wrote {out}")

    log.info("\n" + "="*72)
    log.info("DONE — three pools written to data/pools/")
    log.info("="*72)


if __name__ == '__main__':
    random.seed(SEED); np.random.seed(SEED)
    main()
