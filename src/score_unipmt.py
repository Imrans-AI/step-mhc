RESULTS = 'results/bench_common/'   # SET FIRST - HANDOFF_9 section 8
EXPECT  = '3c2e6dfdbdc92746c8978109b9156089f6c7b1a7bc3a63551267a6b10699b7f9'
UP      = '<UNIPMT_DATA>/'
CODE    = 'baselines/UniPMT/code'
import sys, os, hashlib, pickle, re
import numpy as np, pandas as pd, torch
# PyG 2.6.1 tracer expects typing re-exported from torch.fx._symbolic_trace (removed in torch 2.10)
import typing as _t, torch.fx._symbolic_trace as _st
for _n in ('List','Dict','Tuple','Optional','Union','Any','Callable','Set'):
    if not hasattr(_st, _n): setattr(_st, _n, getattr(_t, _n))
ROOT=os.getcwd()

F = RESULTS + 'common_testset_v5.csv'
assert hashlib.sha256(open(F,'rb').read()).hexdigest() == EXPECT, 'HASH MISMATCH'
d = pd.read_csv(F); d['row_id'] = d.index
peps = sorted(set(d.peptide)); tcrs = sorted(set(d.B3)); alls = sorted(set(d.allele))
pi = {s:i for i,s in enumerate(peps)}; ti = {s:i for i,s in enumerate(tcrs)}; mi = {s:i for i,s in enumerate(alls)}
print(f'nodes: p={len(peps)} t={len(tcrs)} m={len(alls)}')

# ---- features (mean-pooled ESM, their convention; verified cos=1.000000) ----
cp = torch.load('data/esm_cache/esm_t33_650M_pep.pt', map_location='cpu', weights_only=False)
cb = torch.load('data/esm_cache/esm_t33_650M_tcr.pt', map_location='cpu', weights_only=False)['B3']
nw = torch.load(RESULTS+'unipmt_new_esm.pt', map_location='cpu', weights_only=False)
def pemb(s): return (cp[s] if s in cp else nw['pep'][s]).numpy().mean(0)
def temb(s): return (cb[s] if s in cb else nw['tcr'][s]).numpy().mean(0)
P = np.stack([pemb(s) for s in peps]).astype(np.float64)
T = np.stack([temb(s) for s in tcrs]).astype(np.float64)

ps = pd.read_csv(UP+'data/pmt_pmt/pseudoseqs.csv')
conv = lambda a: re.sub(r'^HLA-([ABC])(\d+:\d+)$', r'HLA-\1*\2', a)
ps = ps.set_index('mhc')
M = np.stack([ps.loc[conv(a)].values for a in alls]).astype(np.float64)
print(f'features: P{P.shape} T{T.shape} M{M.shape}')

meta_dir = RESULTS+'unipmt/meta'; os.makedirs(meta_dir, exist_ok=True)
np.save(meta_dir+'/p_features.npy', P); np.save(meta_dir+'/t_features.npy', T)
np.save(meta_dir+'/m_features.npy', M)

# ---- config + model ----
os.chdir(CODE)
sys.path.insert(0, os.getcwd())
import config.config as config
config.task = 'pmt'
from model import MolGNN
from torch_geometric.data import HeteroData
import torch_geometric.transforms as TG

def build(with_pt_edges):
    g = HeteroData()
    g['p'].x = torch.tensor(range(len(peps)))
    g['t'].x = torch.tensor(range(len(tcrs)))
    g['m'].x = torch.tensor(range(len(alls)))
    pm = [[],[]]; pt = [[],[]]; mt = [[],[]]
    seen_pm=set(); seen_mt=set()
    for _,r in d.iterrows():
        a,b = pi[r.peptide], mi[r.allele]
        if (a,b) not in seen_pm: pm[0].append(a); pm[1].append(b); seen_pm.add((a,b))
        c = ti[r.B3]
        if (b,c) not in seen_mt: mt[0].append(b); mt[1].append(c); seen_mt.add((b,c))
        if with_pt_edges and r.label==1: pt[0].append(a); pt[1].append(c)
    g['p','pm_link','m'].edge_index = torch.tensor(pm, dtype=torch.int64)
    g['p','pt_link','t'].edge_index = torch.tensor(pt, dtype=torch.int64) if pt[0] else torch.zeros((2,0),dtype=torch.int64)
    g['m','mt_link','t'].edge_index = torch.tensor(mt, dtype=torch.int64)
    return TG.ToUndirected()(g)

out = {}
for tag, flag in [('coldstart', False), ('with_pos_edges', True)]:
    g = build(flag)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = MolGNN(os.path.join(ROOT, meta_dir), len(peps), len(tcrs), len(alls),
               config.hidden_size, g.metadata(), dev)
    sd = torch.load(UP+'model/model_pmt_pmt.pt', map_location='cpu')
    missing, unexpected = m.load_state_dict(sd, strict=False)
    print(f'\n[{tag}] missing={len(missing)} unexpected={len(unexpected)}')
    if missing: print('  missing:', missing[:5])
    m.to(dev).eval()
    with torch.no_grad():
        emb = m.gnn_learn(g)
        pe = emb['p'][torch.tensor([pi[s] for s in d.peptide])]
        te = emb['t'][torch.tensor([ti[s] for s in d.B3])]
        me = emb['m'][torch.tensor([mi[s] for s in d.allele])]
        s = m.pt_pred(pe, te, me).squeeze().cpu().numpy()
    out[tag] = s
    print(f'  scores n={len(s)} range [{s.min():.4f}, {s.max():.4f}] mean {s.mean():.4f}')

os.chdir(ROOT)
for tag, s in out.items():
    p = f'{RESULTS}scores_unipmt_{tag}.csv'
    pd.DataFrame({'row_id':d.row_id,'label':d.label,'score':s}).to_csv(p, index=False)
    print('wrote', p)
print('\nNO METRICS COMPUTED - prereg S5.')
