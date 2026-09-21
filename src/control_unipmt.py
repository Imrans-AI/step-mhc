RESULTS = 'results/bench_common/controls/unipmt/'   # SET FIRST - HANDOFF_9 section 8
UP='<EXTERNAL_DATA>/unipmt_drive/'; CODE='baselines/UniPMT/code'
import sys, os, pickle
import numpy as np, pandas as pd, torch
import typing as _t, torch.fx._symbolic_trace as _st
for _n in ('List','Dict','Tuple','Optional','Union','Any','Callable','Set'):
    if not hasattr(_st,_n): setattr(_st,_n,getattr(_t,_n))
ROOT=os.getcwd(); os.makedirs(RESULTS, exist_ok=True)
rng=np.random.default_rng(42)
U=UP+'data/pmt_pmt/'

pos=pd.read_csv(U+'edges_pmt_test.csv')[['Peptide','MHC','TCR']]
pos=pos.drop_duplicates()
peps=sorted(set(pos.Peptide))
rows=[]
for _,r in pos.iterrows():
    rows.append(dict(p=r.Peptide,m=r.MHC,t=r.TCR,label=1))
    others=[q for q in peps if q!=r.Peptide]
    for q in rng.choice(others, size=min(5,len(others)), replace=False):
        rows.append(dict(p=q,m=r.MHC,t=r.TCR,label=0))   # swap peptide, keep MHC+TCR
ctl=pd.DataFrame(rows).drop_duplicates(['p','m','t'])
print(f'control: {len(ctl)} rows | {int(ctl.label.sum())} pos | {ctl.p.nunique()} peptides')
ctl.to_csv(RESULTS+'control_rows.csv', index=False)

pdict=pickle.load(open(U+'meta/p_dict.pkl','rb'))
tdict=pickle.load(open(U+'meta/t_dict.pkl','rb'))
mdict=pickle.load(open(U+'meta/m_dict.pkl','rb'))
st=pickle.load(open(U+'meta/statics.pkl','rb'))

os.chdir(CODE); sys.path.insert(0, os.getcwd())
import config.config as config
config.task='pmt'
from model import MolGNN
from torch_geometric.data import HeteroData
import torch_geometric.transforms as TG

g=HeteroData()
g['p'].x=torch.tensor(range(st['p_num'])); g['t'].x=torch.tensor(range(st['t_num'])); g['m'].x=torch.tensor(range(st['m_num']))
tr=pd.read_csv(U+'edges_pmt_train.csv')                      # TRAIN edges only - no test leakage
pm=list({(pdict[a],mdict[b]) for a,b in zip(tr.Peptide,tr.MHC)})
mt=list({(mdict[b],tdict[c]) for b,c in zip(tr.MHC,tr.TCR)})
pt=list({(pdict[a],tdict[c]) for a,c in zip(tr.Peptide,tr.TCR)})
g['p','pm_link','m'].edge_index=torch.tensor(list(zip(*pm)),dtype=torch.int64)
g['p','pt_link','t'].edge_index=torch.tensor(list(zip(*pt)),dtype=torch.int64)
g['m','mt_link','t'].edge_index=torch.tensor(list(zip(*mt)),dtype=torch.int64)
g=TG.ToUndirected()(g)
print('graph edges: pm',len(pm),'pt',len(pt),'mt',len(mt))

dev='cuda' if torch.cuda.is_available() else 'cpu'
m=MolGNN(os.path.join(ROOT,'..','unipmt_drive','data','pmt_pmt','meta') if False else UP+'data/pmt_pmt/meta',
         st['p_num'], st['t_num'], st['m_num'], config.hidden_size, g.metadata(), dev)
miss,unexp=m.load_state_dict(torch.load(UP+'model/model_pmt_pmt.pt',map_location='cpu'),strict=False)
print('missing',len(miss),'unexpected',len(unexp))
m.to(dev).eval()
with torch.no_grad():
    emb=m.gnn_learn(g)
    pe=emb['p'][torch.tensor([pdict[x] for x in ctl.p])]
    te_=emb['t'][torch.tensor([tdict[x] for x in ctl.t])]
    me=emb['m'][torch.tensor([mdict[x] for x in ctl.m])]
    s=m.pt_pred(pe,te_,me).squeeze().cpu().numpy()
os.chdir(ROOT)
from sklearn.metrics import roc_auc_score, average_precision_score
print(f'\nUniPMT POSITIVE CONTROL (their own pmt_pmt test set, our scoring path)')
print(f'  ROC {roc_auc_score(ctl.label,s):.4f} | PR {average_precision_score(ctl.label,s):.4f} | n={len(s)}')
pd.DataFrame({'label':ctl.label,'score':s}).to_csv(RESULTS+'control_scores.csv',index=False)
