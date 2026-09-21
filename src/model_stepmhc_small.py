"""Smaller, capacity-controlled STEP-MHC for stability on ~30k rows / 26 epitopes.
Same two-branch (TCRxEpi + PepxHLA) design + structure prior, ~2M params instead of ~24.8M."""
import numpy as np, torch, torch.nn as nn
from pathlib import Path

class BasicBlock(nn.Module):
    def __init__(self, cin, cout, downsample=None):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False); self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False); self.b2 = nn.BatchNorm2d(cout)
        self.down = downsample; self.relu = nn.ReLU(inplace=False)
    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        o = self.relu(self.b1(self.c1(x))); o = self.b2(self.c2(o))
        return self.relu(o + idt)

class SmallNet(nn.Module):
    """3 shallow stages: 32 -> 64 -> 128, one block each."""
    def __init__(self, in_channels=5):
        super().__init__(); self.cin = 32
        self.stem = nn.Sequential(nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(32), nn.ReLU(False))
        self.l1 = self._make(32); self.l2 = self._make(64); self.l3 = self._make(128)
        self.pool = nn.AdaptiveAvgPool2d(1)
    def _make(self, cout):
        ds = None
        if self.cin != cout:
            ds = nn.Sequential(nn.Conv2d(self.cin, cout, 1, bias=False), nn.BatchNorm2d(cout))
        blk = BasicBlock(self.cin, cout, ds); self.cin = cout
        return blk
    def forward(self, x):
        return self.pool(self.l3(self.l2(self.l1(self.stem(x))))).flatten(1)  # [B,128]

class GlobalAttn(nn.Module):
    def __init__(self, d=7):
        super().__init__(); self.ln = nn.LayerNorm(d); self.at = nn.MultiheadAttention(d, 1, batch_first=True)
    def forward(self, x):
        x = self.ln(x.float()).unsqueeze(1); o, _ = self.at(x, x, x); return (o + x).squeeze(1)

class STEPMHCModelSmall(nn.Module):
    def __init__(self, use_structure_priors=True, prior_path=None, prior_shape=(73, 12)):
        super().__init__(); self.use_prior = use_structure_priors
        cw = torch.ones(1, 1, *prior_shape)
        if use_structure_priors and prior_path and Path(prior_path).exists():
            d = np.load(prior_path); cw = torch.from_numpy(d["cw_te"]).float().view(1, 1, *prior_shape)
            print(f"loaded TCRxEpi structure prior from {d.get('n_structures','?')} structures")
        else:
            print("structure prior: UNIFORM (placeholder until extraction)")
        self.register_buffer("cw_te", cw)
        self.branch_te = SmallNet(5); self.branch_ph = SmallNet(5); self.gattn = GlobalAttn(7)
        self.drop = nn.Dropout(0.4)
        self.head = nn.Sequential(nn.Linear(128 + 128 + 7, 64), nn.ReLU(False), nn.Dropout(0.4), nn.Linear(64, 1))
    def forward(self, te, ph, g):
        if self.use_prior: te = te * (1.0 + self.cw_te)
        f = torch.cat([self.branch_te(te), self.branch_ph(ph), self.gattn(g)], 1)
        return self.head(self.drop(f)).squeeze(1)

if __name__ == "__main__":
    import sys; sys.path.insert(0, str(Path(__file__).parent))
    from dataset_stepmhc import StepMHCDataset
    from torch.utils.data import DataLoader
    ds = StepMHCDataset("data/splits/unseen_tcr/fold0/test.csv")
    te, ph, g, y = next(iter(DataLoader(ds, batch_size=8)))
    m = STEPMHCModelSmall(use_structure_priors=False)
    out = m(te, ph, g)
    print("params:", f"{sum(p.numel() for p in m.parameters() if p.requires_grad):,}")
    print("out:", tuple(out.shape), "NaN:", torch.isnan(out).any().item())
