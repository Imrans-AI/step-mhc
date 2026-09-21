"""Heatmap panels: the explicit interaction representation and the structural contact prior.
Vector PDF for Illustrator; 600 dpi PNG for checking."""
import sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
sys.path.insert(0,'.')
from src.dataset_stepmhc import _imap, CAPS, LOOPS, L_TCR, L_PEP, L_HLA, PROPS

plt.rcParams.update({'font.family':'Nimbus Sans','font.size':7,'axes.linewidth':0.35,
                     'xtick.major.width':0.35,'ytick.major.width':0.35,
                     'xtick.labelsize':6,'ytick.labelsize':6,'axes.titlesize':7.5})
CMAP = LinearSegmentedColormap.from_list('daisy', ['#FFFFFF','#C9D9EC','#A3C9E8','#8377A8','#4A3F6B'])
CMAP2 = LinearSegmentedColormap.from_list('contact', ['#FFFFFF','#F6E3EC','#EBA8C3','#B8628C','#6B2F4A'])

# VERIFIED real training-pool positive (VDJdb, binder=1, HLA-A*02:01 / GILGFVFTL)
TCR = dict(A1='DSASNY', A2='IRSNVGE', A3='CAASGGGSQGNLIF',
           B1='LNHDA', B2='SQIVND', B3='CASSIRSQETQYF')
PEP, HLA = 'GILGFVFTL', 'YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY'
ts_pad = "".join(str(TCR[c])[:CAPS[c]].ljust(CAPS[c],'-') for c in LOOPS)   # model input
ts = "".join(str(TCR[c]) for c in LOOPS)                                     # occupied rows only

bounds, o = [], 0
for k in LOOPS: bounds.append((k, o, o+len(str(TCR[k])))); o += len(str(TCR[k]))
L_SHOW = o

fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.6),
                         gridspec_kw=dict(width_ratios=[1.0,1.1], wspace=0.42))

# a — TCR x peptide interaction map (hydrophobicity channel)
te = _imap(ts, PEP, L_SHOW, L_PEP)[:, :, :len(PEP)]
im = axes[0].imshow(te[0], aspect='auto', cmap=CMAP, interpolation='nearest')
axes[0].set_title('TCR $\\times$ peptide map\n(hydrophobicity channel)', pad=4)
axes[0].set_xlabel('peptide position')
axes[0].set_ylabel('CDR residues', labelpad=14)
axes[0].set_xticks(range(0,len(PEP),2)); axes[0].set_xticklabels(range(1,len(PEP)+1,2))
for k,a,b in bounds:
    axes[0].axhline(b-0.5, color='#888', lw=0.5)
    axes[0].text(-1.1, (a+b)/2, k, ha='right', va='center', fontsize=5.4, color='#333')
axes[0].set_yticks([])
plt.colorbar(im, ax=axes[0], fraction=0.045, pad=0.02).ax.tick_params(labelsize=5)

# b — peptide x HLA map
ph = _imap(PEP, HLA, L_PEP, L_HLA)[:, :len(PEP), :]
im2 = axes[1].imshow(ph[0], aspect='auto', cmap=CMAP, interpolation='nearest')
axes[1].set_title('peptide $\\times$ HLA map', pad=4)
axes[1].set_xlabel('HLA pseudo-sequence (34)')
axes[1].set_xticks(range(0,L_HLA,5)); axes[1].set_xticklabels(range(1,L_HLA+1,5)); axes[1].set_ylabel('peptide position'); axes[1].set_yticks(range(0,len(PEP),2)); axes[1].set_yticklabels(range(1,len(PEP)+1,2))
plt.colorbar(im2, ax=axes[1], fraction=0.045, pad=0.02).ax.tick_params(labelsize=5)

for ax,l in zip(axes,'ab'):
    ax.text(-0.16, 1.14, l, transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')
plt.savefig('figures/Fig_maps.pdf', bbox_inches='tight')
plt.savefig('figures/Fig_maps.png', dpi=600, bbox_inches='tight')
print('wrote figures/Fig_maps.{pdf,png}')
