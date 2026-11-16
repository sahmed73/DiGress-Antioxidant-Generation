#!/usr/bin/env python3
"""Plot the verified numeric history using released CSVs."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
with (root/'data/training_history.csv').open() as f:train=list(csv.DictReader(f))
with (root/'data/validation_history.csv').open() as f:val=list(csv.DictReader(f))
plt.rcParams.update({'font.sans-serif':['Arial','DejaVu Sans'],'font.size':18,'axes.labelsize':20,'axes.spines.top':False,'axes.spines.right':False,'xtick.direction':'out','ytick.direction':'out','pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
for field,label,color in [('atom_cross_entropy','Atom CE','#007e87'),('bond_cross_entropy','Bond CE','#d08b36')]:axes[0].plot([int(r['epoch']) for r in train],[float(r[field]) for r in train],label=label,color=color)
axes[0].set(xlabel='Epoch (zero-based)',ylabel='Training cross-entropy');axes[0].legend(frameon=False)
axes[1].plot([int(r['epoch']) for r in val],[float(r['negative_log_likelihood']) for r in val],color='#526171',marker='o',ms=4)
best=min(val,key=lambda r:float(r['negative_log_likelihood']));axes[1].scatter([int(best['epoch'])],[float(best['negative_log_likelihood'])],color='#007e87',marker='*',s=150,zorder=3,label='Checkpoint: epoch 399');axes[1].set(xlabel='Epoch (zero-based)',ylabel='Validation NLL',yscale='log');axes[1].legend(frameon=False,fontsize=14)
for ext in ('png','pdf','svg'):fig.savefig(root/'figures'/f'training_history.{ext}',dpi=500)
plt.close(fig)
