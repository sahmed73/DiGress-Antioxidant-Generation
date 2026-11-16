#!/usr/bin/env python3
"""Redraw diversity figures from the released numeric results."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];out=root/'figures'
stats=json.loads((out/'diversity_metrics.json').read_text())
with (out/'nearest_training_neighbors.csv').open() as f:
 records=[[r['cohort'],r['query_smiles'],r['nearest_training_smiles'],float(r['tanimoto'])] for r in csv.DictReader(f)]
n_unique=stats['generated_unique_nonempty_scaffolds']
n_novel=stats['generated_unique_scaffolds_absent_from_training']
plt.rcParams.update({'font.sans-serif':['Arial','DejaVu Sans'],'font.size':18,'axes.labelsize':20,'axes.spines.top':False,'axes.spines.right':False,'xtick.direction':'out','ytick.direction':'out','pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
for cohort,color in [('Generated','#007e87'),('Held-out validation','#d08b36')]:
 vals=sorted(r[3] for r in records if r[0]==cohort);axes[0].step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',label=cohort,color=color)
axes[0].set(xlabel='Nearest-training Morgan Tanimoto',ylabel='Cumulative fraction',xlim=(0,1),ylim=(0,1));axes[0].legend(frameon=False,fontsize=14)
axes[1].bar(['Seen in training','Absent from training'],[n_unique-n_novel,n_novel],color=['#526171','#007e87']);axes[1].set(ylabel='Unique Murcko scaffolds')
for ext in ('png','pdf','svg'):fig.savefig(out/f'chemical_diversity.{ext}',dpi=500)
plt.close(fig);print(json.dumps(stats,indent=2),flush=True)
