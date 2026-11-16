#!/usr/bin/env python3
"""Recompute recorded sample metrics and draw figures; no GPU/checkpoint required."""
import csv, gzip, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
RDLogger.DisableLog('rdApp.*')
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
with gzip.open(ROOT/'data/training_smiles.csv.gz','rt') as f: training=list(csv.DictReader(f))
train={r['smiles'] for r in training if r['split']=='train'}
with gzip.open(ROOT/'data/vun_benchmark_samples.csv.gz','rt') as f: samples=list(csv.DictReader(f))
assert len(samples)==30000
assert [int(r['sample_index']) for r in samples]==list(range(30000))
expected=json.loads((ROOT/'data/vun_benchmark_three_runs.json').read_text()); results=[]
for i in range(3):
 rows=samples[i*10000:(i+1)*10000];valid=[r['canonical_smiles'] for r in rows if r['is_valid']=='1'];unique=set(valid);novel=unique-train
 values=[100*len(valid)/len(rows),100*len(unique)/len(valid),100*len(novel)/len(unique)]
 for key,value in zip(('validity','uniqueness','novelty'),values): assert np.isclose(value,expected['metrics'][key]['replicate_percentages'][i])
 results.append(values)
arr=np.array(results); summary={k:{'mean_percent':float(arr[:,i].mean()),'sample_std_percent':float(arr[:,i].std(ddof=1))} for i,k in enumerate(('validity','uniqueness','novelty'))}
(OUT/'recomputed_metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
plt.rcParams.update({'font.sans-serif':['Arial','DejaVu Sans'],'font.size':18,'axes.labelsize':20,'axes.spines.top':False,'axes.spines.right':False,'xtick.direction':'out','ytick.direction':'out','pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(8,5),layout='constrained'); xx=np.arange(3); means=arr.mean(axis=0)
ax.bar(xx,means,color=['#007e87','#526171','#d08b36'],width=.58)
for row in arr:ax.scatter(xx,row,s=28,facecolor='white',edgecolor='black',zorder=3)
for x,y in zip(xx,means):ax.text(x,y+2,f'{y:.2f}%',ha='center',fontsize=16)
ax.set(xticks=xx,xticklabels=['Validity','Uniqueness','Novelty'],ylabel='Fraction (%)',ylim=(0,111));ax.set_yticks([0,20,40,60,80,100])
for ext in ('png','pdf','svg'):fig.savefig(OUT/f'generation_quality.{ext}',dpi=500)
plt.close(fig)
pattern=Chem.MolFromSmarts('[c][OX2H]'); chosen=[];seen=set()
for row in samples:
 s=row['canonical_smiles']
 if row['is_valid']!='1' or row['n_components']!='1' or s in train or s in seen:continue
 mol=Chem.MolFromSmiles(s)
 if mol is None or not mol.HasSubstructMatch(pattern):continue
 seen.add(s);chosen.append((row['sample_index'],s,mol))
 if len(chosen)==8:break
fig,axes=plt.subplots(2,4,figsize=(14,7),layout='constrained')
for ax,(idx,s,mol) in zip(axes.flat,chosen):
 ax.imshow(Draw.MolToImage(mol,size=(450,350)));ax.set_axis_off();ax.set_title(f'Sample {idx}',fontsize=18)
for ext in ('png','pdf','svg'):fig.savefig(OUT/f'generated_examples.{ext}',dpi=500)
plt.close(fig)
with (OUT/'illustrated_molecules.csv').open('w') as f:
 writer=csv.writer(f);writer.writerow(['sample_index','canonical_smiles']);writer.writerows((i,s) for i,s,m in chosen)
print(json.dumps(summary,indent=2));print('All recorded replicate metrics reproduced.')
