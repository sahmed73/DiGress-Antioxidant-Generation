#!/usr/bin/env python3
"""A transparent memorization control: sample training SMILES with replacement."""
import csv,gzip,json,random
from pathlib import Path
import numpy as np
from rdkit import Chem,RDLogger
RDLogger.DisableLog('rdApp.*');root=Path(__file__).resolve().parents[1]
with gzip.open(root/'data/training_smiles.csv.gz','rt') as f:pool=[r['smiles'] for r in csv.DictReader(f) if r['split']=='train']
reference=set()
for s in pool:
 m=Chem.MolFromSmiles(s);assert m is not None
 reference.add(Chem.MolToSmiles(m,canonical=True))
rows=[]
for seed in (42,43,44):
 rng=random.Random(seed);raw=[rng.choice(pool) for _ in range(10000)];valid=[]
 for s in raw:
  m=Chem.MolFromSmiles(s);assert m is not None
  parts=Chem.GetMolFrags(m,asMols=True,sanitizeFrags=True);valid.append(Chem.MolToSmiles(max(parts,key=lambda m:m.GetNumAtoms()),canonical=True))
 # Check canonicalized identities against the same recorded reference.
 novelty=100*len(set(valid)-reference)/len(set(valid))
 rows.append({'seed':seed,'raw_samples':10000,'validity_percent':100.,'uniqueness_percent':100*len(set(valid))/len(valid),'novelty_percent':novelty})
with (root/'data/resampling_baseline.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary={k:{'mean_percent':float(np.mean([r[k+'_percent'] for r in rows])),'sample_std_percent':float(np.std([r[k+'_percent'] for r in rows],ddof=1))} for k in ('validity','uniqueness','novelty')}
(root/'data/resampling_baseline.json').write_text(json.dumps({'method':'Training-set sampling with replacement; memorization control, not a learned generative competitor','samples_per_replicate':10000,'seeds':[42,43,44],'metrics':summary},indent=2)+'\n');print(json.dumps(summary,indent=2))
