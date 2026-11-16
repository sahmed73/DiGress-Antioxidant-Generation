#!/usr/bin/env python3
"""Scaffolds for all valid unique generations; sampled nearest neighbors vs all training molecules."""
import argparse,csv,gzip,json,random
from pathlib import Path
import numpy as np
from rdkit import Chem,DataStructs,RDLogger,rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--queries',type=int,default=1000);p.add_argument('--seed',type=int,default=42);args=p.parse_args()
root=Path(__file__).resolve().parents[1];out=root/'figures';out.mkdir(exist_ok=True)
RDLogger.DisableLog('rdApp.*');rng=random.Random(args.seed)
with gzip.open(root/'data/training_smiles.csv.gz','rt') as f:rows=list(csv.DictReader(f))
train=sorted({r['smiles'] for r in rows if r['split']=='train'});val=sorted({r['smiles'] for r in rows if r['split']=='val'})
with gzip.open(root/'data/vun_benchmark_samples.csv.gz','rt') as f:gen=sorted({r['canonical_smiles'] for r in csv.DictReader(f) if r['is_valid']=='1'})
fpgen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
def scaffold(m):return MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=False)
trainfps=[];trainscaff=set()
for i,s in enumerate(train):
 m=Chem.MolFromSmiles(s);assert m is not None,s
 trainfps.append(fpgen.GetFingerprint(m));trainscaff.add(scaffold(m))
 if (i+1)%25000==0:print('Training molecules processed:',i+1,flush=True)
genscaff=[]
for s in gen:
 m=Chem.MolFromSmiles(s);assert m is not None,s;genscaff.append(scaffold(m))
nonempty=[s for s in genscaff if s];unique=set(nonempty);novel=unique-trainscaff
stats={'rdkit_version':rdBase.rdkitVersion,'seed':args.seed,'fingerprint':'Morgan radius 2, 2048 bits, chirality enabled','training_reference_molecules':len(train),'training_unique_nonempty_scaffolds':len(trainscaff-{''}),'generated_valid_unique_molecules':len(gen),'generated_nonempty_scaffold_molecules':len(nonempty),'generated_unique_nonempty_scaffolds':len(unique),'generated_unique_scaffolds_absent_from_training':len(novel),'novel_unique_scaffold_fraction':len(novel)/len(unique),'generated_molecule_fraction_with_scaffold_absent_from_training':sum(s not in trainscaff for s in nonempty)/len(nonempty),'generated_empty_scaffolds':len(genscaff)-len(nonempty)}
records=[]
for cohort,smiles in [('Generated',gen),('Held-out validation',val)]:
 queries=rng.sample(smiles,min(args.queries,len(smiles)));sims=[]
 for j,s in enumerate(queries):
  fp=fpgen.GetFingerprint(Chem.MolFromSmiles(s));similarities=DataStructs.BulkTanimotoSimilarity(fp,trainfps);k=int(np.argmax(similarities));score=float(similarities[k]);sims.append(score);records.append([cohort,s,train[k],score])
 stats[cohort]={'query_count':len(sims),'mean_nearest_tanimoto':float(np.mean(sims)),'median_nearest_tanimoto':float(np.median(sims)),'fraction_at_least_0_8':float(np.mean(np.array(sims)>=.8))}
 print(cohort,stats[cohort],flush=True)
with (out/'nearest_training_neighbors.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['cohort','query_smiles','nearest_training_smiles','tanimoto']);w.writerows(records)
(out/'diversity_metrics.json').write_text(json.dumps(stats,indent=2)+'\n')
import runpy
runpy.run_path(str(root/'scripts/plot_diversity.py'),run_name='__main__')
