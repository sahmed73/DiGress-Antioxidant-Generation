#!/usr/bin/env python3
"""Extract numeric history and source hashes without releasing raw cluster logs."""
import argparse,csv,hashlib,json,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('log',type=Path);args=p.parse_args()
text=args.log.read_text(errors='replace');root=Path(__file__).resolve().parents[1]
train=re.findall(r'^Epoch (\d+): X_CE: ([\d.]+) -- E_CE: ([\d.]+) -- y_CE: (-?[\d.]+)',text,re.M)
val=re.findall(r'^Epoch (\d+): Val NLL ([\d.]+) -- Val Atom type KL ([\d.]+) --  Val Edge type KL: ([\d.]+)',text,re.M)
assert len(train)==500 and [int(x[0]) for x in train]==list(range(500))
assert len({x[0] for x in val})==len(val)
for name,fields,rows in [('training_history.csv',['epoch','atom_cross_entropy','bond_cross_entropy'],[r[:3] for r in train]),('validation_history.csv',['epoch','negative_log_likelihood','atom_kl','bond_kl'],val)]:
 with (root/'data'/name).open('w',newline='') as f:w=csv.writer(f);w.writerow(fields);w.writerows(rows)
meta={'source_log':'PhenOnlyUnConditional/02_train_extrafeat/slurm-74442.out','source_sha256':hashlib.sha256(args.log.read_bytes()).hexdigest(),'run_name':'uncond_v2_feat_e500','train_epochs':len(train),'validation_evaluations':len(val),'minimum_logged_validation_nll_epoch':int(min(val,key=lambda r:float(r[1]))[0]),'validation_epoch_0_is_initial_sanity_check':True,'note':'Values are rounded as printed in the source log; training CE and validation NLL are different objectives.'}
(root/'data/training_history_provenance.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,indent=2))
