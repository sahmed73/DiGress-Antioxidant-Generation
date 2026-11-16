# Reproduction levels

## Recorded analysis: runnable without model weights

Install `requirements-analysis.txt` and run `python scripts/reproduce.py` from the root. It verifies replicate metrics from the included samples and training split, saves the overview and illustrated molecules, and writes `figures/recomputed_metrics.json`. The original validity flags are replayed; invalid graph tensors are not supplied for re-sanitization.

## Training: adapted workflow, external dependencies and GPU required

The included `02_training_and_generation/code` preserves the dataset adapter, training launcher, generation code, source evaluation code and selected DiGress source/configuration files. The original DiGress MIT notice is retained. Install dependencies compatible with `configs/recorded_environment.json`; this is an environment record, not a tested portable lockfile. Obtain external runtime dependencies and allocate compute resources before running substantial preprocessing or GPU training.

`graph_preparation/prepare.py` rebuilds graph files from the included SMILES and preserves the `split` column. Its configuration writes to `02_training_and_generation/data/graph_splits`, where the training launcher expects them. The adapter supports C/N/O graphs with implicit hydrogens and at most 50 heavy atoms. Graph preparation and training have not been rerun in this portfolio; the historical binary graph files are excluded.

The training launcher installs the custom dataset adapter and patches the bundled working copy of DiGress before launching it. Work on a disposable checkout for those steps. Launch `python 02_training_and_generation/code/training/train.py --help` in the configured training environment, then configure a GPU allocation. Consult `configs/recorded_training.yaml` rather than assuming every environment's default matches the historical run.

## Generation: checkpoint required

The historical generation configuration refers to `02_training_and_generation/data/model/digress_epoch399.ckpt`, which is not distributed here. A trained compatible checkpoint and the graph training split are required to run the generation code. Configure their paths in `generation/config.py` and consult the launcher's command-line help. No download URL is invented; the checkpoint can be added to a separately approved release later. The 30,000 recorded sample benchmark remains reproducible from the packaged records without those assets.

Fresh training/sampling and bitwise agreement with the historical campaign have not been demonstrated. The [verified training history](training_history.md) is now included; `python scripts/plot_training.py` reproduces it. [Diversity and memorization-control analyses](diversity.md) use the released data with their own commands.
