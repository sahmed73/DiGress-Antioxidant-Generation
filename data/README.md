# Data and provenance

These are independent local copies from the corrected project's training/generation stage, not generated example data:

| File | Purpose |
|---|---|
| `training_smiles.csv.gz` | 194,800 SMILES with recorded train/val/test labels; source `digress_training_set.csv.gz` |
| `training_split_counts.json` | Counts computed from those labels |
| `vun_benchmark_samples.csv.gz` | 30,000 recorded samples: index, validity flag, canonical SMILES, component count and failure type |
| `vun_benchmark_three_runs.json` / `.csv` | Original recorded replicate counts and summary statistics |

Source stage: `Paper_Workflow_Corrected/02_training_and_generation/data`. Data are copied without altering the originals. `SHA256SUMS` records hashes of the packaged inputs. The extraction excludes the full retrospectively curated generation library, graph tensors, model weights, publisher literature, raw simulation data and unrelated downstream studies.

The split is a random molecular partition, not a scaffold or prospective split. The included graph-preparation adapter now honors the recorded split column when rebuilding graph files; it preserves successful input order rather than assigning groups according to multiprocessing completion order. Thus graph file byte identity is not asserted. The historical model, evaluation and saved samples were not rerun or replaced.

## Additional verified evaluations

`training_history.csv`, `validation_history.csv` and `training_history_provenance.json` contain the extracted training record and source-log checksum. `resampling_baseline.csv` and `.json` record the three training-resampling controls. Methods are documented under `docs/`. The computed scaffold summary and exact nearest-neighbor query/match table are under `figures/`; no new molecule generation or training was performed.
