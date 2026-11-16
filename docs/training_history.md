# Verified training history

The numeric history is extracted from `PhenOnlyUnConditional/02_train_extrafeat/slurm-74442.out`, which identifies the run as `uncond_v2_feat_e500` and records the expected dataset sizes and 500 configured epochs. The source-file hash and extraction details are in `data/training_history_provenance.json`; raw cluster logs are not included. `scripts/extract_training_history.py LOG_PATH` regenerates the numeric tables from an authorized source log.

`training_history.csv` contains 500 epoch summaries of atom and bond cross-entropy. These are printed component metrics, rounded to three decimals, not the validation objective. `validation_history.csv` contains 21 negative-log-likelihood and KL summaries, including the initial epoch-0 sanity check before training and subsequent 25-epoch evaluations. The plot places training CE and validation NLL in different panels; the NLL axis uses a logarithmic scale to show the initial value and later trajectory.

The lowest logged validation NLL is 290.80 at epoch 399 (zero-based). The bundled checkpoint callback monitors `val/epoch_NLL` in minimization mode, and the preserved sampling model is the epoch-399 checkpoint. The log also records 82.88% strict validity over 10,000 final test draws. That single draw evaluation is distinct from the separately recorded three-replicate benchmark reported in the overview. Epoch 399 is selected by validation likelihood, not by maximizing the later benchmark novelty.

Run `python scripts/plot_training.py` to redraw the figure from the released numeric tables. These are historical results from the recorded run; no new training was performed.
