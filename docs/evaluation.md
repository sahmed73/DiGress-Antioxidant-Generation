# Evaluation definitions

The benchmark contains 30,000 raw draws, grouped consecutively into three replicates of 10,000 from one fixed checkpoint. The recorded source evaluator builds atom/bond graphs, runs strict RDKit sanitization without charge/valence corrections, and represents a sanitizable disconnected graph by its largest fragment. This convention follows the source DiGress molecular benchmark, not the stricter downstream connected-candidate filter.

- **Validity:** valid samples / raw draws.
- **Uniqueness:** distinct canonical SMILES / valid samples, separately within each replicate.
- **Novelty:** distinct valid canonical SMILES absent from the complete training split / distinct valid canonical SMILES.

The mean and sample standard deviation summarize sampling variability for one checkpoint. They do not quantify training-run variability or confidence in biological/chemical efficacy. The recorded JSON additionally contains Student t confidence intervals across the three replicates; an unbounded approximation can exceed 100%, so the overview figure shows the actual replicate values instead.

The portable replay recomputes counts from recorded validity flags and canonical SMILES; it does not reconstruct the original failed graphs, which are not included, or regenerate molecules. Training membership comes from the included split-labelled SMILES. Candidate motifs can be checked with RDKit; exact-match novelty alone does not establish scaffold novelty or absence from PubChem. Antioxidant activity, toxicity and synthesizability require separate evaluation.
