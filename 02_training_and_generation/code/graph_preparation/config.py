from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUT_CSV = str(
    Path(__file__).resolve().parents[3]
    / "data/training_smiles.csv.gz"
)
OUTPUT_DIR = str(
    Path(__file__).resolve().parents[3]
    / "02_training_and_generation/data/graph_splits"
)
MAX_HEAVY_ATOMS = 50
TRAIN_FRAC = 0.9
VAL_FRAC = 0.05
RANDOM_SEED = 42
