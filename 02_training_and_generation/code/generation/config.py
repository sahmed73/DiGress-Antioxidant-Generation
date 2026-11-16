from pathlib import Path

HERE = Path(__file__).resolve().parent
DIGRESS_ROOT = Path(
    str(Path(__file__).resolve().parents[3] / "02_training_and_generation/code/DiGress")
)
DIGRESS_SRC = DIGRESS_ROOT / "src"
CHECKPOINT_PATH = str(
    Path(__file__).resolve().parents[3]
    / "02_training_and_generation/data/model/digress_epoch399.ckpt"
)
TRAIN_PT = str(
    Path(__file__).resolve().parents[3]
    / "02_training_and_generation/data/graph_splits/train.pt"
)
N_GENERATE = 5000
GENERATE_BATCH_SIZE = 2000
SATURATION_THRESHOLD = 30
SATURATION_PATIENCE = 3
NOVELTY_TANIMOTO_CUTOFF = 0.4
MAX_HEAVY_ATOMS = 50
RANDOM_SEED = 42
OUTPUT_DIR = str(
    Path(__file__).resolve().parents[3]
    / "02_training_and_generation/outputs/new_generation"
)
