from pathlib import Path

HERE = Path(__file__).resolve().parent
DIGRESS_ROOT = Path(
    str(Path(__file__).resolve().parents[3] / "02_training_and_generation/code/DiGress")
)
DIGRESS_SRC = DIGRESS_ROOT / "src"
DIGRESS_CONF = DIGRESS_ROOT / "configs"
DATA_DIR = Path(
    str(
        Path(__file__).resolve().parents[3]
        / "02_training_and_generation/data/graph_splits"
    )
)
RUNS_DIR = Path(
    str(
        Path(__file__).resolve().parents[3]
        / "02_training_and_generation/outputs/training"
    )
)
LR = 0.0001
EPOCHS = 500
BATCH_SIZE = 16
WEIGHT_DECAY = 1e-12
GRAD_CLIP = 1.0
RANDOM_SEED = 42
NUM_WORKERS = 4
T = 500
HIDDEN_DIM = 256
N_LAYERS = 6
N_HEADS = 8
VAL_EVERY = 25
SAMPLE_EVERY_VAL = 2
LOG_EVERY = 50
GPUS = 2
