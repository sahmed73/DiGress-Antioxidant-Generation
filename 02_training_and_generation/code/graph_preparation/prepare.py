import argparse
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path
import torch
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config

ATOM_ENCODER = {"C": 0, "N": 1, "O": 2}
BOND_ENCODER = {
    Chem.rdchem.BondType.SINGLE: 1,
    Chem.rdchem.BondType.DOUBLE: 2,
    Chem.rdchem.BondType.TRIPLE: 3,
    Chem.rdchem.BondType.AROMATIC: 4,
}
N_ATOM_TYPES = 3
N_BOND_TYPES = 5
N_WORKERS = 8
LOG_INTERVAL = 10000


def ts():
    return time.strftime("[%H:%M:%S]")


def _init_worker():
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")


def mol_to_pyg(smiles):
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.RemoveHs(mol)
    if mol is None:
        return None
    n_atoms = mol.GetNumAtoms()
    if n_atoms > config.MAX_HEAVY_ATOMS or n_atoms < 2:
        return None
    x_rows = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym not in ATOM_ENCODER:
            return None
        row = [0] * N_ATOM_TYPES
        row[ATOM_ENCODER[sym]] = 1
        x_rows.append(row)
    src, dst, edge_feats = ([], [], [])
    for bond in mol.GetBonds():
        i, j = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        btype = BOND_ENCODER.get(bond.GetBondType())
        if btype is None:
            return None
        feat = [0] * N_BOND_TYPES
        feat[btype] = 1
        for u, v in [(i, j), (j, i)]:
            src.append(u)
            dst.append(v)
            edge_feats.append(feat)
    if not src:
        return None
    return {
        "x_rows": x_rows,
        "src": src,
        "dst": dst,
        "edge_feats": edge_feats,
        "smiles": Chem.MolToSmiles(mol),
    }


def dict_to_data(d):
    from torch_geometric.data import Data

    data = Data(
        x=torch.tensor(d["x_rows"], dtype=torch.float),
        edge_index=torch.tensor([d["src"], d["dst"]], dtype=torch.long),
        edge_attr=torch.tensor(d["edge_feats"], dtype=torch.float),
        y=torch.zeros(1, 0, dtype=torch.float),
    )
    data.smiles = d["smiles"]
    return data


def main(dry_run=False):
    import pandas as pd

    t0 = time.time()
    print(f"{ts()} {'=' * 64}")
    print(f"{ts()} Data Preparation — DiGress Phenolic Unconditional")
    print(f"{ts()} Input    : {config.INPUT_CSV}")
    print(f"{ts()} Output   : {config.OUTPUT_DIR}")
    print(
        f"{ts()} Split    : {config.TRAIN_FRAC}/{config.VAL_FRAC}/{round(1 - config.TRAIN_FRAC - config.VAL_FRAC, 2)}"
    )
    print(f"{ts()} Workers  : {N_WORKERS}")
    print(f"{ts()} {'=' * 64}", flush=True)
    df = pd.read_csv(config.INPUT_CSV)
    if dry_run:
        df = df.head(2000)
        print(f"{ts()} [dry-run] using {len(df):,} rows")
    if "smiles" not in df.columns:
        print(f"{ts()} ERROR: missing column 'smiles'")
        return
    df = df.dropna(subset=["smiles"])
    total = len(df)
    print(f"{ts()} Loaded {total:,} molecules", flush=True)
    smiles_list = list(df["smiles"])
    print(f"\n{ts()} Converting to PyG graphs...", flush=True)
    data_list = []
    recorded_splits = list(df["split"]) if "split" in df.columns else None
    split_labels = []
    skipped = 0
    done = 0
    with mp.Pool(N_WORKERS, initializer=_init_worker) as pool:
        for result in pool.imap(mol_to_pyg, smiles_list, chunksize=500):
            done += 1
            if result is None:
                skipped += 1
            else:
                data_list.append(dict_to_data(result))
                if recorded_splits is not None:
                    split_labels.append(recorded_splits[done - 1])
            if done % LOG_INTERVAL == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta_s = (total - done) / rate if rate > 0 else float("inf")
                pct = 100.0 * done / total
                print(
                    f"[{time.strftime('%H:%M:%S')}] {done:>9,} / {total:,} ({pct:5.1f}%) | converted={len(data_list):,} skipped={skipped:,} | elapsed={elapsed / 60:.2f}min | ETA={eta_s / 60:.2f}min",
                    flush=True,
                )
    elapsed = time.time() - t0
    print(
        f"\n{ts()} Converted: {len(data_list):,}  Skipped: {skipped:,}  Time: {elapsed / 60:.2f}min",
        flush=True,
    )
    if recorded_splits is not None:
        if not set(split_labels) <= {"train", "val", "test"}:
            raise ValueError("Unrecognized split label")
        splits = {k: [g for g, label in zip(data_list, split_labels) if label == k]
                  for k in ("train", "val", "test")}
    else:
        random.seed(config.RANDOM_SEED)
        random.shuffle(data_list)
        n = len(data_list)
        n_train = int(n * config.TRAIN_FRAC)
        n_val = int(n * config.VAL_FRAC)
        splits = {"train": data_list[:n_train], "val": data_list[n_train:n_train+n_val],
                  "test": data_list[n_train+n_val:]}
    for split, lst in splits.items():
        print(f"{ts()} {split:5s}: {len(lst):,}")
    if not dry_run:
        out = Path(config.OUTPUT_DIR)
        out.mkdir(parents=True, exist_ok=True)
        for split, lst in splits.items():
            torch.save(lst, out / f"{split}.pt")
            print(f"{ts()} Saved {len(lst):,} → {out / split}.pt")
    else:
        print(f"\n{ts()} [dry-run] sample x shape: {data_list[0].x.shape}")
    print(f"\n{ts()} Done — {(time.time() - t0) / 60:.2f}min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
