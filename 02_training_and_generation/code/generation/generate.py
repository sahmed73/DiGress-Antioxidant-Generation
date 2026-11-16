import argparse
import csv
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
HERE = Path(__file__).resolve().parent
DIGRESS_SRC = Path(
    str(
        Path(__file__).resolve().parents[3]
        / "02_training_and_generation/code/DiGress/src"
    )
)
sys.path.insert(0, str(HERE))
import config

_PHENOL_PAT = None


def ts():
    return time.strftime("[%H:%M:%S]")


def is_phenolic(smi):
    from rdkit import Chem

    global _PHENOL_PAT
    if _PHENOL_PAT is None:
        _PHENOL_PAT = Chem.MolFromSmarts("[c][OX2H]")
    mol = Chem.MolFromSmiles(smi)
    return mol is not None and mol.HasSubstructMatch(_PHENOL_PAT)


def validate_smiles(smiles_list):
    from rdkit import Chem

    valid = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        if len(Chem.GetMolFrags(mol)) > 1:
            continue
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            continue
        valid.append(Chem.MolToSmiles(mol))
    return valid


def ao_filter(smiles_list):
    from rdkit import Chem

    out = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        if mol.GetNumHeavyAtoms() > config.MAX_HEAVY_ATOMS:
            continue
        if is_phenolic(s):
            out.append(s)
    return out


def build_train_fps():
    import torch, random
    from rdkit import Chem
    from rdkit.Chem import AllChem

    p = Path(config.TRAIN_PT)
    if not p.exists():
        return []
    data = torch.load(p, weights_only=False)
    random.seed(config.RANDOM_SEED)
    sample = random.sample(data, min(10000, len(data)))
    fps = []
    for d in sample:
        smi = getattr(d, "smiles", None)
        if smi:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
    return fps


def is_novel(smi, train_fps):
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    mol = Chem.MolFromSmiles(smi)
    if mol is None or not train_fps:
        return True
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return (
        max(DataStructs.BulkTanimotoSimilarity(fp, train_fps), default=0.0)
        < config.NOVELTY_TANIMOTO_CUTOFF
    )


def load_model(checkpoint_path):
    sys.path.insert(0, str(DIGRESS_SRC.parent))
    sys.path.insert(0, str(DIGRESS_SRC))
    import torch
    from diffusion_model_discrete import DiscreteDenoisingDiffusion
    from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
    from metrics.molecular_metrics import SamplingMolecularMetrics

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    dataset_infos = ckpt["hyper_parameters"]["dataset_infos"]
    train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
    sampling_metrics = SamplingMolecularMetrics(dataset_infos, None)
    model = DiscreteDenoisingDiffusion.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
        train_metrics=train_metrics,
        sampling_metrics=sampling_metrics,
    )
    model.visualization_tools = None
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def sample_batch(model, batch_size):
    import torch

    with torch.no_grad():
        samples = model.sample_batch(
            batch_id=0,
            batch_size=batch_size,
            num_nodes=None,
            save_final=0,
            keep_chain=0,
            number_chain_steps=model.number_chain_steps,
        )
    return samples


def samples_to_smiles(samples, model):
    from rdkit import Chem
    from rdkit.Chem import RWMol, BondType

    BOND_DECODER = {
        1: BondType.SINGLE,
        2: BondType.DOUBLE,
        3: BondType.TRIPLE,
        4: BondType.AROMATIC,
    }
    ATOM_DECODER = model.dataset_info.atom_decoder
    smiles_out = []
    for atom_types, edge_types in samples:
        n = atom_types.shape[0]
        rw = RWMol()
        for i in range(n):
            rw.AddAtom(Chem.Atom(ATOM_DECODER[atom_types[i].item()]))
        for i in range(n):
            for j in range(i + 1, n):
                bt_int = edge_types[i, j].item()
                if bt_int > 0 and bt_int in BOND_DECODER:
                    rw.AddBond(i, j, BOND_DECODER[bt_int])
        try:
            mol = rw.GetMol()
            Chem.SanitizeMol(mol)
            smiles_out.append(Chem.MolToSmiles(mol))
        except Exception:
            pass
    return smiles_out


def run_generation(model, train_fps, dry_run, out_dir):
    t0 = time.time()
    all_novel, seen = ([], set())
    saturation_counter = 0
    print(f"\n{ts()} {'─' * 62}")
    print(f"{ts()} Generating unconditional phenolic molecules...", flush=True)
    for iteration in range(1, 10000):
        if len(all_novel) >= config.N_GENERATE:
            break
        samples = sample_batch(model, config.GENERATE_BATCH_SIZE)
        raw = samples_to_smiles(samples, model)
        valid = validate_smiles(raw)
        filtered = ao_filter(valid)
        unique = [s for s in filtered if s not in seen]
        seen.update(unique)
        novel = [s for s in unique if is_novel(s, train_fps)]
        all_novel.extend(novel)
        print(
            f"{ts()} iter={iteration} | raw={len(raw)} valid={len(valid)} phenolic={len(filtered)} novel={len(novel)} cumul={len(all_novel)}",
            flush=True,
        )
        if len(novel) < config.SATURATION_THRESHOLD:
            saturation_counter += 1
            if saturation_counter >= config.SATURATION_PATIENCE:
                print(f"{ts()} Saturated — stopping")
                break
        else:
            saturation_counter = 0
        if dry_run and iteration >= 2:
            break
    elapsed = (time.time() - t0) / 60
    print(f"{ts()} DONE: {len(all_novel):,} novel molecules in {elapsed:.1f}min")
    if not dry_run and all_novel:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_csv = out_dir / f"generated_{timestamp}.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["smiles"])
            w.writerows([[s] for s in all_novel])
        print(f"{ts()} Saved {len(all_novel):,} → {out_csv}")
    return all_novel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=config.CHECKPOINT_PATH)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.checkpoint:
        print("ERROR: set CHECKPOINT_PATH in config.py or pass --checkpoint")
        sys.exit(1)
    t_total = time.time()
    print(f"{ts()} {'=' * 62}")
    print(f"{ts()} DiGress Generation — Phenolic Unconditional")
    print(f"{ts()} Checkpoint : {args.checkpoint}", flush=True)
    model = load_model(args.checkpoint)
    print(f"{ts()} Model loaded", flush=True)
    train_fps = build_train_fps()
    print(f"{ts()} Train FP reference: {len(train_fps):,}", flush=True)
    out_dir = Path(config.OUTPUT_DIR)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    mols = run_generation(model, train_fps, args.dry_run, out_dir)
    print(f"\n{ts()} {'=' * 62}")
    print(f"{ts()} Total: {len(mols):,} novel unique phenolic molecules")
    print(f"{ts()} Total time: {(time.time() - t_total) / 60:.1f}min")


if __name__ == "__main__":
    main()
