#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem, RDLogger

HERE = Path(__file__).resolve().parent
GENERATE_DIR = Path(
    str(
        Path(__file__).resolve().parents[3]
        / "02_training_and_generation/code/generation"
    )
)
sys.path.insert(0, str(GENERATE_DIR))
import generate

RDLogger.DisableLog("rdApp.*")
BOND_DECODER = {
    1: Chem.rdchem.BondType.SINGLE,
    2: Chem.rdchem.BondType.DOUBLE,
    3: Chem.rdchem.BondType.TRIPLE,
    4: Chem.rdchem.BondType.AROMATIC,
}


def ts() -> str:
    return time.strftime("[%H:%M:%S]")


def canonicalize_smiles(smiles: str | None) -> str | None:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def strict_smiles(atom_types, edge_types, atom_decoder):
    rw = Chem.RWMol()
    try:
        for atom_type in atom_types:
            rw.AddAtom(Chem.Atom(atom_decoder[int(atom_type.item())]))
        n_atoms = int(atom_types.shape[0])
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                bond_index = int(edge_types[i, j].item())
                bond_type = BOND_DECODER.get(bond_index)
                if bond_type is not None:
                    rw.AddBond(i, j, bond_type)
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        n_components = len(fragments)
        largest = max(
            fragments, default=mol, key=lambda fragment: fragment.GetNumAtoms()
        )
        Chem.SanitizeMol(largest)
        smiles = Chem.MolToSmiles(largest, canonical=True)
        return (smiles, n_components, None)
    except Exception as exc:
        return (None, None, type(exc).__name__)


def load_training_smiles(train_pt: Path) -> set[str]:
    print(f"{ts()} Loading the complete training split: {train_pt}", flush=True)
    training_data = torch.load(train_pt, map_location="cpu", weights_only=False)
    training_smiles = set()
    missing = 0
    for datum in training_data:
        canonical = canonicalize_smiles(getattr(datum, "smiles", None))
        if canonical is None:
            missing += 1
        else:
            training_smiles.add(canonical)
    print(
        f"{ts()} Training reference: {len(training_data):,} records, {len(training_smiles):,} unique canonical SMILES, {missing:,} missing/invalid",
        flush=True,
    )
    del training_data
    return training_smiles


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054):
    if total == 0:
        return (None, None)
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return (center - half_width, center + half_width)


def metric_record(numerator: int, denominator: int) -> dict:
    low, high = wilson_interval(numerator, denominator)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "fraction": numerator / denominator if denominator else None,
        "percent": 100.0 * numerator / denominator if denominator else None,
        "wilson_95_percent_low": 100.0 * low if low is not None else None,
        "wilson_95_percent_high": 100.0 * high if high is not None else None,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-pt", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=25000)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    parser.add_argument("--run-tag", default="epoch399_vun_25000")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n_samples <= 0 or args.batch_size <= 0:
        raise ValueError("n-samples and batch-size must be positive")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.train_pt.is_file():
        raise FileNotFoundError(args.train_pt)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_csv = args.output_dir / f"{args.run_tag}_samples.csv"
    summary_json = args.output_dir / f"{args.run_tag}_summary.json"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"{ts()} DiGress independent V/U/N benchmark", flush=True)
    print(f"{ts()} Checkpoint : {args.checkpoint}", flush=True)
    print(f"{ts()} Samples    : {args.n_samples:,}", flush=True)
    print(f"{ts()} Batch size : {args.batch_size:,}", flush=True)
    print(f"{ts()} Seed       : {args.seed}", flush=True)
    print(f"{ts()} CUDA       : {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"{ts()} GPU        : {torch.cuda.get_device_name(0)}", flush=True)
    training_smiles = load_training_smiles(args.train_pt)
    model = generate.load_model(str(args.checkpoint))
    atom_decoder = model.dataset_info.atom_decoder
    print(f"{ts()} Model loaded; atom decoder={atom_decoder}", flush=True)
    valid_smiles = []
    component_counts = []
    invalid_count = 0
    started = time.time()
    with sample_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_index",
                "is_valid",
                "canonical_smiles",
                "n_components",
                "error_type",
            ],
        )
        writer.writeheader()
        sampled = 0
        batch_index = 0
        while sampled < args.n_samples:
            batch_index += 1
            current_batch = min(args.batch_size, args.n_samples - sampled)
            samples = generate.sample_batch(model, current_batch)
            for offset, (atom_types, edge_types) in enumerate(samples):
                smiles, n_components, error_type = strict_smiles(
                    atom_types, edge_types, atom_decoder
                )
                sample_index = sampled + offset
                is_valid = smiles is not None
                if is_valid:
                    valid_smiles.append(smiles)
                    component_counts.append(n_components)
                else:
                    invalid_count += 1
                writer.writerow(
                    {
                        "sample_index": sample_index,
                        "is_valid": int(is_valid),
                        "canonical_smiles": smiles or "",
                        "n_components": (
                            n_components if n_components is not None else ""
                        ),
                        "error_type": error_type or "",
                    }
                )
            sampled += current_batch
            handle.flush()
            elapsed_minutes = (time.time() - started) / 60.0
            print(
                f"{ts()} batch={batch_index} sampled={sampled:,}/{args.n_samples:,} valid={len(valid_smiles):,} invalid={invalid_count:,} elapsed={elapsed_minutes:.1f} min",
                flush=True,
            )
            del samples
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    unique_valid = set(valid_smiles)
    novel_unique = unique_valid.difference(training_smiles)
    training_matches = unique_valid.intersection(training_smiles)
    validity = metric_record(len(valid_smiles), args.n_samples)
    uniqueness = metric_record(len(unique_valid), len(valid_smiles))
    novelty = metric_record(len(novel_unique), len(unique_valid))
    component_summary = {
        "single_component_valid": sum((count == 1 for count in component_counts)),
        "multi_component_valid": sum((count > 1 for count in component_counts)),
        "minimum": min(component_counts) if component_counts else None,
        "mean": float(np.mean(component_counts)) if component_counts else None,
        "maximum": max(component_counts) if component_counts else None,
    }
    summary = {
        "protocol": {
            "checkpoint": str(args.checkpoint.resolve()),
            "training_split": str(args.train_pt.resolve()),
            "n_samples": args.n_samples,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "validity_definition": "strict RDKit sanitization; no correction; largest fragment represents sanitizable disconnected graphs",
            "uniqueness_definition": "unique canonical SMILES divided by valid samples",
            "novelty_definition": "unique valid canonical SMILES absent from the complete training split divided by unique valid canonical SMILES",
            "filters_before_metrics": "none",
        },
        "counts": {
            "raw_samples": args.n_samples,
            "valid_samples": len(valid_smiles),
            "invalid_samples": invalid_count,
            "unique_valid_samples": len(unique_valid),
            "novel_unique_valid_samples": len(novel_unique),
            "unique_valid_training_matches": len(training_matches),
            "unique_training_reference": len(training_smiles),
        },
        "metrics": {"validity": validity, "uniqueness": uniqueness, "novelty": novelty},
        "components": component_summary,
        "elapsed_minutes": (time.time() - started) / 60.0,
        "sample_csv": str(sample_csv.resolve()),
    }
    with summary_json.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"{ts()} RESULTS", flush=True)
    print(
        f"{ts()} Validity   = {validity['percent']:.4f}% ({validity['numerator']:,}/{validity['denominator']:,})",
        flush=True,
    )
    print(
        f"{ts()} Uniqueness = {uniqueness['percent']:.4f}% ({uniqueness['numerator']:,}/{uniqueness['denominator']:,})",
        flush=True,
    )
    print(
        f"{ts()} Novelty    = {novelty['percent']:.4f}% ({novelty['numerator']:,}/{novelty['denominator']:,})",
        flush=True,
    )
    print(f"{ts()} Summary    : {summary_json}", flush=True)
    print(f"{ts()} Samples    : {sample_csv}", flush=True)


if __name__ == "__main__":
    main()
