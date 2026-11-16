#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[3]
        / "02_training_and_generation/code/generation"
    ),
)
import generate
from evaluate_vun import load_training_smiles, metric_record, strict_smiles


def ts() -> str:
    return time.strftime("[%H:%M:%S]")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-pt", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=25000)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    parser.add_argument("--run-tag", default="epoch399_vun_25000")
    parser.add_argument(
        "--resume-run-tag",
        default=None,
        help="Reuse committed batches/RNG state from another run tag while writing new final outputs",
    )
    return parser.parse_args()


def atomic_torch_save(payload, destination: Path):
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def atomic_json_save(payload, destination: Path):
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def restore_rng_state(state: dict):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def set_initial_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batch_csv_path(batch_dir: Path, batch_index: int) -> Path:
    return batch_dir / f"batch_{batch_index:05d}.csv"


def state_path(batch_dir: Path, batch_index: int) -> Path:
    return batch_dir / f"state_{batch_index:05d}.pt"


def find_last_commit(batch_dir: Path):
    committed = []
    for state_file in batch_dir.glob("state_*.pt"):
        try:
            batch_index = int(state_file.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        if batch_csv_path(batch_dir, batch_index).is_file():
            committed.append(batch_index)
    if not committed:
        return (0, 0, None)
    last_batch = max(committed)
    expected = set(range(1, last_batch + 1))
    if set(committed) != expected:
        missing = sorted(expected.difference(committed))
        raise RuntimeError(f"Non-contiguous committed batches; missing {missing}")
    state = torch.load(
        state_path(batch_dir, last_batch), map_location="cpu", weights_only=False
    )
    return (last_batch, int(state["sampled"]), state)


def write_batch_atomically(destination: Path, rows: list[dict]):
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="") as handle:
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
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def collect_committed_rows(batch_dir: Path, last_batch: int):
    valid_smiles = []
    component_counts = []
    raw_count = 0
    for batch_index in range(1, last_batch + 1):
        with batch_csv_path(batch_dir, batch_index).open(newline="") as handle:
            for row in csv.DictReader(handle):
                raw_count += 1
                if row["is_valid"] == "1":
                    valid_smiles.append(row["canonical_smiles"])
                    if row["n_components"]:
                        component_counts.append(int(row["n_components"]))
    return (raw_count, valid_smiles, component_counts)


def write_combined_samples(batch_dir: Path, last_batch: int, destination: Path):
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    fieldnames = [
        "sample_index",
        "is_valid",
        "canonical_smiles",
        "n_components",
        "error_type",
    ]
    with temporary.open("w", newline="") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
        writer.writeheader()
        for batch_index in range(1, last_batch + 1):
            with batch_csv_path(batch_dir, batch_index).open(
                newline=""
            ) as input_handle:
                writer.writerows(csv.DictReader(input_handle))
        output_handle.flush()
        os.fsync(output_handle.fileno())
    os.replace(temporary, destination)


def finalize(
    args, batch_dir: Path, last_batch: int, summary_json: Path, sample_csv: Path
):
    raw_count, valid_smiles, component_counts = collect_committed_rows(
        batch_dir, last_batch
    )
    if raw_count != args.n_samples:
        raise RuntimeError(
            f"Expected {args.n_samples:,} committed samples, found {raw_count:,}"
        )
    print(
        f"{ts()} All raw samples committed; loading complete training reference",
        flush=True,
    )
    training_smiles = load_training_smiles(args.train_pt)
    unique_valid = set(valid_smiles)
    novel_unique = unique_valid.difference(training_smiles)
    training_matches = unique_valid.intersection(training_smiles)
    summary = {
        "protocol": {
            "checkpoint": str(args.checkpoint.resolve()),
            "training_split": str(args.train_pt.resolve()),
            "n_samples": args.n_samples,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "restartable": True,
            "validity_definition": "strict RDKit sanitization; no correction; largest fragment represents sanitizable disconnected graphs",
            "uniqueness_definition": "unique canonical SMILES divided by valid samples",
            "novelty_definition": "unique valid canonical SMILES absent from the complete training split divided by unique valid canonical SMILES",
            "filters_before_metrics": "none",
        },
        "counts": {
            "raw_samples": raw_count,
            "valid_samples": len(valid_smiles),
            "invalid_samples": raw_count - len(valid_smiles),
            "unique_valid_samples": len(unique_valid),
            "novel_unique_valid_samples": len(novel_unique),
            "unique_valid_training_matches": len(training_matches),
            "unique_training_reference": len(training_smiles),
        },
        "metrics": {
            "validity": metric_record(len(valid_smiles), raw_count),
            "uniqueness": metric_record(len(unique_valid), len(valid_smiles)),
            "novelty": metric_record(len(novel_unique), len(unique_valid)),
        },
        "components": {
            "single_component_valid": sum((value == 1 for value in component_counts)),
            "multi_component_valid": sum((value > 1 for value in component_counts)),
        },
    }
    write_combined_samples(batch_dir, last_batch, sample_csv)
    atomic_json_save(summary, summary_json)
    print(f"{ts()} FINAL RESULTS", flush=True)
    for name in ("validity", "uniqueness", "novelty"):
        metric = summary["metrics"][name]
        print(
            f"{ts()} {name.capitalize():10s} = {metric['percent']:.4f}% ({metric['numerator']:,}/{metric['denominator']:,})",
            flush=True,
        )
    print(f"{ts()} Summary: {summary_json}", flush=True)
    print(f"{ts()} Samples: {sample_csv}", flush=True)


def main():
    args = parse_args()
    if args.n_samples <= 0 or args.batch_size <= 0:
        raise ValueError("n-samples and batch-size must be positive")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.train_pt.is_file():
        raise FileNotFoundError(args.train_pt)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    batch_run_tag = args.resume_run_tag or args.run_tag
    batch_dir = args.output_dir / f"{batch_run_tag}_batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    summary_json = args.output_dir / f"{args.run_tag}_summary.json"
    sample_csv = args.output_dir / f"{args.run_tag}_samples.csv"
    if summary_json.is_file() and sample_csv.is_file():
        summary = json.loads(summary_json.read_text())
        print(f"{ts()} Benchmark is already complete: {summary_json}")
        for name in ("validity", "uniqueness", "novelty"):
            metric = summary["metrics"][name]
            print(f"{name.capitalize():10s} = {metric['percent']:.4f}%")
        return
    print(f"{ts()} Loading checkpoint {args.checkpoint}", flush=True)
    model = generate.load_model(str(args.checkpoint))
    atom_decoder = model.dataset_info.atom_decoder
    print(f"{ts()} Model loaded on {torch.cuda.get_device_name(0)}", flush=True)
    last_batch, sampled, previous_state = find_last_commit(batch_dir)
    if previous_state is None:
        set_initial_seed(args.seed)
        print(f"{ts()} Starting new benchmark with seed {args.seed}", flush=True)
    else:
        if previous_state["checkpoint"] != str(args.checkpoint.resolve()):
            raise RuntimeError(
                "Checkpoint differs from the checkpoint stored in resume state"
            )
        previous_target = int(previous_state["target_samples"])
        if previous_target > args.n_samples or sampled > args.n_samples:
            raise RuntimeError(
                f"Cannot resume {sampled:,}/{previous_target:,} state with the smaller target {args.n_samples:,}"
            )
        restore_rng_state(previous_state["rng"])
        print(
            f"{ts()} RESUME: {sampled:,}/{args.n_samples:,} samples committed through batch {last_batch}",
            flush=True,
        )
    while sampled < args.n_samples:
        batch_index = last_batch + 1
        current_batch = min(args.batch_size, args.n_samples - sampled)
        started = time.time()
        samples = generate.sample_batch(model, current_batch)
        rows = []
        for offset, (atom_types, edge_types) in enumerate(samples):
            smiles, n_components, error_type = strict_smiles(
                atom_types, edge_types, atom_decoder
            )
            rows.append(
                {
                    "sample_index": sampled + offset,
                    "is_valid": int(smiles is not None),
                    "canonical_smiles": smiles or "",
                    "n_components": n_components if n_components is not None else "",
                    "error_type": error_type or "",
                }
            )
        write_batch_atomically(batch_csv_path(batch_dir, batch_index), rows)
        sampled += current_batch
        state = {
            "batch_index": batch_index,
            "sampled": sampled,
            "target_samples": args.n_samples,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "checkpoint": str(args.checkpoint.resolve()),
            "rng": capture_rng_state(),
        }
        atomic_torch_save(state, state_path(batch_dir, batch_index))
        last_batch = batch_index
        valid_in_batch = sum((row["is_valid"] for row in rows))
        print(
            f"{ts()} COMMITTED batch={batch_index} samples={current_batch:,} valid={valid_in_batch:,} progress={sampled:,}/{args.n_samples:,} batch_time={(time.time() - started) / 60.0:.1f} min",
            flush=True,
        )
        del samples, rows
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    finalize(args, batch_dir, last_batch, summary_json, sample_csv)


if __name__ == "__main__":
    main()
