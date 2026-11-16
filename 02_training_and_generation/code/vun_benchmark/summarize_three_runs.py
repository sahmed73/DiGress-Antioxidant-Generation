#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from evaluate_vun import load_training_smiles

T_975_DF2 = 4.302652729696142


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--train-pt", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--replicate-size", type=int, default=10000)
    parser.add_argument("--replicates", type=int, default=3)
    return parser.parse_args()


def percentage(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator


def main():
    args = parse_args()
    expected = args.replicate_size * args.replicates
    with args.samples.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected:,} samples, found {len(rows):,}")
    indices = [int(row["sample_index"]) for row in rows]
    if indices != list(range(expected)):
        raise RuntimeError("Sample indices are not contiguous and ordered from zero")
    training_smiles = load_training_smiles(args.train_pt)
    replicate_records = []
    for replicate_index in range(args.replicates):
        start = replicate_index * args.replicate_size
        stop = start + args.replicate_size
        block = rows[start:stop]
        valid = [row["canonical_smiles"] for row in block if row["is_valid"] == "1"]
        unique_valid = set(valid)
        novel = unique_valid.difference(training_smiles)
        record = {
            "replicate": replicate_index + 1,
            "sample_start": start,
            "sample_stop_exclusive": stop,
            "raw_samples": len(block),
            "valid_samples": len(valid),
            "unique_valid_samples": len(unique_valid),
            "novel_unique_valid_samples": len(novel),
            "validity_percent": percentage(len(valid), len(block)),
            "uniqueness_percent": percentage(len(unique_valid), len(valid)),
            "novelty_percent": percentage(len(novel), len(unique_valid)),
        }
        replicate_records.append(record)
    metrics = {}
    for metric in ("validity", "uniqueness", "novelty"):
        values = [record[f"{metric}_percent"] for record in replicate_records]
        mean = statistics.mean(values)
        sample_sd = statistics.stdev(values)
        margin = T_975_DF2 * sample_sd / math.sqrt(args.replicates)
        metrics[metric] = {
            "replicate_percentages": values,
            "mean_percent": mean,
            "sample_standard_deviation_percent": sample_sd,
            "student_t_95_margin_percent": margin,
            "student_t_95_ci_percent": [mean - margin, mean + margin],
        }
    summary = {
        "protocol": {
            "checkpoint_runs": 1,
            "sampling_replicates": args.replicates,
            "samples_per_replicate": args.replicate_size,
            "total_samples": expected,
            "confidence_interval": "two-sided 95% Student t interval across sampling replicates; df=2",
            "interpretation": "sampling variability for one fixed checkpoint; not training-run variability",
        },
        "replicates": replicate_records,
        "metrics": metrics,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(replicate_records[0]))
        writer.writeheader()
        writer.writerows(replicate_records)
    for metric, result in metrics.items():
        low, high = result["student_t_95_ci_percent"]
        print(
            f"{metric.capitalize():10s} = {result['mean_percent']:.4f}% (95% t-CI {low:.4f}--{high:.4f}%)"
        )


if __name__ == "__main__":
    main()
