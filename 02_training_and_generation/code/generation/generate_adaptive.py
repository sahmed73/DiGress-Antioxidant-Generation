import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config
import generate

EMA_DECAY = 0.6
MARGINAL_EPS = 0.02
PATIENCE = 3
MAX_ITERS = 750
MAX_SAMPLES = 1500000
SAT_FIELDS = [
    "iter",
    "n_sampled",
    "valid",
    "uniq_phenolic",
    "local_new",
    "cum_local_unique",
    "new_and_novel",
    "cum_novel",
    "local_novelty_rate",
    "global_novelty_frac",
    "marginal_yield",
    "ema_marginal",
    "low_streak",
]
GEN_FIELDS = ["smiles", "iter", "is_locally_new", "is_globally_novel"]


def ts():
    return time.strftime("[%H:%M:%S]")


def load_previous_smiles(csv_paths):
    seen = set()
    for p in csv_paths:
        p = Path(p)
        if not p.exists():
            print(f"WARNING: --resume file not found, skipping: {p}")
            continue
        with open(p, newline="") as f:
            reader = csv.DictReader(f)
            if "smiles" not in (reader.fieldnames or []):
                print(f"WARNING: no 'smiles' column in {p}, skipping")
                continue
            for row in reader:
                s = row["smiles"].strip()
                if s:
                    seen.add(s)
        print(f"{ts()} Loaded {len(seen):,} prior SMILES from {p.name}")
    return seen


def append_iter_rows(f, writer, uniq_phenolic, local_new_set, novel_flag, it):
    for s in uniq_phenolic:
        writer.writerow(
            {
                "smiles": s,
                "iter": it,
                "is_locally_new": s in local_new_set,
                "is_globally_novel": novel_flag[s],
            }
        )
    f.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=config.CHECKPOINT_PATH)
    parser.add_argument(
        "--resume",
        nargs="+",
        metavar="CSV",
        help="One or more prior generated_adaptive_*.csv files. Their SMILES pre-populate `seen` so this run only collects molecules not already generated.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.checkpoint:
        print("ERROR: set CHECKPOINT_PATH in config.py or pass --checkpoint")
        sys.exit(1)
    t0 = time.time()
    print(f"{ts()} {'=' * 64}")
    print(f"{ts()} DiGress Adaptive Generation — Phenolic Unconditional")
    print(f"{ts()} Checkpoint : {args.checkpoint}")
    if args.resume:
        print(f"{ts()} Resume mode: {len(args.resume)} prior CSV(s)")
    print(
        f"{ts()} EMA_DECAY={EMA_DECAY}  MARGINAL_EPS={MARGINAL_EPS}  PATIENCE={PATIENCE}  MAX_ITERS={MAX_ITERS}  MAX_SAMPLES={MAX_SAMPLES:,}",
        flush=True,
    )
    seen = set()
    if args.resume:
        seen = load_previous_smiles(args.resume)
        print(
            f"{ts()} Resume: {len(seen):,} molecules pre-loaded into seen set — this run will only collect molecules not in any prior run.",
            flush=True,
        )
    model = generate.load_model(args.checkpoint)
    print(f"{ts()} Model loaded", flush=True)
    train_fps = generate.build_train_fps()
    print(f"{ts()} Train FP reference: {len(train_fps):,}", flush=True)
    out_dir = Path(config.OUTPUT_DIR)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    ts_tag = time.strftime("%Y%m%d_%H%M%S")
    gen_csv = out_dir / f"generated_adaptive_{ts_tag}.csv"
    sat_csv = out_dir / f"saturation_{ts_tag}.csv"
    if not args.dry_run:
        with open(sat_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=SAT_FIELDS).writeheader()
        gen_fh = open(gen_csv, "w", newline="")
        gen_writer = csv.DictWriter(gen_fh, fieldnames=GEN_FIELDS)
        gen_writer.writeheader()
        gen_fh.flush()
    bs = config.GENERATE_BATCH_SIZE
    cum_novel = 0
    ema = None
    low_streak = 0
    n_sampled = 0
    stop_reason = "max_iters"
    print(f"\n{ts()} {'-' * 64}")
    print(f"{ts()} Sampling adaptively...", flush=True)
    try:
        for it in range(1, MAX_ITERS + 1):
            samples = generate.sample_batch(model, bs)
            raw = generate.samples_to_smiles(samples, model)
            valid = generate.validate_smiles(raw)
            phenolic = generate.ao_filter(valid)
            n_sampled += bs
            uniq_phenolic = list(dict.fromkeys(phenolic))
            novel_flag = {s: generate.is_novel(s, train_fps) for s in uniq_phenolic}
            local_new_list = [s for s in uniq_phenolic if s not in seen]
            local_new_set = set(local_new_list)
            seen.update(local_new_set)
            new_and_novel = [s for s in local_new_list if novel_flag[s]]
            cum_novel += len(new_and_novel)
            global_novel_batch = sum(novel_flag.values())
            denom = max(1, len(uniq_phenolic))
            local_novelty_rate = len(local_new_list) / denom
            global_novelty_frac = global_novel_batch / denom
            marginal_yield = len(new_and_novel) / bs
            ema = (
                marginal_yield
                if ema is None
                else EMA_DECAY * ema + (1 - EMA_DECAY) * marginal_yield
            )
            if ema < MARGINAL_EPS:
                low_streak += 1
            else:
                low_streak = 0
            print(
                f"{ts()} iter={it} | valid={len(valid)} uniq_phen={len(uniq_phenolic)} local_new={len(local_new_list)} new&novel={len(new_and_novel)} cum_novel={cum_novel} | local_nov={local_novelty_rate:.3f} global_nov={global_novelty_frac:.3f} marg={marginal_yield:.4f} ema={ema:.4f} streak={low_streak}",
                flush=True,
            )
            if not args.dry_run:
                with open(sat_csv, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=SAT_FIELDS).writerow(
                        {
                            "iter": it,
                            "n_sampled": n_sampled,
                            "valid": len(valid),
                            "uniq_phenolic": len(uniq_phenolic),
                            "local_new": len(local_new_list),
                            "cum_local_unique": len(seen),
                            "new_and_novel": len(new_and_novel),
                            "cum_novel": cum_novel,
                            "local_novelty_rate": round(local_novelty_rate, 4),
                            "global_novelty_frac": round(global_novelty_frac, 4),
                            "marginal_yield": round(marginal_yield, 5),
                            "ema_marginal": round(ema, 5),
                            "low_streak": low_streak,
                        }
                    )
                append_iter_rows(
                    gen_fh, gen_writer, uniq_phenolic, local_new_set, novel_flag, it
                )
            if low_streak >= PATIENCE:
                stop_reason = "saturated"
                break
            if n_sampled >= MAX_SAMPLES:
                stop_reason = "max_samples"
                break
            if args.dry_run and it >= 2:
                stop_reason = "dry_run"
                break
    finally:
        if not args.dry_run:
            gen_fh.close()
    elapsed = (time.time() - t0) / 60
    print(f"\n{ts()} {'=' * 64}")
    print(f"{ts()} STOP ({stop_reason}) after {n_sampled:,} sampled")
    print(
        f"{ts()} Collected {cum_novel:,} new unique novel phenolics THIS run in {elapsed:.1f}min"
    )
    if not args.dry_run:
        print(f"{ts()} SMILES     -> {gen_csv}")
        print(f"{ts()} Saturation -> {sat_csv}")
    if args.resume:
        print(f"{ts()} To chain another run: --resume {gen_csv}")


if __name__ == "__main__":
    main()
