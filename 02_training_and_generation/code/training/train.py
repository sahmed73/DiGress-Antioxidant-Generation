import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config

DIGRESS_ROOT = config.DIGRESS_ROOT
DIGRESS_SRC = config.DIGRESS_SRC
DIGRESS_CONF = config.DIGRESS_CONF
DATA_DIR = config.DATA_DIR
RUNS_DIR = config.RUNS_DIR
DIGRESS_DATA_RAW = DIGRESS_ROOT / "data" / "ao_phenolic_bde" / "raw"


def ts():
    return time.strftime("[%H:%M:%S]")


def check_prerequisites():
    errors = []
    for s in ("train", "val", "test"):
        p = DATA_DIR / f"{s}.pt"
        if not p.exists():
            errors.append(f"  Missing: {p}")
    if errors:
        print(f"{ts()} ERROR:")
        [print(e) for e in errors]
        sys.exit(1)
    print(f"{ts()} Prerequisites OK")


def install_dataset():
    dst = DIGRESS_SRC / "datasets" / "ao_phenolic_dataset.py"
    shutil.copy2(HERE / "dataset.py", dst)
    print(f"{ts()} Installed dataset.py → {dst}")
    dst_yaml = DIGRESS_CONF / "dataset" / "ao_phenolic_bde.yaml"
    shutil.copy2(HERE / "ao_phenolic_bde.yaml", dst_yaml)
    print(f"{ts()} Installed ao_phenolic_bde.yaml → {dst_yaml}")


def patch_main_py():
    main_py = DIGRESS_SRC / "main.py"
    bak = main_py.with_suffix(".py.bak")
    if not bak.exists():
        shutil.copy2(main_py, bak)
    if "ao_phenolic_dataset" in bak.read_text():
        print(
            f"{ts()} WARNING: backup main.py.bak is already patched — not pristine. Restore a clean main.py before continuing."
        )
        return
    shutil.copy2(bak, main_py)
    text = main_py.read_text()
    old = '    else:\n        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))'
    new = '    elif dataset_config["name"] == "ao_phenolic_bde":\n        from datasets.ao_phenolic_dataset import (\n            AOPhenolicDataModule, AOPhenolicInfos)\n        from metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics\n        from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete\n        from diffusion.extra_features import ExtraFeatures, DummyExtraFeatures\n        from diffusion.extra_features_molecular import ExtraMolecularFeatures\n        from analysis.visualization import MolecularVisualization\n        datamodule    = AOPhenolicDataModule(cfg)\n        dataset_infos = AOPhenolicInfos(datamodule, cfg)\n        if cfg.model.extra_features is not None:\n            extra_features  = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)\n            domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)\n        else:\n            extra_features  = DummyExtraFeatures()\n            domain_features = DummyExtraFeatures()\n        dataset_infos.compute_input_output_dims(\n            datamodule=datamodule,\n            extra_features=extra_features,\n            domain_features=domain_features)\n        train_metrics    = TrainMolecularMetricsDiscrete(dataset_infos)\n        sampling_metrics = SamplingMolecularMetrics(dataset_infos, None)\n        visualization_tools = MolecularVisualization(\n            cfg.dataset.remove_h, dataset_infos=dataset_infos)\n        model_kwargs = {\n            "dataset_infos": dataset_infos,\n            "train_metrics": train_metrics,\n            "sampling_metrics": sampling_metrics,\n            "visualization_tools": visualization_tools,\n            "extra_features": extra_features,\n            "domain_features": domain_features,\n        }\n\n    else:\n        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))'
    if old not in text:
        print(
            f"{ts()} WARNING: patch target not found in main.py — check DiGress version"
        )
        return
    main_py.write_text(text.replace(old, new))
    print(f"{ts()} Patched main.py with ao_phenolic_bde branch")


def setup_data(dry_run):
    processed = DIGRESS_DATA_RAW.parent / "processed"
    if processed.exists():
        shutil.rmtree(processed)
        print(f"{ts()} Cleared processed cache")
    DIGRESS_DATA_RAW.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        src = (DATA_DIR / f"{split}.pt").resolve()
        dst = DIGRESS_DATA_RAW / f"{split}.pt"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if dry_run:
            import torch

            data = torch.load(src, weights_only=False)
            torch.save(data[:200], dst)
        else:
            os.symlink(src, dst)
        print(f"{ts()} {('Copied(dry)' if dry_run else 'Linked')} {dst.name}")


DEFAULTS = {
    "epochs": 500,
    "lr": 0.0001,
    "batch_size": 16,
    "T": 500,
    "n_layers": 6,
    "hidden_dim": 256,
}


def build_run_name(tag):
    parts = [tag, f"e{config.EPOCHS}"]
    if config.LR != DEFAULTS["lr"]:
        parts.append(f"lr{config.LR:.0e}")
    if config.BATCH_SIZE != DEFAULTS["batch_size"]:
        parts.append(f"bs{config.BATCH_SIZE}")
    if config.T != DEFAULTS["T"]:
        parts.append(f"t{config.T}")
    if config.N_LAYERS != DEFAULTS["n_layers"]:
        parts.append(f"L{config.N_LAYERS}")
    if config.HIDDEN_DIM != DEFAULTS["hidden_dim"]:
        parts.append(f"h{config.HIDDEN_DIM}")
    return "_".join(parts)


def build_overrides(run_name, run_dir, dry_run):
    return [
        "dataset=ao_phenolic_bde",
        f"general.name={run_name}",
        "general.wandb=disabled",
        f"general.gpus={config.GPUS}",
        f"model.n_layers={config.N_LAYERS}",
        f"model.diffusion_steps={config.T}",
        f"model.hidden_dims.dx={config.HIDDEN_DIM}",
        f"model.hidden_dims.de={max(32, config.HIDDEN_DIM // 4)}",
        f"model.hidden_dims.dy={max(32, config.HIDDEN_DIM // 4)}",
        f"model.hidden_mlp_dims.X={config.HIDDEN_DIM}",
        f"model.hidden_mlp_dims.E={max(64, config.HIDDEN_DIM // 2)}",
        f"model.hidden_mlp_dims.y={max(64, config.HIDDEN_DIM // 2)}",
        f"model.hidden_dims.n_head={config.N_HEADS}",
        f"train.n_epochs={(2 if dry_run else config.EPOCHS)}",
        f"train.batch_size={config.BATCH_SIZE}",
        f"train.lr={config.LR}",
        f"train.weight_decay={config.WEIGHT_DECAY}",
        f"train.clip_grad={config.GRAD_CLIP}",
        f"train.num_workers={config.NUM_WORKERS}",
        f"train.seed={config.RANDOM_SEED}",
        f"train.save_model={('False' if dry_run else 'True')}",
        f"general.check_val_every_n_epochs={(1 if dry_run else config.VAL_EVERY)}",
        f"general.samples_to_generate={(32 if dry_run else 512)}",
        f"general.sample_every_val={config.SAMPLE_EVERY_VAL}",
        f"general.log_every_steps={config.LOG_EVERY}",
        "general.chains_to_save=0",
        "general.final_model_chains_to_save=0",
        "general.val_check_interval=null",
        f"hydra.run.dir={run_dir}",
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    t0 = time.time()
    run_name = build_run_name(args.tag)
    run_dir = RUNS_DIR / run_name
    print(f"{ts()} {'=' * 62}")
    print(f"{ts()} DiGress Training — Phenolic Unconditional")
    print(f"{ts()} run_name={run_name}")
    print(f"{ts()} EPOCHS={config.EPOCHS}  LR={config.LR}")
    print(f"{ts()} {'=' * 62}", flush=True)
    check_prerequisites()
    run_dir.mkdir(parents=True, exist_ok=True)
    install_dataset()
    patch_main_py()
    setup_data(args.dry_run)
    overrides = build_overrides(run_name, run_dir, args.dry_run)
    cmd = f"cd {DIGRESS_ROOT} && python src/main.py " + " ".join(overrides)
    json.dump(
        {
            "run_name": run_name,
            "tag": args.tag,
            "epochs": config.EPOCHS,
            "lr": config.LR,
            "dry_run": args.dry_run,
        },
        open(run_dir / "config_snapshot.json", "w"),
        indent=2,
    )
    print(f"\n{ts()} Run dir : {run_dir}")
    print(f"{ts()} Launching DiGress training...", flush=True)
    rc = 0
    try:
        proc = subprocess.run(cmd, shell=True, check=False)
        rc = proc.returncode
    except KeyboardInterrupt:
        rc = 1
    elapsed = time.time() - t0
    status = "SUCCESS" if rc == 0 else f"FAILED (exit {rc})"
    json.dump(
        {"status": status, "elapsed_minutes": round(elapsed / 60, 2)},
        open(run_dir / "run_summary.json", "w"),
        indent=2,
    )
    print(f"\n{ts()} {status} — {elapsed / 60:.1f} min")
    if rc != 0:
        sys.exit(rc)


if __name__ == "__main__":
    main()
