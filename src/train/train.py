"""Train one arm. The only entry point for a training run.

    python -m src.train.train --arm A --config configs/kaggle.yaml \
        --cache-root /kaggle/input/datasets/rah098/fyp-dr-eyepacs-224/processed \
        --run-id phase3_baseline_resnet18 --epochs 8

WHAT THIS SCRIPT REFUSES TO DO
------------------------------
* **Touch the test set.** `load_arm_splits` returns three frames and only two are passed
  to `build_loaders`, which defaults `include_test=False`. Evaluating on test is a
  separate script run once, deliberately (R3).
* **Run without the leakage gate.** CLAUDE.md §7 requires `tests/test_no_leakage.py` to
  pass at the top of every training script. `--skip-gate` exists for a machine with no
  pytest and prints a warning that lands in the run log.
* **Start without writing down what it is doing.** `runs/<run_id>/config.yaml` is written
  before the first batch, so an interrupted run is still a documented one (R6).

Everything reported comes out of `runs/<run_id>/metrics.json` (R4).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.dataset import AugmentConfig
from src.data.manifest import class_counts, load_arm_splits, read_header
from src.data.sampler import build_loaders, describe_balance
from src.eval.metrics import compute_all
from src.models.factory import ModelConfig, build_model, describe, normalisation
from src.train.loop import evaluate, fit
from src.train.losses import build_loss
from src.train.schedulers import build_optimizer, cosine_with_warmup, preview_schedule

REPO = Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def merge(base: dict, *overlays: dict) -> dict:
    """Shallow-merge per top-level block, which is how the config files are structured."""
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for o in overlays:
        for k, v in o.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k].update(v)
            else:
                out[k] = v
    return out


def to_primitive(value):
    """Recursively coerce a value to plain builtins, for `yaml.safe_dump`.

    WHY THIS EXISTS (DECISION-025). `yaml.SafeDumper` dispatches on EXACT type, not
    `isinstance`, so anything that merely subclasses a builtin is refused:

        yaml.representer.RepresenterError: ('cannot represent an object', '2.10.0+cu128')

    `torch.__version__` is a `TorchVersion`, a `str` subclass. So is torchvision's. The
    fix is not to special-case torch: it is that **nothing reaches safe_dump unless it is
    a primitive**, because the next exotic type will be a numpy scalar from a config, a
    `Path`, or an enum from a library that has not been added yet.
    """
    import enum

    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, enum.Enum):
        return to_primitive(value.value)
    if isinstance(value, dict):
        return {to_primitive(k): to_primitive(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_primitive(v) for v in value]
    if isinstance(value, bool):          # before int: bool is an int subclass
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, str):           # str SUBCLASSES land here: TorchVersion, etc.
        return str(value)
    if hasattr(value, "item"):           # numpy / torch scalars
        try:
            return to_primitive(value.item())
        except Exception:                # noqa: BLE001
            pass
    return str(value)                    # Path, dataclass, anything else: legible, dumpable


def metrics_by_dataset(val_df, idx, y_true, y_pred, *, seed: int = 42,
                       bootstrap_n: int = 200) -> dict:
    """Per-origin metrics for a pooled arm, or {} when there is only one origin.

    DECISION-007: arm F's gain has to survive being split by source, because
    P(aptos | grade) runs 0.065 at grade 0 to 0.291 at grade 4 and "APTOS camera implies
    severe" is a shortcut worth taking. This is the only consumer of the index that
    `DRDataset.__getitem__` returns, and it is joined with `.iloc` — positional, per that
    method's contract.

    Split out of main() so it can be tested. Arm F is the last arm to run, so this would
    otherwise first execute at the end of the ablation.
    """
    rows = val_df.iloc[idx]
    origins = sorted(rows["dataset"].unique())
    if len(origins) < 2:
        return {}

    out = {}
    for ds in origins:
        mask = (rows["dataset"] == ds).to_numpy()
        out[str(ds)] = compute_all(y_true[mask], y_pred[mask], split=f"val[{ds}]",
                                   bootstrap_n=bootstrap_n, seed=seed)
    return out


def leakage_gate(skip: bool = False) -> None:
    """CLAUDE.md §7 — the leakage tests run before any training run, every time."""
    if skip:
        print("WARNING: leakage gate SKIPPED by --skip-gate. This run's provenance is "
              "weaker than every other run's; say so wherever its numbers appear.")
        return

    print("leakage gate: tests/test_no_leakage.py ...", flush=True)
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_no_leakage.py", "-q"],
                       cwd=REPO)
    if r.returncode != 0:
        raise SystemExit(
            "leakage tests FAILED — nothing trains until they are green (CLAUDE.md §7)"
        )


def build_parser() -> argparse.ArgumentParser:
    """The parser, separate from main(), so an invocation can be checked without
    running it (DECISION-022)."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arm", required=True, help="A B C D E or F")
    ap.add_argument("--config", type=Path, default=REPO / "configs/base.yaml",
                    help="environment overlay; base.yaml is always loaded underneath")
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--runs-root", type=Path, default=REPO / "runs")
    ap.add_argument("--epochs", type=int, help="override train.epochs")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--arch", help="override model.arch")
    ap.add_argument("--limit-train", type=int,
                    help="use only the first N TRAIN rows — smoke tests only, and it is "
                         "recorded in the run config so the result cannot be mistaken "
                         "for a full run")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--skip-gate", action="store_true")
    ap.add_argument("--splits-root", type=Path,
                    help="read split CSVs from here instead of data/splits/. For the "
                         "end-to-end smoke run only; recorded in the run config")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="build the model without pretrained weights. Smoke runs only - "
                         "it changes what the run measures, and it is recorded")
    ap.add_argument("--log-every", type=int, default=0)
    return ap


def main() -> int:
    ap = build_parser()
    args = ap.parse_args()

    arm = args.arm.upper()
    t_start = time.time()

    # ---- config ------------------------------------------------------------------
    cfg = load_yaml(REPO / "configs/base.yaml")
    if Path(args.config).resolve() != (REPO / "configs/base.yaml").resolve():
        cfg = merge(cfg, load_yaml(args.config))
    arm_file = REPO / f"configs/arm_{arm.lower()}.yaml"
    if not arm_file.exists():
        ap.error(f"no config for arm {arm} at {arm_file}")
    cfg = merge(cfg, load_yaml(arm_file))

    train_cfg = dict(cfg.get("train", {}))
    if args.epochs:
        train_cfg["epochs"] = args.epochs
    if args.batch_size:
        train_cfg["batch_size"] = args.batch_size
    if args.lr:
        train_cfg["lr"] = args.lr

    seed = args.seed if args.seed is not None else int(cfg.get("project", {}).get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    model_cfg = ModelConfig(**{k: v for k, v in cfg.get("model", {}).items()
                               if k in ModelConfig.__dataclass_fields__})
    if args.arch:
        model_cfg = ModelConfig(**{**model_cfg.__dict__, "arch": args.arch})
    if args.no_pretrained:
        model_cfg = ModelConfig(**{**model_cfg.__dict__, "pretrained": False})

    run_dir = Path(args.runs_root) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"RUN {args.run_id}   arm {arm}   seed {seed}")
    print("=" * 78)

    leakage_gate(args.skip_gate)

    # ---- data --------------------------------------------------------------------
    if args.splits_root:
        print(f"\n!! --splits-root {args.splits_root}: NOT the committed partition. "
              "This is a smoke run, not a result.")
    # test frame deliberately unused (R3)
    train_df, val_df, _test_df = load_arm_splits(arm, args.splits_root)
    if args.limit_train:
        train_df = train_df.head(args.limit_train).reset_index(drop=True)
        print(f"\n!! --limit-train {args.limit_train}: this is a SMOKE TEST, not a result")

    print(f"\ntrain {len(train_df):>6}   val {len(val_df):>6}   "
          f"(test withheld — R3)")
    print("train class counts:", class_counts(train_df).to_dict())
    print("val   class counts:", class_counts(val_df).to_dict())

    # ---- model -------------------------------------------------------------------
    model = build_model(model_cfg)
    mean, std = normalisation(model)
    print("\n" + describe(model, model_cfg))

    # ---- loaders -----------------------------------------------------------------
    imbalance = dict(cfg.get("imbalance", {}))
    aug = AugmentConfig.from_yaml(cfg)
    loaders = build_loaders(
        train_df, val_df, None, args.cache_root,
        sampler_kind=imbalance.get("sampler", "none"),
        batch_size=int(train_cfg.get("batch_size", 32)),
        num_workers=args.num_workers,
        seed=seed,
        mean=mean, std=std,
        augment=aug,
        image_size=int(cfg.get("preprocess", {}).get("image_size", 224)),
        pin_memory=bool(cfg.get("compute", {}).get("pin_memory", False)),
    )

    missing = loaders["train"].dataset.check_cache(limit=5)
    if missing:
        raise SystemExit(
            f"cache is incomplete, e.g. {missing}. Run "
            "`python -m src.data.reconcile_cache` — do not train against a partial cache."
        )

    print("\nbalance (natural vs drawn):")
    print(describe_balance(train_df, loaders["train"].sampler).to_string())

    # ---- loss, optimiser, schedule -----------------------------------------------
    criterion = build_loss(imbalance, train_df,
                           label_smoothing=float(train_cfg.get("label_smoothing", 0.1)))
    optimizer = build_optimizer(model, train_cfg.get("optimizer", "adamw"),
                                lr=float(train_cfg.get("lr", 3e-4)),
                                weight_decay=float(train_cfg.get("weight_decay", 1e-4)))
    epochs = int(train_cfg.get("epochs", 30))
    warmup = float(train_cfg.get("warmup_epochs", 2))
    warmup = min(warmup, max(0.0, epochs - 1))       # a short schedule keeps a real decay
    scheduler = cosine_with_warmup(optimizer, len(loaders["train"]), epochs, warmup)

    print(f"\nloss      : {type(criterion).__name__}")
    print(f"optimiser : {train_cfg.get('optimizer', 'adamw')} lr={train_cfg.get('lr')} "
          f"wd={train_cfg.get('weight_decay')}")
    print(f"schedule  : cosine, {warmup} warmup epoch(s) of {epochs}; LR per epoch:")
    print("            " + " ".join(
        f"{v:.2e}" for v in preview_schedule(len(loaders["train"]), epochs, warmup,
                                             float(train_cfg.get("lr", 3e-4)))))

    # ---- write the run config BEFORE training (R6) --------------------------------
    run_config = {
        "run_id": args.run_id,
        "arm": arm,
        "seed": seed,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": _git_sha(),
        "cache_root": str(args.cache_root),
        "limit_train": args.limit_train,
        "skip_gate": args.skip_gate,
        "splits_root": str(args.splits_root) if args.splits_root else None,
        "is_smoke_run": bool(args.splits_root or args.no_pretrained or args.limit_train),
        "model": model_cfg.__dict__,
        "train": train_cfg,
        "imbalance": imbalance,
        "augment": aug.__dict__,
        "normalisation": {"mean": list(mean), "std": list(std), "source": "timm default_cfg"},
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "split_provenance": {n: read_header(n, args.splits_root)
                             for n in ("train", "val")},
        "versions": _versions(),
    }
    # EVERYTHING is coerced, not just the versions. safe_dump refuses any type it does
    # not know exactly, and the run config is assembled from a dozen sources.
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(to_primitive(run_config), sort_keys=False), encoding="utf-8"
    )
    print(f"\nrun config -> {run_dir / 'config.yaml'}")

    # ---- train --------------------------------------------------------------------
    state = fit(
        model, loaders["train"], loaders["val"], criterion, optimizer, scheduler,
        epochs=epochs,
        run_dir=run_dir,
        device=args.device,
        amp=bool(cfg.get("compute", {}).get("amp", True)),
        grad_clip=float(train_cfg.get("grad_clip", 1.0)),
        patience=int(train_cfg.get("early_stopping", {}).get("patience", 7)),
        head=model_cfg.head,
        freeze_warmup_epochs=model_cfg.freeze_warmup_epochs,
        log_every=args.log_every,
    )

    # ---- final validation metrics (R4) --------------------------------------------
    dev = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu"
                       else "cpu")
    y_true, y_pred, idx = evaluate(model, loaders["val"], dev, head=model_cfg.head)
    metrics = compute_all(
        y_true, y_pred, split="val",
        referable_threshold=int(cfg.get("eval", {}).get("referable_threshold", 2)),
        bootstrap_n=int(cfg.get("eval", {}).get("bootstrap_n", 1000)),
        bootstrap_ci_level=float(cfg.get("eval", {}).get("bootstrap_ci", 0.95)),
        seed=seed,
    )
    metrics.update({
        "run_id": args.run_id,
        "arm": arm,
        "best_epoch": state.best_epoch,
        "best_val_qwk_during_training": state.best_qwk,
        "stopped_early": state.stopped_early,
        "stop_reason": state.stop_reason,
        "epochs_run": len(state.history),
        "wall_clock_minutes": (time.time() - t_start) / 60,
        "is_smoke_test": bool(args.limit_train or args.splits_root or args.no_pretrained),
    })

    # Arm F reports by source dataset — this is what the returned index is for.
    by_ds = metrics_by_dataset(loaders["val"].dataset.df, idx, y_true, y_pred, seed=seed)
    if by_ds:
        metrics["by_dataset"] = by_ds

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"val QWK           {metrics['qwk']:.4f}   "
          f"[{metrics['qwk_ci']['lo']:.4f}, {metrics['qwk_ci']['hi']:.4f}]")
    print(f"val accuracy      {metrics['accuracy']:.4f}")
    print(f"val balanced acc  {metrics['balanced_accuracy']:.4f}")
    print(f"referable sens    {metrics['referable']['sensitivity']:.4f}   "
          f"spec {metrics['referable']['specificity']:.4f}")
    print("\nconfusion matrix (rows = truth, cols = predicted):")
    for i, row in enumerate(metrics["confusion_matrix"]):
        print(f"  {i}  " + " ".join(f"{v:>6}" for v in row))
    if metrics["collapse"]["collapsed"]:
        print("\n*** COLLAPSE DETECTED — stop and diagnose (CLAUDE.md §2) ***")
        for r in metrics["collapse"]["reasons"]:
            print(f"    {r}")
    print(f"\nmetrics -> {run_dir / 'metrics.json'}")
    print("=" * 78)

    return 0


def _git_sha() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                           capture_output=True, text=True, check=True)
        return r.stdout.strip()
    except Exception:
        return "unknown"


def _versions() -> dict:
    import cv2
    import numpy
    import pandas
    import timm
    import torchvision
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "cv2": cv2.__version__,
        "cuda": torch.version.cuda or "cpu",
    }


if __name__ == "__main__":
    sys.exit(main())
