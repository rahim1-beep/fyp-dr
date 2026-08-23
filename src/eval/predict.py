"""Write a run's RAW validation outputs to `runs/<run_id>/val_outputs.npz`.

WHY THIS EXISTS. Threshold optimisation needs the model's **continuous** output per
image. `metrics.json` stores only integer predictions — the confusion matrix — so no
operating point can be recovered from the committed artefacts. Arm E's cut points, and
every arm's sensitivity/specificity curve, were unreachable without re-running inference.

Two ways in, and the project needs both:

  * **automatic**, from `train.py`, for every run from now on. `save_val_outputs` is
    called on the restored best checkpoint, so the outputs describe the model that was
    SELECTED rather than the one left in memory at the last epoch.
  * **backfill**, via this CLI, for the five Phase 4 arms that already ran. It reloads
    `best.pth`, rebuilds the model from the run's own `config.yaml`, and re-runs
    validation. Inference only: no training, no GPU strictly required, a couple of
    minutes for the whole ablation.

The file is small — 5,268 floats for an ordinal head, 5,268x5 for a softmax one, about
100-200 KB — so it is a tracked artefact rather than a regenerable one. Once it exists,
every threshold and operating-point question is answerable locally, forever, with no
checkpoint and no GPU.

    python -m src.eval.predict --run-dir runs/phase4_arm_e_resnet18 \\
        --cache-root /kaggle/working/cache/processed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]

OUTPUTS_FILE = "val_outputs.npz"


def save_val_outputs(run_dir: Path, outputs: np.ndarray, y_true: np.ndarray,
                     indices: np.ndarray, head: str) -> Path:
    """Write the npz. `indices` are POSITIONAL into the val manifest (DRDataset's
    contract), so predictions can be joined back with `.iloc`."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / OUTPUTS_FILE
    np.savez_compressed(
        path,
        outputs=np.asarray(outputs, dtype=np.float32),
        y_true=np.asarray(y_true, dtype=np.int64),
        indices=np.asarray(indices, dtype=np.int64),
        head=np.array(head),
    )
    return path


def predict_from_checkpoint(run_dir: Path, cache_root: Path, *, device: str = "auto",
                            batch_size: int = 64, num_workers: int = 0) -> dict:
    """Reload a finished run's best checkpoint and re-run validation.

    The model is rebuilt from the run's OWN `config.yaml`, not from the current defaults:
    a run analysed under a different architecture or head than it was trained with would
    produce numbers about a model that never existed.
    """
    from src.data.dataset import AugmentConfig  # noqa: F401  (kept explicit for clarity)
    from src.data.manifest import load_arm_splits
    from src.data.sampler import build_loaders
    from src.models.factory import ModelConfig, build_model, normalisation
    from src.train.loop import evaluate

    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    ckpt_path = run_dir / "best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"{ckpt_path} does not exist. The checkpoint is not fetched by "
            "src.data.fetch_run (it is gitignored and large), so this has to run where "
            "the checkpoint is — on Kaggle, with the notebook's output mounted."
        )

    model_cfg = ModelConfig(**{k: v for k, v in cfg["model"].items()
                               if k in ModelConfig.__dataclass_fields__})
    # Never download weights here: the checkpoint supplies them.
    model_cfg = ModelConfig(**{**model_cfg.__dict__, "pretrained": False})
    model = build_model(model_cfg)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    print(f"  loaded epoch {ckpt.get('epoch')} (val QWK {ckpt.get('val_qwk'):.4f})")

    mean, std = normalisation(model)
    _train, val_df, _test = load_arm_splits(cfg["arm"])

    loaders = build_loaders(_train, val_df, None, cache_root,
                            sampler_kind="none", batch_size=batch_size,
                            num_workers=num_workers, mean=mean, std=std,
                            image_size=int(cfg.get("preprocess", {}).get("image_size", 224)))

    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                       else ("cpu" if device == "auto" else device))
    model.to(dev)

    y_true, y_pred, idx, raw = evaluate_with_raw(model, loaders["val"], dev,
                                                 head=model_cfg.head)
    path = save_val_outputs(run_dir, raw, y_true, idx, model_cfg.head)
    return {"run_dir": str(run_dir), "n": int(len(y_true)), "head": model_cfg.head,
            "path": str(path), "epoch": ckpt.get("epoch")}


@torch.no_grad()
def evaluate_with_raw(model, loader, device, head: str = "softmax"):
    """Like `loop.evaluate`, but also returns the raw model output per image.

    Kept here rather than changing `evaluate`'s return signature, which the training loop
    calls every epoch and which several tests pin.
    """
    from src.train.losses import predictions_from

    model.eval()
    ys, ps, ix, raws = [], [], [], []
    for images, labels, idx in loader:
        images = images.to(device, non_blocking=True)
        out = model(images).float()
        ys.append(labels.numpy())
        ps.append(predictions_from(out, head).cpu().numpy())
        ix.append(idx.numpy())
        raws.append(out.cpu().numpy())

    return (np.concatenate(ys), np.concatenate(ps), np.concatenate(ix),
            np.concatenate(raws))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run-dir", type=Path, required=True, action="append",
                    help="runs/<run_id>/ holding config.yaml and best.pth; repeatable")
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0)
    return ap


def main() -> int:
    args = build_parser().parse_args()

    failures = []
    for run_dir in args.run_dir:
        print(f"\n--- {run_dir.name} ---")
        try:
            info = predict_from_checkpoint(run_dir, args.cache_root,
                                           device=args.device,
                                           batch_size=args.batch_size,
                                           num_workers=args.num_workers)
            print(f"  wrote {info['path']}  ({info['n']} images, head {info['head']})")
        except Exception as exc:                 # noqa: BLE001 - reported, not hidden
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            failures.append(run_dir.name)

    print()
    if failures:
        print(f"FAILED for {failures}")
        return 1
    print(f"OK - {len(args.run_dir)} run(s) now carry {OUTPUTS_FILE}")
    print("Threshold analysis is now a local, GPU-free step:")
    print("  python -m src.eval.thresholds --run-dir runs/<run_id>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
