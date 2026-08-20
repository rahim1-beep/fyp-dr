"""The training loop: one epoch of fitting, one of evaluation, early stopping on val QWK.

Load-bearing choices, all of them rules from CLAUDE.md rather than preferences:

**Early stopping and checkpoint selection watch validation QWK, not loss and not
accuracy** (§5). On a 73.5%-grade-0 validation set, accuracy peaks early for a model that
is learning to ignore the rare classes, and loss can fall while QWK falls with it.

**The test set is never opened here** (R3). `evaluate` takes whatever loader it is given,
and `fit` is only ever handed train and val. There is no `test` argument.

**The collapse check runs every epoch, on validation** (§2). A run that has collapsed
should be visible at epoch 3, not at the end of a 2-hour session — and it is recorded in
the history rather than only printed.

**Everything is written as it happens.** `train_log.csv` is appended per epoch and the
best checkpoint is saved when it improves, so a session killed at epoch 20 leaves 20
usable epochs of evidence rather than nothing.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.eval.metrics import (
    accuracy,
    balanced_accuracy,
    detect_collapse,
    quadratic_weighted_kappa,
)
from src.train.losses import predictions_from
from src.train.schedulers import lr_of


@dataclass
class TrainState:
    """Everything the loop needs to survive being interrupted and inspected."""

    best_qwk: float = -2.0          # QWK can be negative; -2 is below any real value
    best_epoch: int = -1
    epochs_without_improvement: int = 0
    history: list[dict] = field(default_factory=list)
    stopped_early: bool = False
    stop_reason: str = ""


def _device(prefer: str = "auto") -> torch.device:
    if prefer == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(prefer)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    head: str = "softmax",
    thresholds: list[float] | None = None,
    amp: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the model over a loader. Returns (y_true, y_pred, indices).

    The indices come back so predictions can be joined to manifest rows
    (`loader.dataset.df.iloc[idx]`) — arm F needs metrics split by source dataset, and
    error analysis needs to know which image a mistake was on.
    """
    model.eval()
    ys, ps, ix = [], [], []

    autocast = torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda")
    for images, labels, idx in loader:
        images = images.to(device, non_blocking=True)
        with autocast:
            out = model(images)
        preds = predictions_from(out.float(), head, thresholds)
        ys.append(labels.numpy())
        ps.append(preds.cpu().numpy())
        ix.append(idx.numpy())

    return np.concatenate(ys), np.concatenate(ps), np.concatenate(ix)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    *,
    scaler: torch.amp.GradScaler | None = None,
    grad_clip: float = 1.0,
    head: str = "softmax",
    log_every: int = 0,
) -> dict:
    """One pass over the training loader. Returns loss and train-set metrics.

    The train metrics are computed on the predictions made DURING the epoch, while the
    weights were still changing, so they are not a clean measurement of the final model.
    They are here to show the loop is learning at all, and they are labelled `train_` so
    nobody mistakes them for a result.
    """
    model.train()
    total_loss, n_batches = 0.0, 0
    ys, ps = [], []
    t0 = time.time()

    for step, (images, labels, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,
                            enabled=scaler is not None and device.type == "cuda"):
            out = model(images)
            loss = criterion(out, labels)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"loss became {loss.item()} at step {step}. Training on a non-finite loss "
                "produces NaN weights within a few steps and every metric afterwards is "
                "meaningless — stop and diagnose."
            )

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()          # per STEP, not per epoch — see schedulers.py

        total_loss += float(loss.detach())
        n_batches += 1
        ys.append(labels.detach().cpu().numpy())
        ps.append(predictions_from(out.detach().float(), head).cpu().numpy())

        if log_every and step % log_every == 0:
            print(f"    step {step:>5}/{len(loader)}  loss {float(loss):.4f}  "
                  f"lr {lr_of(optimizer):.2e}", flush=True)

    y, p = np.concatenate(ys), np.concatenate(ps)
    return {
        "train_loss": total_loss / max(1, n_batches),
        "train_qwk": quadratic_weighted_kappa(y, p),
        "train_acc": accuracy(y, p),
        "train_seconds": time.time() - t0,
    }


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    *,
    epochs: int,
    run_dir: Path,
    device: str = "auto",
    amp: bool = True,
    grad_clip: float = 1.0,
    patience: int = 7,
    head: str = "softmax",
    freeze_warmup_epochs: int = 0,
    log_every: int = 0,
) -> TrainState:
    """Train, selecting on validation QWK. The test set is not an argument here (R3)."""
    from src.models.factory import assert_fully_trainable, freeze_backbone, unfreeze_all

    dev = _device(device)
    model.to(dev)

    use_amp = amp and dev.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp) if use_amp else None

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.csv"
    ckpt_path = run_dir / "best.pth"

    state = TrainState()

    if freeze_warmup_epochs:
        freeze_backbone(model)
        print(f"backbone frozen for {freeze_warmup_epochs} warmup epoch(s)")

    print(f"device: {dev}  |  amp: {use_amp}  |  epochs: {epochs}  |  "
          f"patience: {patience} (on val QWK)")

    for epoch in range(epochs):
        if freeze_warmup_epochs and epoch == freeze_warmup_epochs:
            unfreeze_all(model)
            print(f"epoch {epoch}: backbone unfrozen — full fine-tune from here")
        if epoch >= freeze_warmup_epochs:
            # R7 / failure mode 2. Cheap, and it catches a warmup that never ended.
            assert_fully_trainable(model)

        tr = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, dev,
                             scaler=scaler, grad_clip=grad_clip, head=head,
                             log_every=log_every)

        y_true, y_pred, _ = evaluate(model, val_loader, dev, head=head, amp=use_amp)
        collapse = detect_collapse(y_true, y_pred)

        row = {
            "epoch": epoch,
            "lr": lr_of(optimizer),
            **tr,
            "val_qwk": quadratic_weighted_kappa(y_true, y_pred),
            "val_acc": accuracy(y_true, y_pred),
            "val_balanced_acc": balanced_accuracy(y_true, y_pred),
            "val_collapsed": collapse.collapsed,
            "val_collapse_reasons": "; ".join(collapse.reasons),
        }
        state.history.append(row)

        improved = row["val_qwk"] > state.best_qwk
        if improved:
            state.best_qwk = row["val_qwk"]
            state.best_epoch = epoch
            state.epochs_without_improvement = 0
            torch.save({"epoch": epoch, "val_qwk": row["val_qwk"],
                        "state_dict": model.state_dict()}, ckpt_path)
        else:
            state.epochs_without_improvement += 1

        print(
            f"epoch {epoch:>3}  loss {row['train_loss']:.4f}  "
            f"val_qwk {row['val_qwk']:.4f}  val_acc {row['val_acc']:.4f}  "
            f"val_bal {row['val_balanced_acc']:.4f}  lr {row['lr']:.2e}  "
            f"{tr['train_seconds'] / 60:.1f}m"
            + ("  <- best" if improved else "")
            + (f"  COLLAPSE: {collapse}" if collapse.collapsed else ""),
            flush=True,
        )

        # Append after every epoch, so an interrupted session still leaves its evidence.
        _append_row(log_path, row)

        if state.epochs_without_improvement >= patience:
            state.stopped_early = True
            state.stop_reason = (
                f"val QWK did not improve for {patience} epochs "
                f"(best {state.best_qwk:.4f} at epoch {state.best_epoch})"
            )
            print(f"early stop: {state.stop_reason}")
            break

    if state.best_epoch >= 0 and ckpt_path.exists():
        best = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(best["state_dict"])
        print(f"restored best checkpoint: epoch {best['epoch']}, "
              f"val QWK {best['val_qwk']:.4f}")

    (run_dir / "history.json").write_text(
        json.dumps(state.history, indent=2), encoding="utf-8"
    )
    return state


def _append_row(path: Path, row: dict) -> None:
    import csv

    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            w.writeheader()
        w.writerow(row)
