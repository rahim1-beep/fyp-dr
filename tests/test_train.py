"""Tests for src/train/ — losses, schedule, and the loop.

The loop is exercised end to end on a tiny synthetic cache through the REAL `DRDataset`
and `build_loaders`, not a stub. A training loop that is only ever tested against fake
tensors is a training loop nobody has run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import yaml

from src.data.dataset import AugmentConfig
from src.data.sampler import build_loaders
from src.models.factory import ModelConfig, build_model
from src.train.losses import (
    FocalLoss,
    OrdinalRegressionLoss,
    build_loss,
    class_weights_from,
    predictions_from,
)
from src.train.loop import evaluate, fit, train_one_epoch
from src.train.schedulers import (
    build_optimizer,
    cosine_with_warmup,
    lr_of,
    preview_schedule,
)
from src.train.train import merge
from tests.test_dataset import _write_split

REPO = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------------
# Class weights — train split only (R2)
# ----------------------------------------------------------------------------------

def _train_frame(labels):
    return pd.DataFrame({"label": labels, "split": "train",
                         "image_path": [f"data/data/{i}_left.jpeg" for i in range(len(labels))],
                         "patient_id": [str(i) for i in range(len(labels))],
                         "dataset": "eyepacs"})


def test_class_weights_are_inverse_frequency_and_mean_one():
    """Normalised to mean 1 so arm C changes the BALANCE between classes without also
    changing the overall gradient scale — otherwise it is running a different effective
    learning rate from arm A and the comparison confounds the two."""
    df = _train_frame([0] * 400 + [1] * 100 + [2] * 200 + [3] * 50 + [4] * 25)
    w = class_weights_from(df)

    assert w.shape == (5,)
    assert float(w.mean()) == pytest.approx(1.0)
    assert w[4] > w[0], "the rarest class must get the largest weight"
    assert float(w[4] / w[0]) == pytest.approx(400 / 25, rel=1e-5)


def test_effective_number_weighting_is_gentler_than_inverse_frequency():
    """On a 36:1 imbalance raw inverse frequency hands grade 4 a weight ~36x grade 0 and
    the loss becomes a few hundred images shouting."""
    df = _train_frame([0] * 3600 + [1] * 400 + [2] * 800 + [3] * 200 + [4] * 100)
    inv = class_weights_from(df, "inverse_frequency")
    eff = class_weights_from(df, "effective_number")
    assert (inv[4] / inv[0]) > (eff[4] / eff[0])


def test_class_weights_refuse_a_split_with_a_missing_class():
    df = _train_frame([0] * 10 + [1] * 5 + [2] * 5 + [3] * 5)
    with pytest.raises(ValueError, match="no examples of class"):
        class_weights_from(df)


# ----------------------------------------------------------------------------------
# build_loss
# ----------------------------------------------------------------------------------

@pytest.mark.parametrize("arm", list("abcdef"))
def test_every_arm_config_builds_its_loss(arm):
    cfg = yaml.safe_load((REPO / f"configs/arm_{arm}.yaml").read_text(encoding="utf-8"))
    df = _train_frame([0] * 40 + [1] * 10 + [2] * 20 + [3] * 5 + [4] * 5)
    loss = build_loss(cfg["imbalance"], df)
    assert isinstance(loss, nn.Module)


def test_class_weights_plus_sampler_is_refused():
    """Both correct the same imbalance. Together they double-count it and no one can
    attribute the result to either — which is the entire point of arms C and D."""
    with pytest.raises(ValueError, match="double-count"):
        build_loss({"arm": "X", "loss": "cross_entropy", "class_weights": True,
                    "sampler": "weighted_random"}, _train_frame([0, 1, 2, 3, 4] * 5))


def test_weighted_loss_requires_the_train_frame():
    with pytest.raises(ValueError, match="TRAIN split"):
        build_loss({"loss": "cross_entropy", "class_weights": True, "sampler": "none"})


def test_arm_a_gets_label_smoothing():
    cfg = yaml.safe_load((REPO / "configs/arm_a.yaml").read_text(encoding="utf-8"))
    loss = build_loss(cfg["imbalance"], None, label_smoothing=0.1)
    assert isinstance(loss, nn.CrossEntropyLoss)
    assert loss.label_smoothing == 0.1
    assert loss.weight is None


def test_arm_e_gets_the_ordinal_loss():
    cfg = yaml.safe_load((REPO / "configs/arm_e.yaml").read_text(encoding="utf-8"))
    assert isinstance(build_loss(cfg["imbalance"], None), OrdinalRegressionLoss)


def test_an_unknown_loss_raises():
    with pytest.raises(ValueError, match="expected one of"):
        build_loss({"loss": "hinge"})


# ----------------------------------------------------------------------------------
# Focal / ordinal behaviour
# ----------------------------------------------------------------------------------

def test_focal_down_weights_the_easy_examples():
    """The property focal loss exists for: confident-correct predictions contribute far
    less than they would under plain CE."""
    logits = torch.tensor([[8.0, 0.0, 0.0, 0.0, 0.0]])     # very confident, correct
    target = torch.tensor([0])

    ce = nn.CrossEntropyLoss()(logits, target)
    focal = FocalLoss(gamma=2.0)(logits, target)
    assert float(focal) < float(ce) / 10


def test_focal_barely_touches_a_hard_example():
    logits = torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.0]])     # nearly uniform
    target = torch.tensor([4])
    ce = nn.CrossEntropyLoss()(logits, target)
    focal = FocalLoss(gamma=2.0)(logits, target)
    assert float(focal) > 0.4 * float(ce)


def test_ordinal_loss_penalises_by_distance():
    """Softmax CE scores 'predicted 0, truth 4' the same as 'predicted 3, truth 4'. This
    is arm E's whole reason to exist."""
    loss = OrdinalRegressionLoss("smooth_l1")
    near = loss(torch.tensor([[3.0]]), torch.tensor([4]))
    far = loss(torch.tensor([[0.0]]), torch.tensor([4]))
    assert float(far) > float(near)


def test_negative_focal_gamma_is_refused():
    with pytest.raises(ValueError, match="gamma"):
        FocalLoss(gamma=-1.0)


# ----------------------------------------------------------------------------------
# predictions_from
# ----------------------------------------------------------------------------------

def test_softmax_predictions_are_the_argmax():
    out = torch.tensor([[0.1, 0.2, 5.0, 0.0, 0.1], [9.0, 0, 0, 0, 0]])
    assert predictions_from(out, "softmax").tolist() == [2, 0]


def test_ordinal_predictions_cut_at_the_midpoints_by_default():
    out = torch.tensor([[-0.4], [0.6], [1.7], [2.5], [3.9], [99.0]])
    assert predictions_from(out, "ordinal_regression").tolist() == [0, 1, 2, 3, 4, 4]


def test_ordinal_thresholds_are_overridable_for_phase_4():
    """Arm E's cut points get optimised on VALIDATION (R3); the default is the honest
    un-tuned starting point, not a tuned one."""
    out = torch.tensor([[0.9]])
    assert predictions_from(out, "ordinal_regression").item() == 1
    assert predictions_from(out, "ordinal_regression", [1.2, 2.2, 3.2, 4.2]).item() == 0


# ----------------------------------------------------------------------------------
# Optimiser and schedule
# ----------------------------------------------------------------------------------

def test_norms_and_biases_are_excluded_from_weight_decay():
    model = build_model(ModelConfig(arch="resnet18", pretrained=False))
    opt = build_optimizer(model, "adamw", lr=3e-4, weight_decay=1e-4)
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["weight_decay"] == 1e-4
    assert opt.param_groups[1]["weight_decay"] == 0.0
    assert len(opt.param_groups[1]["params"]) > 0


def test_the_schedule_warms_up_then_decays():
    lrs = preview_schedule(steps_per_epoch=50, epochs=10, warmup_epochs=2, lr=1e-3)
    assert lrs[0] < lrs[1] < lrs[2]                 # ramping
    assert lrs[2] == pytest.approx(1e-3, rel=0.02)  # peak at the end of warmup
    assert all(a >= b for a, b in zip(lrs[2:], lrs[3:]))   # then monotonically down
    assert lrs[-1] < 0.1 * lrs[2]


def test_the_schedule_never_reaches_exactly_zero():
    """A schedule that hits zero spends its last epoch not training, and the final
    checkpoint is identical to the one before it."""
    lrs = preview_schedule(steps_per_epoch=20, epochs=6, warmup_epochs=1, lr=1e-3)
    assert lrs[-1] > 0


def test_the_schedule_steps_per_batch_not_per_epoch():
    model = nn.Linear(4, 5)
    opt = build_optimizer(model, "adamw", lr=1e-3)
    sched = cosine_with_warmup(opt, steps_per_epoch=100, epochs=5, warmup_epochs=1)

    start = lr_of(opt)
    for _ in range(10):
        opt.step()
        sched.step()
    assert lr_of(opt) != start, "LR did not move within an epoch — this is a staircase"


def test_a_warmup_as_long_as_the_run_is_refused():
    opt = build_optimizer(nn.Linear(2, 2), "adamw")
    with pytest.raises(ValueError, match="never decays"):
        cosine_with_warmup(opt, 10, epochs=3, warmup_epochs=3)


def test_an_empty_loader_is_refused():
    opt = build_optimizer(nn.Linear(2, 2), "adamw")
    with pytest.raises(ValueError, match="loader is empty"):
        cosine_with_warmup(opt, 0, epochs=3)


# ----------------------------------------------------------------------------------
# The loop, end to end through the real Dataset
# ----------------------------------------------------------------------------------

class _Tiny(nn.Module):
    """A real but minuscule network, so an epoch is milliseconds."""

    def __init__(self, n_out=5):
        super().__init__()
        self.net = nn.Sequential(nn.AdaptiveAvgPool2d(4), nn.Flatten(),
                                 nn.Linear(3 * 16, n_out))

    def forward(self, x):
        return self.net(x)


def _tiny_partition(tmp_path, n_train=40, n_val=20):
    root = tmp_path / "cache"
    labels = [i % 5 for i in range(n_train)]
    train = _write_split(root, n=n_train, labels=labels, split="train", first_patient=0)
    val = _write_split(root, n=n_val, labels=[i % 5 for i in range(n_val)],
                       split="val", first_patient=10_000)
    return train, val, root


def test_one_epoch_runs_and_reports_train_metrics(tmp_path):
    train, val, root = _tiny_partition(tmp_path)
    loaders = build_loaders(train, val, None, root, batch_size=8,
                            augment=AugmentConfig(enabled=False))
    model = _Tiny()
    opt = build_optimizer(model, "adamw", lr=1e-3)
    sched = cosine_with_warmup(opt, len(loaders["train"]), epochs=2, warmup_epochs=0)

    out = train_one_epoch(model, loaders["train"], nn.CrossEntropyLoss(), opt, sched,
                          torch.device("cpu"))
    assert set(out) >= {"train_loss", "train_qwk", "train_acc"}
    assert np.isfinite(out["train_loss"])


def test_fit_writes_a_log_a_checkpoint_and_a_history(tmp_path):
    train, val, root = _tiny_partition(tmp_path)
    loaders = build_loaders(train, val, None, root, batch_size=8,
                            augment=AugmentConfig(enabled=False))
    model = _Tiny()
    opt = build_optimizer(model, "adamw", lr=1e-3)
    sched = cosine_with_warmup(opt, len(loaders["train"]), epochs=3, warmup_epochs=0)
    run_dir = tmp_path / "run"

    state = fit(model, loaders["train"], loaders["val"], nn.CrossEntropyLoss(), opt,
                sched, epochs=3, run_dir=run_dir, device="cpu", amp=False, patience=99)

    assert len(state.history) == 3
    assert (run_dir / "train_log.csv").exists()
    assert (run_dir / "history.json").exists()
    assert (run_dir / "best.pth").exists()

    rows = pd.read_csv(run_dir / "train_log.csv")
    assert len(rows) == 3
    assert {"epoch", "val_qwk", "val_balanced_acc", "val_collapsed"} <= set(rows.columns)


def test_the_log_is_written_every_epoch_not_at_the_end(tmp_path):
    """A session killed at epoch 20 must leave 20 epochs of evidence, not nothing."""
    train, val, root = _tiny_partition(tmp_path)
    loaders = build_loaders(train, val, None, root, batch_size=8,
                            augment=AugmentConfig(enabled=False))
    model = _Tiny()
    opt = build_optimizer(model, "adamw", lr=1e-3)
    sched = cosine_with_warmup(opt, len(loaders["train"]), epochs=2, warmup_epochs=0)
    run_dir = tmp_path / "run"

    seen = []

    class Spy(nn.CrossEntropyLoss):
        def forward(self, *a, **k):
            seen.append((run_dir / "train_log.csv").exists())
            return super().forward(*a, **k)

    fit(model, loaders["train"], loaders["val"], Spy(), opt, sched, epochs=2,
        run_dir=run_dir, device="cpu", amp=False, patience=99)

    # by the time the second epoch is running, epoch 0's row is already on disk
    assert seen[-1] is True


def test_early_stopping_fires_on_val_qwk(tmp_path):
    """Not on loss and not on accuracy (§5). On a 73.5%-grade-0 val set accuracy peaks
    early for a model learning to ignore the rare classes."""
    train, val, root = _tiny_partition(tmp_path)
    loaders = build_loaders(train, val, None, root, batch_size=8,
                            augment=AugmentConfig(enabled=False))
    model = _Tiny()
    opt = build_optimizer(model, "adamw", lr=0.0)      # frozen: QWK cannot improve
    sched = cosine_with_warmup(opt, len(loaders["train"]), epochs=20, warmup_epochs=0)

    state = fit(model, loaders["train"], loaders["val"], nn.CrossEntropyLoss(), opt,
                sched, epochs=20, run_dir=tmp_path / "run", device="cpu", amp=False,
                patience=2)

    assert state.stopped_early
    assert "val QWK did not improve" in state.stop_reason
    assert len(state.history) < 20


def test_the_best_checkpoint_is_restored_at_the_end(tmp_path):
    train, val, root = _tiny_partition(tmp_path)
    loaders = build_loaders(train, val, None, root, batch_size=8,
                            augment=AugmentConfig(enabled=False))
    model = _Tiny()
    opt = build_optimizer(model, "adamw", lr=1e-2)
    sched = cosine_with_warmup(opt, len(loaders["train"]), epochs=4, warmup_epochs=0)
    run_dir = tmp_path / "run"

    state = fit(model, loaders["train"], loaders["val"], nn.CrossEntropyLoss(), opt,
                sched, epochs=4, run_dir=run_dir, device="cpu", amp=False, patience=99)

    saved = torch.load(run_dir / "best.pth", map_location="cpu", weights_only=False)
    assert saved["epoch"] == state.best_epoch
    for k, v in model.state_dict().items():
        assert torch.equal(v, saved["state_dict"][k]), (
            "the in-memory model is not the best checkpoint — the final evaluation would "
            "measure the LAST epoch rather than the selected one"
        )


def test_a_collapse_is_recorded_in_the_history(tmp_path):
    """Visible at epoch 3, not at the end of a two-hour session (§2)."""
    root = tmp_path / "cache"
    train = _write_split(root, n=40, labels=[i % 5 for i in range(40)], split="train")
    val = _write_split(root, n=25, labels=[0] * 21 + [1, 2, 3, 4], split="val",
                       first_patient=10_000)
    loaders = build_loaders(train, val, None, root, batch_size=8,
                            augment=AugmentConfig(enabled=False))

    class AlwaysZero(nn.Module):
        """Predicts grade 0 for everything — the collapse the detection rule exists for."""

        def __init__(self):
            super().__init__()
            self.dead = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            out = torch.zeros(x.shape[0], 5, device=x.device)
            out[:, 0] = 10.0
            return out + self.dead * x.mean()

    model = AlwaysZero()
    opt = build_optimizer(nn.Linear(2, 2), "adamw", lr=0.0)
    sched = cosine_with_warmup(opt, len(loaders["train"]), epochs=1, warmup_epochs=0)

    state = fit(model, loaders["train"], loaders["val"], nn.CrossEntropyLoss(), opt,
                sched, epochs=1, run_dir=tmp_path / "run", device="cpu", amp=False)

    assert state.history[0]["val_collapsed"] is True
    assert "never predicted" in state.history[0]["val_collapse_reasons"]


def test_a_non_finite_loss_stops_the_run(tmp_path):
    """NaN weights follow within a few steps and every metric after that is meaningless."""
    train, val, root = _tiny_partition(tmp_path)
    loaders = build_loaders(train, val, None, root, batch_size=8,
                            augment=AugmentConfig(enabled=False))

    class NanLoss(nn.Module):
        def forward(self, out, target):
            return out.sum() * float("nan")

    model = _Tiny()
    opt = build_optimizer(model, "adamw", lr=1e-3)
    sched = cosine_with_warmup(opt, len(loaders["train"]), epochs=1, warmup_epochs=0)

    with pytest.raises(RuntimeError, match="non-finite loss|loss became"):
        train_one_epoch(model, loaders["train"], NanLoss(), opt, sched,
                        torch.device("cpu"))


def test_evaluate_returns_indices_that_join_back_to_manifest_rows(tmp_path):
    """`loader.dataset.df.iloc[idx]` is the documented join. Arm F's per-origin metrics
    depend on it being right."""
    train, val, root = _tiny_partition(tmp_path)
    loaders = build_loaders(train, val, None, root, batch_size=8,
                            augment=AugmentConfig(enabled=False))

    y, p, idx = evaluate(_Tiny(), loaders["val"], torch.device("cpu"))
    assert len(y) == len(p) == len(idx) == len(val)

    rows = loaders["val"].dataset.df.iloc[idx]
    assert rows["label"].tolist() == y.tolist()


def test_fit_never_receives_a_test_loader(tmp_path):
    """R3, structurally: `fit` has no test argument at all."""
    import inspect

    assert "test" not in inspect.signature(fit).parameters


# ----------------------------------------------------------------------------------
# Config merge
# ----------------------------------------------------------------------------------

def test_config_merge_is_per_block_so_an_arm_only_states_what_it_changes():
    base = {"train": {"epochs": 30, "lr": 3e-4}, "model": {"arch": "resnet18"}}
    arm = {"model": {"head": "softmax", "num_outputs": 5}}
    out = merge(base, arm)
    assert out["train"]["epochs"] == 30
    assert out["model"] == {"arch": "resnet18", "head": "softmax", "num_outputs": 5}
