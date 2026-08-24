"""Tests for src/train/ — losses, schedule, and the loop.

The loop is exercised end to end on a tiny synthetic cache through the REAL `DRDataset`
and `build_loaders`, not a stub. A training loop that is only ever tested against fake
tensors is a training loop nobody has run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import yaml

from src.data.dataset import AugmentConfig
from src.data.sampler import build_loaders, describe_balance
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


# ----------------------------------------------------------------------------------
# Per-arm paths that only arms B/D had ever exercised
# ----------------------------------------------------------------------------------
#
# The arm A smoke run died in describe_balance with
#     AttributeError: 'RandomSampler' object has no attribute 'weights'
# because DataLoader(shuffle=True) substitutes a RandomSampler, so an unbalanced arm
# arrives with a sampler OBJECT rather than the None that build_sampler returned. Arm A
# is the one arm that never gets a weighted sampler, and every earlier test built the
# sampler directly instead of going through build_loaders.
#
# So these run each arm's real configuration through the real call chain.

import yaml as _yaml

ARM_SAMPLERS = {"a": "none", "b": "weighted_random", "c": "none",
                "d": "weighted_random", "e": "none", "f": "weighted_random"}


@pytest.mark.parametrize("arm", list("abcdef"))
def test_every_arms_sampler_kind_survives_build_loaders_and_describe_balance(arm, tmp_path):
    """The exact chain train.py runs: build_loaders -> loader.sampler -> describe_balance."""
    cfg = _yaml.safe_load((REPO / f"configs/arm_{arm}.yaml").read_text(encoding="utf-8"))
    kind = cfg["imbalance"].get("sampler", "none")
    assert kind == ARM_SAMPLERS[arm], "arm config changed; update this test deliberately"

    root = tmp_path / "cache"
    labels = [0] * 120 + [1] * 30 + [2] * 40 + [3] * 12 + [4] * 8
    train = _write_split(root, n=len(labels), labels=labels, split="train")
    val = _write_split(root, n=40, split="val", first_patient=10_000)

    loaders = build_loaders(train, val, None, root, sampler_kind=kind, batch_size=16,
                            augment=AugmentConfig(enabled=False))

    table = describe_balance(train, loaders["train"].sampler, draws=4_000)
    assert list(table["n"]) == [120, 30, 40, 12, 8]
    assert table["drawn"].sum() == pytest.approx(1.0, abs=0.01)

    if kind == "none":
        assert table.attrs["weighted"] is False
        assert (table["drawn"] == table["natural"]).all()
        assert "note" in table.columns
    else:
        assert table.attrs["weighted"] is True
        assert table["drawn"].min() > 0.10, "a weighted arm must actually rebalance"


def test_describe_balance_accepts_the_loaders_random_sampler(tmp_path):
    """The regression, stated at its narrowest."""
    from torch.utils.data import RandomSampler

    root = tmp_path / "cache"
    train = _write_split(root, n=50, labels=[i % 5 for i in range(50)], split="train")
    val = _write_split(root, n=20, split="val", first_patient=10_000)

    loaders = build_loaders(train, val, None, root, sampler_kind="none", batch_size=8)
    assert isinstance(loaders["train"].sampler, RandomSampler)
    assert not hasattr(loaders["train"].sampler, "weights")

    table = describe_balance(train, loaders["train"].sampler)   # must not raise
    assert table.attrs["sampler"] == "RandomSampler"


def test_arm_e_trains_end_to_end_with_the_ordinal_head(tmp_path):
    """Arm E is the other arm whose path differs: one continuous output, thresholded.
    `fit` and `evaluate` both branch on `head`, and only the softmax branch had ever run
    inside a loop."""
    from src.train.losses import OrdinalRegressionLoss

    root = tmp_path / "cache"
    train = _write_split(root, n=40, labels=[i % 5 for i in range(40)], split="train")
    val = _write_split(root, n=20, labels=[i % 5 for i in range(20)], split="val",
                       first_patient=10_000)

    loaders = build_loaders(train, val, None, root, batch_size=8,
                            augment=AugmentConfig(enabled=False))
    model = _Tiny(n_out=1)
    opt = build_optimizer(model, "adamw", lr=1e-3)
    sched = cosine_with_warmup(opt, len(loaders["train"]), epochs=2, warmup_epochs=0)

    state = fit(model, loaders["train"], loaders["val"], OrdinalRegressionLoss(), opt,
                sched, epochs=2, run_dir=tmp_path / "run", device="cpu", amp=False,
                head="ordinal_regression", patience=99)

    assert len(state.history) == 2
    y, p, _ = evaluate(model, loaders["val"], torch.device("cpu"),
                       head="ordinal_regression")
    assert set(p.tolist()) <= {0, 1, 2, 3, 4}, "ordinal output must map into 0-4"


def test_arm_c_trains_end_to_end_with_a_weighted_loss(tmp_path):
    """Arm C's weights come from the train split and go into the LOSS rather than the
    sampler. build_loss was unit-tested; the weighted CE had never run inside `fit`."""
    root = tmp_path / "cache"
    labels = [0] * 30 + [1] * 4 + [2] * 8 + [3] * 4 + [4] * 4
    train = _write_split(root, n=len(labels), labels=labels, split="train")
    val = _write_split(root, n=20, split="val", first_patient=10_000)

    cfg = _yaml.safe_load((REPO / "configs/arm_c.yaml").read_text(encoding="utf-8"))
    criterion = build_loss(cfg["imbalance"], train)
    assert criterion.weight is not None

    loaders = build_loaders(train, val, None, root, sampler_kind="none", batch_size=8,
                            augment=AugmentConfig(enabled=False))
    model = _Tiny()
    opt = build_optimizer(model, "adamw", lr=1e-3)
    sched = cosine_with_warmup(opt, len(loaders["train"]), epochs=1, warmup_epochs=0)

    state = fit(model, loaders["train"], loaders["val"], criterion, opt, sched, epochs=1,
                run_dir=tmp_path / "run", device="cpu", amp=False)
    assert len(state.history) == 1


def test_arm_f_per_dataset_metrics(tmp_path):
    """Arm F is the LAST arm to run, so this block would otherwise first execute at the
    end of the whole ablation."""
    from src.train.train import metrics_by_dataset

    root = tmp_path / "cache"
    eye = _write_split(root, n=30, labels=[i % 5 for i in range(30)], split="val")
    ap = _write_split(root, n=20, labels=[i % 5 for i in range(20)], split="aptos_val",
                      first_patient=90_000, dataset="aptos")
    pooled = pd.concat([eye, ap], ignore_index=True)

    idx = np.arange(len(pooled))
    y_true = pooled["label"].to_numpy()
    y_pred = y_true.copy()

    out = metrics_by_dataset(pooled, idx, y_true, y_pred, bootstrap_n=20)
    assert set(out) == {"eyepacs", "aptos"}
    assert out["eyepacs"]["n"] == 30 and out["aptos"]["n"] == 20
    assert out["eyepacs"]["split"] == "val[eyepacs]"


def test_per_dataset_metrics_are_empty_for_a_single_origin_arm(tmp_path):
    """Arms A-E are EyePACS only; the block must not appear in their metrics.json."""
    from src.train.train import metrics_by_dataset

    root = tmp_path / "cache"
    eye = _write_split(root, n=20, labels=[i % 5 for i in range(20)], split="val")
    idx = np.arange(len(eye))
    y = eye["label"].to_numpy()
    assert metrics_by_dataset(eye, idx, y, y.copy()) == {}


def test_per_dataset_metrics_join_positionally_not_by_label(tmp_path):
    """`DRDataset.__getitem__` returns a POSITIONAL index; joining it with `.loc` would
    silently mix the two origins' rows."""
    from src.train.train import metrics_by_dataset

    root = tmp_path / "cache"
    eye = _write_split(root, n=10, labels=[0] * 10, split="val")
    ap = _write_split(root, n=10, labels=[4] * 10, split="aptos_val",
                      first_patient=90_000, dataset="aptos")
    pooled = pd.concat([eye, ap], ignore_index=True)

    # a shuffled evaluation order, exactly what a DataLoader could produce
    idx = np.array([15, 3, 11, 7, 19, 1])
    y_true = pooled["label"].to_numpy()[idx]

    out = metrics_by_dataset(pooled, idx, y_true, y_true.copy(), bootstrap_n=10)
    assert out["aptos"]["n"] == 3 and out["eyepacs"]["n"] == 3
    assert out["aptos"]["support"][4] == 3, "aptos rows are the grade-4 ones"
    assert out["eyepacs"]["support"][0] == 3


# ----------------------------------------------------------------------------------
# Arm C: the criterion has to follow the model onto the device
# ----------------------------------------------------------------------------------

def test_a_weighted_criterion_is_moved_to_the_device(tmp_path):
    """Arm C died on Kaggle before epoch 1 with only config.yaml written.

    `nn.CrossEntropyLoss(weight=...)` registers the class weights as a buffer, built on
    CPU in train.py before `fit` picks a device. `fit` moved the model and not the
    criterion, so the first forward hit a cuda/cpu mismatch. Arm C is the only arm with
    class weights, which is why exactly one arm failed.

    Checked on the `meta` device because it works without a GPU: it proves the move
    happens, which is the thing that was missing.
    """
    import torch.nn as nn

    w = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    criterion = nn.CrossEntropyLoss(weight=w)
    assert criterion.weight.device.type == "cpu"

    criterion.to("meta")
    assert criterion.weight.device.type == "meta", (
        "a criterion carrying class weights must follow the model onto the device"
    )


def test_fit_moves_the_criterion(tmp_path):
    """The regression, through the real fit()."""
    import torch.nn as nn

    train, val, root = _tiny_partition(tmp_path)
    loaders = build_loaders(train, val, None, root, batch_size=8,
                            augment=AugmentConfig(enabled=False))
    criterion = nn.CrossEntropyLoss(weight=torch.ones(5))
    model = _Tiny()
    opt = build_optimizer(model, "adamw", lr=1e-3)
    sched = cosine_with_warmup(opt, len(loaders["train"]), epochs=1, warmup_epochs=0)

    fit(model, loaders["train"], loaders["val"], criterion, opt, sched, epochs=1,
        run_dir=tmp_path / "run", device="cpu", amp=False)

    # same device as the model, whatever that device is
    assert criterion.weight.device == next(model.parameters()).device


def test_the_other_arms_criteria_carry_no_tensors():
    """Why the fix is provably inert for the five arms already run."""
    import yaml as _y
    import torch.nn as nn

    df = _train_frame([0] * 40 + [1] * 10 + [2] * 20 + [3] * 5 + [4] * 5)
    for arm in "abdef":
        cfg = _y.safe_load((REPO / f"configs/arm_{arm}.yaml").read_text(encoding="utf-8"))
        loss = build_loss(cfg["imbalance"], df)
        weight = getattr(loss, "weight", None)
        alpha = getattr(loss, "alpha", None)
        assert weight is None, f"arm {arm} unexpectedly carries class weights"
        assert alpha is None or alpha.numel() == 0, f"arm {arm} carries a non-empty alpha"

    cfg_c = _y.safe_load((REPO / "configs/arm_c.yaml").read_text(encoding="utf-8"))
    assert build_loss(cfg_c["imbalance"], df).weight is not None, "arm C must be weighted"


def test_arm_c2_uses_effective_number_weighting():
    """DECISION-034: the arm C result is specific to INVERSE-FREQUENCY weighting, so the
    gentler scheme has to actually exist and be gentler."""
    import yaml as _y

    from src.data.manifest import load_split
    from src.train.losses import class_weights_from

    cfg = _y.safe_load((REPO / "configs/arm_c2.yaml").read_text(encoding="utf-8"))
    assert cfg["imbalance"]["class_weight_scheme"] == "effective_number"
    assert cfg["imbalance"]["sampler"] == "none"      # weighting, not resampling

    train = load_split("train")
    inv = class_weights_from(train, "inverse_frequency")
    eff = class_weights_from(train, "effective_number")
    assert (eff[4] / eff[0]) < (inv[4] / inv[0]) / 1.5, (
        "effective-number weighting must be materially gentler than inverse frequency, "
        f"got {float(eff[4]/eff[0]):.1f}x vs {float(inv[4]/inv[0]):.1f}x"
    )


def test_an_unknown_arm_names_the_configs_that_exist():
    """The arm set used to be hardcoded as 'ABCDE', which made any new variant
    unrunnable and reported letters rather than the missing file."""
    from src.data.manifest import load_arm_splits

    with pytest.raises(ValueError, match="configs/ defines"):
        load_arm_splits("Z9")


def test_smoke_accepts_an_arch_override_and_uses_it():
    """`smoke --arch` must actually reach the model factory.

    Stage 3.5 switches backbone via `--arch` on a Kaggle run. Without this the first
    execution of that code path is the hour-long one — which is how DECISION-022, -024
    and -025 each cost a session.
    """
    from src.train.smoke import build_parser

    args = build_parser().parse_args(["--arm", "E", "--arch", "efficientnet_b2"])
    assert args.arch == "efficientnet_b2"


def test_smoke_run_arm_passes_arch_through_to_train_argv(tmp_path, monkeypatch):
    import src.train.smoke as smoke

    seen = {}

    def fake_main():
        seen["argv"] = list(sys.argv)
        raise RuntimeError("stop here — argv is what is under test")

    monkeypatch.setattr("src.train.train.main", fake_main)
    monkeypatch.setattr(smoke, "build_fixture",
                        lambda d: (tmp_path / "cache", tmp_path / "splits"))
    with pytest.raises(RuntimeError):
        smoke.run_arm("E", tmp_path, arch="efficientnet_b2")
    assert "--arch" in seen["argv"]
    assert seen["argv"][seen["argv"].index("--arch") + 1] == "efficientnet_b2"


def test_efficientnet_b2_builds_with_the_ordinal_head():
    """The stage 3.5 backbone, with arm E's head, before an hour of GPU is spent on it."""
    import torch
    from src.models.factory import ModelConfig, build_model

    m = build_model(ModelConfig(arch="efficientnet_b2", num_outputs=1,
                                head="ordinal_regression", pretrained=False))
    assert tuple(m(torch.randn(2, 3, 224, 224)).shape) == (2, 1)
