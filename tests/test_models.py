"""Tests for src/models/factory.py.

`pretrained=True` downloads weights, so every test here builds with `pretrained=False`.
Architecture, head shape, normalisation metadata and the freeze/unfreeze contract are all
observable without the weights; whether the download works is a Kaggle question, checked
by the first notebook run, not a unit test.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.models.factory import (
    ModelConfig,
    assert_fully_trainable,
    build_model,
    describe,
    freeze_backbone,
    input_size,
    load_model_config,
    normalisation,
    smoke_forward,
    trainable_fraction,
    unfreeze_all,
)

REPO = Path(__file__).resolve().parents[1]
CPU = ModelConfig(arch="resnet18", pretrained=False)


# ----------------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------------

def test_base_config_model_block_loads():
    cfg = load_model_config(REPO / "configs" / "base.yaml")
    assert cfg.arch == "resnet18" and cfg.pretrained is True
    assert cfg.freeze_warmup_epochs == 0


@pytest.mark.parametrize("arm,head,outputs", [
    ("arm_a", "softmax", 5), ("arm_b", "softmax", 5), ("arm_c", "softmax", 5),
    ("arm_d", "softmax", 5), ("arm_e", "ordinal_regression", 1),
    ("arm_f", "softmax", 5),
])
def test_every_arm_config_produces_a_valid_model_config(arm, head, outputs):
    cfg = load_model_config(REPO / "configs" / "base.yaml",
                            REPO / "configs" / f"{arm}.yaml")
    assert cfg.head == head and cfg.num_outputs == outputs


def test_a_softmax_head_with_the_wrong_width_is_refused():
    with pytest.raises(ValueError, match="one per DR grade"):
        ModelConfig(head="softmax", num_outputs=1)


def test_the_ordinal_head_must_be_single_output():
    """Arm E's whole point is one continuous output thresholded on validation (R3)."""
    with pytest.raises(ValueError, match="single continuous output"):
        ModelConfig(head="ordinal_regression", num_outputs=5)


def test_an_unknown_head_is_refused():
    with pytest.raises(ValueError, match="not one of"):
        ModelConfig(head="sigmoid_multilabel", num_outputs=5)


def test_a_long_freeze_is_refused_as_a_training_strategy():
    """freeze_warmup_epochs is a WARMUP. Frozen-only training is failure mode 2 and is
    not expressible through this config."""
    with pytest.raises(ValueError, match="WARMUP"):
        ModelConfig(freeze_warmup_epochs=30)


def test_an_unknown_model_key_is_refused(tmp_path):
    """A typo'd key that silently does nothing is how a config drifts away from the
    behaviour it claims to describe."""
    import yaml

    path = tmp_path / "typo.yaml"
    path.write_text(yaml.safe_dump({"model": {"architecture": "resnet18"}}),
                    encoding="utf-8")
    with pytest.raises(KeyError, match="unknown key"):
        load_model_config(path)


# ----------------------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------------------

def test_the_head_is_five_wide_and_the_forward_pass_matches():
    model = build_model(CPU)
    assert smoke_forward(model, 224, batch=2) == torch.Size([2, 5])


def test_the_ordinal_arm_produces_one_output():
    model = build_model(ModelConfig(arch="resnet18", pretrained=False,
                                    head="ordinal_regression", num_outputs=1))
    assert smoke_forward(model, 224, batch=3) == torch.Size([3, 1])


def test_efficientnet_b0_builds_too():
    """The primary architecture, not just the baseline."""
    model = build_model(ModelConfig(arch="efficientnet_b0", pretrained=False))
    assert smoke_forward(model, 224, batch=2) == torch.Size([2, 5])


def test_a_model_is_fully_trainable_on_return():
    """R7 / failure mode 2. A run that spends thirty epochs with a frozen backbone looks
    completely normal in the logs."""
    model = build_model(CPU)
    assert trainable_fraction(model) == 1.0
    assert_fully_trainable(model)


# ----------------------------------------------------------------------------------
# Normalisation comes from timm, never from a literal
# ----------------------------------------------------------------------------------

def test_normalisation_comes_from_the_model_not_a_constant():
    model = build_model(CPU)
    mean, std = normalisation(model)
    assert len(mean) == 3 and len(std) == 3
    assert all(0.0 < v < 1.0 for v in mean + std)


def test_normalisation_is_channel_asymmetric_so_channel_order_matters():
    """If mean were (0.5, 0.5, 0.5) an RGB/BGR swap would be invisible. It is not:
    ImageNet's per-channel means differ, so the order the Dataset delivers is load-bearing
    and a swap shifts every input systematically."""
    mean, std = normalisation(build_model(CPU))
    assert mean[0] != mean[2], "a symmetric mean would hide a channel swap"


def test_normalisation_raises_rather_than_falling_back_to_imagenet():
    class Bare(nn.Module):
        def forward(self, x):
            return x

    with pytest.raises(AttributeError, match="default_cfg"):
        normalisation(Bare())


def test_input_size_is_224():
    assert input_size(build_model(CPU)) == 224


# ----------------------------------------------------------------------------------
# Freeze / unfreeze
# ----------------------------------------------------------------------------------

def test_freeze_backbone_leaves_only_the_head_trainable():
    model = build_model(CPU)
    n = freeze_backbone(model)
    assert n > 0
    assert 0.0 < trainable_fraction(model) < 0.01
    head = model.get_classifier()
    assert all(p.requires_grad for p in head.parameters())


def test_a_frozen_model_fails_the_pre_epoch_assertion():
    model = build_model(CPU)
    freeze_backbone(model)
    with pytest.raises(RuntimeError, match="frozen-only training"):
        assert_fully_trainable(model)


def test_unfreeze_restores_everything():
    model = build_model(CPU)
    freeze_backbone(model)
    unfreeze_all(model)
    assert trainable_fraction(model) == 1.0
    assert_fully_trainable(model)


def test_describe_reports_the_normalisation_it_will_actually_use():
    """The run log has to carry the architecture and its normalisation, so a run's
    settings are recoverable from its own output (R6)."""
    model = build_model(CPU)
    text = describe(model, CPU)
    assert "resnet18" in text and "timm default_cfg" in text
    assert "100.0% trainable" in text
