"""The checkpoint convention, tested against checkpoints the real training loop wrote.

WHY THIS FILE EXISTS. `notebooks/phase5_gradcam.py` loaded weights with

    model.load_state_dict(state["model"] if "model" in state else state)

`"model"` is a key that has never existed in this project — `src/train/loop.py` has always
written `"state_dict"`. The fallback then handed `load_state_dict` the whole outer dict
(`epoch`, `val_qwk`, `state_dict`), which fails with a RuntimeError that reads like an
architecture mismatch rather than like a wrong key.

Every test passed, because **not one of them loaded a checkpoint the real training loop
had written**. `tests/test_train.py` checked the SAVE side; the eval loaders happened to
be right; the notebook was never executed locally at all. So the tests here go through
`src/train/smoke.py`, which runs the real `train.main()` end to end on a synthetic
fixture and writes a real `best.pth` — the artefact, not a hand-built dict that would
just re-assert whatever convention the test author had in mind.

DECISION-049.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from src.models.factory import (CHECKPOINT_STATE_KEY, ModelConfig, build_model,
                                load_checkpoint)

SRC = Path(__file__).resolve().parents[1] / "src"
NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"


@pytest.fixture(scope="module")
def real_checkpoint(tmp_path_factory) -> Path:
    """A `best.pth` written by the real training loop, via the real `train.main()`."""
    from src.train.smoke import run_arm

    work = tmp_path_factory.mktemp("ckpt")
    run_arm("E", work, epochs=1, keep=True)
    found = sorted(work.rglob("best.pth"))
    assert found, "the smoke run produced no checkpoint"
    return found[0]


def test_the_training_loop_writes_the_documented_keys(real_checkpoint):
    raw = torch.load(real_checkpoint, map_location="cpu", weights_only=False)
    assert isinstance(raw, dict)
    assert CHECKPOINT_STATE_KEY in raw
    assert {"epoch", "val_qwk"} <= set(raw)


def test_the_key_the_notebook_guessed_does_not_exist(real_checkpoint):
    """`"model"` was invented. If it ever becomes real, this test should be revisited
    deliberately rather than the two conventions quietly coexisting."""
    raw = torch.load(real_checkpoint, map_location="cpu", weights_only=False)
    assert "model" not in raw


def test_the_shared_loader_restores_a_real_checkpoint(real_checkpoint):
    """Exactly what the Phase 5 notebook does: rebuild from the run's OWN config, then
    load the weights."""
    run_dir = real_checkpoint.parent
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    model = build_model(ModelConfig(
        arch=cfg["model"]["arch"], num_outputs=cfg["model"]["num_outputs"],
        head=cfg["model"]["head"], pretrained=False))

    ckpt = load_checkpoint(real_checkpoint, model)

    assert ckpt["epoch"] is not None
    saved = ckpt[CHECKPOINT_STATE_KEY]
    for k, v in model.state_dict().items():
        assert torch.equal(v, saved[k]), f"{k} was not restored"


def test_the_loaded_model_runs_a_forward_pass(real_checkpoint):
    """Loading without erroring is not the same as loading correctly."""
    run_dir = real_checkpoint.parent
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    model = build_model(ModelConfig(
        arch=cfg["model"]["arch"], num_outputs=cfg["model"]["num_outputs"],
        head=cfg["model"]["head"], pretrained=False))
    load_checkpoint(real_checkpoint, model)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, cfg["model"]["num_outputs"])
    assert torch.isfinite(out).all()


def test_the_loader_returns_the_dict_without_a_model(real_checkpoint):
    ckpt = load_checkpoint(real_checkpoint)
    assert CHECKPOINT_STATE_KEY in ckpt


def test_a_wrong_key_raises_naming_what_was_actually_found(tmp_path):
    """The failure the notebook produced read like an architecture mismatch. It should
    say 'you used the wrong key' and list the keys that are there."""
    bad = tmp_path / "bad.pth"
    torch.save({"epoch": 3, "model": {"w": torch.zeros(1)}}, bad)
    with pytest.raises(KeyError) as e:
        load_checkpoint(bad)
    msg = str(e.value)
    assert CHECKPOINT_STATE_KEY in msg
    assert "epoch" in msg and "model" in msg, "the message must name what it found"


def test_a_bare_state_dict_is_refused_rather_than_half_working(tmp_path):
    bare = tmp_path / "bare.pth"
    torch.save({"w": torch.zeros(1)}, bare)
    with pytest.raises(KeyError):
        load_checkpoint(bare)


def test_nothing_loads_a_checkpoint_except_through_the_shared_loader():
    """The structural countermeasure: one convention, one definition.

    Three call sites each hardcoded the key; two were right and one was invented. A
    fourth site would have been a fourth coin flip, so `load_state_dict` on a checkpoint
    is confined to `src/models/factory.load_checkpoint`.
    """
    offenders = []
    for path in list(SRC.rglob("*.py")) + list(NOTEBOOKS.rglob("*.py")):
        if path.name == "factory.py":
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or "load_state_dict" not in stripped:
                continue
            offenders.append(f"{path.name}:{i}: {stripped}")
    assert not offenders, (
        "load a checkpoint with src.models.factory.load_checkpoint, which owns the "
        "convention:\n  " + "\n  ".join(offenders)
    )
