"""The end-to-end test: the REAL `src.train.train.main()`, start to finish. DECISION-025.

Three consecutive Kaggle sessions died inside `main()` on code no unit test reached:

    1. describe_balance   'RandomSampler' object has no attribute 'weights'
    2. yaml.safe_dump     cannot represent an object '2.10.0+cu128'

Both would have been caught in seconds by running `main()` once on anything. Everything
`main()` calls was green the whole time; `main()` itself had never executed.

These are the slowest tests in the suite, at roughly a minute for the pair, and they are
worth every second of it. Arm A and arm F are chosen because they are the two most
different paths: A has no sampler and one origin, F has a weighted sampler and two, and
between them they touch every branch in `main()`. Arms B-E are covered at unit level in
`test_train.py` and by `python -m src.train.smoke --all-arms`, which is what to run before
a real session.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.train.smoke import build_fixture, run_arm
from src.train.train import to_primitive

REPO = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------------
# main(), end to end
# ----------------------------------------------------------------------------------

@pytest.mark.parametrize("arm", ["A", "F"])
def test_train_main_runs_end_to_end(arm, tmp_path):
    """Arm A: no sampler, one origin. Arm F: weighted sampler, two origins."""
    metrics = run_arm(arm, tmp_path, epochs=1, keep=True)

    assert metrics["arm"] == arm
    assert metrics["n"] > 0
    assert len(metrics["confusion_matrix"]) == 5
    assert metrics["is_smoke_test"] is True
    assert "qwk" in metrics and "qwk_ci" in metrics


def test_the_run_config_is_valid_yaml_with_the_real_versions(tmp_path):
    """THE regression. torch.__version__ is a TorchVersion, a str SUBCLASS, and
    yaml.SafeDumper dispatches on exact type rather than isinstance."""
    run_arm("A", tmp_path, epochs=1, keep=True)

    cfg_path = tmp_path / "A" / "runs" / "smoke_a" / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    assert cfg["versions"]["torch"], "torch version missing from the dumped config"
    assert isinstance(cfg["versions"]["torch"], str)
    assert cfg["is_smoke_run"] is True
    assert cfg["split_provenance"]["train"], "provenance headers must survive the dump"
    assert cfg["normalisation"]["source"] == "timm default_cfg"


def test_arm_f_metrics_carry_the_per_origin_breakdown(tmp_path):
    """DECISION-007: arm F's gain has to survive being split by source."""
    metrics = run_arm("F", tmp_path, epochs=1, keep=True)
    assert "by_dataset" in metrics
    assert set(metrics["by_dataset"]) == {"eyepacs", "aptos"}


def test_arm_a_metrics_have_no_per_origin_breakdown(tmp_path):
    metrics = run_arm("A", tmp_path, epochs=1, keep=True)
    assert "by_dataset" not in metrics


# ----------------------------------------------------------------------------------
# to_primitive — so a different exotic type cannot recur
# ----------------------------------------------------------------------------------

def test_every_runtime_derived_value_dumps():
    """The versions block built from the live interpreter, dumped for real."""
    from src.train.train import _versions

    yaml.safe_dump(to_primitive(_versions()))       # must not raise


def test_a_str_subclass_is_coerced():
    class Weird(str):
        pass

    with pytest.raises(yaml.representer.RepresenterError):
        yaml.safe_dump({"v": Weird("2.10.0+cu128")})
    assert yaml.safe_load(yaml.safe_dump(to_primitive({"v": Weird("2.10.0+cu128")})))["v"] \
        == "2.10.0+cu128"


def test_numpy_scalars_are_coerced():
    import numpy as np

    payload = {"i": np.int64(7), "f": np.float32(0.5), "b": np.bool_(True)}
    with pytest.raises(yaml.representer.RepresenterError):
        yaml.safe_dump(payload)

    out = yaml.safe_load(yaml.safe_dump(to_primitive(payload)))
    assert out == {"i": 7, "f": 0.5, "b": True}


def test_paths_and_nested_structures_are_coerced():
    payload = {"p": Path("/kaggle/working"), "nested": {"list": [Path("a"), (1, 2)]},
               "set": {3, 1}}
    out = yaml.safe_load(yaml.safe_dump(to_primitive(payload)))
    assert isinstance(out["p"], str)
    assert isinstance(out["nested"]["list"][0], str)
    assert out["nested"]["list"][1] == [1, 2]
    assert sorted(out["set"]) == [1, 3]


def test_plain_values_survive_unchanged():
    payload = {"a": 1, "b": 2.5, "c": "x", "d": True, "e": None, "f": [1, "y"]}
    assert to_primitive(payload) == payload


def test_a_dataclass_config_block_dumps():
    """model_cfg.__dict__ and augment.__dict__ both go into the run config."""
    from src.data.dataset import AugmentConfig
    from src.models.factory import ModelConfig

    payload = {"model": ModelConfig().__dict__, "augment": AugmentConfig().__dict__}
    out = yaml.safe_load(yaml.safe_dump(to_primitive(payload)))
    assert out["model"]["arch"] == "resnet18"
    assert out["augment"]["scale"] == [0.9, 1.1]        # tuple -> list


# ----------------------------------------------------------------------------------
# The fixture itself
# ----------------------------------------------------------------------------------

def test_the_fixture_is_patient_disjoint(tmp_path):
    """A smoke fixture that leaked would teach the pipeline nothing (R1)."""
    from src.data.manifest import load_split

    _cache, splits = build_fixture(tmp_path)
    frames = {n: load_split(n, splits)
              for n in ("train", "val", "test", "aptos_train", "aptos_val", "aptos_test")}

    roles = {"train": ["train", "aptos_train"], "val": ["val", "aptos_val"],
             "test": ["test", "aptos_test"]}
    ids = {r: set().union(*(set(frames[n]["patient_id"]) for n in names))
           for r, names in roles.items()}

    assert not ids["train"] & ids["val"]
    assert not ids["train"] & ids["test"]
    assert not ids["val"] & ids["test"]


def test_the_fixture_has_all_five_classes_in_every_split(tmp_path):
    """Otherwise the sampler refuses it and the smoke would fail for the wrong reason."""
    from src.data.manifest import class_counts, load_split

    _cache, splits = build_fixture(tmp_path)
    for name in ("train", "val", "test", "aptos_train", "aptos_val", "aptos_test"):
        counts = class_counts(load_split(name, splits))
        assert (counts > 0).all(), f"{name} is missing a class: {counts.to_dict()}"


def test_the_fixture_cache_matches_what_cache_relpath_expects(tmp_path):
    """If the fixture wrote paths the real code would not look for, the smoke would pass
    against a shape the pipeline never sees."""
    from src.data.manifest import load_split
    from src.data.preprocess import cache_relpath

    cache, splits = build_fixture(tmp_path)
    for name in ("train", "aptos_train"):
        df = load_split(name, splits)
        for row in df.head(5).itertuples(index=False):
            assert (cache / cache_relpath(row.image_path, row.dataset)).exists()
