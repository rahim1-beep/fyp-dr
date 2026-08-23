"""Tests for src/data/fetch_run.py.

`--from-dir` exists partly so these can exercise everything except the network call: a
fake kernel-output tree goes in, and the finding, verifying and copying all run for real.
The download itself is one function with one job, and it is the only part not covered.

The checks that matter here are the cross-file ones. Three artefacts that each parse
individually but describe different runs is the failure no per-file check can see, and it
is exactly what mixing up two arms in Phase 4 would produce.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.data.fetch_run import (
    FetchError,
    find_run_dir,
    install,
    skipped_artefacts,
    verify,
)

RUN_ID = "phase3_baseline_resnet18"


def _run(root: Path, run_id: str = RUN_ID, *, epochs: int = 8, best: int = 6,
         nest: str = "runs", smoke: bool = False, with_ckpt: bool = True) -> Path:
    """A plausible kernel output tree."""
    d = root / nest / run_id if nest else root / run_id
    d.mkdir(parents=True, exist_ok=True)

    (d / "metrics.json").write_text(json.dumps({
        "run_id": run_id, "arm": "A", "split": "val", "n": 5268,
        "qwk": 0.6138, "qwk_ci": {"lo": 0.5859, "hi": 0.6421, "point": 0.6138},
        "accuracy": 0.7965, "balanced_accuracy": 0.4163,
        "per_class_recall": [0.982, 0.0, 0.376, 0.316, 0.407],
        "support": [3882, 357, 788, 133, 108],
        "predicted_counts": [4657, 0, 478, 68, 65],
        "confusion_matrix": [[0] * 5 for _ in range(5)],
        "referable": {"sensitivity": 0.5131, "specificity": 0.9804},
        "collapse": {"collapsed": False, "level": "warning", "reasons": [],
                     "warnings": ["grade 1 never predicted"]},
        "best_epoch": best, "epochs_run": epochs, "is_smoke_test": smoke,
    }), encoding="utf-8")

    (d / "config.yaml").write_text(yaml.safe_dump({
        "run_id": run_id, "arm": "A", "seed": 42, "git": "dfdfe0d",
        "model": {"arch": "resnet18"},
    }), encoding="utf-8")

    pd.DataFrame({
        "epoch": list(range(epochs)),
        "train_loss": [1.0 - 0.05 * i for i in range(epochs)],
        "val_qwk": [0.025, 0.511, 0.549, 0.546, 0.597, 0.595, 0.614, 0.614][:epochs],
        "val_acc": [0.7] * epochs,
        "val_balanced_acc": [0.4] * epochs,
    }).to_csv(d / "train_log.csv", index=False)

    if with_ckpt:
        (d / "best.pth").write_bytes(b"\x00" * 512)
    return d


# ----------------------------------------------------------------------------------
# Finding the run directory
# ----------------------------------------------------------------------------------

def test_finds_the_run_under_runs(tmp_path):
    _run(tmp_path)
    assert find_run_dir(tmp_path, RUN_ID) == tmp_path / "runs" / RUN_ID


def test_finds_the_run_at_the_output_root(tmp_path):
    _run(tmp_path, nest="")
    assert find_run_dir(tmp_path, RUN_ID) == tmp_path / RUN_ID


def test_finds_the_run_nested_deeper(tmp_path):
    """The output layout is not a stable interface (DECISION-023)."""
    _run(tmp_path, nest="kaggle/working/runs")
    assert find_run_dir(tmp_path, RUN_ID).name == RUN_ID


def test_a_directory_with_the_right_name_but_no_metrics_is_not_the_run(tmp_path):
    (tmp_path / "somewhere" / RUN_ID).mkdir(parents=True)
    with pytest.raises(FetchError, match="no run directory"):
        find_run_dir(tmp_path, RUN_ID)


def test_the_error_lists_what_the_output_actually_holds(tmp_path):
    _run(tmp_path, run_id="some_other_run")
    with pytest.raises(FetchError, match="what the output contains"):
        find_run_dir(tmp_path, RUN_ID)


def test_the_shallower_copy_wins_when_two_match(tmp_path):
    _run(tmp_path, nest="runs")
    _run(tmp_path, nest="runs/archive/old/runs")
    assert find_run_dir(tmp_path, RUN_ID) == tmp_path / "runs" / RUN_ID


# ----------------------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------------------

def test_a_good_run_verifies(tmp_path):
    d = _run(tmp_path)
    s = verify(d, RUN_ID)
    assert s["arm"] == "A" and s["epochs"] == 8 and s["best_epoch"] == 6
    assert s["qwk"] == pytest.approx(0.6138)


def test_a_missing_file_is_refused(tmp_path):
    d = _run(tmp_path)
    (d / "train_log.csv").unlink()
    with pytest.raises(FetchError, match="missing"):
        verify(d, RUN_ID)


def test_unparseable_metrics_is_refused(tmp_path):
    d = _run(tmp_path)
    (d / "metrics.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(FetchError, match="does not parse"):
        verify(d, RUN_ID)


def test_a_metrics_file_for_a_different_run_is_refused(tmp_path):
    d = _run(tmp_path)
    m = json.loads((d / "metrics.json").read_text())
    m["run_id"] = "phase4_arm_b"
    (d / "metrics.json").write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(FetchError, match="not 'phase3_baseline_resnet18'"):
        verify(d, RUN_ID)


def test_a_config_for_a_different_run_is_refused(tmp_path):
    d = _run(tmp_path)
    cfg = yaml.safe_load((d / "config.yaml").read_text())
    cfg["run_id"] = "phase4_arm_c"
    (d / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(FetchError, match="config.yaml is for run_id"):
        verify(d, RUN_ID)


def test_missing_metrics_keys_are_refused(tmp_path):
    d = _run(tmp_path)
    m = json.loads((d / "metrics.json").read_text())
    del m["qwk_ci"], m["referable"]
    (d / "metrics.json").write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(FetchError, match="missing key"):
        verify(d, RUN_ID)


def test_missing_train_log_columns_are_refused(tmp_path):
    d = _run(tmp_path)
    df = pd.read_csv(d / "train_log.csv").drop(columns=["val_balanced_acc"])
    df.to_csv(d / "train_log.csv", index=False)
    with pytest.raises(FetchError, match="missing column"):
        verify(d, RUN_ID)


def test_an_empty_train_log_is_refused(tmp_path):
    d = _run(tmp_path)
    pd.read_csv(d / "train_log.csv").iloc[0:0].to_csv(d / "train_log.csv", index=False)
    with pytest.raises(FetchError, match="no rows"):
        verify(d, RUN_ID)


# --- the cross-file checks, which are the point -------------------------------------

def test_files_from_different_runs_are_caught_by_best_epoch(tmp_path):
    """Each file parses; together they are incoherent. Mixing up two arms in Phase 4
    produces exactly this."""
    d = _run(tmp_path, epochs=8, best=19)
    with pytest.raises(FetchError, match="best_epoch 19"):
        verify(d, RUN_ID)


def test_files_from_different_runs_are_caught_by_epoch_count(tmp_path):
    d = _run(tmp_path, epochs=8, best=6)
    m = json.loads((d / "metrics.json").read_text())
    m["epochs_run"] = 30
    (d / "metrics.json").write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(FetchError, match="epochs_run 30"):
        verify(d, RUN_ID)


def test_a_smoke_run_is_refused_as_a_result(tmp_path):
    """`is_smoke_test` exists so a synthetic run cannot be quoted as one (DECISION-025);
    it should not be silently installed into runs/ either."""
    d = _run(tmp_path, smoke=True)
    with pytest.raises(FetchError, match="is_smoke_test"):
        verify(d, RUN_ID)


def test_all_problems_are_reported_at_once(tmp_path):
    """One fix per round trip to Kaggle would be a slow way to work."""
    d = _run(tmp_path, epochs=8, best=99)
    m = json.loads((d / "metrics.json").read_text())
    del m["referable"]
    m["run_id"] = "other"
    (d / "metrics.json").write_text(json.dumps(m), encoding="utf-8")

    with pytest.raises(FetchError) as excinfo:
        verify(d, RUN_ID)
    text = str(excinfo.value)
    assert "missing key" in text and "not 'phase3" in text and "best_epoch" in text


# ----------------------------------------------------------------------------------
# Installing
# ----------------------------------------------------------------------------------

def test_install_copies_the_three_files_and_no_checkpoint(tmp_path):
    d = _run(tmp_path)
    target = tmp_path / "repo" / "runs" / RUN_ID

    copied = install(d, target)
    assert sorted(copied) == ["config.yaml", "metrics.json", "train_log.csv"]
    assert sorted(p.name for p in target.iterdir()) == sorted(copied)
    assert not (target / "best.pth").exists(), "checkpoints must never be copied"


def test_the_checkpoint_is_reported_as_skipped(tmp_path):
    d = _run(tmp_path)
    assert skipped_artefacts(d) == ["best.pth"]


def test_install_refuses_to_overwrite_a_committed_run(tmp_path):
    """A committed run is the record of that run (R6)."""
    d = _run(tmp_path)
    target = tmp_path / "repo" / "runs" / RUN_ID
    install(d, target)

    with pytest.raises(FetchError, match="already exists"):
        install(d, target)

    install(d, target, force=True)      # deliberate replacement is allowed


def test_nothing_is_written_when_verification_fails(tmp_path):
    """The ordering that matters: verify the download, then write. A partial result must
    never half-land in runs/, where it would look committed."""
    d = _run(tmp_path)
    (d / "metrics.json").write_text("{bad", encoding="utf-8")
    target = tmp_path / "repo" / "runs" / RUN_ID

    with pytest.raises(FetchError):
        verify(d, RUN_ID)
    assert not target.exists()


# ----------------------------------------------------------------------------------
# The CLI
# ----------------------------------------------------------------------------------

def test_the_cli_runs_end_to_end_from_a_directory(tmp_path, capsys):
    """`--from-dir` covers everything except the network call."""
    from src.data.fetch_run import main

    _run(tmp_path / "download")
    runs_root = tmp_path / "repo" / "runs"

    import sys
    old = sys.argv
    sys.argv = ["src.data.fetch_run", "--kernel", "rah098/x", "--run-id", RUN_ID,
                "--from-dir", str(tmp_path / "download"), "--runs-root", str(runs_root)]
    try:
        rc = main()
    finally:
        sys.argv = old

    assert rc == 0
    assert (runs_root / RUN_ID / "metrics.json").exists()
    assert not (runs_root / RUN_ID / "best.pth").exists()

    out = capsys.readouterr().out
    assert "skipping" in out and "best.pth" in out
    assert "FETCH OK" in out


def test_the_cli_returns_nonzero_and_writes_nothing_on_a_bad_run(tmp_path):
    from src.data.fetch_run import main

    d = _run(tmp_path / "download")
    (d / "config.yaml").write_text("run_id: someone_elses_run\n", encoding="utf-8")
    runs_root = tmp_path / "repo" / "runs"

    import sys
    old = sys.argv
    sys.argv = ["src.data.fetch_run", "--kernel", "rah098/x", "--run-id", RUN_ID,
                "--from-dir", str(tmp_path / "download"), "--runs-root", str(runs_root)]
    try:
        rc = main()
    finally:
        sys.argv = old

    assert rc == 1
    assert not (runs_root / RUN_ID).exists()


def test_multiple_run_ids_in_one_call(tmp_path):
    """Phase 4 is six arms; a notebook may hold several."""
    from src.data.fetch_run import main

    for rid in ("arm_a", "arm_b"):
        _run(tmp_path / "download", run_id=rid)
    runs_root = tmp_path / "repo" / "runs"

    import sys
    old = sys.argv
    sys.argv = ["src.data.fetch_run", "--kernel", "rah098/x",
                "--run-id", "arm_a", "--run-id", "arm_b",
                "--from-dir", str(tmp_path / "download"), "--runs-root", str(runs_root)]
    try:
        rc = main()
    finally:
        sys.argv = old

    assert rc == 0
    assert (runs_root / "arm_a" / "metrics.json").exists()
    assert (runs_root / "arm_b" / "metrics.json").exists()


def test_the_parser_is_exposed_for_notebook_check():
    """DECISION-022: a CLI whose parser lives inside main() cannot be validated."""
    from src.data.fetch_run import build_parser

    p = build_parser()
    p.parse_args(["--kernel", "rah098/x", "--run-id", "y"])
    with pytest.raises(SystemExit):
        p.parse_args(["--run-id", "y"])          # --kernel is required


def test_optional_artefacts_are_copied_when_present(tmp_path):
    """val_outputs.npz makes threshold analysis a local step (DECISION-030), so it has to
    come back with the run."""
    import numpy as np

    d = _run(tmp_path)
    np.savez_compressed(d / "val_outputs.npz", outputs=np.zeros((4, 5)),
                        y_true=np.zeros(4, int), indices=np.arange(4), head=np.array("softmax"))
    (d / "thresholds.json").write_text("{}", encoding="utf-8")

    target = tmp_path / "repo" / "runs" / RUN_ID
    copied = install(d, target)

    assert "val_outputs.npz" in copied and "thresholds.json" in copied
    assert (target / "val_outputs.npz").exists()
    assert not (target / "best.pth").exists()


def test_a_run_without_the_optional_artefacts_still_installs(tmp_path):
    """Runs from before val_outputs.npz existed are still valid records."""
    d = _run(tmp_path)
    copied = install(d, tmp_path / "repo" / "runs" / RUN_ID)
    assert sorted(copied) == ["config.yaml", "metrics.json", "train_log.csv"]
