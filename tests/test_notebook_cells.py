"""Every notebook command line is accepted by the module it calls. DECISION-022.

THE FAILURE THIS EXISTS FOR. Three Kaggle sessions were lost at the packaging step rather
than at the work. The last died after **8.1 hours** on:

    archive_cache.py: error: argument cmd: invalid choice: '/kaggle/working/processed'

A missing `pack` subcommand. Rejected in 0.0 seconds, after the build was finished,
reconciliation had passed, and `READY TO ARCHIVE` had printed. 13 hours across three
sessions, none of it lost to the modelling.

Notebook cells were the only executable code in this project that nothing imported,
nothing type-checked, and no test ran. They are also the most expensive place to be
wrong: they run once, last, after hours of compute, on a machine that then wipes itself.

These tests are cheap and they are not a substitute for `--self-test`, which actually
executes the pack path. Both run in cell 1 of every notebook, because the copy that runs
on Kaggle is a PASTE, and the paste is what drifted.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.data.notebook_check import (
    api_calls_in,
    check,
    check_api_call,
    collect_api_calls,
    collect_invocations,
    invocations_in,
    self_test,
)

REPO = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------------
# The real cells
# ----------------------------------------------------------------------------------

def _all_invocations():
    return collect_invocations()


def test_the_notebooks_contain_invocations_to_check():
    """A checker that silently finds nothing would pass forever."""
    invs = _all_invocations()
    assert len(invs) >= 8, f"only found {len(invs)} invocations — the extractor is broken"
    assert {i.notebook for i in invs} >= {"phase2_build_cache", "phase3_baseline"}


@pytest.mark.parametrize("inv", _all_invocations(), ids=str)
def test_every_notebook_command_line_is_accepted_by_its_module(inv):
    why = check(inv)
    assert why is None, f"{inv}\n  -> {why}"


@pytest.mark.parametrize("call", collect_api_calls(),
                         ids=lambda c: f"{c[0]}:{c[1]}:{c[4]}")
def test_every_direct_call_into_src_binds_to_the_real_signature(call):
    _notebook, _cell, _line, module, symbol, node = call
    why = check_api_call(module, symbol, node)
    assert why is None, f"{module}.{symbol} in cell {_cell}\n  -> {why}"


def test_the_archive_path_actually_runs():
    """Not an inspection. pack -> verify -> unpack, executed on a throwaway tree.

    The 8.1-hour failure was in code that had unit tests and a CLI that had never been
    invoked the way the notebook invoked it.
    """
    assert self_test() == []


def test_every_module_a_notebook_shells_into_exposes_build_parser():
    """`build_parser()` is what makes the check possible. A new CLI module that keeps its
    parser inside main() silently opts out of it."""
    import importlib

    for inv in _all_invocations():
        mod = inv.module
        if mod in {"pytest", "kaggle"}:
            continue
        assert hasattr(importlib.import_module(mod), "build_parser"), (
            f"{mod} has no build_parser(); split the parser out of main() so "
            f"{inv.notebook} cell {inv.cell} can be validated"
        )


# ----------------------------------------------------------------------------------
# The checker itself must be able to FAIL — otherwise it proves nothing
# ----------------------------------------------------------------------------------

def _one(source: str):
    return invocations_in(source, "synthetic", 1)[0]


def test_the_exact_failure_that_cost_8_hours_is_caught():
    """The missing `pack` subcommand, reproduced verbatim."""
    broken = _one('''
run([
    sys.executable, "-m", "src.data.archive_cache",
    "--cache-root", OUT,
    "--out", ARCHIVE,
    "--expect-images", "38788",
])
''')
    why = check(broken)
    assert why is not None
    assert "invalid choice" in why or "cmd" in why


def test_the_corrected_invocation_passes():
    ok = _one('''
run([
    sys.executable, "-m", "src.data.archive_cache", "pack",
    "--cache-root", OUT,
    "--out", ARCHIVE,
    "--expect-images", "38788",
])
''')
    assert check(ok) is None


def test_a_misspelled_flag_is_caught():
    bad = _one('''
run([sys.executable, "-m", "src.data.reconcile_cache", "--cache-dir", OUT])
''')
    assert check(bad) is not None


def test_a_missing_required_flag_is_caught():
    """`--cache-root` is required; leaving it out is accepted by nothing."""
    bad = _one('''
run([sys.executable, "-m", "src.train.train", "--arm", "A", "--run-id", "x"])
''')
    why = check(bad)
    assert why is not None and "cache-root" in why


def test_a_bad_choice_value_is_caught():
    bad = _one('''
run([sys.executable, "-m", "src.data.archive_cache", "compress",
     "--cache-root", OUT, "--out", ARCHIVE])
''')
    assert check(bad) is not None


def test_a_nonexistent_pytest_target_is_caught():
    bad = _one('''
run([sys.executable, "-m", "pytest", "tests/test_does_not_exist.py", "-q"])
''')
    assert check(bad) is not None


def test_a_module_that_does_not_exist_is_caught():
    bad = _one('''
run([sys.executable, "-m", "src.data.nope", "--x"])
''')
    assert check(bad) is not None


def test_a_wrong_keyword_on_a_direct_call_is_caught():
    """The Phase 3 `unpack(...)` class of mistake, which argparse never sees."""
    calls = api_calls_in(
        'from src.data.archive_cache import unpack\n'
        'unpack(z, dest, expected_images=38788)\n',
        "synthetic", 1)
    assert len(calls) == 1
    why = check_api_call(calls[0][3], calls[0][4], calls[0][5])
    assert why is not None and "expected_images" in why


def test_a_correct_direct_call_passes():
    calls = api_calls_in(
        'from src.data.archive_cache import unpack\n'
        'unpack(z, dest, expect_images=38788)\n',
        "synthetic", 1)
    assert check_api_call(calls[0][3], calls[0][4], calls[0][5]) is None


def test_a_renamed_function_is_caught():
    calls = api_calls_in(
        'from src.data.kaggle_paths import resolve_dataset\n'
        'resolve_dataset("x")\n',
        "synthetic", 1)
    why = check_api_call(calls[0][3], calls[0][4], calls[0][5])
    assert why is not None and "resolve_dataset" in why


# ----------------------------------------------------------------------------------
# Extraction details
# ----------------------------------------------------------------------------------

def test_runtime_values_are_placeheld_not_guessed():
    inv = _one('run([sys.executable, "-m", "src.data.archive_cache", "pack", '
               '"--cache-root", OUT, "--out", WORK / "x.zip"])')
    assert inv.unresolved == 2
    assert check(inv) is None


def test_non_python_commands_are_ignored():
    assert invocations_in('run(["ls", "-la"])', "synthetic", 1) == []


def test_cells_parse_as_python():
    """A cell that does not parse cannot be checked, and would fail on paste anyway."""
    import importlib

    for name in ("phase2_build_cache", "phase3_baseline"):
        mod = importlib.import_module(f"notebooks.{name}")
        for i, cell in enumerate(mod.CELLS, start=1):
            ast.parse(cell)     # raises SyntaxError with the cell's own line numbers
