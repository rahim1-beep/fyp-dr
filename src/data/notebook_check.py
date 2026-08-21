"""Validate every command line a notebook cell will run, without running any of it.

WHY THIS EXISTS (DECISION-022). Three Kaggle sessions were lost at the packaging step,
not at the work. The last one died after **8.1 hours** on

    archive_cache.py: error: argument cmd: invalid choice: '/kaggle/working/processed'

— a missing `pack` subcommand, rejected by argparse in 0.0 seconds. The build was
correct. Reconciliation had passed. `READY TO ARCHIVE` had printed. Everything that
mattered was already done, and it was thrown away by an argument list nobody had ever
executed.

Notebook cells are the least-tested code in the project and the most expensive to get
wrong: they are strings, they run once, they run last, and they run after hours of
compute. Nothing imports them, so nothing type-checks them and no test exercised them.

WHAT THIS DOES. It reads `CELLS` out of each `notebooks/*.py` and checks two things, neither of which
runs any work:

  1. Every `[sys.executable, "-m", "<module>", ...]` list handed to `run(...)` or
     `subprocess.run` is fed to the target module's real `build_parser()`, with a
     placeholder substituted for anything not knowable statically (a `Path` variable, an
     f-string). argparse either accepts it or it does not.
  2. Every direct call into `src.*` that a cell imports — `unpack(...)`,
     `resolve_input(...)` — is bound against the real signature. Not every mistake is a
     subprocess, and a dropped argument there fails at the same point in the session as a
     bad argv, where argparse never sees it.
  3. Every CLI module in `CLI_MODULES` exposes `build_parser()`, and every
     `python -m <module> ...` example in its own docstring is accepted by that parser.
     Docstring usage is what an operator copies, and it rots silently when a flag is
     renamed.

It runs in two places, and it needs both:

  * `tests/test_notebook_cells.py` — protects the copy in the repo.
  * Cell 1 of every notebook, via `--all` — protects the copy that was PASTED, which is
    the one that actually runs on Kaggle and the one that drifted.

`--self-test` additionally exercises the pack/verify/unpack path for real on a three-file
temporary directory. That is the check that would have saved the 8.1 hours: an execution,
not an inspection.

    python -m src.data.notebook_check --all --self-test
"""

from __future__ import annotations

import argparse
import ast
import importlib
import io
import contextlib
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = REPO / "notebooks"

# A stand-in for any argument whose value is only known at runtime (OUT, CACHE, WORK /
# "x.csv", an f-string). argparse cares about the SHAPE of the command line, not the
# values, so a placeholder is enough — and using one keeps the check honest about what it
# can and cannot see.
PLACEHOLDER = "__RUNTIME_VALUE__"
BACKSLASH = chr(92)


class Invocation:
    def __init__(self, notebook: str, cell: int, line: int, argv: list[str],
                 unresolved: int):
        self.notebook, self.cell, self.line = notebook, cell, line
        self.argv, self.unresolved = argv, unresolved

    @property
    def module(self) -> str | None:
        if len(self.argv) >= 3 and self.argv[1] == "-m":
            return self.argv[2]
        return None

    @property
    def rest(self) -> list[str]:
        return self.argv[3:]

    def __str__(self) -> str:
        shown = " ".join(self.rest[:8]) + (" ..." if len(self.rest) > 8 else "")
        return f"{self.notebook} cell {self.cell} line {self.line}: {self.module} {shown}"


def _literal(node: ast.AST) -> str:
    """Best-effort static value of one list element."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    # sys.executable -> the interpreter slot
    if (isinstance(node, ast.Attribute) and node.attr == "executable"
            and isinstance(node.value, ast.Name) and node.value.id == "sys"):
        return "python"
    return PLACEHOLDER


def invocations_in(source: str, notebook: str, cell_no: int) -> list[Invocation]:
    """Every subprocess-style argv list in one cell."""
    tree = ast.parse(source)
    found: list[Invocation] = []

    # A command line assigned to a name (PACK_CMD = [...]) and run later, possibly in
    # another cell. Cell 5 does exactly this on purpose, so that the list that executes
    # is the list cell 1 validated -- which means the static checker has to see it too,
    # or the safest-written cell would be the only unchecked one.
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.List):
            argv = [_literal(e) for e in node.value.elts]
            if argv and argv[0] == "python":
                found.append(Invocation(notebook, cell_no, node.lineno, argv,
                                        sum(1 for a in argv if a is PLACEHOLDER)))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        name = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name not in {"run", "check_call", "check_output", "Popen"}:
            continue
        if not node.args or not isinstance(node.args[0], ast.List):
            continue

        argv = [_literal(e) for e in node.args[0].elts]
        if not argv or argv[0] != "python":
            continue
        found.append(Invocation(notebook, cell_no, node.lineno, argv,
                                sum(1 for a in argv if a is PLACEHOLDER)))
    return found


def collect_invocations(notebook_names: list[str] | None = None) -> list[Invocation]:
    """Read `CELLS` out of each notebook module and extract every invocation."""
    names = notebook_names or [
        p.stem for p in sorted(NOTEBOOK_DIR.glob("*.py"))
        if p.stem not in {"__init__", "make_bundle"}
    ]

    out: list[Invocation] = []
    for name in names:
        mod = importlib.import_module(f"notebooks.{name}")
        cells = getattr(mod, "CELLS", None)
        if not cells:
            continue
        for i, cell in enumerate(cells, start=1):
            out += invocations_in(cell, name, i)
    return out


def check(inv: Invocation) -> str | None:
    """None if the invocation is accepted; otherwise the reason it is not."""
    module = inv.module
    if module is None:
        return f"not a `-m module` invocation: {inv.argv[:4]}"

    # pytest is not ours; check the target file exists rather than its arguments.
    if module == "pytest":
        targets = [a for a in inv.rest if a.endswith(".py") or "/" in a]
        for t in targets:
            if not (REPO / t).exists():
                return f"pytest target does not exist: {t}"
        return None

    if module == "kaggle":
        return None                      # third-party CLI, not importable to a parser here

    try:
        mod = importlib.import_module(module)
    except Exception as exc:             # noqa: BLE001 - reported, not hidden
        return f"module {module} does not import: {exc!r}"

    builder = getattr(mod, "build_parser", None)
    if builder is None:
        return (f"module {module} has no build_parser(), so its command line cannot be "
                "validated. Split the parser out of main().")

    parser = builder()
    buf = io.StringIO()
    try:
        with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
            parser.parse_args(inv.rest)
    except SystemExit:
        msg = buf.getvalue().strip().splitlines()
        return msg[-1] if msg else "argparse rejected the arguments"
    except Exception as exc:             # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    return None


# ----------------------------------------------------------------------------------
# Direct calls into repo functions — the OTHER way a cell can be wrong
# ----------------------------------------------------------------------------------

def api_calls_in(source: str, notebook: str, cell_no: int) -> list[tuple]:
    """(notebook, cell, line, module, symbol, node) for every call to something the cell
    imported from `src.*`.

    Not every mistake is a subprocess. Phase 3 cell 1 calls `unpack(...)` and
    `resolve_input(...)` directly; a renamed keyword or a dropped argument there fails at
    exactly the same place in the session as a bad argv, and argparse never sees it.
    """
    tree = ast.parse(source)

    imported: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src."):
            for alias in node.names:
                imported[alias.asname or alias.name] = node.module

    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            mod = imported.get(node.func.id)
            if mod:
                out.append((notebook, cell_no, node.lineno, mod, node.func.id, node))
    return out


def collect_api_calls(notebook_names: list[str] | None = None) -> list[tuple]:
    names = notebook_names or [
        q.stem for q in sorted(NOTEBOOK_DIR.glob("*.py"))
        if q.stem not in {"__init__", "make_bundle"}
    ]
    out = []
    for name in names:
        mod = importlib.import_module(f"notebooks.{name}")
        for i, cell in enumerate(getattr(mod, "CELLS", []) or [], start=1):
            out += api_calls_in(cell, name, i)
    return out


def check_api_call(module: str, symbol: str, node: ast.Call) -> str | None:
    """None if the call binds against the real signature, else why it does not."""
    import inspect

    try:
        mod = importlib.import_module(module)
    except Exception as exc:             # noqa: BLE001
        return f"module {module} does not import: {exc!r}"

    fn = getattr(mod, symbol, None)
    if fn is None:
        return f"{module} has no attribute {symbol!r}"
    if not callable(fn):
        return None

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None

    # Values do not matter; arity and keyword NAMES do. *args in a cell would make this
    # unknowable, so it is skipped rather than guessed at.
    if any(isinstance(a, ast.Starred) for a in node.args):
        return None
    if any(k.arg is None for k in node.keywords):
        return None

    args = [PLACEHOLDER] * len(node.args)
    kwargs = {k.arg: PLACEHOLDER for k in node.keywords}
    try:
        sig.bind(*args, **kwargs)
    except TypeError as exc:
        return f"{symbol}{sig} does not accept this call: {exc}"
    return None


def validate_argv(argv: list) -> str | None:
    """Check one already-built command line, in-process. None if it is accepted.

    This is the version a NOTEBOOK calls, on the exact list it is about to run, before it
    runs anything. The static checker protects the repo's copy of the cells; this protects
    the copy that was pasted into Kaggle, which is the one that actually executes and the
    one that drifted.
    """
    argv = [str(a) for a in argv]
    if len(argv) < 3 or argv[1] != "-m":
        return f"not a `-m module` invocation: {argv[:3]}"
    inv = Invocation("in-session", 0, 0, ["python"] + argv[1:], 0)
    return check(inv)


# ----------------------------------------------------------------------------------
# CLI modules and their own docstring examples
# ----------------------------------------------------------------------------------

# Every module in the project with a command line. A CLI that is not in this list is not
# checked, so adding one here is part of adding one at all.
CLI_MODULES = (
    "src.data.preprocess",
    "src.data.reconcile_cache",
    "src.data.archive_cache",
    "src.data.fetch_run",
    "src.data.notebook_check",
    "src.train.train",
    "src.train.smoke",
)


def usage_examples(module: str) -> list[list[str]]:
    """`python -m <module> ...` invocations found in a module's own docstring.

    Docstring usage is documentation, and documentation rots: `--expect-gb` gets renamed,
    a subcommand is added, and the example that told the next person how to run the thing
    silently becomes wrong. These are the examples an operator copies, so they are checked
    against the same parser as everything else.

    Continuation backslashes are joined; a leading `python` or `.venv/Scripts/python` is
    normalised away.
    """
    mod = importlib.import_module(module)
    doc = mod.__doc__ or ""

    lines, buf = [], ""
    for raw in doc.splitlines():
        line = raw.strip()
        if buf:
            buf = buf[:-1].rstrip() + " " + line if buf.endswith(BACKSLASH) else buf
            if not line.endswith(BACKSLASH):
                lines.append(buf)
                buf = ""
            else:
                buf = buf[:-1].rstrip() + " " if not buf.endswith(BACKSLASH) else buf
            continue
        if "-m " + module in line and line.endswith(BACKSLASH):
            buf = line
        elif "-m " + module in line:
            lines.append(line)

    out = []
    for line in lines:
        # A trailing comment is normal in a usage example and is not an argument.
        if "#" in line:
            line = line[:line.index("#")]
        parts = line.replace(BACKSLASH, " ").split()
        if "-m" not in parts:
            continue
        i = parts.index("-m")
        if i + 1 >= len(parts) or parts[i + 1] != module:
            continue
        out.append(parts[i + 2:])
    return out


def check_cli_modules(modules=CLI_MODULES) -> list[tuple[str, str]]:
    """(module, problem) for every CLI module that cannot be validated, plus every
    docstring example its own parser rejects."""
    problems = []
    for module in modules:
        try:
            mod = importlib.import_module(module)
        except Exception as exc:             # noqa: BLE001
            problems.append((module, f"does not import: {exc!r}"))
            continue

        builder = getattr(mod, "build_parser", None)
        if builder is None:
            problems.append((module, "has no build_parser(); split the parser out of "
                                     "main() so its command line can be validated"))
            continue

        for argv in usage_examples(module):
            parser = builder()
            buf = io.StringIO()
            try:
                with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
                    parser.parse_args(argv)
            except SystemExit:
                msg = buf.getvalue().strip().splitlines()
                problems.append((
                    module,
                    f"docstring example is not accepted by its own parser: "
                    f"`{' '.join(argv)}` -> {msg[-1] if msg else 'rejected'}",
                ))
            except Exception as exc:         # noqa: BLE001
                problems.append((module, f"docstring example raised {type(exc).__name__}: {exc}"))
    return problems


def self_test() -> list[str]:
    """Actually run pack -> verify -> unpack on a throwaway tree.

    An inspection proves the arguments are shaped right. This proves the code path works,
    which is the thing that was never true before it ran for real on 8 hours of output.
    """
    from src.data.archive_cache import pack, unpack, verify_archive

    problems: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "processed"
        (root / "eyepacs").mkdir(parents=True)
        for i in range(3):
            (root / "eyepacs" / f"{i}_left.jpg").write_bytes(b"\xff\xd8\xff" + bytes(64))
        for s in ("_cache_provenance.json", "processed_stats.csv", "aptos_stats.csv"):
            (root / s).write_text("{}", encoding="utf-8")

        archive = Path(tmp) / "toy.zip"
        try:
            pack(root, archive, expect_images=3, progress_every=0)
            problems += verify_archive(archive, expect_images=3, expect_gb=None)
            out = unpack(archive, Path(tmp) / "x", expect_images=3)
            if sum(1 for _ in out.rglob("*.jpg")) != 3:
                problems.append("unpack did not restore 3 images")
        except Exception as exc:         # noqa: BLE001
            problems.append(f"{type(exc).__name__}: {exc}")
    return problems


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--notebook", action="append", default=[],
                    help="notebook module stem, e.g. phase2_build_cache; repeatable")
    ap.add_argument("--all", action="store_true", help="every notebooks/*.py")
    ap.add_argument("--self-test", action="store_true",
                    help="also run pack/verify/unpack for real on a toy directory")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if not args.all and not args.notebook:
        args.all = True

    invs = collect_invocations(None if args.all else args.notebook)
    print(f"checking {len(invs)} command line(s) from "
          f"{len(set(i.notebook for i in invs))} notebook(s)\n")

    failures = []
    for inv in invs:
        why = check(inv)
        mark = "ok  " if why is None else "FAIL"
        note = f"  ({inv.unresolved} runtime value(s))" if inv.unresolved else ""
        print(f"  [{mark}] {inv}{note}")
        if why is not None:
            print(f"         -> {why}")
            failures.append((inv, why))

    api = collect_api_calls(None if args.all else args.notebook)
    if api:
        print(f"\nchecking {len(api)} direct call(s) into src.* ...")
    for notebook, cell_no, line, module, symbol, node in api:
        why = check_api_call(module, symbol, node)
        mark = "ok  " if why is None else "FAIL"
        print(f"  [{mark}] {notebook} cell {cell_no} line {line}: {module}.{symbol}(...)")
        if why is not None:
            print(f"         -> {why}")
            failures.append((f"{module}.{symbol}", why))

    cli_problems = check_cli_modules()
    print(f"\nchecking {len(CLI_MODULES)} CLI module(s) and their docstring examples ...")
    for module in CLI_MODULES:
        bad = [w for m, w in cli_problems if m == module]
        print(f"  [{'FAIL' if bad else 'ok  '}] {module}")
        for w in bad:
            print(f"         -> {w}")
    failures += [(m, w) for m, w in cli_problems]

    if args.self_test:
        print("\nself-test: pack -> verify -> unpack on a 3-file directory ...")
        problems = self_test()
        for p in problems:
            print(f"  FAIL {p}")
        if problems:
            failures.append(("self-test", "; ".join(problems)))
        else:
            print("  ok")

    print()
    if failures:
        print(f"NOTEBOOK CHECK FAILED - {len(failures)} problem(s). Do not start a run.")
        return 1
    print("NOTEBOOK CHECK PASSED - every command line is accepted by its module's parser.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
