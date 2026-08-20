"""Enumerate a Kaggle dataset's full file listing without downloading it.

`kaggle datasets files` returns filenames and sizes only — no image bytes. Paging through
the whole listing yields a complete remote manifest, which is what lets the CSV-to-disk
reconciliation in `inspect.py` run on a machine that will never hold the 35.3 GB dataset.

EyePACS is 176 pages at the 200-item maximum, so this takes a few minutes. The output is
gitignored (it lives under data/raw/); regenerate it with this script rather than
committing it.

Usage:
    python -m src.data.list_remote --slug dreamer07/eyepacs \
        --out data/raw/eyepacs/remote_listing.csv
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

PAGE_SIZE = 200
TOKEN_PREFIX = "Next Page Token = "
HEADER = "name,size,creationDate"


def fetch_page(slug: str, token: str | None) -> tuple[list[str], str | None]:
    cmd = ["kaggle", "datasets", "files", slug, "--page-size", str(PAGE_SIZE), "--format", "csv"]
    if token:
        cmd += ["--page-token", token]

    proc = subprocess.run(cmd, capture_output=True, text=True, shell=(sys.platform == "win32"))
    if proc.returncode != 0:
        raise SystemExit(f"kaggle CLI failed (exit {proc.returncode}):\n{proc.stderr}")

    rows: list[str] = []
    next_token: str | None = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or line == HEADER:
            continue
        if line.startswith(TOKEN_PREFIX):
            next_token = line[len(TOKEN_PREFIX):].strip()
            continue
        rows.append(line)
    return rows, next_token


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True, help="e.g. dreamer07/eyepacs")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-pages", type=int, default=400, help="safety cap")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[str] = []
    token: str | None = None
    for page in range(1, args.max_pages + 1):
        rows, token = fetch_page(args.slug, token)
        all_rows.extend(rows)
        print(f"page {page}: {len(rows)} rows (total {len(all_rows)})", flush=True)
        if not token:
            print(f"complete — no further page token after page {page}")
            break
    else:
        raise SystemExit(f"Hit the {args.max_pages}-page safety cap; listing may be incomplete.")

    # Written without a header, matching what inspect.py's load_listing() expects.
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        for row in all_rows:
            w.writerow(next(csv.reader([row])))

    print(f"\n{len(all_rows):,} entries -> {args.out}")


if __name__ == "__main__":
    main()
