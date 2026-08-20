"""Download a handful of named images from a Kaggle dataset, without the other 35.3 GB.

EyePACS is never downloaded whole to this machine (CLAUDE.md §3). But a few things do
need real image bytes locally: the Phase 2 QA contact sheet, and later the ~50-image
demo sample the web app is exercised against (`paths.sample_images` in configs/local.yaml).

WHY THIS EXISTS RATHER THAN `kaggle datasets download -f`:

    The kaggle CLI (1.7.4.5) does not URL-encode the path separators in a nested file
    path. It requests

        .../datasets/download/dreamer07/eyepacs/data/data/3829_left.jpeg?filename=data%2F...

    which the API reads as a malformed route and answers 404 — for every file in this
    dataset, including `trainLabels.csv/trainLabels.csv`, which the CLI downloaded
    successfully in an earlier session. Percent-encoding the path fixes it:

        .../datasets/download/dreamer07/eyepacs?filename=data%2Fdata%2F3829_left.jpeg   -> 200

    So this module calls the endpoint directly. If a future kaggle release fixes the
    encoding, this can go away; until then `-f` on a nested path is broken and it is not
    worth rediscovering that.

Downloaded sizes are checked against `data/raw/eyepacs/remote_listing.csv` when it is
available, so a truncated or HTML-error-page download cannot masquerade as an image.

Usage:
    python -m src.data.fetch_sample --manifest docs/phase2_qa_sample.csv \
        --slug dreamer07/eyepacs --out data/raw/qa
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import zipfile
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[2]
API = "https://www.kaggle.com/api/v1/datasets/download"


def credentials() -> tuple[str, str]:
    """Kaggle credentials from KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json."""
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if user and key:
        return user, key

    path = Path(os.path.expanduser("~/.kaggle/kaggle.json"))
    if not path.exists():
        raise FileNotFoundError(
            f"No Kaggle credentials: set KAGGLE_USERNAME/KAGGLE_KEY or place {path}"
        )
    c = json.loads(path.read_text(encoding="utf-8"))
    return c["username"], c["key"]


def expected_sizes(listing: Path) -> dict[str, int]:
    """{path: bytes} from a remote listing, for verifying downloads. Empty if absent."""
    if not listing.exists():
        return {}
    df = pd.read_csv(listing, header=None, names=["name", "size", "created"])
    return dict(zip(df["name"], df["size"].astype(int)))


def fetch(slug: str, remote_path: str, dest: Path, auth: tuple[str, str]) -> int:
    """Download one file, unwrapping the ZIP the API sometimes returns. Returns bytes
    written to `dest`.

    SECOND KAGGLE QUIRK: the endpoint's response format depends on file size. Small files
    come back raw (`FF D8` for a JPEG); files over roughly 1 MB come back as a ZIP
    ARCHIVE containing the one file (`50 4B`, "PK"), with `Content-Type: image/jpeg`
    either way. Written straight to disk, the large ones are ZIPs with a .jpeg extension
    — `cv2.imdecode` returns None, and the byte count looks like a truncated download
    because the ZIP is compressed. Sniff the magic bytes and extract.
    """
    # quote(..., safe="") is the whole point — the separators MUST be percent-encoded.
    url = f"{API}/{slug}?filename={quote(remote_path, safe='')}"
    r = requests.get(url, auth=auth, allow_redirects=True, timeout=180)
    r.raise_for_status()

    ctype = r.headers.get("Content-Type", "")
    if ctype.startswith("text/html"):
        raise RuntimeError(
            f"{remote_path}: server returned HTML, not a file. Usually an auth or "
            f"licence-acceptance problem — open https://www.kaggle.com/datasets/{slug} "
            "and accept the dataset's terms."
        )

    body = r.content
    if body[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(body)) as zf:
            names = zf.namelist()
            if len(names) != 1:
                raise RuntimeError(
                    f"{remote_path}: expected a 1-file ZIP, got {len(names)}: {names[:5]}"
                )
            body = zf.read(names[0])

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)
    return len(body)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--manifest", type=Path, required=True,
                    help="CSV with an image_path column (read with comment='#')")
    ap.add_argument("--slug", default="dreamer07/eyepacs")
    ap.add_argument("--out", type=Path, required=True,
                    help="root to mirror the remote paths under")
    ap.add_argument("--listing", type=Path,
                    default=REPO / "data/raw/eyepacs/remote_listing.csv")
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest, comment="#")
    if "image_path" not in df.columns:
        ap.error(f"{args.manifest} has no image_path column")
    paths = df["image_path"].tolist()

    auth = credentials()
    sizes = expected_sizes(args.listing)

    print(f"slug     : {args.slug}")
    print(f"manifest : {args.manifest}  ({len(paths)} images)")
    print(f"out      : {args.out}")
    if sizes:
        want = sum(sizes.get(p, 0) for p in paths)
        print(f"expected : {want / 1024**2:.1f} MB total\n")

    failures, total = [], 0
    for i, rel in enumerate(paths, start=1):
        dest = args.out / rel
        if dest.exists() and not args.force:
            total += dest.stat().st_size
            print(f"[{i:>3}/{len(paths)}] cached  {rel}")
            continue
        try:
            n = fetch(args.slug, rel, dest, auth)
        except Exception as exc:                      # noqa: BLE001 - reported, not hidden
            failures.append((rel, repr(exc)))
            print(f"[{i:>3}/{len(paths)}] FAILED  {rel}: {exc}")
            continue

        want = sizes.get(rel)
        if want is not None and n != want:
            failures.append((rel, f"size mismatch: got {n}, listing says {want}"))
            print(f"[{i:>3}/{len(paths)}] SIZE!!  {rel}: got {n}, expected {want}")
            continue

        total += n
        print(f"[{i:>3}/{len(paths)}] ok      {rel}  ({n / 1024:.1f} KB)")

    print(f"\ndownloaded/present: {len(paths) - len(failures)}/{len(paths)}  "
          f"({total / 1024**2:.1f} MB)")
    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for rel, why in failures:
            print(f"  {rel}: {why}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
