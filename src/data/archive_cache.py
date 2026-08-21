"""Pack the preprocessed cache into ONE file, and prove it is complete before trusting it.

WHY THIS EXISTS (DECISION-021). Kaggle caps a notebook's OUTPUT at 500 files. The Phase 2
build wrote 38,788 correct images to `/kaggle/working/processed/` and the packaging step
kept **499 of them**. Nothing failed: the build was right, reconciliation passed, and the
truncation happened afterwards, silently, at the boundary between the session and the
saved output. `/kaggle/working` is then wiped, so 2.5 hours were gone.

The cap counts FILES, not bytes, and a dataset built from a single archive is one file. So
the cache is zipped, the archive is verified while the session is still alive, and only
then is the loose tree deleted.

STORED, NOT DEFLATED. The payload is 38,788 JPEGs, which are already entropy-coded —
deflate spends minutes to save low single-digit percent. `ZIP_STORED` writes at disk speed
and the archive comes out the same size as the tree.

VERIFY IN-SESSION, ALWAYS. `verify_archive` reads the central directory back and checks
the image count and the total size. That check has to run BEFORE the loose tree is deleted
and before the session ends, because after that there is nothing left to compare against
— which is exactly how the first attempt got away with keeping 1.3% of the data.

    python -m src.data.archive_cache pack --cache-root /kaggle/working/processed \\
        --out /kaggle/working/fyp-dr-eyepacs-224.zip --expect-images 38788
    python -m src.data.archive_cache unpack --archive /kaggle/input/.../x.zip \\
        --dest /kaggle/working
"""

from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path

IMAGE_SUFFIX = ".jpg"
# Everything that is not an image but must travel with the cache, so the published
# dataset is self-describing: which splits it was reconciled against, which images were
# flagged, and what preprocessing config produced the pixels.
SIDECARS = ("_cache_provenance.json", "processed_stats.csv", "aptos_stats.csv")


def collect(cache_root: Path) -> tuple[list[Path], list[Path]]:
    """(images, everything else) under the cache root."""
    root = Path(cache_root)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a directory")

    images, others = [], []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        (images if p.suffix.lower() == IMAGE_SUFFIX else others).append(p)
    return images, others


def pack(
    cache_root: Path,
    dest: Path,
    *,
    arcname: str = "processed",
    expect_images: int | None = None,
    progress_every: int = 5000,
) -> dict:
    """Zip the cache into `dest`. Returns a summary dict; raises before writing anything
    if the tree does not hold the expected number of images.

    The expectation is checked on the SOURCE first. Packing a short tree and then
    verifying the archive against the same short tree would agree with itself perfectly.
    """
    cache_root = Path(cache_root)
    dest = Path(dest)

    images, others = collect(cache_root)
    if expect_images is not None and len(images) != expect_images:
        raise RuntimeError(
            f"{cache_root} holds {len(images)} {IMAGE_SUFFIX} files, expected "
            f"{expect_images}. Do NOT archive this — find out what is missing with "
            "`python -m src.data.reconcile_cache` first."
        )

    found = {p.name for p in others}
    missing = [s for s in SIDECARS if s not in found]
    if missing:
        print(f"WARNING: sidecar(s) not in the cache: {missing}. The published dataset "
              "will not describe itself.")

    dest.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    total = len(images) + len(others)

    # ZIP64 explicitly: >0.8 GB and ~38k entries is close enough to the classic limits
    # that relying on the default would be a bet.
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for i, p in enumerate(images + others, start=1):
            z.write(p, f"{arcname}/{p.relative_to(cache_root).as_posix()}")
            if progress_every and i % progress_every == 0:
                print(f"  {i}/{total} ...", flush=True)

    size = dest.stat().st_size
    return {
        "archive": str(dest),
        "images": len(images),
        "sidecars": len(others),
        "entries": total,
        "bytes": size,
        "gb": size / 1024 ** 3,
        "seconds": time.time() - t0,
        "arcname": arcname,
    }


def verify_archive(
    archive: Path,
    *,
    expect_images: int,
    expect_gb: float | None = None,
    gb_tolerance: float = 0.10,
    expect_sidecars: tuple[str, ...] = SIDECARS,
    deep: bool = False,
) -> list[str]:
    """Read the archive back and check it. Returns a list of problems; empty means good.

    Reads the CENTRAL DIRECTORY rather than trusting what was just written — that is what
    a consumer will read, and it is the thing that would have caught 499 files.
    """
    archive = Path(archive)
    problems: list[str] = []

    if not archive.exists():
        return [f"{archive} does not exist"]

    with zipfile.ZipFile(archive) as z:
        infos = [i for i in z.infolist() if not i.is_dir()]
        images = [i for i in infos if i.filename.lower().endswith(IMAGE_SUFFIX)]
        names = {Path(i.filename).name for i in infos}

        if len(images) != expect_images:
            problems.append(
                f"archive holds {len(images)} images, expected {expect_images} "
                f"({expect_images - len(images)} missing)"
            )

        for s in expect_sidecars:
            if s not in names:
                problems.append(f"sidecar {s} is not in the archive")

        empty = [i.filename for i in infos if i.file_size == 0]
        if empty:
            problems.append(f"{len(empty)} zero-byte entr(ies), e.g. {empty[:3]}")

        uncompressed = sum(i.file_size for i in infos)
        if expect_gb is not None:
            got_gb = uncompressed / 1024 ** 3
            if abs(got_gb - expect_gb) > gb_tolerance:
                problems.append(
                    f"archive holds {got_gb:.3f} GB of data, expected about "
                    f"{expect_gb:.3f} GB (tolerance {gb_tolerance})"
                )

        if deep:
            # CRC-checks every entry. Minutes on 38k files, so it is opt-in; worth it
            # once, before the thing is published and the source is deleted.
            bad = z.testzip()
            if bad is not None:
                problems.append(f"CRC failure at {bad}")

    return problems


def summarise_archive(archive: Path) -> dict:
    with zipfile.ZipFile(archive) as z:
        infos = [i for i in z.infolist() if not i.is_dir()]
        images = [i for i in infos if i.filename.lower().endswith(IMAGE_SUFFIX)]
        return {
            "entries": len(infos),
            "images": len(images),
            "uncompressed_gb": sum(i.file_size for i in infos) / 1024 ** 3,
            "archive_gb": Path(archive).stat().st_size / 1024 ** 3,
        }


def unpack(archive: Path, dest: Path, *, expect_images: int | None = None) -> Path:
    """Extract the archive and return the directory the cache landed in.

    Used at the top of every training notebook. The count check is repeated on the
    EXTRACTED tree, because "the archive was complete" and "the extraction completed" are
    different claims and the second one is the one training depends on.
    """
    archive, dest = Path(archive), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive) as z:
        roots = {Path(i.filename).parts[0] for i in z.infolist() if i.filename.strip("/")}
        z.extractall(dest)

    out = dest / sorted(roots)[0] if len(roots) == 1 else dest
    if expect_images is not None:
        n = sum(1 for _ in out.rglob(f"*{IMAGE_SUFFIX}"))
        if n != expect_images:
            raise RuntimeError(
                f"extracted {n} images to {out}, expected {expect_images}. The archive "
                "is truncated or the extraction was interrupted — do not train on it."
            )
    return out


def build_parser() -> argparse.ArgumentParser:
    """The parser, separate from main(), so an invocation can be checked without
    running it. `src/data/notebook_check.py` and `tests/test_notebook_cells.py` both
    parse notebook command lines against this."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pack", help="zip a cache tree into one file, then verify it")
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--arcname", default="processed")
    p.add_argument("--expect-images", type=int, default=38788)
    p.add_argument("--expect-gb", type=float, default=0.836)
    p.add_argument("--gb-tolerance", type=float, default=0.10)
    p.add_argument("--deep", action="store_true", help="CRC-check every entry (slow)")

    v = sub.add_parser("verify", help="check an existing archive")
    v.add_argument("--archive", type=Path, required=True)
    v.add_argument("--expect-images", type=int, default=38788)
    v.add_argument("--expect-gb", type=float, default=0.836)
    v.add_argument("--gb-tolerance", type=float, default=0.10)
    v.add_argument("--deep", action="store_true")

    u = sub.add_parser("unpack", help="extract an archive for training")
    u.add_argument("--archive", type=Path, required=True)
    u.add_argument("--dest", type=Path, required=True)
    u.add_argument("--expect-images", type=int, default=38788)

    return ap


def main() -> int:
    args = build_parser().parse_args()

    if args.cmd == "unpack":
        out = unpack(args.archive, args.dest, expect_images=args.expect_images)
        print(f"extracted to {out}")
        return 0

    if args.cmd == "pack":
        print(f"packing {args.cache_root} -> {args.out}")
        info = pack(args.cache_root, args.out, arcname=args.arcname,
                    expect_images=args.expect_images)
        print(f"\n{info['entries']} entries ({info['images']} images + "
              f"{info['sidecars']} sidecars), {info['gb']:.3f} GB, "
              f"{info['seconds'] / 60:.1f} min")
        archive = args.out
    else:
        archive = args.archive

    print(f"\nverifying {archive} by reading its central directory back ...")
    problems = verify_archive(archive, expect_images=args.expect_images,
                              expect_gb=args.expect_gb,
                              gb_tolerance=args.gb_tolerance, deep=args.deep)
    s = summarise_archive(archive)
    print(f"  entries      : {s['entries']}")
    print(f"  images       : {s['images']}")
    print(f"  data         : {s['uncompressed_gb']:.3f} GB")
    print(f"  archive file : {s['archive_gb']:.3f} GB")

    if problems:
        print(f"\nARCHIVE VERIFICATION FAILED - {len(problems)} problem(s):")
        for x in problems:
            print(f"  - {x}")
        print("\nDO NOT delete the loose tree and DO NOT publish this.")
        return 1

    print("\nARCHIVE VERIFIED - complete, self-describing, and one file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
