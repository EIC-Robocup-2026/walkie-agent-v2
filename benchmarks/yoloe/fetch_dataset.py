#!/usr/bin/env python3
"""Download the YOLOE benchmark image set from Google Drive and prep it for run_bench.

Pulls every image from the shared Drive folder into ./dataset/ (gitignored), then
converts any HEIC/HEIF files to JPG so the benchmark can read them everywhere
(OpenCV / ultralytics can't open HEIC; phones in the dataset shoot a mix of JPG and
HEIC).

The folder must be link-shared ("Anyone with the link"); it is, since the team set
it up for this benchmark.

Usage:
    python fetch_dataset.py                      # default folder + ./dataset
    python fetch_dataset.py --out dataset
    python fetch_dataset.py --url <drive-folder-url>
    python fetch_dataset.py --keep-heic          # keep originals next to the .jpg

Requires: gdown, pillow-heif, pillow  (see requirements.txt)
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

# Shared Drive folder with the 2026-06-30 benchmark photos.
DEFAULT_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1P5x46uxtH0VJKjugdC3jHM88llx5Q6hz"
)

HEIC_EXTS = {".heic", ".heif"}


def download_folder(url: str, out: Path, max_retries: int = 4, pace: float = 1.5) -> None:
    """Download every file in the Drive folder, resiliently.

    Google rate-limits anonymous bulk access ("Cannot retrieve the public link ...
    had many accesses"), and gdown's one-shot folder download aborts (and re-fetches
    everything) on the first throttled file. So we enumerate the folder once, then
    download each file individually: skip files already on disk, retry with backoff,
    pace requests, and tolerate per-file failures. Re-running resumes the gaps.
    """
    try:
        import gdown
    except Exception as exc:
        sys.exit(f"error: gdown not installed ({exc}). pip install -r requirements.txt")

    out.mkdir(parents=True, exist_ok=True)
    files = (
        gdown.download_folder(url=url, skip_download=True, quiet=True, use_cookies=False)
        or []
    )
    print(f"folder lists {len(files)} file(s); downloading into {out}")

    ok = skipped = failed = 0
    missing: list[str] = []
    for i, f in enumerate(files, 1):
        dest = out / f.path  # f.path is the name (relative) inside the folder
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size > 0:
            skipped += 1
            continue

        for attempt in range(1, max_retries + 1):
            try:
                res = gdown.download(
                    id=f.id, output=str(dest), quiet=True, resume=True, use_cookies=False
                )
                if not res:
                    raise RuntimeError("gdown returned no path")
                ok += 1
                break
            except Exception as exc:
                if attempt == max_retries:
                    failed += 1
                    missing.append(f.path)
                    print(f"  [{i}/{len(files)}] FAILED {f.path}: {exc}")
                else:
                    # Back off harder each try to let Drive's throttle cool down.
                    time.sleep(min(60.0, 5.0 * attempt) + random.uniform(0, 2))
        # Pace requests a little to avoid tripping the anti-abuse throttle.
        time.sleep(pace + random.uniform(0, 0.8))
        if i % 10 == 0:
            print(f"  progress {i}/{len(files)}  ok={ok} skip={skipped} fail={failed}")

    print(f"download summary: ok={ok} skipped={skipped} failed={failed} of {len(files)}")
    if missing:
        shown = ", ".join(missing[:20]) + (" ..." if len(missing) > 20 else "")
        print(f"  still missing ({len(missing)} — re-run this script to resume): {shown}")


def convert_heic(out: Path, keep_originals: bool) -> int:
    heics = [p for p in out.rglob("*") if p.suffix.lower() in HEIC_EXTS]
    if not heics:
        return 0
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
        from PIL import Image, ImageOps
    except Exception as exc:
        print(
            f"warning: cannot convert HEIC ({exc}); install pillow-heif. "
            f"{len(heics)} HEIC file(s) left as-is.",
            file=sys.stderr,
        )
        return 0

    converted = 0
    for p in heics:
        jpg = p.with_suffix(".jpg")
        if jpg.exists():
            continue
        try:
            img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
            img.save(jpg, quality=95)
            converted += 1
            if not keep_originals:
                p.unlink()
        except Exception as exc:
            print(f"  HEIC convert failed for {p.name}: {exc}", file=sys.stderr)
    print(f"converted {converted} HEIC -> JPG" + ("" if keep_originals else " (originals removed)"))
    return converted


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=DEFAULT_FOLDER_URL, help="Drive folder URL")
    p.add_argument("--out", default="dataset", help="output folder")
    p.add_argument("--keep-heic", action="store_true", help="keep HEIC originals next to the .jpg")
    args = p.parse_args(argv)

    out = Path(args.out).resolve()
    download_folder(args.url, out)
    convert_heic(out, args.keep_heic)

    imgs = [
        p
        for p in out.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]
    print(f"ready: {len(imgs)} usable image(s) in {out}")
    print(f"next:  python run_bench.py --images {args.out} --out outputs")


if __name__ == "__main__":
    main()
