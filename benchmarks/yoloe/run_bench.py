#!/usr/bin/env python3
"""Local YOLOE detection benchmark over a folder of photos.

Runs the on-robot YOLOE open-vocabulary detector **locally** (via `ultralytics`,
*not* the remote walkie-ai-server), prompting it with the production RoboCup class
list (``WALKIE_EXPLORE_INTERESTED_CLASSES``). For every image it records what YOLOE
detected and where, draws an annotated overlay, and writes per-image + aggregate
reports so you can eyeball "from any photo, what object does YOLOE see, and where".

There is no ground truth in this benchmark — it is a perception audit, not mAP.

Examples
--------
    # default: prompted with the production class list, weights auto-downloaded
    python run_bench.py --images dataset --out outputs

    # point at the *exact* weights walkie-ai-server uses for a faithful comparison
    python run_bench.py --images dataset --weights /path/to/server-yoloe.pt

    # raw / prompt-free: let YOLOE use its own built-in vocabulary (use a -pf weight)
    python run_bench.py --images dataset --prompt-free --weights yoloe-11l-seg-pf.pt

    # custom prompts and a lower confidence floor
    python run_bench.py --images dataset --classes "cup,bottle,can,apple" --conf 0.15

Outputs (under --out)
---------------------
    annotated/<name>.jpg   overlay with boxes + labels + masks
    detections.json        structured per-image detections
    detections.csv         one row per detection (easy to pivot)
    summary.md             class-frequency histogram, empties, latency, config used

Install (in an isolated venv — do NOT pollute the robot's uv env):
    python -m venv .venv && . .venv/bin/activate
    pip install -r requirements.txt
    # plus a torch build matching your hardware (CPU example):
    #   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import tomllib
from collections import Counter
from pathlib import Path

# --------------------------------------------------------------------------- #
# Production prompt set (kept in sync with the robot's config).
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[2]

# Where the on-robot perception loop's open-vocab class list lives. The package was
# renamed walkie_graphs -> realtime_explore; the knob followed. We try both so the
# benchmark keeps working across that rename, then fall back to a frozen copy.
_CONFIG_CANDIDATES = [
    (REPO_ROOT / "services/realtime_explore/config.toml", "WALKIE_EXPLORE_INTERESTED_CLASSES"),
    (REPO_ROOT / "services/walkie_graphs/config.toml", "WALKIE_GRAPHS_INTERESTED_CLASSES"),
]

# Frozen fallback (matches services/realtime_explore/config.toml at time of writing)
# so the script still runs if the repo config moves again.
_FALLBACK_CLASSES = (
    "sofa,door,chair,armchair,table,stool,shirt,sponge,bowl,cup,fork,knife,plate,"
    "spoon,coke,ice tea,milk,orange juice,red bull,water bottle,bread,cornflakes,"
    "instant noodles,potato,tomato soup,apple,avocado,lemon,orange,chips,cookies,"
    "gum,mixed nuts,pringles,hand cream,soap,toothpaste,dish rack,shelf,monitor,"
    "bed,pillow"
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}


def load_production_classes() -> tuple[list[str], str]:
    """Return (class list, source description), preferring the live repo config."""
    for cfg_path, key in _CONFIG_CANDIDATES:
        if not cfg_path.exists():
            continue
        try:
            data = tomllib.loads(cfg_path.read_text())
        except Exception:
            continue
        # The knob may sit at top level or under any table.
        raw = data.get(key)
        if raw is None:
            for table in data.values():
                if isinstance(table, dict) and key in table:
                    raw = table[key]
                    break
        if raw:
            classes = _parse_classes(raw)
            if classes:
                return classes, f"{cfg_path.relative_to(REPO_ROOT)} :: {key}"
    return _parse_classes(_FALLBACK_CLASSES), "frozen fallback (repo config not found)"


def _parse_classes(raw: str) -> list[str]:
    return [c.strip() for c in str(raw).split(",") if c.strip()]


def load_classes_file(path: str) -> tuple[list[str], str]:
    """Load a prompt list from a file: one class per line (or comma-separated).

    Blank lines and ``#`` comments are ignored, so a file can document its source.
    """
    p = Path(path)
    if not p.exists():
        sys.exit(f"error: --classes-file not found: {p}")
    classes: list[str] = []
    for line in p.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        classes.extend(_parse_classes(line))  # handles comma-separated or single name
    if not classes:
        sys.exit(f"error: --classes-file has no class names: {p}")
    # de-dupe, preserve order
    seen: set[str] = set()
    uniq = [c for c in classes if not (c in seen or seen.add(c))]
    return uniq, f"file: {p.name}"


# --------------------------------------------------------------------------- #
# Image discovery (with optional HEIC support).
# --------------------------------------------------------------------------- #


def _register_heif() -> bool:
    try:
        import pillow_heif  # type: ignore

        pillow_heif.register_heif_opener()
        return True
    except Exception:
        return False


def discover_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        sys.exit(f"error: --images dir not found: {images_dir}")
    paths = sorted(
        p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not paths:
        sys.exit(f"error: no images ({sorted(IMAGE_EXTS)}) under {images_dir}")
    return paths


def load_image(path: Path):
    """Open as an RGB PIL image (ultralytics treats PIL as RGB correctly)."""
    from PIL import Image, ImageOps

    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # honour phone rotation metadata
    return img.convert("RGB")


# --------------------------------------------------------------------------- #
# Model.
# --------------------------------------------------------------------------- #


def pick_device(requested: str | None) -> str:
    if requested:
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def build_model(weights: str, classes: list[str], prompt_free: bool):
    try:
        from ultralytics import YOLOE
    except Exception as exc:  # pragma: no cover - import-time guard
        sys.exit(
            "error: could not import ultralytics.YOLOE "
            f"({exc}).\nInstall it: pip install -r requirements.txt"
        )

    model = YOLOE(weights)
    if not prompt_free:
        # Text-prompt open-vocab: give YOLOE the class names + their text embeddings.
        # The canonical API is set_classes(names, get_text_pe(names)); some ultralytics
        # versions accept set_classes(names) alone. Try the documented form, fall back.
        try:
            model.set_classes(classes, model.get_text_pe(classes))
        except TypeError:
            model.set_classes(classes)
    return model


# --------------------------------------------------------------------------- #
# Run.
# --------------------------------------------------------------------------- #


def run(args: argparse.Namespace) -> None:
    images_dir = Path(args.images).resolve()
    out_dir = Path(args.out).resolve()
    ann_dir = out_dir / "annotated"
    ann_dir.mkdir(parents=True, exist_ok=True)

    heif_ok = _register_heif()
    images = discover_images(images_dir)
    if any(p.suffix.lower() in {".heic", ".heif"} for p in images) and not heif_ok:
        print(
            "warning: HEIC/HEIF images present but pillow-heif is not installed; "
            "they will be skipped. `pip install pillow-heif` or run fetch_dataset.py "
            "to convert them first.",
            file=sys.stderr,
        )

    if args.classes:
        classes, classes_src = _parse_classes(args.classes), "--classes CLI override"
    elif args.classes_file:
        classes, classes_src = load_classes_file(args.classes_file)
    else:
        classes, classes_src = load_production_classes()

    device = pick_device(args.device)
    exclude_terms = [t.strip().lower() for t in (args.exclude or "").split(",") if t.strip()]

    print(f"images:   {len(images)} under {images_dir}")
    print(f"weights:  {args.weights}")
    print(f"device:   {device}")
    print(f"mode:     {'PROMPT-FREE (built-in vocab)' if args.prompt_free else 'prompted'}")
    if not args.prompt_free:
        print(f"classes:  {len(classes)} from {classes_src}")
    print(f"conf:     {args.conf}   iou: {args.iou}   imgsz: {args.imgsz}")
    if exclude_terms:
        print(f"exclude:  {', '.join(exclude_terms)}  (substring, case-insensitive)")
    print("-" * 60)

    model = build_model(args.weights, classes, args.prompt_free)

    import cv2  # imported lazily so --help works without it

    records: list[dict] = []
    class_counter: Counter[str] = Counter()
    excluded_counter: Counter[str] = Counter()
    latencies: list[float] = []
    empty_images: list[str] = []

    def is_excluded(label: str) -> bool:
        low = label.lower()
        return any(term in low for term in exclude_terms)

    for i, path in enumerate(images, 1):
        try:
            img = load_image(path)
        except Exception as exc:
            print(f"[{i:>3}/{len(images)}] {path.name}: SKIP (open failed: {exc})")
            continue

        w, h = img.size
        t0 = time.perf_counter()
        results = model.predict(
            img,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=device,
            retina_masks=True,
            verbose=False,
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt_ms)

        r = results[0]
        names = r.names  # idx -> label (the prompt list, or built-in vocab)
        dets: list[dict] = []
        keep_idx: list[int] = []  # box indices that survive the --exclude filter
        if r.boxes is not None:
            for bi, box in enumerate(r.boxes):
                cls_idx = int(box.cls.item())
                label = names.get(cls_idx, str(cls_idx)) if isinstance(names, dict) else str(cls_idx)
                conf = float(box.conf.item())
                if is_excluded(label):
                    excluded_counter[label] += 1
                    continue
                keep_idx.append(bi)
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / float(w * h) if w and h else 0.0
                dets.append(
                    {
                        "class": label,
                        "confidence": round(conf, 4),
                        "bbox_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                        "bbox_norm": [
                            round(x1 / w, 4),
                            round(y1 / h, 4),
                            round(x2 / w, 4),
                            round(y2 / h, 4),
                        ],
                        "area_ratio": round(area_ratio, 4),
                    }
                )
                class_counter[label] += 1

        # Sort detections by confidence (most confident first) for readability.
        dets.sort(key=lambda d: d["confidence"], reverse=True)

        # Annotated overlay — drop the excluded boxes/masks from the drawing too.
        if r.boxes is None or len(r.boxes) == 0:
            annotated = r.plot()  # nothing detected at all
        elif not keep_idx:
            annotated = r.orig_img.copy()  # every box excluded -> bare BGR frame
        elif len(keep_idx) == len(r.boxes):
            annotated = r.plot()  # nothing excluded -> draw everything
        else:
            annotated = r[keep_idx].plot()  # subset survives
        cv2.imwrite(str(ann_dir / f"{path.stem}.jpg"), annotated)

        if not dets:
            empty_images.append(path.name)

        records.append(
            {
                "image": path.name,
                "path": str(path.relative_to(images_dir)),
                "width": w,
                "height": h,
                "latency_ms": round(dt_ms, 1),
                "n_detections": len(dets),
                "detections": dets,
            }
        )

        top = ", ".join(f"{d['class']}({d['confidence']:.2f})" for d in dets[:5]) or "—"
        print(f"[{i:>3}/{len(images)}] {path.name}: {len(dets):>2} det  {top}")

    _write_reports(
        out_dir, records, class_counter, excluded_counter, latencies, empty_images,
        classes_src, args, device, exclude_terms,
    )
    print("-" * 60)
    if exclude_terms:
        print(f"excluded:  {sum(excluded_counter.values())} detections dropped via --exclude")
    print(f"done. reports + annotated images in: {out_dir}")


# --------------------------------------------------------------------------- #
# Reports.
# --------------------------------------------------------------------------- #


def _write_reports(
    out_dir, records, class_counter, excluded_counter, latencies, empty_images,
    classes_src, args, device, exclude_terms,
):
    meta = {
        "weights": args.weights,
        "device": device,
        "prompt_free": args.prompt_free,
        "classes_source": classes_src if not args.prompt_free else "built-in vocab (prompt-free)",
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "exclude": ",".join(exclude_terms) if exclude_terms else None,
        "n_images": len(records),
        "n_detections": sum(r["n_detections"] for r in records),
        "n_detections_excluded": sum(excluded_counter.values()),
        "n_images_no_detection": len(empty_images),
    }

    (out_dir / "detections.json").write_text(
        json.dumps({"meta": meta, "images": records}, indent=2)
    )

    with (out_dir / "detections.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["image", "class", "confidence", "x1", "y1", "x2", "y2", "area_ratio"])
        for rec in records:
            for d in rec["detections"]:
                x1, y1, x2, y2 = d["bbox_xyxy"]
                wr.writerow([rec["image"], d["class"], d["confidence"], x1, y1, x2, y2, d["area_ratio"]])

    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    lines = ["# YOLOE detection benchmark", ""]
    lines.append("## Run config")
    for k, v in meta.items():
        lines.append(f"- **{k}**: {v}")
    lines.append(f"- **avg_latency_ms**: {avg_lat:.1f}")
    lines.append("")
    lines.append("## What YOLOE saw (class frequency)")
    lines.append("")
    if class_counter:
        lines.append("| class | detections | images |")
        lines.append("|---|---:|---:|")
        images_per_class: Counter[str] = Counter()
        for rec in records:
            for cls in {d["class"] for d in rec["detections"]}:
                images_per_class[cls] += 1
        for cls, n in class_counter.most_common():
            lines.append(f"| {cls} | {n} | {images_per_class[cls]} |")
    else:
        lines.append("_No detections at all — check weights / prompts / conf threshold._")
    lines.append("")
    if exclude_terms:
        n_excl = sum(excluded_counter.values())
        lines.append(f"## Dropped by --exclude `{','.join(exclude_terms)}` ({n_excl} detections)")
        lines.append("")
        if excluded_counter:
            lines.append("| class | dropped |")
            lines.append("|---|---:|")
            for cls, n in excluded_counter.most_common():
                lines.append(f"| {cls} | {n} |")
        else:
            lines.append("_nothing matched the exclude terms._")
        lines.append("")
    lines.append(f"## Images with no detection ({len(empty_images)})")
    lines.append("")
    lines.append(", ".join(empty_images) if empty_images else "_none — every image got at least one box._")
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines))


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Local YOLOE open-vocab detection benchmark (perception audit).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--images", default="dataset", help="folder of images to run over")
    p.add_argument("--out", default="outputs", help="output folder for reports + overlays")
    p.add_argument(
        "--weights",
        default="yoloe-11l-seg.pt",
        help="YOLOE checkpoint. Point at the server's exact .pt for a faithful "
        "comparison; ultralytics auto-downloads named checkpoints.",
    )
    p.add_argument(
        "--classes",
        default=None,
        help="comma-separated prompt list (overrides the repo's production class list)",
    )
    p.add_argument(
        "--classes-file",
        default=None,
        help="path to a prompt list file (one class per line; # comments ok). "
        "Overridden by --classes; overrides the production list.",
    )
    p.add_argument(
        "--prompt-free",
        action="store_true",
        help="skip text prompts; use YOLOE's built-in vocabulary (use a -pf weight)",
    )
    p.add_argument(
        "--exclude",
        default=None,
        help="comma-separated labels to drop from results AND overlays "
        "(case-insensitive substring match, e.g. 'floor,person'). Useful to mute "
        "the prompt-free vocab's floor/scene/person noise. Dropped counts are still "
        "reported in summary.md so nothing is silently hidden.",
    )
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    p.add_argument("--imgsz", type=int, default=640, help="inference image size")
    p.add_argument(
        "--device",
        default=None,
        help="torch device (e.g. cuda:0, cpu, mps). Default: auto-detect.",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
