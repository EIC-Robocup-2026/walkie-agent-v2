#!/usr/bin/env python3
"""Local YOLOE detection benchmark over a VIDEO (e.g. the robot's ZED head camera).

Samples frames from a video, runs the same local YOLOE open-vocab detector used by
``run_bench.py`` (prompted with a class list), and writes an annotated output video
plus per-frame detections + an aggregate summary. Shares model/prompt loading with
``run_bench.py``.

Examples
--------
    # official Incheon 2026 prompts, sample 2 fps
    python run_video_bench.py --video videos/zed_head.mp4 --out outputs_video \
        --classes-file incheon2026_objects.txt --sample-fps 2

    # robot's production list, every frame, cap at 500 frames
    python run_video_bench.py --video videos/zed_head.mp4 --out outputs_video \
        --sample-fps 0 --max-frames 500

    # ZED side-by-side stereo: detect on the left image only
    python run_video_bench.py --video videos/zed_head.mp4 --crop left

Outputs (under --out)
---------------------
    annotated.mp4        the sampled frames with boxes/labels/masks drawn
    detections.json      per-sampled-frame detections (frame idx + timestamp)
    detections.csv       one row per detection
    summary.md           class-frequency histogram, empty frames, config, timing
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

# Reuse model + prompt loading from the image benchmark (import-light: no torch
# pulled at import time).
import run_bench


def _open_video(path: Path):
    import cv2

    if not path.exists():
        sys.exit(f"error: --video not found: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        sys.exit(f"error: could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    return cap, fps, n, w, h


def _crop(frame, mode: str):
    if mode == "none":
        return frame
    w = frame.shape[1]
    half = w // 2
    return frame[:, :half] if mode == "left" else frame[:, half:]


def run(args: argparse.Namespace) -> None:
    import cv2

    video = Path(args.video).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prompt set: same precedence as run_bench (--classes > --classes-file > prod list).
    if args.classes:
        classes, classes_src = run_bench._parse_classes(args.classes), "--classes CLI override"
    elif args.classes_file:
        classes, classes_src = run_bench.load_classes_file(args.classes_file)
    else:
        classes, classes_src = run_bench.load_production_classes()

    device = run_bench.pick_device(args.device)
    exclude_terms = [t.strip().lower() for t in (args.exclude or "").split(",") if t.strip()]

    def is_excluded(label: str) -> bool:
        low = label.lower()
        return any(term in low for term in exclude_terms)

    cap, src_fps, n_frames, w, h = _open_video(video)
    src_fps = src_fps if src_fps > 0 else 30.0
    stride = 1 if args.sample_fps <= 0 else max(1, round(src_fps / args.sample_fps))
    out_fps = src_fps if args.sample_fps <= 0 else float(args.sample_fps)
    cw = w // 2 if args.crop != "none" else w
    looks_stereo = w >= 2 * h and h > 0

    print(f"video:    {video.name}  {w}x{h}  {src_fps:.1f} fps  {n_frames} frames "
          f"(~{n_frames / src_fps:.1f}s)" if n_frames else f"video:    {video.name}")
    if looks_stereo and args.crop == "none":
        print("note:     frame looks like ZED side-by-side stereo (w >= 2*h); "
              "pass --crop left to detect on a single eye.")
    print(f"sampling: every {stride} frame(s)  (~{out_fps:.1f} fps out)"
          + (f", capped at {args.max_frames}" if args.max_frames else ""))
    print(f"weights:  {args.weights}   device: {device}")
    print(f"mode:     {'PROMPT-FREE' if args.prompt_free else 'prompted'}"
          + ("" if args.prompt_free else f"   classes: {len(classes)} from {classes_src}"))
    print(f"conf:     {args.conf}  iou: {args.iou}  imgsz: {args.imgsz}")
    if exclude_terms:
        print(f"exclude:  {', '.join(exclude_terms)}  (substring, case-insensitive)")
    print("-" * 60)

    model = run_bench.build_model(args.weights, classes, args.prompt_free)

    writer = None
    records: list[dict] = []
    class_counter: Counter[str] = Counter()
    excluded_counter: Counter[str] = Counter()
    latencies: list[float] = []
    empty = 0
    idx = -1
    sampled = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        idx += 1
        if idx % stride != 0:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        frame = _crop(frame, args.crop)

        t0 = time.perf_counter()
        results = model.predict(
            frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
            device=device, retina_masks=True, verbose=False,
        )
        latencies.append((time.perf_counter() - t0) * 1000.0)
        r = results[0]
        names = r.names
        dets: list[dict] = []
        keep_idx: list[int] = []  # box indices surviving the --exclude filter
        if r.boxes is not None:
            for bi, box in enumerate(r.boxes):
                ci = int(box.cls.item())
                label = names.get(ci, str(ci)) if isinstance(names, dict) else str(ci)
                conf = float(box.conf.item())
                if is_excluded(label):
                    excluded_counter[label] += 1
                    continue
                keep_idx.append(bi)
                x1, y1, x2, y2 = (round(float(v), 1) for v in box.xyxy[0].tolist())
                dets.append({"class": label, "confidence": round(conf, 4),
                             "bbox_xyxy": [x1, y1, x2, y2]})
                class_counter[label] += 1
        dets.sort(key=lambda d: d["confidence"], reverse=True)

        # Drop excluded boxes/masks from the drawn frame too.
        if r.boxes is None or len(r.boxes) == 0:
            annotated = r.plot()
        elif not keep_idx:
            annotated = r.orig_img.copy()
        elif len(keep_idx) == len(r.boxes):
            annotated = r.plot()
        else:
            annotated = r[keep_idx].plot()
        if writer is None:
            ah, aw = annotated.shape[:2]
            writer = cv2.VideoWriter(
                str(out_dir / "annotated.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (aw, ah),
            )
        writer.write(annotated)

        if not dets:
            empty += 1
        records.append({"frame": idx, "time_s": round(idx / src_fps, 3),
                        "n_detections": len(dets), "detections": dets})
        sampled += 1
        if sampled % 20 == 0 or sampled == 1:
            top = ", ".join(f"{d['class']}({d['confidence']:.2f})" for d in dets[:4]) or "—"
            print(f"  frame {idx:>6} (t={idx/src_fps:6.1f}s)  {len(dets):>2} det  {top}")
        if args.max_frames and sampled >= args.max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()

    _write_reports(out_dir, records, class_counter, excluded_counter, latencies, empty,
                   classes_src, args, device, exclude_terms,
                   {"video": video.name, "src_fps": round(src_fps, 2), "resolution": f"{cw}x{h}",
                    "sampled_frames": sampled})
    print("-" * 60)
    if exclude_terms:
        print(f"excluded:  {sum(excluded_counter.values())} detections dropped via --exclude")
    print(f"done. annotated.mp4 + reports in: {out_dir}")


def _write_reports(out_dir, records, class_counter, excluded_counter, latencies, empty,
                   classes_src, args, device, exclude_terms, vinfo):
    meta = {
        **vinfo,
        "weights": args.weights, "device": device, "prompt_free": args.prompt_free,
        "classes_source": classes_src if not args.prompt_free else "built-in vocab (prompt-free)",
        "sample_fps": args.sample_fps, "conf": args.conf, "iou": args.iou, "imgsz": args.imgsz,
        "exclude": ",".join(exclude_terms) if exclude_terms else None,
        "n_detections": sum(r["n_detections"] for r in records),
        "n_detections_excluded": sum(excluded_counter.values()),
        "frames_no_detection": empty,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
    }
    (out_dir / "detections.json").write_text(json.dumps({"meta": meta, "frames": records}, indent=2))

    with (out_dir / "detections.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["frame", "time_s", "class", "confidence", "x1", "y1", "x2", "y2"])
        for rec in records:
            for d in rec["detections"]:
                x1, y1, x2, y2 = d["bbox_xyxy"]
                wr.writerow([rec["frame"], rec["time_s"], d["class"], d["confidence"], x1, y1, x2, y2])

    lines = ["# YOLOE video detection benchmark", "", "## Run config"]
    for k, v in meta.items():
        lines.append(f"- **{k}**: {v}")
    lines += ["", "## What YOLOE saw (class frequency over sampled frames)", ""]
    if class_counter:
        frames_per_class: Counter[str] = Counter()
        for rec in records:
            for cls in {d["class"] for d in rec["detections"]}:
                frames_per_class[cls] += 1
        lines += ["| class | detections | frames |", "|---|---:|---:|"]
        for cls, c in class_counter.most_common():
            lines.append(f"| {cls} | {c} | {frames_per_class[cls]} |")
    else:
        lines.append("_No detections — check prompts / conf / weights._")
    if exclude_terms:
        n_excl = sum(excluded_counter.values())
        lines += ["", f"## Dropped by --exclude `{','.join(exclude_terms)}` ({n_excl} detections)", ""]
        if excluded_counter:
            lines += ["| class | dropped |", "|---|---:|"]
            for cls, c in excluded_counter.most_common():
                lines.append(f"| {cls} | {c} |")
        else:
            lines.append("_nothing matched the exclude terms._")
    lines += ["", f"_{meta['sampled_frames']} frames sampled; "
              f"{empty} had no detection._", ""]
    (out_dir / "summary.md").write_text("\n".join(lines))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Local YOLOE open-vocab detection benchmark over a video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video", required=True, help="path to the input video")
    p.add_argument("--out", default="outputs_video", help="output folder")
    p.add_argument("--weights", default="yoloe-11l-seg.pt", help="YOLOE checkpoint")
    p.add_argument("--classes", default=None, help="comma-separated prompt list")
    p.add_argument("--classes-file", default=None, help="prompt list file (one per line)")
    p.add_argument("--prompt-free", action="store_true", help="use YOLOE built-in vocab")
    p.add_argument("--exclude", default=None,
                   help="comma-separated labels to drop from results AND the annotated "
                   "video (case-insensitive substring, e.g. 'floor,person')")
    p.add_argument("--sample-fps", type=float, default=2.0,
                   help="frames per second to sample (0 = every frame)")
    p.add_argument("--max-frames", type=int, default=0, help="cap sampled frames (0 = no cap)")
    p.add_argument("--crop", choices=["none", "left", "right"], default="none",
                   help="for side-by-side stereo: detect on one half")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None, help="cuda:0 / cpu / mps (default auto)")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
