# YOLOE perception benchmark

A small, self-contained harness to answer one question: **for any photo, what does
our YOLOE detect it as, and where?**

It runs YOLOE **locally** (via `ultralytics`, *not* the remote `walkie-ai-server`),
prompted with the robot's production class list
(`WALKIE_EXPLORE_INTERESTED_CLASSES`, the same open-vocab prompts the on-robot
`realtime_explore` loop uses). For every image it writes the detected classes +
boxes and an annotated overlay. There is **no ground truth** — this is a perception
audit, not an mAP score.

## Why it's isolated

The robot brain (`walkie-agent-v2`) is a thin client; the heavy vision models live
on the GPU server. This benchmark deliberately keeps `ultralytics` / `torch` **out
of the robot's `uv` env** — install it in its own venv so nothing here touches the
on-robot lockfile.

## Setup

```bash
cd benchmarks/yoloe
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

`ultralytics` pulls `torch` automatically. **On a CPU-only / AMD-iGPU machine** the
default may be a CUDA wheel that won't import (`libcudnn.so.9 not found`) — install a
CPU torch first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

On the NVIDIA GPU box the default CUDA wheel is correct and it'll use the GPU
automatically.

## 1. Fetch the dataset

```bash
python fetch_dataset.py            # -> ./dataset (gitignored)
```

Downloads the shared Drive folder and converts the HEIC photos to JPG (the set is a
mix of JPG + HEIC from three phones; OpenCV can't read HEIC). Override with
`--url <folder>` / `--out <dir>`; `--keep-heic` keeps the originals.

## 2. Run the benchmark

```bash
python run_bench.py --images dataset --out outputs
```

- **Faithful "our YOLOE":** the default weights (`yoloe-11l-seg.pt`) are the public
  Ultralytics checkpoint. For a true apples-to-apples with the robot, point at the
  **exact weights walkie-ai-server uses**:
  `python run_bench.py --weights /path/to/server-yoloe.pt`
- **Raw perception:** `--prompt-free --weights yoloe-11l-seg-pf.pt` lets YOLOE use
  its own built-in vocabulary instead of the RoboCup prompt list.
- **Other knobs:** `--classes "cup,bottle,can"`, `--conf 0.15`, `--imgsz 1024`,
  `--device cpu|cuda:0|mps`.

## Outputs (under `--out`)

| file | what |
|---|---|
| `annotated/<name>.jpg` | overlay: boxes + labels + masks — eyeball "what + where" |
| `detections.json` | structured per-image detections (bbox in px + normalized, area, latency) |
| `detections.csv` | one row per detection — easy to pivot in a spreadsheet |
| `summary.md` | class-frequency histogram, images with no detection, latency, the exact config used |

## Notes

- The prompt list is read live from `services/realtime_explore/config.toml`
  (falls back to the old `services/walkie_graphs/config.toml`, then a frozen copy),
  so it stays in sync with the robot. Override per-run with `--classes`.
- YOLOE is open-vocab but **prompt-driven and brand-blind** — it keys on generic
  visual descriptors, not brand names. Expect "coke" to surface as `can` / `red can`
  rather than `coke`. The production grasp path works around this with descriptor
  expansion + CLIP rerank (`tasks/skills/grasp.py`); this benchmark intentionally
  shows the *raw* prompted detector so you can see where that gap is.
