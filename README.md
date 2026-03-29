# OOH Billboard Dataset and Detection Toolkit

This project helps you build a billboard detection dataset and run model training/inference with a simple CLI.

It includes:
- `ooh_parser.py` for collecting and auto-labeling outdoor advertising images.
- `main.py` for interactive training and single-image inference.

## Who this is for

If you are new to YOLO and dataset preparation, this repository gives you a practical end-to-end flow:
1. Collect and auto-label billboard images.
2. Train a model on your labeled data.
3. Run detection on your own photos and save previews.

## Main features

- Multi-source image crawling (Bing, Yandex, DuckDuckGo, Wikimedia, Flickr, Openverse when available).
- RU/EN query expansion.
- Multi-object labeling (multiple billboards per image).
- YOLO-World first, CV2 fallback when needed.
- Accepted/Review/Trash split to reduce bad labels.
- Interactive launcher (`main.py`) for training or inference.
- Reuse local `.pt` weights instead of downloading every run.

## Project structure

- `ooh_parser.py`: dataset collection and auto-labeling.
- `main.py`: interactive train/infer launcher.
- `dataset_ooh/accepted/images`: accepted training images.
- `dataset_ooh/accepted/labels`: YOLO labels.
- `dataset_ooh/accepted/dataset.yaml`: training dataset config.
- `dataset_ooh/accepted/preview`: visualization previews.
- `dataset_ooh/review`: uncertain samples.
- `dataset_ooh/trash`: rejected samples.

## Requirements

- Python 3.10+
- macOS/Linux/Windows
- Recommended for training: Apple Silicon MPS or CUDA GPU (CPU also works)

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Step 1: Build dataset with `ooh_parser.py`

Quick start (interactive wizard):

```bash
python3 ooh_parser.py --wizard
```

Profile-based run:

```bash
python3 ooh_parser.py --count 1000 --profile balanced --detector auto --clean-existing
```

Useful options:
- `--max-boxes-per-image` limits how many billboard boxes are saved per image.
- `--profile fast|balanced|quality` controls speed/quality defaults.
- `--clean-existing` re-checks existing accepted samples.
- `--only-clean` only re-checks, no new download.

## Step 2: Train or run inference with `main.py`

Start launcher:

```bash
python3 main.py
```

You will choose one of two modes:
1. Train on `dataset.yaml`.
2. Use an existing YOLO `.pt` model.

After model selection, choose an input image path.
The preview result is saved next to the source image as:
- `<image_name>_billboards_preview.jpg`

## Low-resource training (Apple M1, 8GB RAM)

`main.py` now supports lightweight training defaults:
- training profile `lite` (default in train mode)
- lower `imgsz`
- smaller `batch`
- lower dataloader `workers`
- reduced `patience`
- `single_cls=True`, `cache=False`, `plots=False`, `amp=False`

These settings reduce memory pressure and system load.

## Reusing already downloaded models

Both training and inference can use local `.pt` files from the project directory.
This avoids repeated downloads.

In `main.py`:
- choose local model list when prompted.
- you can pick already downloaded weights such as `yolov8s-worldv2.pt` or `yolov8m-worldv2.pt`.
- you can also provide a custom path to your own YOLO8/YOLO11/YOLO12 `.pt` file.

## Troubleshooting

If YOLO model loading fails:
- verify environment is active: `source .venv/bin/activate`
- reinstall dependencies: `pip install -r requirements.txt`
- choose another local `.pt` file in `main.py`

If crawling returns too few URLs:
- increase `--pool-multiplier`
- add broader queries with `--queries`
- rerun (the parser continues from current progress)

If labels are still noisy:
- keep `review/` and manually clean edge cases
- retrain after cleaning for better precision
