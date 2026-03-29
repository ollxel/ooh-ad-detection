#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import hashlib
import os
import random
import re
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import requests
from tqdm import tqdm

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    RICH_AVAILABLE = True
except Exception:
    Console = None
    Panel = None
    Table = None
    RICH_AVAILABLE = False

DEFAULT_QUERIES = [
    "outdoor advertising",
    "out-of-home ad",
    "ad billboard",
    "street billboard advertisement",
    "digital billboard ad",
    "city OOH advertising",
    "bus stop billboard ad",
    "roadside commercial billboard",
    "наружная реклама билборд",
    "рекламный щит на улице",
    "цифровой билборд реклама",
    "городская наружная реклама",
]

BING_NEGATIVE_TERMS = (
    "-logo -icon -clipart -vector -svg -book -cover -portrait -headshot "
    "-infographic -presentation -template -mockup -render"
)

URL_BLOCKLIST_TOKENS = (
    "logo",
    "clipart",
    "icon",
    "vector",
    "book-cover",
    "book_cover",
    "headshot",
    "portrait",
    "infographic",
    "presentation",
    "template",
    "mockup",
    "sticker",
)

CLASS_NAME = "billboard"
MIN_IMAGE_SIDE = 320
MAX_IMAGE_BYTES = 25_000_000
BING_PAGE_SIZE = 35
WIKIMEDIA_PAGE_SIZE = 50
YANDEX_PAGE_SIZE = 40
OPENVERSE_PAGE_SIZE = 40
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)
DDGS_IMPORT_WARNED = False

PROFILE_PRESETS = {
    "fast": {
        "yolo_world_model": "yolov8s-worldv2.pt",
        "yolo_imgsz": 768,
        "yolo_accept_conf": 0.18,
        "yolo_review_conf": 0.09,
        "max_boxes_per_image": 4,
        "pool_multiplier": 6,
    },
    "balanced": {
        "yolo_world_model": "yolov8m-worldv2.pt",
        "yolo_imgsz": 960,
        "yolo_accept_conf": 0.16,
        "yolo_review_conf": 0.08,
        "max_boxes_per_image": 6,
        "pool_multiplier": 7,
    },
    "quality": {
        "yolo_world_model": "yolov8l-worldv2.pt",
        "yolo_imgsz": 1280,
        "yolo_accept_conf": 0.14,
        "yolo_review_conf": 0.06,
        "max_boxes_per_image": 8,
        "pool_multiplier": 9,
    },
}


@dataclass
class RunStats:
    downloaded: int = 0
    download_failed: int = 0
    duplicates: int = 0
    accepted: int = 0
    review: int = 0
    rejected: int = 0


@dataclass
class DownloadResult:
    url: str
    ok: bool
    image: np.ndarray | None = None
    digest: str | None = None
    reason: str = ""


@dataclass
class DetectionResult:
    bbox: tuple[int, int, int, int] | None
    score: float
    backend: str
    bboxes: list[tuple[int, int, int, int]]
    box_scores: list[float]

    @property
    def has_boxes(self) -> bool:
        return bool(self.bboxes)


@dataclass
class PathCounters:
    accepted: int
    review: int
    trash: int


@dataclass
class CleanStats:
    kept: int = 0
    moved_to_review: int = 0
    moved_to_trash: int = 0


def init_face_cascade() -> cv2.CascadeClassifier | None:
    try:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if not cascade_path.exists():
            return None
        cascade = cv2.CascadeClassifier(str(cascade_path))
        if cascade.empty():
            return None
        return cascade
    except Exception:
        return None


FACE_CASCADE = init_face_cascade()


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = float(iw * ih)
    if inter <= 0:
        return 0.0

    area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
    area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def dedupe_boxes_with_nms(
    boxes: Sequence[tuple[int, int, int, int]],
    scores: Sequence[float],
    iou_threshold: float,
    max_boxes: int,
) -> tuple[list[tuple[int, int, int, int]], list[float]]:
    if not boxes:
        return [], []

    candidates = sorted(
        zip(boxes, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    kept_boxes: list[tuple[int, int, int, int]] = []
    kept_scores: list[float] = []

    for box, score in candidates:
        if any(box_iou(box, selected) >= iou_threshold for selected in kept_boxes):
            continue
        kept_boxes.append(box)
        kept_scores.append(score)
        if len(kept_boxes) >= max_boxes:
            break

    return kept_boxes, kept_scores


class BillboardDetector:
    def __init__(
        self,
        detector_mode: str,
        yolo_model_name: str,
        yolo_accept_conf: float,
        yolo_review_conf: float,
        yolo_iou: float,
        yolo_imgsz: int,
        device: str,
        max_boxes_per_image: int,
    ) -> None:
        self.detector_mode = detector_mode
        self.yolo_model_name = yolo_model_name
        self.yolo_accept_conf = yolo_accept_conf
        self.yolo_review_conf = yolo_review_conf
        self.yolo_iou = yolo_iou
        self.yolo_imgsz = yolo_imgsz
        self.device = device
        self.max_boxes_per_image = max(1, int(max_boxes_per_image))

        self._yolo_world = None
        self._yolo_generic = None
        self._yolo_world_ready = False
        self._yolo_unavailable_reason = ""

        self._prompt_classes = [
            "billboard",
            "advertising billboard",
            "outdoor advertisement",
            "digital billboard",
            "ad poster board",
            "street advertisement board",
            "bus stop advertisement",
            "roadside billboard",
        ]

    def _init_yolo_world(self) -> None:
        if self._yolo_world_ready:
            return
        self._yolo_world_ready = True

        world_error = ""
        try:
            from ultralytics import YOLOWorld

            self._yolo_world = YOLOWorld(self.yolo_model_name)
            self._yolo_world.set_classes(self._prompt_classes)
            return
        except Exception as exc:
            world_error = str(exc)
            self._yolo_world = None

        # Fallback: generic YOLO (например, собственные веса YOLO11/YOLO12 с классом billboard).
        try:
            from ultralytics import YOLO

            self._yolo_generic = YOLO(self.yolo_model_name)
            self._yolo_unavailable_reason = (
                f"YOLO-World недоступен ({world_error}), включен generic YOLO."
            )
        except Exception as generic_exc:
            self._yolo_generic = None
            self._yolo_unavailable_reason = f"{world_error}; generic YOLO: {generic_exc}"

    def _detect_with_yolo_world(
        self,
        image: np.ndarray,
    ) -> tuple[list[tuple[int, int, int, int]], list[float]]:
        self._init_yolo_world()
        if self._yolo_world is None:
            return [], []

        try:
            results = self._yolo_world.predict(
                source=image,
                conf=max(0.01, self.yolo_review_conf * 0.75),
                iou=self.yolo_iou,
                imgsz=self.yolo_imgsz,
                device=self.device,
                verbose=False,
            )
        except Exception:
            return [], []

        if not results:
            return [], []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return [], []

        detected_boxes: list[tuple[int, int, int, int]] = []
        detected_scores: list[float] = []
        img_h, img_w = image.shape[:2]
        img_area = float(img_h * img_w)

        for box in boxes:
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1_i = max(0, min(int(round(x1)), img_w - 2))
            y1_i = max(0, min(int(round(y1)), img_h - 2))
            x2_i = max(x1_i + 1, min(int(round(x2)), img_w - 1))
            y2_i = max(y1_i + 1, min(int(round(y2)), img_h - 1))

            bw = x2_i - x1_i
            bh = y2_i - y1_i
            area_ratio = (bw * bh) / max(img_area, 1.0)
            aspect = bw / max(bh, 1)

            if bw < 28 or bh < 28:
                continue
            if area_ratio < 0.015 or area_ratio > 0.9:
                continue
            if aspect < 0.2 or aspect > 6.0:
                continue

            detected_boxes.append((x1_i, y1_i, x2_i, y2_i))
            detected_scores.append(conf)

        if not detected_boxes:
            return [], []

        return dedupe_boxes_with_nms(
            boxes=detected_boxes,
            scores=detected_scores,
            iou_threshold=0.55,
            max_boxes=self.max_boxes_per_image,
        )

    def _detect_with_generic_yolo(
        self,
        image: np.ndarray,
    ) -> tuple[list[tuple[int, int, int, int]], list[float]]:
        self._init_yolo_world()
        if self._yolo_generic is None:
            return [], []

        try:
            results = self._yolo_generic.predict(
                source=image,
                conf=max(0.01, self.yolo_review_conf * 0.75),
                iou=self.yolo_iou,
                imgsz=self.yolo_imgsz,
                device=self.device,
                verbose=False,
            )
        except Exception:
            return [], []

        if not results:
            return [], []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return [], []

        names = results[0].names
        multi_class = isinstance(names, (dict, list, tuple)) and len(names) > 1
        billboard_tokens = ("billboard", "advert", "ad ", "poster", "sign", "ooh")

        detected_boxes: list[tuple[int, int, int, int]] = []
        detected_scores: list[float] = []
        img_h, img_w = image.shape[:2]
        img_area = float(img_h * img_w)

        for box in boxes:
            conf = float(box.conf[0].item())
            cls_idx = int(box.cls[0].item()) if box.cls is not None else -1

            if multi_class:
                cls_name = ""
                if isinstance(names, dict):
                    cls_name = str(names.get(cls_idx, "")).lower()
                elif isinstance(names, (list, tuple)) and 0 <= cls_idx < len(names):
                    cls_name = str(names[cls_idx]).lower()
                if not any(token in cls_name for token in billboard_tokens):
                    continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1_i = max(0, min(int(round(x1)), img_w - 2))
            y1_i = max(0, min(int(round(y1)), img_h - 2))
            x2_i = max(x1_i + 1, min(int(round(x2)), img_w - 1))
            y2_i = max(y1_i + 1, min(int(round(y2)), img_h - 1))

            bw = x2_i - x1_i
            bh = y2_i - y1_i
            area_ratio = (bw * bh) / max(img_area, 1.0)
            aspect = bw / max(bh, 1)
            if bw < 28 or bh < 28:
                continue
            if area_ratio < 0.012 or area_ratio > 0.9:
                continue
            if aspect < 0.2 or aspect > 6.0:
                continue

            detected_boxes.append((x1_i, y1_i, x2_i, y2_i))
            detected_scores.append(conf)

        if not detected_boxes:
            return [], []

        return dedupe_boxes_with_nms(
            boxes=detected_boxes,
            scores=detected_scores,
            iou_threshold=0.55,
            max_boxes=self.max_boxes_per_image,
        )

    def detect(self, image: np.ndarray) -> DetectionResult:
        mode = self.detector_mode
        if mode == "auto":
            mode = "yolo-world"

        if mode == "yolo-world":
            ml_boxes, ml_scores = self._detect_with_yolo_world(image)
            if ml_boxes:
                return DetectionResult(
                    bbox=ml_boxes[0],
                    score=ml_scores[0],
                    backend="yolo-world",
                    bboxes=ml_boxes,
                    box_scores=ml_scores,
                )

            generic_boxes, generic_scores = self._detect_with_generic_yolo(image)
            if generic_boxes:
                return DetectionResult(
                    bbox=generic_boxes[0],
                    score=generic_scores[0],
                    backend="yolo-generic",
                    bboxes=generic_boxes,
                    box_scores=generic_scores,
                )

            cv2_boxes, cv2_scores = detect_billboard_bboxes_cv2(
                image,
                max_boxes=self.max_boxes_per_image,
            )
            if cv2_boxes:
                return DetectionResult(
                    bbox=cv2_boxes[0],
                    score=cv2_scores[0],
                    backend="cv2-fallback",
                    bboxes=cv2_boxes,
                    box_scores=cv2_scores,
                )
            return DetectionResult(
                bbox=None,
                score=0.0,
                backend="none",
                bboxes=[],
                box_scores=[],
            )

        cv2_boxes, cv2_scores = detect_billboard_bboxes_cv2(
            image,
            max_boxes=self.max_boxes_per_image,
        )
        return DetectionResult(
            bbox=cv2_boxes[0] if cv2_boxes else None,
            score=cv2_scores[0] if cv2_scores else 0.0,
            backend="cv2",
            bboxes=cv2_boxes,
            box_scores=cv2_scores,
        )

    @property
    def yolo_available(self) -> bool:
        self._init_yolo_world()
        return (self._yolo_world is not None) or (self._yolo_generic is not None)

    @property
    def yolo_unavailable_reason(self) -> str:
        return self._yolo_unavailable_reason

    @property
    def yolo_backend_label(self) -> str:
        if self._yolo_world is not None:
            return "YOLO-World"
        if self._yolo_generic is not None:
            return "YOLO(generic)"
        return "none"


def parse_args() -> argparse.Namespace:
    default_workers = min(16, max(4, (os.cpu_count() or 8)))

    parser = argparse.ArgumentParser(
        description=(
            "Собирает фото наружной рекламы, фильтрует нерелевантный мусор, "
            "находит billboard bbox через YOLO-World/cv2 и сохраняет YOLO-датасет."
        )
    )
    basic = parser.add_argument_group("Быстрый старт")
    basic.add_argument(
        "--count",
        type=int,
        default=None,
        help="Желаемое итоговое количество изображений в accepted/images.",
    )
    basic.add_argument(
        "--profile",
        choices=["fast", "balanced", "quality"],
        default="balanced",
        help="Профиль качества/скорости: fast | balanced | quality.",
    )
    basic.add_argument(
        "--wizard",
        action="store_true",
        help="Интерактивный мастер (минимум вопросов, минимум флагов).",
    )
    basic.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset_ooh"),
        help="Папка с датасетом.",
    )
    basic.add_argument(
        "--detector",
        choices=["auto", "yolo-world", "cv2"],
        default="auto",
        help="Детектор bbox: auto(yolo-world->cv2 fallback) | yolo-world | cv2.",
    )
    basic.add_argument(
        "--clean-existing",
        action="store_true",
        help="Перечистить уже собранный accepted (обновить bbox, вынести мусор).",
    )
    basic.add_argument(
        "--only-clean",
        action="store_true",
        help="Только перечистить existing accepted и выйти, без нового парсинга.",
    )

    advanced = parser.add_argument_group("Продвинутые настройки")
    advanced.add_argument(
        "--queries",
        nargs="+",
        default=DEFAULT_QUERIES,
        help="Базовые поисковые запросы.",
    )
    advanced.add_argument(
        "--pool-multiplier",
        type=int,
        default=PROFILE_PRESETS["balanced"]["pool_multiplier"],
        help="Во сколько раз собрать URL больше, чем нужно новых accepted-кадров.",
    )
    advanced.add_argument(
        "--timeout",
        type=int,
        default=15,
        help="Таймаут скачивания одной картинки, сек.",
    )
    advanced.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help="Количество параллельных потоков скачивания.",
    )
    advanced.add_argument(
        "--max-side",
        type=int,
        default=1280,
        help="Максимальная сторона сохраняемых изображений.",
    )
    advanced.add_argument(
        "--jpeg-quality",
        type=int,
        default=82,
        help="JPEG quality для accepted/review/trash (0-100).",
    )
    advanced.add_argument(
        "--min-det-score",
        type=float,
        default=1.9,
        help="Минимальный score CV2-детекции для уверенного accepted.",
    )
    advanced.add_argument(
        "--yolo-world-model",
        default=PROFILE_PRESETS["balanced"]["yolo_world_model"],
        help=(
            "Вес модели: YOLO-World (например: yolov8m-worldv2.pt) "
            "или свои YOLO11/YOLO12 веса (generic fallback)."
        ),
    )
    advanced.add_argument(
        "--yolo-accept-conf",
        type=float,
        default=PROFILE_PRESETS["balanced"]["yolo_accept_conf"],
        help="Порог confidence YOLO-World для accepted.",
    )
    advanced.add_argument(
        "--yolo-review-conf",
        type=float,
        default=PROFILE_PRESETS["balanced"]["yolo_review_conf"],
        help="Порог confidence YOLO-World для review.",
    )
    advanced.add_argument(
        "--yolo-iou",
        type=float,
        default=0.45,
        help="IoU threshold для YOLO-World inference.",
    )
    advanced.add_argument(
        "--yolo-imgsz",
        type=int,
        default=PROFILE_PRESETS["balanced"]["yolo_imgsz"],
        help="Размер изображения для YOLO-World inference.",
    )
    advanced.add_argument(
        "--device",
        default="cpu",
        help="Устройство для YOLO-World, например cpu / 0.",
    )
    advanced.add_argument(
        "--max-boxes-per-image",
        type=int,
        default=PROFILE_PRESETS["balanced"]["max_boxes_per_image"],
        help="Максимум bbox (биллбордов), сохраняемых на одном изображении.",
    )
    advanced.add_argument(
        "--review-threshold",
        type=float,
        default=-0.45,
        help="Порог relevance score для отправки в review вместо trash.",
    )
    return parser.parse_args()


def ask_for_count() -> int:
    while True:
        value = input("Сколько изображений нужно иметь в accepted/images? ").strip()
        if not value:
            print("Введите число больше 0.")
            continue
        if not value.isdigit():
            print("Нужно целое положительное число.")
            continue

        number = int(value)
        if number <= 0:
            print("Число должно быть больше 0.")
            continue
        return number


def print_cli_header() -> None:
    title = "OOH Parser | Multi-billboard AutoLabel"
    subtitle = "YOLO-World + CV2 fallback, multi-source crawler"

    if RICH_AVAILABLE and Console is not None and Panel is not None:
        console = Console()
        console.print(
            Panel.fit(
                f"[bold cyan]{title}[/bold cyan]\n[dim]{subtitle}[/dim]",
                border_style="cyan",
            )
        )
        return

    print("=" * len(title))
    print(title)
    print(subtitle)
    print("=" * len(title))


def ask_choice(prompt: str, options: list[str], default_index: int = 0) -> str:
    default_index = max(0, min(default_index, len(options) - 1))
    while True:
        print(prompt)
        for idx, option in enumerate(options, start=1):
            marker = " (default)" if (idx - 1) == default_index else ""
            print(f"  {idx}. {option}{marker}")
        raw = input("Выбор: ").strip()
        if not raw:
            return options[default_index]
        if raw.isdigit():
            value = int(raw)
            if 1 <= value <= len(options):
                return options[value - 1]
        print("Введите номер из списка.")


def was_flag_provided(*flags: str) -> bool:
    argv = sys.argv[1:]
    for flag in flags:
        if flag in argv:
            return True
        prefix = f"{flag}="
        if any(arg.startswith(prefix) for arg in argv):
            return True
    return False


def apply_profile_overrides(args: argparse.Namespace) -> argparse.Namespace:
    preset = PROFILE_PRESETS.get(args.profile, PROFILE_PRESETS["balanced"])

    if not was_flag_provided("--yolo-world-model"):
        args.yolo_world_model = preset["yolo_world_model"]
    if not was_flag_provided("--yolo-imgsz"):
        args.yolo_imgsz = int(preset["yolo_imgsz"])
    if not was_flag_provided("--yolo-accept-conf"):
        args.yolo_accept_conf = float(preset["yolo_accept_conf"])
    if not was_flag_provided("--yolo-review-conf"):
        args.yolo_review_conf = float(preset["yolo_review_conf"])
    if not was_flag_provided("--max-boxes-per-image"):
        args.max_boxes_per_image = int(preset["max_boxes_per_image"])
    if not was_flag_provided("--pool-multiplier"):
        args.pool_multiplier = int(preset["pool_multiplier"])

    return args


def run_wizard_if_needed(args: argparse.Namespace) -> argparse.Namespace:
    enable_wizard = args.wizard or len(sys.argv) == 1
    if not enable_wizard:
        return args
    if not sys.stdin.isatty():
        return args

    if args.count is None:
        args.count = ask_for_count()

    if not was_flag_provided("--profile"):
        args.profile = ask_choice(
            "Профиль качества/скорости:",
            options=["fast", "balanced", "quality"],
            default_index=1,
        )

    if not was_flag_provided("--detector"):
        args.detector = ask_choice(
            "Режим детектора:",
            options=["auto", "yolo-world", "cv2"],
            default_index=0,
        )

    if not (was_flag_provided("--clean-existing") or was_flag_provided("--only-clean")):
        clean_mode = ask_choice(
            "Что сделать с уже собранным accepted?",
            options=["skip-clean", "clean-existing", "only-clean"],
            default_index=0,
        )
        if clean_mode == "clean-existing":
            args.clean_existing = True
        elif clean_mode == "only-clean":
            args.clean_existing = True
            args.only_clean = True

    return args


def print_run_plan(args: argparse.Namespace, target_count: int, dataset_root: Path) -> None:
    if RICH_AVAILABLE and Console is not None and Table is not None:
        console = Console()
        table = Table(title="Run Plan", show_header=True, header_style="bold magenta")
        table.add_column("Option", style="cyan", no_wrap=True)
        table.add_column("Value", style="white")
        table.add_row("dataset", str(dataset_root))
        table.add_row("target accepted", str(target_count))
        table.add_row("profile", args.profile)
        table.add_row("detector", args.detector)
        table.add_row("model", args.yolo_world_model)
        table.add_row("max boxes/image", str(args.max_boxes_per_image))
        table.add_row("workers", str(args.workers))
        table.add_row("pool multiplier", str(args.pool_multiplier))
        table.add_row("max side", str(args.max_side))
        table.add_row("jpeg quality", str(args.jpeg_quality))
        console.print(table)
        return

    print("[plan]")
    print(f"  dataset: {dataset_root}")
    print(f"  target accepted: {target_count}")
    print(f"  profile: {args.profile}")
    print(f"  detector: {args.detector}")
    print(f"  model: {args.yolo_world_model}")
    print(f"  max boxes/image: {args.max_boxes_per_image}")
    print(f"  workers: {args.workers}")
    print(f"  pool multiplier: {args.pool_multiplier}")
    print(f"  max side: {args.max_side}")
    print(f"  jpeg quality: {args.jpeg_quality}")


def prepare_dirs(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "accepted": root / "accepted",
        "images": root / "accepted" / "images",
        "labels": root / "accepted" / "labels",
        "preview": root / "accepted" / "preview",
        "review": root / "review",
        "review_images": root / "review" / "images",
        "review_preview": root / "review" / "preview",
        "trash": root / "trash",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def next_index(directory: Path, prefix: str) -> int:
    pattern = re.compile(rf"{re.escape(prefix)}_(\d+)\.jpg$", re.IGNORECASE)
    max_idx = 0

    for path in directory.glob(f"{prefix}_*.jpg"):
        match = pattern.match(path.name)
        if not match:
            continue
        max_idx = max(max_idx, int(match.group(1)))

    return max_idx + 1


def init_counters(paths: dict[str, Path]) -> PathCounters:
    return PathCounters(
        accepted=next_index(paths["images"], "img"),
        review=next_index(paths["review_images"], "review"),
        trash=next_index(paths["trash"], "trash"),
    )


def count_jpg_files(directory: Path) -> int:
    return sum(1 for _ in directory.glob("*.jpg"))


def save_jpeg(path: Path, image: np.ndarray, quality: int) -> None:
    quality = int(max(20, min(quality, 100)))
    ok = cv2.imwrite(
        str(path),
        image,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            quality,
            cv2.IMWRITE_JPEG_OPTIMIZE,
            1,
        ],
    )
    if not ok:
        raise RuntimeError(f"Не удалось сохранить файл: {path}")


def resize_max_side(image: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return image

    h, w = image.shape[:2]
    largest = max(h, w)
    if largest <= max_side:
        return image

    scale = max_side / float(largest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def expand_queries(queries: Sequence[str]) -> list[str]:
    expanded: list[str] = []
    cyrillic_re = re.compile(r"[а-яА-Я]")

    for query in queries:
        clean = query.strip()
        if not clean:
            continue

        expanded.append(clean)
        if cyrillic_re.search(clean):
            expanded.append(f"{clean} наружная реклама фото")
            expanded.append(f"{clean} реклама на щите улица")
            expanded.append(f"{clean} outdoor billboard street photo")
        else:
            expanded.append(f"{clean} street billboard photo")
            expanded.append(f"{clean} outdoor advertisement roadside photo")
            expanded.append(f"{clean} наружная реклама билборд фото")

    deduped: list[str] = []
    seen: set[str] = set()
    for item in expanded:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)

    # Слишком длинный список запросов сильно замедляет поиск.
    return deduped[:24]


def normalize_extracted_url(raw: str) -> str:
    url = html.unescape(raw).strip().strip('"').strip("'")
    url = url.replace("\\u002f", "/").replace("\\u003a", ":").replace("\\/", "/")
    if url.startswith("//"):
        return f"https:{url}"
    return url


def extract_bing_urls(page_html: str) -> list[str]:
    patterns = (
        r'murl&quot;:&quot;(.*?)&quot;',
        r'"murl":"(.*?)"',
    )
    extracted: list[str] = []
    for pattern in patterns:
        extracted.extend(re.findall(pattern, page_html))

    normalized: list[str] = []
    for raw in extracted:
        url = normalize_extracted_url(raw)
        if url.startswith("http://") or url.startswith("https://"):
            normalized.append(url)
    return normalized


def extract_yandex_urls(page_html: str) -> list[str]:
    patterns = (
        r'img_href&quot;:&quot;(.*?)&quot;',
        r'"img_href":"(.*?)"',
        r'origin&quot;:\{&quot;w&quot;:\d+,&quot;h&quot;:\d+,&quot;url&quot;:&quot;(.*?)&quot;',
        r'"origin":\{"w":\d+,"h":\d+,"url":"(.*?)"',
    )

    extracted: list[str] = []
    for pattern in patterns:
        extracted.extend(re.findall(pattern, page_html))

    normalized: list[str] = []
    for raw in extracted:
        url = normalize_extracted_url(raw)
        if url.startswith("http://") or url.startswith("https://"):
            normalized.append(url)
    return normalized


def collect_bing_image_urls(query: str, max_results: int, timeout: int) -> list[str]:
    if max_results <= 0:
        return []

    urls: list[str] = []
    seen: set[str] = set()

    page = 0
    max_pages = max(4, (max_results // BING_PAGE_SIZE) + 8)
    search_query = f"{query} {BING_NEGATIVE_TERMS}"

    while len(urls) < max_results and page < max_pages:
        params = {
            "q": search_query,
            "first": page * BING_PAGE_SIZE + 1,
            "count": BING_PAGE_SIZE,
            "adlt": "off",
            "form": "HDRSC2",
        }

        page_html = None
        for attempt in range(3):
            try:
                response = requests.get(
                    "https://www.bing.com/images/search",
                    params=params,
                    headers={"User-Agent": USER_AGENT},
                    timeout=timeout,
                )
                if response.status_code == 200:
                    page_html = response.text
                    break
                if response.status_code in {403, 429}:
                    time.sleep(1.0 + (attempt * 1.2) + random.uniform(0.2, 0.7))
                    continue
                break
            except requests.RequestException:
                time.sleep(0.6 + random.uniform(0.2, 0.5))

        if not page_html:
            break

        page_urls = extract_bing_urls(page_html)
        if not page_urls:
            break

        added_in_page = 0
        for image_url in page_urls:
            if image_url in seen:
                continue
            seen.add(image_url)
            urls.append(image_url)
            added_in_page += 1
            if len(urls) >= max_results:
                break

        if added_in_page == 0:
            break

        page += 1
        time.sleep(random.uniform(0.25, 0.8))

    return urls


def collect_yandex_image_urls(query: str, max_results: int, timeout: int) -> list[str]:
    if max_results <= 0:
        return []

    urls: list[str] = []
    seen: set[str] = set()
    max_pages = max(3, (max_results // YANDEX_PAGE_SIZE) + 5)
    search_query = f"{query} {BING_NEGATIVE_TERMS}"

    for page in range(max_pages):
        if len(urls) >= max_results:
            break

        params = {
            "text": search_query,
            "p": page,
            "isize": "large",
        }
        page_html = None
        for attempt in range(3):
            try:
                response = requests.get(
                    "https://yandex.com/images/search",
                    params=params,
                    headers={
                        "User-Agent": USER_AGENT,
                        "Accept-Language": "ru,en;q=0.8",
                    },
                    timeout=timeout,
                    allow_redirects=True,
                )
                if response.status_code == 200 and response.text:
                    page_html = response.text
                    break
                if response.status_code in {403, 429}:
                    time.sleep(1.0 + (attempt * 1.4) + random.uniform(0.2, 0.8))
                    continue
                break
            except requests.RequestException:
                time.sleep(0.6 + random.uniform(0.2, 0.7))

        if not page_html:
            break

        page_urls = extract_yandex_urls(page_html)
        if not page_urls:
            break

        page_added = 0
        for image_url in page_urls:
            if image_url in seen:
                continue
            seen.add(image_url)
            urls.append(image_url)
            page_added += 1
            if len(urls) >= max_results:
                break

        if page_added == 0:
            break

        time.sleep(random.uniform(0.25, 0.7))

    return urls


def collect_duckduckgo_image_urls(query: str, max_results: int, timeout: int) -> list[str]:
    if max_results <= 0:
        return []

    global DDGS_IMPORT_WARNED
    try:
        from ddgs import DDGS
    except Exception:
        if not DDGS_IMPORT_WARNED:
            print(
                "[warn] Источник DuckDuckGo отключен: пакет `ddgs` не установлен. "
                "Установите `pip install ddgs` для расширения источников."
            )
            DDGS_IMPORT_WARNED = True
        return []

    urls: list[str] = []
    seen: set[str] = set()

    try:
        with DDGS(timeout=timeout) as ddgs:
            try:
                results = ddgs.images(
                    query,
                    safesearch="off",
                    size="Large",
                    max_results=max_results,
                )
            except TypeError:
                results = ddgs.images(query, max_results=max_results)

            for item in results:
                image_url = item.get("image") or item.get("url")
                if not image_url or image_url in seen:
                    continue
                seen.add(image_url)
                urls.append(image_url)
                if len(urls) >= max_results:
                    break
    except Exception:
        return urls

    return urls


def collect_flickr_feed_urls(query: str, max_results: int, timeout: int) -> list[str]:
    if max_results <= 0:
        return []

    tags = re.findall(r"[a-zA-Zа-яА-Я0-9]+", query.lower())
    tags = [tag for tag in tags if len(tag) >= 3][:6]
    if not tags:
        return []

    try:
        response = requests.get(
            "https://www.flickr.com/services/feeds/photos_public.gne",
            params={
                "format": "json",
                "nojsoncallback": 1,
                "tags": ",".join(tags),
            },
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
        if response.status_code != 200:
            return []
        payload = response.json()
    except Exception:
        return []

    urls: list[str] = []
    seen: set[str] = set()
    for item in payload.get("items", []):
        media_url = item.get("media", {}).get("m")
        if not media_url:
            continue

        # Пытаемся сразу брать более крупные версии, если они доступны.
        variants = [media_url]
        if "_m." in media_url:
            variants = [
                media_url.replace("_m.", "_b."),
                media_url.replace("_m.", "_c."),
                media_url.replace("_m.", "_z."),
                media_url,
            ]

        for variant in variants:
            if variant in seen:
                continue
            seen.add(variant)
            urls.append(variant)
            if len(urls) >= max_results:
                return urls

    return urls


def collect_wikimedia_image_urls(query: str, max_results: int, timeout: int) -> list[str]:
    if max_results <= 0:
        return []

    urls: list[str] = []
    seen: set[str] = set()
    offset: int | None = 0
    rounds = 0
    max_rounds = max(4, (max_results // WIKIMEDIA_PAGE_SIZE) + 8)

    wiki_query = (
        f"{query} -logo -icon -book -cover -portrait -headshot "
        "-infographic -template -mockup"
    )

    while len(urls) < max_results and rounds < max_rounds:
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": wiki_query,
            "gsrnamespace": 6,
            "gsrlimit": WIKIMEDIA_PAGE_SIZE,
            "prop": "imageinfo",
            "iiprop": "url",
            "format": "json",
        }
        if offset:
            params["gsroffset"] = offset

        try:
            response = requests.get(
                "https://commons.wikimedia.org/w/api.php",
                params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=timeout,
            )
            if response.status_code in {403, 429}:
                rounds += 1
                time.sleep(0.8 + random.uniform(0.2, 0.7))
                continue
            if response.status_code != 200:
                break
            payload = response.json()
        except requests.RequestException:
            break
        except ValueError:
            break

        pages = payload.get("query", {}).get("pages", {})
        if not pages:
            break

        for page_data in pages.values():
            image_info = page_data.get("imageinfo") or []
            if not image_info:
                continue
            image_url = image_info[0].get("url")
            if not image_url:
                continue
            if image_url in seen:
                continue
            seen.add(image_url)
            urls.append(image_url)
            if len(urls) >= max_results:
                break

        next_offset = payload.get("continue", {}).get("gsroffset")
        if next_offset is None:
            break

        offset = int(next_offset)
        rounds += 1
        time.sleep(random.uniform(0.15, 0.45))

    return urls


def collect_openverse_image_urls(query: str, max_results: int, timeout: int) -> list[str]:
    if max_results <= 0:
        return []

    urls: list[str] = []
    seen: set[str] = set()
    page = 1
    rounds = 0
    max_pages = max(3, (max_results // OPENVERSE_PAGE_SIZE) + 5)

    while len(urls) < max_results and rounds < max_pages:
        page_size = min(OPENVERSE_PAGE_SIZE, max_results - len(urls))
        if page_size <= 0:
            break

        try:
            response = requests.get(
                "https://api.openverse.org/v1/images/",
                params={
                    "q": query,
                    "page": page,
                    "page_size": page_size,
                    "mature": "false",
                },
                headers={"User-Agent": USER_AGENT},
                timeout=timeout,
            )
            if response.status_code in {403, 429}:
                rounds += 1
                time.sleep(0.8 + random.uniform(0.2, 0.7))
                continue
            if response.status_code != 200:
                break
            payload = response.json()
        except requests.RequestException:
            break
        except ValueError:
            break

        items = payload.get("results") or []
        if not items:
            break

        added = 0
        for item in items:
            image_url = (
                item.get("url")
                or item.get("thumbnail")
                or item.get("thumbnail_large")
                or item.get("thumbnail_url")
            )
            if not image_url:
                continue
            if not (image_url.startswith("http://") or image_url.startswith("https://")):
                continue
            if image_url in seen:
                continue
            seen.add(image_url)
            urls.append(image_url)
            added += 1
            if len(urls) >= max_results:
                break

        if added == 0:
            break

        page += 1
        rounds += 1
        time.sleep(random.uniform(0.15, 0.45))

    return urls


def collect_urls_for_query_parallel(
    query: str,
    query_target: int,
    timeout: int,
) -> tuple[list[str], dict[str, int]]:
    need_bing = max(12, int(query_target * 0.28))
    need_yandex = max(10, int(query_target * 0.22))
    need_duck = max(8, int(query_target * 0.16))
    need_wikimedia = max(7, int(query_target * 0.13))
    need_openverse = max(7, int(query_target * 0.13))
    allocated = need_bing + need_yandex + need_duck + need_wikimedia + need_openverse
    need_flickr = max(5, query_target - allocated)

    tasks = [
        ("bing", collect_bing_image_urls, need_bing),
        ("yandex", collect_yandex_image_urls, need_yandex),
        ("duck", collect_duckduckgo_image_urls, need_duck),
        ("wikimedia", collect_wikimedia_image_urls, need_wikimedia),
        ("openverse", collect_openverse_image_urls, need_openverse),
        ("flickr", collect_flickr_feed_urls, need_flickr),
    ]

    source_urls: dict[str, list[str]] = {name: [] for name, _, _ in tasks}
    source_counts: dict[str, int] = {name: 0 for name, _, _ in tasks}

    executor = ThreadPoolExecutor(max_workers=len(tasks))
    try:
        future_to_name = {
            executor.submit(fn, query=query, max_results=max_results, timeout=timeout): name
            for name, fn, max_results in tasks
        }
        done, not_done = wait(
            tuple(future_to_name.keys()),
            timeout=max(20, timeout * 5),
        )
        for future in done:
            source_name = future_to_name[future]
            try:
                urls = future.result()
            except Exception:
                urls = []
            source_urls[source_name] = urls
            source_counts[source_name] = len(urls)
        for future in not_done:
            future.cancel()
            source_name = future_to_name[future]
            source_urls[source_name] = []
            source_counts[source_name] = 0
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    # Перемешиваем source-листы отдельно, чтобы не было перекоса по одному движку.
    for source_list in source_urls.values():
        random.shuffle(source_list)

    merged: list[str] = []
    max_len = max((len(source_list) for source_list in source_urls.values()), default=0)
    ordered_sources = ["bing", "yandex", "duck", "openverse", "wikimedia", "flickr"]
    for idx in range(max_len):
        for source_name in ordered_sources:
            source_list = source_urls.get(source_name, [])
            if idx < len(source_list):
                merged.append(source_list[idx])

    return merged, source_counts


def is_likely_spam_url(url: str) -> bool:
    lower = url.lower()
    if lower.endswith(".svg") or ".svg?" in lower:
        return True
    return any(token in lower for token in URL_BLOCKLIST_TOKENS)


def collect_image_urls(
    queries: Sequence[str],
    need_count: int,
    pool_multiplier: int,
    timeout: int,
    known_urls: set[str],
) -> list[str]:
    expanded_queries = expand_queries(queries)
    if not expanded_queries:
        return []

    target_pool = max(need_count * pool_multiplier, len(expanded_queries) * 40)
    per_query = max(50, target_pool // len(expanded_queries))

    print(
        f"[info] Сбор URL: цель ~{target_pool}, запросов={len(expanded_queries)}, "
        f"до {per_query} на запрос"
    )

    urls: list[str] = []
    seen_urls: set[str] = set(known_urls)
    shuffled_queries = list(expanded_queries)
    random.shuffle(shuffled_queries)

    for query in shuffled_queries:
        if len(urls) >= target_pool:
            break

        query_target = min(per_query, target_pool - len(urls))
        merged_source_urls, source_counts = collect_urls_for_query_parallel(
            query=query,
            query_target=query_target,
            timeout=timeout,
        )

        added = 0
        for source_url in merged_source_urls:
            if source_url in seen_urls:
                continue
            if is_likely_spam_url(source_url):
                continue
            seen_urls.add(source_url)
            urls.append(source_url)
            added += 1
            if len(urls) >= target_pool:
                break

        print(
            f"[info] '{query}': +{added} URL "
            f"(bing={source_counts.get('bing', 0)}, "
            f"yandex={source_counts.get('yandex', 0)}, "
            f"duck={source_counts.get('duck', 0)}, "
            f"openverse={source_counts.get('openverse', 0)}, "
            f"wikimedia={source_counts.get('wikimedia', 0)}, "
            f"flickr={source_counts.get('flickr', 0)})"
        )

    if len(urls) < target_pool:
        print(
            "[warn] URL собрано меньше целевого пула. "
            "Можно поднять --pool-multiplier или расширить --queries."
        )

    random.shuffle(urls)
    return urls


def download_image(url: str, timeout: int) -> tuple[np.ndarray | None, bytes | None]:
    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
            allow_redirects=True,
        )
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        if content_type and ("image" not in content_type and "octet-stream" not in content_type):
            return None, None

        raw = response.content
        if len(raw) < 8_000 or len(raw) > MAX_IMAGE_BYTES:
            return None, None

        np_buf = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
        if image is None:
            return None, None

        h, w = image.shape[:2]
        if min(h, w) < MIN_IMAGE_SIDE:
            return None, None

        return image, raw
    except Exception:
        return None, None


def fetch_download_result(url: str, timeout: int, max_side: int) -> DownloadResult:
    image, raw = download_image(url=url, timeout=timeout)
    if image is None or raw is None:
        return DownloadResult(url=url, ok=False, reason="download_failed")

    image = resize_max_side(image, max_side=max_side)
    digest = hashlib.sha1(raw).hexdigest()
    return DownloadResult(url=url, ok=True, image=image, digest=digest)


def estimate_colorfulness(image: np.ndarray) -> float:
    b, g, r = cv2.split(image.astype(np.float32))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    std_root = np.sqrt(np.var(rg) + np.var(yb))
    mean_root = np.sqrt((np.mean(rg) ** 2) + (np.mean(yb) ** 2))
    return float(std_root + 0.3 * mean_root)


def estimate_skin_ratio(image: np.ndarray) -> float:
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    skin = (cr > 133) & (cr < 173) & (cb > 77) & (cb < 127)
    return float(np.mean(skin))


def estimate_face_ratio(gray: np.ndarray) -> float:
    if FACE_CASCADE is None:
        return 0.0

    h, w = gray.shape[:2]
    scale = min(1.0, 640.0 / max(h, w))
    small = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    faces = FACE_CASCADE.detectMultiScale(
        small,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(24, 24),
    )
    if len(faces) == 0:
        return 0.0

    total_face_area = float(sum(fw * fh for _, _, fw, fh in faces))
    img_area = float(small.shape[0] * small.shape[1])
    return total_face_area / max(img_area, 1.0)


def estimate_textish_ratio(gray: np.ndarray) -> float:
    thr = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        13,
    )
    line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    textish = cv2.morphologyEx(thr, cv2.MORPH_OPEN, line_kernel, iterations=1)
    return float(np.mean(textish > 0))


def estimate_palette_ratio(image: np.ndarray) -> float:
    quant = image // 32
    bins = quant[:, :, 0] * 64 + quant[:, :, 1] * 8 + quant[:, :, 2]
    unique_bins = len(np.unique(bins))
    return unique_bins / 512.0


def score_relevance(image: np.ndarray) -> tuple[float, dict[str, float]]:
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 70, 180)
    edge_density = float(np.mean(edges > 0))
    white_ratio = float(np.mean((sat < 28) & (val > 215)))
    colorfulness = estimate_colorfulness(image)
    skin_ratio = estimate_skin_ratio(image)
    textish_ratio = estimate_textish_ratio(gray)
    palette_ratio = estimate_palette_ratio(image)

    face_ratio = 0.0
    if skin_ratio > 0.12:
        face_ratio = estimate_face_ratio(gray)

    aspect = w / max(h, 1)

    score = 0.0
    score += 0.55 if 0.5 <= aspect <= 2.8 else -0.45
    score += min(edge_density * 5.5, 0.95)

    if edge_density < 0.018:
        score -= 0.8

    if white_ratio > 0.60:
        score -= 1.15
    elif white_ratio > 0.45:
        score -= 0.55

    if colorfulness < 18:
        score -= 0.45
    elif colorfulness > 35:
        score += 0.4

    if palette_ratio < 0.12:
        score -= 0.6
    elif palette_ratio > 0.22:
        score += 0.35

    if skin_ratio > 0.32:
        score -= 1.2
    elif skin_ratio > 0.22:
        score -= 0.6

    if face_ratio > 0.10:
        score -= 1.3
    elif face_ratio > 0.04:
        score -= 0.6

    if textish_ratio > 0.28:
        score -= 0.8
    elif textish_ratio > 0.18:
        score -= 0.45

    metrics = {
        "edge_density": edge_density,
        "white_ratio": white_ratio,
        "colorfulness": colorfulness,
        "skin_ratio": skin_ratio,
        "face_ratio": face_ratio,
        "textish_ratio": textish_ratio,
        "palette_ratio": palette_ratio,
    }
    return score, metrics


def score_contour_candidate(
    contour: np.ndarray,
    sat: np.ndarray,
    val: np.ndarray,
    edges: np.ndarray,
    img_w: int,
    img_h: int,
    img_area: float,
    global_sat: float,
    global_val: float,
    source_bonus: float,
) -> tuple[int, int, int, int, float] | None:
    area = cv2.contourArea(contour)
    if area < img_area * 0.02 or area > img_area * 0.78:
        return None

    x, y, bw, bh = cv2.boundingRect(contour)
    if bw < img_w * 0.07 or bh < img_h * 0.07:
        return None

    aspect = bw / max(bh, 1)
    if aspect < 0.25 or aspect > 4.8:
        return None

    rect_area = float(bw * bh)
    fill_ratio = area / max(rect_area, 1.0)
    if fill_ratio < 0.2:
        return None

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / max(hull_area, 1.0)
    if solidity < 0.5:
        return None

    contour_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)

    local_sat = float(cv2.mean(sat, mask=contour_mask)[0])
    local_val = float(cv2.mean(val, mask=contour_mask)[0])

    pad = int(max(6, min(bw, bh) * 0.12))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(img_w, x + bw + pad), min(img_h, y + bh + pad)

    ring_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    ring_mask[y0:y1, x0:x1] = 255
    ring_mask = cv2.bitwise_and(ring_mask, cv2.bitwise_not(contour_mask))

    if cv2.countNonZero(ring_mask) > 80:
        outer_sat = float(cv2.mean(sat, mask=ring_mask)[0])
        outer_val = float(cv2.mean(val, mask=ring_mask)[0])
    else:
        outer_sat = global_sat
        outer_val = global_val

    perimeter = cv2.arcLength(contour, True)
    approx_len = (
        len(cv2.approxPolyDP(contour, 0.03 * perimeter, True))
        if perimeter > 0
        else 99
    )
    if approx_len == 4:
        poly_bonus = 1.2
    elif approx_len <= 6:
        poly_bonus = 0.45
    else:
        poly_bonus = -0.45

    edge_roi = edges[y : y + bh, x : x + bw]
    edge_density = float(np.mean(edge_roi > 0)) if edge_roi.size else 0.0

    border_t = max(2, int(min(bw, bh) * 0.04))
    top_density = float(np.mean(edges[y : y + border_t, x : x + bw] > 0))
    bottom_density = float(np.mean(edges[y + bh - border_t : y + bh, x : x + bw] > 0))
    left_density = float(np.mean(edges[y : y + bh, x : x + border_t] > 0))
    right_density = float(np.mean(edges[y : y + bh, x + bw - border_t : x + bw] > 0))
    side_hits = sum(d > 0.04 for d in (top_density, bottom_density, left_density, right_density))
    perimeter_density = (top_density + bottom_density + left_density + right_density) / 4.0
    if perimeter_density < 0.03:
        return None
    perimeter_bonus = perimeter_density * 2.0 + (0.7 if side_hits >= 3 else -0.7)

    sat_boost = max(0.0, (local_sat - global_sat) / 80.0)
    val_boost = max(0.0, (local_val - global_val) / 90.0)
    white_bonus = max(0.0, (local_val - 165.0) / 70.0) * max(0.0, (85.0 - local_sat) / 85.0)
    ring_contrast = max(0.0, (local_sat - outer_sat) / 65.0) + max(
        0.0, (local_val - outer_val) / 75.0
    )
    area_ratio = area / img_area

    score = (
        source_bonus
        + area_ratio * 2.1
        + fill_ratio * 1.15
        + sat_boost
        + val_boost
        + white_bonus * 1.1
        + ring_contrast
        + poly_bonus
        + perimeter_bonus
        + min(edge_density * 2.2, 0.9)
    )

    border_hits = int(x <= 2) + int(y <= 2) + int(x + bw >= img_w - 2) + int(y + bh >= img_h - 2)
    if border_hits >= 2 and area_ratio > 0.35 and local_sat < 40:
        return None
    if border_hits >= 3:
        score -= 0.9

    if area_ratio > 0.68:
        score -= 0.75
    if area_ratio < 0.04:
        score -= 0.55
    if side_hits < 3 and area_ratio < 0.08:
        return None

    center_y = (y + (bh / 2.0)) / max(float(img_h), 1.0)
    if center_y > 0.75 and area_ratio < 0.1:
        return None
    if center_y > 0.72 and area_ratio < 0.12:
        score -= 1.4
    if center_y > 0.82 and area_ratio < 0.18:
        score -= 0.8

    return x, y, bw, bh, score


def detect_billboard_bboxes_cv2(
    image: np.ndarray,
    max_boxes: int = 4,
) -> tuple[list[tuple[int, int, int, int]], list[float]]:
    orig_h, orig_w = image.shape[:2]
    proc_scale = min(1.0, 960.0 / max(orig_h, orig_w))

    if proc_scale < 1.0:
        proc = cv2.resize(
            image,
            (int(orig_w * proc_scale), int(orig_h * proc_scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        proc = image

    h, w = proc.shape[:2]
    img_area = float(h * w)

    hsv = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    global_sat = float(np.mean(sat))
    global_val = float(np.mean(val))

    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 70, 190)
    edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)

    vivid = (
        ((sat > 90) & (val > 65))
        | ((sat > 65) & (val > 145))
        | ((sat > 35) & (val > 205))
    ).astype(np.uint8) * 255

    val_bg = cv2.GaussianBlur(val, (0, 0), 9)
    contrast = (cv2.absdiff(val, val_bg) > 22).astype(np.uint8) * 255

    color_mask = cv2.bitwise_and(vivid, contrast)
    color_mask = cv2.morphologyEx(
        color_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
        iterations=2,
    )
    color_mask = cv2.morphologyEx(
        color_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
        iterations=1,
    )

    panel_mask = (
        ((val > 150) & (sat < 90))
        | ((val > 180) & (sat < 120))
    ).astype(np.uint8) * 255
    panel_mask = cv2.morphologyEx(
        panel_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13)),
        iterations=2,
    )
    panel_mask = cv2.morphologyEx(
        panel_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
        iterations=1,
    )

    white_mask = ((val > 175) & (sat < 45)).astype(np.uint8) * 255
    white_mask = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
        iterations=2,
    )
    white_mask = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1,
    )

    geom_mask = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)),
        iterations=2,
    )

    candidates: list[tuple[int, int, int, int, float]] = []

    color_contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in color_contours:
        candidate = score_contour_candidate(
            contour=contour,
            sat=sat,
            val=val,
            edges=edges,
            img_w=w,
            img_h=h,
            img_area=img_area,
            global_sat=global_sat,
            global_val=global_val,
            source_bonus=0.45,
        )
        if candidate is not None:
            candidates.append(candidate)

    panel_contours, _ = cv2.findContours(panel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in panel_contours:
        candidate = score_contour_candidate(
            contour=contour,
            sat=sat,
            val=val,
            edges=edges,
            img_w=w,
            img_h=h,
            img_area=img_area,
            global_sat=global_sat,
            global_val=global_val,
            source_bonus=0.35,
        )
        if candidate is not None:
            candidates.append(candidate)

    white_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in white_contours:
        candidate = score_contour_candidate(
            contour=contour,
            sat=sat,
            val=val,
            edges=edges,
            img_w=w,
            img_h=h,
            img_area=img_area,
            global_sat=global_sat,
            global_val=global_val,
            source_bonus=0.55,
        )
        if candidate is not None:
            candidates.append(candidate)

    geom_contours, _ = cv2.findContours(geom_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in geom_contours:
        candidate = score_contour_candidate(
            contour=contour,
            sat=sat,
            val=val,
            edges=edges,
            img_w=w,
            img_h=h,
            img_area=img_area,
            global_sat=global_sat,
            global_val=global_val,
            source_bonus=0.25,
        )
        if candidate is not None:
            candidates.append(candidate)

    if not candidates:
        return [], []

    sorted_candidates = sorted(candidates, key=lambda item: item[4], reverse=True)
    best_score = sorted_candidates[0][4]

    if best_score < 1.45:
        return [], []

    final_boxes: list[tuple[int, int, int, int]] = []
    final_scores: list[float] = []

    for cand_x, cand_y, cand_w, cand_h, cand_score in sorted_candidates:
        x1 = int(round(cand_x / proc_scale))
        y1 = int(round(cand_y / proc_scale))
        x2 = int(round((cand_x + cand_w) / proc_scale))
        y2 = int(round((cand_y + cand_h) / proc_scale))

        x1 = max(0, min(x1, orig_w - 2))
        y1 = max(0, min(y1, orig_h - 2))
        x2 = max(x1 + 1, min(x2, orig_w - 1))
        y2 = max(y1 + 1, min(y2, orig_h - 1))

        bw = x2 - x1
        bh = y2 - y1
        if bw < 24 or bh < 24:
            continue

        area_ratio = (bw * bh) / max(float(orig_w * orig_h), 1.0)
        if area_ratio > 0.64:
            continue
        if area_ratio > 0.46 and cand_score < 2.35:
            continue
        if area_ratio < 0.015:
            continue

        final_boxes.append((x1, y1, x2, y2))
        final_scores.append(float(cand_score))

    if not final_boxes:
        return [], []

    return dedupe_boxes_with_nms(
        boxes=final_boxes,
        scores=final_scores,
        iou_threshold=0.48,
        max_boxes=max_boxes,
    )


def detect_billboard_bbox_cv2(image: np.ndarray) -> tuple[tuple[int, int, int, int] | None, float]:
    boxes, scores = detect_billboard_bboxes_cv2(image=image, max_boxes=1)
    if not boxes:
        return None, 0.0
    return boxes[0], scores[0]


def decide_bucket(
    relevance_score: float,
    detection: DetectionResult,
    min_det_score: float,
    review_threshold: float,
    yolo_accept_conf: float,
    yolo_review_conf: float,
) -> str:
    bbox = detection.bbox
    det_score = detection.score
    backend = detection.backend

    if bbox is None:
        if relevance_score >= (review_threshold - 0.15):
            return "review"
        return "trash"

    if backend in {"yolo-world", "yolo-generic"}:
        if det_score >= yolo_accept_conf and relevance_score >= -1.45:
            return "accept"
        if det_score >= max(0.03, yolo_review_conf * 0.9):
            return "review"
        if relevance_score >= review_threshold:
            return "review"
        return "trash"

    if backend == "cv2-fallback":
        if det_score >= (min_det_score + 0.45) and relevance_score >= -0.05:
            return "accept"
        if det_score >= 1.0 or relevance_score >= (review_threshold - 0.2):
            return "review"
        return "trash"

    if backend == "cv2":
        if det_score >= min_det_score and relevance_score >= -0.15:
            return "accept"
        if det_score >= 1.0 or relevance_score >= review_threshold:
            return "review"
        return "trash"

    if relevance_score >= review_threshold:
        return "review"

    return "trash"


def yolo_label_line(bbox: tuple[int, int, int, int], img_w: int, img_h: int) -> str:
    x1, y1, x2, y2 = bbox
    xc = ((x1 + x2) / 2.0) / img_w
    yc = ((y1 + y2) / 2.0) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"


def save_yolo_labels(
    path: Path,
    bboxes: Sequence[tuple[int, int, int, int]],
    img_w: int,
    img_h: int,
) -> None:
    if not bboxes:
        path.write_text("", encoding="utf-8")
        return
    lines = [yolo_label_line(bbox, img_w, img_h) for bbox in bboxes]
    path.write_text("".join(lines), encoding="utf-8")


def save_yolo_label(path: Path, bbox: tuple[int, int, int, int], img_w: int, img_h: int) -> None:
    save_yolo_labels(path=path, bboxes=[bbox], img_w=img_w, img_h=img_h)


def draw_preview(
    image: np.ndarray,
    bboxes: Sequence[tuple[int, int, int, int]],
    box_scores: Sequence[float],
    relevance_score: float,
    det_score: float,
    detector_backend: str,
    mode: str,
) -> np.ndarray:
    preview = image.copy()

    color = (0, 255, 0) if mode == "accept" else (0, 215, 255)
    for idx, bbox in enumerate(bboxes):
        cv2.rectangle(preview, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
        if idx < len(box_scores):
            cv2.putText(
                preview,
                f"{box_scores[idx]:.2f}",
                (bbox[0] + 3, max(18, bbox[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    text = (
        f"{mode} | rel={relevance_score:.2f} det={det_score:.2f} "
        f"boxes={len(bboxes)} [{detector_backend}]"
    )
    cv2.putText(
        preview,
        text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (20, 20, 20),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return preview


def save_support_files(paths: dict[str, Path]) -> None:
    (paths["accepted"] / "classes.txt").write_text(f"{CLASS_NAME}\n", encoding="utf-8")

    dataset_yaml = (
        f"path: {paths['accepted']}\n"
        "train: images\n"
        "val: images\n"
        "names:\n"
        f"  0: {CLASS_NAME}\n"
    )
    (paths["accepted"] / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")


def append_metadata_rows(path: Path, rows: list[tuple[str, str, int, int, int, int]]) -> None:
    if not rows:
        return

    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        if write_header:
            writer.writerow(["filename", "source_url", "x1", "y1", "x2", "y2"])
        writer.writerows(rows)


def prune_metadata_to_existing(metadata_path: Path, images_dir: Path) -> tuple[int, int]:
    if not metadata_path.exists():
        return 0, 0

    existing_names = {path.name for path in images_dir.glob("img_*.jpg")}
    kept_rows: list[dict[str, str]] = []
    total_rows = 0

    try:
        with metadata_path.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                total_rows += 1
                filename = row.get("filename") or ""
                if filename in existing_names:
                    kept_rows.append(row)
    except Exception:
        return 0, 0

    removed = total_rows - len(kept_rows)
    if removed <= 0:
        return len(kept_rows), 0

    if not kept_rows:
        try:
            metadata_path.unlink(missing_ok=True)
        except Exception:
            pass
        return 0, removed

    fieldnames = ["filename", "source_url", "x1", "y1", "x2", "y2"]
    with metadata_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    return len(kept_rows), removed


def load_known_urls(metadata_path: Path) -> set[str]:
    if not metadata_path.exists():
        return set()

    urls: set[str] = set()
    try:
        with metadata_path.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                source_url = row.get("source_url")
                if source_url:
                    urls.add(source_url)
    except Exception:
        return set()

    return urls


def load_existing_hashes(paths: dict[str, Path], include_trash: bool = False) -> set[str]:
    hashes: set[str] = set()
    directories = [paths["images"], paths["review_images"]]
    if include_trash:
        directories.append(paths["trash"])

    for directory in directories:
        for image_path in directory.glob("*.jpg"):
            try:
                data = image_path.read_bytes()
            except Exception:
                continue
            hashes.add(hashlib.sha1(data).hexdigest())

    return hashes


def remove_if_exists(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def clean_existing_dataset(
    paths: dict[str, Path],
    counters: PathCounters,
    detector: BillboardDetector,
    max_side: int,
    jpeg_quality: int,
    min_det_score: float,
    review_threshold: float,
    yolo_accept_conf: float,
    yolo_review_conf: float,
) -> CleanStats:
    stats = CleanStats()
    accepted_images = sorted(paths["images"].glob("img_*.jpg"))

    if not accepted_images:
        return stats

    for image_path in tqdm(accepted_images, desc="Чистка existing accepted", unit="img"):
        image = cv2.imread(str(image_path))
        stem = image_path.stem
        label_path = paths["labels"] / f"{stem}.txt"
        preview_path = paths["preview"] / f"{stem}.jpg"

        if image is None:
            remove_if_exists(image_path)
            remove_if_exists(label_path)
            remove_if_exists(preview_path)
            stats.moved_to_trash += 1
            continue

        image = resize_max_side(image, max_side=max_side)
        relevance_score, _ = score_relevance(image)
        detection = detector.detect(image)
        decision = decide_bucket(
            relevance_score=relevance_score,
            detection=detection,
            min_det_score=min_det_score,
            review_threshold=review_threshold,
            yolo_accept_conf=yolo_accept_conf,
            yolo_review_conf=yolo_review_conf,
        )

        if decision == "accept" and detection.has_boxes:
            save_jpeg(image_path, image, jpeg_quality)
            save_yolo_labels(label_path, detection.bboxes, image.shape[1], image.shape[0])
            preview = draw_preview(
                image,
                detection.bboxes,
                detection.box_scores,
                relevance_score,
                detection.score,
                detector_backend=detection.backend,
                mode="accept",
            )
            save_jpeg(preview_path, preview, jpeg_quality)
            stats.kept += 1
            continue

        if decision == "review":
            review_name = f"review_{counters.review:06d}.jpg"
            counters.review += 1

            save_jpeg(paths["review_images"] / review_name, image, jpeg_quality)
            review_preview = draw_preview(
                image,
                detection.bboxes,
                detection.box_scores,
                relevance_score,
                detection.score,
                detector_backend=detection.backend,
                mode="review",
            )
            save_jpeg(paths["review_preview"] / review_name, review_preview, jpeg_quality)

            remove_if_exists(image_path)
            remove_if_exists(label_path)
            remove_if_exists(preview_path)
            stats.moved_to_review += 1
            continue

        trash_name = f"trash_{counters.trash:06d}.jpg"
        counters.trash += 1
        save_jpeg(paths["trash"] / trash_name, image, jpeg_quality)

        remove_if_exists(image_path)
        remove_if_exists(label_path)
        remove_if_exists(preview_path)
        stats.moved_to_trash += 1

    return stats


def process_urls(
    urls: Sequence[str],
    target_count: int,
    current_accepted_count: int,
    paths: dict[str, Path],
    timeout: int,
    workers: int,
    max_side: int,
    jpeg_quality: int,
    min_det_score: float,
    review_threshold: float,
    yolo_accept_conf: float,
    yolo_review_conf: float,
    detector: BillboardDetector,
    known_urls: set[str],
    known_hashes: set[str],
    counters: PathCounters,
) -> tuple[RunStats, list[tuple[str, str, int, int, int, int]], int]:
    stats = RunStats()
    metadata_rows: list[tuple[str, str, int, int, int, int]] = []
    accepted_total = current_accepted_count

    url_iter = iter(urls)
    pending: dict[object, str] = {}

    def submit_next(executor: ThreadPoolExecutor) -> bool:
        try:
            url = next(url_iter)
        except StopIteration:
            return False

        future = executor.submit(fetch_download_result, url, timeout, max_side)
        pending[future] = url
        return True

    prefetch = min(len(urls), max(16, workers * 3))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for _ in range(prefetch):
            if not submit_next(executor):
                break

        with tqdm(total=len(urls), desc="Скачивание + фильтр + детекция", unit="img") as pbar:
            while pending and accepted_total < target_count:
                done, _ = wait(tuple(pending.keys()), return_when=FIRST_COMPLETED)

                for future in done:
                    source_url = pending.pop(future)
                    pbar.update(1)

                    try:
                        result = future.result()
                    except Exception:
                        stats.download_failed += 1
                        submit_next(executor)
                        continue

                    if (not result.ok) or (result.image is None) or (result.digest is None):
                        stats.download_failed += 1
                        submit_next(executor)
                        continue

                    stats.downloaded += 1

                    if source_url in known_urls or result.digest in known_hashes:
                        stats.duplicates += 1
                        submit_next(executor)
                        continue

                    known_urls.add(source_url)
                    known_hashes.add(result.digest)

                    image = result.image
                    relevance_score, _ = score_relevance(image)
                    detection = detector.detect(image)

                    decision = decide_bucket(
                        relevance_score=relevance_score,
                        detection=detection,
                        min_det_score=min_det_score,
                        review_threshold=review_threshold,
                        yolo_accept_conf=yolo_accept_conf,
                        yolo_review_conf=yolo_review_conf,
                    )

                    if decision == "accept" and detection.has_boxes:
                        filename = f"img_{counters.accepted:06d}.jpg"
                        counters.accepted += 1

                        image_path = paths["images"] / filename
                        label_path = paths["labels"] / f"{Path(filename).stem}.txt"
                        preview_path = paths["preview"] / filename

                        save_jpeg(image_path, image, jpeg_quality)
                        save_yolo_labels(label_path, detection.bboxes, image.shape[1], image.shape[0])
                        preview = draw_preview(
                            image,
                            detection.bboxes,
                            detection.box_scores,
                            relevance_score,
                            detection.score,
                            detector_backend=detection.backend,
                            mode="accept",
                        )
                        save_jpeg(preview_path, preview, jpeg_quality)

                        for bbox in detection.bboxes:
                            metadata_rows.append(
                                (
                                    filename,
                                    source_url,
                                    bbox[0],
                                    bbox[1],
                                    bbox[2],
                                    bbox[3],
                                )
                            )

                        stats.accepted += 1
                        accepted_total += 1
                    elif decision == "review":
                        filename = f"review_{counters.review:06d}.jpg"
                        counters.review += 1

                        review_image_path = paths["review_images"] / filename
                        review_preview_path = paths["review_preview"] / filename

                        save_jpeg(review_image_path, image, jpeg_quality)
                        review_preview = draw_preview(
                            image,
                            detection.bboxes,
                            detection.box_scores,
                            relevance_score,
                            detection.score,
                            detector_backend=detection.backend,
                            mode="review",
                        )
                        save_jpeg(review_preview_path, review_preview, jpeg_quality)
                        stats.review += 1
                    else:
                        filename = f"trash_{counters.trash:06d}.jpg"
                        counters.trash += 1
                        save_jpeg(paths["trash"] / filename, image, jpeg_quality)
                        stats.rejected += 1

                    submit_next(executor)

                    if accepted_total >= target_count:
                        break

        for future in pending:
            future.cancel()

    return stats, metadata_rows, accepted_total


def main() -> int:
    args = parse_args()
    args = run_wizard_if_needed(args)
    args = apply_profile_overrides(args)

    if args.count and args.count > 0:
        target_count = int(args.count)
    else:
        target_count = ask_for_count()

    paths = prepare_dirs(args.output_dir.resolve())
    metadata_path = paths["accepted"] / "metadata.csv"
    print_cli_header()

    print_run_plan(args=args, target_count=target_count, dataset_root=paths["root"])

    detector = BillboardDetector(
        detector_mode=args.detector,
        yolo_model_name=args.yolo_world_model,
        yolo_accept_conf=args.yolo_accept_conf,
        yolo_review_conf=args.yolo_review_conf,
        yolo_iou=args.yolo_iou,
        yolo_imgsz=args.yolo_imgsz,
        device=args.device,
        max_boxes_per_image=args.max_boxes_per_image,
    )

    if args.detector in {"auto", "yolo-world"}:
        if detector.yolo_available:
            print(
                f"[info] Детектор: {detector.yolo_backend_label} "
                f"({args.yolo_world_model}) + cv2 fallback"
            )
        else:
            print(
                "[warn] ML детектор недоступен, использую cv2 fallback. "
                f"Причина: {detector.yolo_unavailable_reason}"
            )
    else:
        print("[info] Детектор: cv2")

    counters = init_counters(paths)

    if args.clean_existing or args.only_clean:
        clean_stats = clean_existing_dataset(
            paths=paths,
            counters=counters,
            detector=detector,
            max_side=args.max_side,
            jpeg_quality=args.jpeg_quality,
            min_det_score=args.min_det_score,
            review_threshold=args.review_threshold,
            yolo_accept_conf=args.yolo_accept_conf,
            yolo_review_conf=args.yolo_review_conf,
        )
        print(
            "[info] Чистка completed: "
            f"kept={clean_stats.kept}, "
            f"review={clean_stats.moved_to_review}, "
            f"trash={clean_stats.moved_to_trash}"
        )
        counters = init_counters(paths)
        kept_rows, removed_rows = prune_metadata_to_existing(metadata_path, paths["images"])
        if removed_rows > 0:
            print(
                "[info] Metadata синхронизирован с accepted/images: "
                f"kept_rows={kept_rows}, removed_rows={removed_rows}"
            )

        if args.only_clean:
            save_support_files(paths)
            print("[info] only-clean завершен.")
            return 0

    current_accepted_count = count_jpg_files(paths["images"])
    kept_rows, removed_rows = prune_metadata_to_existing(metadata_path, paths["images"])
    if removed_rows > 0:
        print(
            "[info] Metadata синхронизирован с accepted/images: "
            f"kept_rows={kept_rows}, removed_rows={removed_rows}"
        )
    print(f"[info] Уже есть accepted/images: {current_accepted_count}")

    if current_accepted_count >= target_count:
        save_support_files(paths)
        print("[info] Цель уже достигнута. Новый парсинг не требуется.")
        return 0

    needed = target_count - current_accepted_count

    known_urls = load_known_urls(metadata_path)
    known_hashes = load_existing_hashes(paths, include_trash=False)

    print(f"[info] Уникальных URL из metadata: {len(known_urls)}")
    print(f"[info] Уникальных hash в dataset: {len(known_hashes)}")

    urls = collect_image_urls(
        queries=args.queries,
        need_count=needed,
        pool_multiplier=args.pool_multiplier,
        timeout=args.timeout,
        known_urls=known_urls,
    )
    if not urls:
        print("[error] Не удалось получить URL для скачивания.")
        return 1

    print(f"[info] Найдено URL для обработки: {len(urls)}")

    stats, metadata_rows, accepted_total = process_urls(
        urls=urls,
        target_count=target_count,
        current_accepted_count=current_accepted_count,
        paths=paths,
        timeout=args.timeout,
        workers=args.workers,
        max_side=args.max_side,
        jpeg_quality=args.jpeg_quality,
        min_det_score=args.min_det_score,
        review_threshold=args.review_threshold,
        yolo_accept_conf=args.yolo_accept_conf,
        yolo_review_conf=args.yolo_review_conf,
        detector=detector,
        known_urls=known_urls,
        known_hashes=known_hashes,
        counters=counters,
    )

    save_support_files(paths)
    append_metadata_rows(metadata_path, metadata_rows)

    print("\n===== Результат =====")
    print(f"Скачано изображений: {stats.downloaded}")
    print(f"Не скачались/не декодировались: {stats.download_failed}")
    print(f"Дубликаты (URL/hash): {stats.duplicates}")
    print(f"Принято в accepted: +{stats.accepted}")
    print(f"Отправлено в review: +{stats.review}")
    print(f"Отправлено в trash: +{stats.rejected}")
    print(f"Итог accepted/images: {accepted_total}")
    print(f"Accepted images: {paths['images']}")
    print(f"Accepted labels: {paths['labels']}")
    print(f"Accepted preview: {paths['preview']}")
    print(f"Review images: {paths['review_images']}")
    print(f"Review preview: {paths['review_preview']}")
    print(f"Trash: {paths['trash']}")

    if accepted_total < target_count:
        print(
            "[warn] До цели не дошли. Можно увеличить --pool-multiplier "
            "или запустить снова (скрипт продолжит без сброса)."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
