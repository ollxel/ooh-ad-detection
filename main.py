#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re

import cv2
import numpy as np
from ultralytics import YOLO

from ooh_parser import BillboardDetector

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
    from rich.table import Table

    RICH_AVAILABLE = True
except Exception:
    RICH_AVAILABLE = False

    class Console:  # type: ignore[override]
        def print(self, *args, **kwargs) -> None:
            cleaned = []
            for value in args:
                if isinstance(value, str):
                    cleaned.append(re.sub(r"\[/?[^\]]+\]", "", value))
                else:
                    cleaned.append(value)
            print(*cleaned)

    class Prompt:  # type: ignore[override]
        @staticmethod
        def ask(
            prompt_text: str,
            choices: list[str] | None = None,
            default: str | None = None,
        ) -> str:
            while True:
                suffix = f" [{default}]" if default is not None else ""
                raw = input(f"{prompt_text}{suffix}: ").strip()
                if not raw and default is not None:
                    raw = default
                if choices and raw not in choices:
                    print(f"Допустимые значения: {', '.join(choices)}")
                    continue
                return raw

    class IntPrompt:  # type: ignore[override]
        @staticmethod
        def ask(prompt_text: str, default: int = 0) -> int:
            while True:
                raw = input(f"{prompt_text} [{default}]: ").strip()
                if not raw:
                    return int(default)
                if raw.isdigit():
                    return int(raw)
                print("Введите целое число.")

    class FloatPrompt:  # type: ignore[override]
        @staticmethod
        def ask(prompt_text: str, default: float = 0.0) -> float:
            while True:
                raw = input(f"{prompt_text} [{default}]: ").strip()
                if not raw:
                    return float(default)
                try:
                    return float(raw)
                except ValueError:
                    print("Введите число.")

    class Confirm:  # type: ignore[override]
        @staticmethod
        def ask(prompt_text: str, default: bool = True) -> bool:
            d = "Y/n" if default else "y/N"
            raw = input(f"{prompt_text} [{d}]: ").strip().lower()
            if not raw:
                return default
            return raw in {"y", "yes", "1", "да", "д"}

    Panel = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]

DEFAULT_DATASET_YAML = Path("dataset_ooh/accepted/dataset.yaml")
DEFAULT_TRAIN_MODEL = "yolo11s.pt"
FALLBACK_TRAIN_MODEL = "yolov8m.pt"
DEFAULT_PRETRAINED_MODEL = Path("yolov8m-worldv2.pt")


def print_header(console: Console) -> None:
    if RICH_AVAILABLE and Panel is not None:
        console.print(
            Panel.fit(
                "[bold cyan]OOH Trainer & Inference[/bold cyan]\n"
                "[dim]Тренировка на твоём датасете или запуск готовой YOLO-модели[/dim]",
                border_style="cyan",
            )
        )
        return

    console.print("==============================================")
    console.print("OOH Trainer & Inference")
    console.print("Тренировка на твоём датасете или запуск готовой YOLO-модели")
    console.print("==============================================")


def choose_mode(console: Console) -> str:
    if RICH_AVAILABLE and Table is not None:
        table = Table(title="Выбери сценарий", header_style="bold magenta")
        table.add_column("№", style="cyan", no_wrap=True)
        table.add_column("Режим", style="white")
        table.add_row("1", "Обучить модель на разметке из ooh_parser (dataset.yaml)")
        table.add_row("2", "Использовать уже скачанную/готовую YOLO модель")
        console.print(table)
    else:
        console.print("Выбери сценарий:")
        console.print("1. Обучить модель на разметке из ooh_parser (dataset.yaml)")
        console.print("2. Использовать уже скачанную/готовую YOLO модель")

    while True:
        choice = Prompt.ask("Твой выбор", choices=["1", "2"], default="1")
        if choice == "1":
            return "train"
        if choice == "2":
            return "pretrained"


def ask_existing_path(console: Console, prompt_text: str, default_path: Path | None = None) -> Path:
    while True:
        raw = Prompt.ask(prompt_text, default=str(default_path) if default_path else "").strip()
        path = Path(raw).expanduser()
        if path.exists():
            return path
        console.print(f"[red]Файл не найден:[/red] {path}")


def choose_pretrained_model(console: Console) -> Path:
    models = sorted(Path.cwd().glob("*.pt"))
    world_models = [m for m in models if "world" in m.name.lower()]

    if world_models:
        default_model = world_models[0]
    elif DEFAULT_PRETRAINED_MODEL.exists():
        default_model = DEFAULT_PRETRAINED_MODEL
    elif models:
        default_model = models[0]
    else:
        default_model = DEFAULT_PRETRAINED_MODEL

    if models:
        if RICH_AVAILABLE and Table is not None:
            table = Table(title="Найденные веса в проекте", header_style="bold green")
            table.add_column("№", style="cyan", no_wrap=True)
            table.add_column("Файл", style="white")
            for idx, model in enumerate(models, start=1):
                table.add_row(str(idx), str(model))
            console.print(table)
        else:
            console.print("Найденные веса в проекте:")
            for idx, model in enumerate(models, start=1):
                console.print(f"{idx}. {model}")

    use_list = Confirm.ask("Выбрать из найденных .pt файлов?", default=True)
    if use_list and models:
        while True:
            raw_idx = Prompt.ask(f"Номер модели (1..{len(models)})", default="1").strip()
            if raw_idx.isdigit():
                idx = int(raw_idx)
                if 1 <= idx <= len(models):
                    return models[idx - 1]
            console.print("[red]Введите корректный номер.[/red]")

    return ask_existing_path(
        console,
        "Путь к весам модели (.pt)",
        default_path=default_model,
    )


def train_on_dataset(
    console: Console,
    dataset_yaml: Path,
    base_model: str,
    epochs: int,
    imgsz: int,
    device: str,
) -> Path:
    console.print("[info] Запуск обучения... это может занять время.")

    model_name = base_model
    try:
        model = YOLO(model_name)
    except Exception as exc:
        console.print(
            f"[yellow]Не удалось загрузить {base_model}: {exc}. "
            f"Переключаюсь на {FALLBACK_TRAIN_MODEL}.[/yellow]"
        )
        model_name = FALLBACK_TRAIN_MODEL
        model = YOLO(model_name)

    model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        project=str(Path("runs") / "ooh_train"),
        name="billboard",
        exist_ok=False,
    )

    best_path: Path | None = None
    trainer = getattr(model, "trainer", None)
    if trainer is not None:
        best_candidate = getattr(trainer, "best", None)
        if best_candidate:
            best_path = Path(str(best_candidate))

    if best_path is None or not best_path.exists():
        # fallback: найти последний best.pt
        candidates = sorted((Path("runs") / "ooh_train").glob("**/weights/best.pt"))
        if candidates:
            best_path = candidates[-1]

    if best_path is None or not best_path.exists():
        raise RuntimeError("После обучения не найден best.pt")

    console.print(f"[green]Обучение завершено. Лучшие веса:[/green] {best_path}")
    return best_path


def render_preview(
    image: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    scores: list[float],
    backend: str,
) -> np.ndarray:
    preview = image.copy()

    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(preview, (x1, y1), (x2, y2), (35, 220, 35), 3)
        if idx < len(scores):
            cv2.putText(
                preview,
                f"{scores[idx]:.2f}",
                (x1 + 4, max(22, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (35, 220, 35),
                2,
                cv2.LINE_AA,
            )

    label = f"billboards={len(boxes)} | backend={backend}"
    cv2.putText(preview, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(preview, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
    return preview


def detect_on_image(
    console: Console,
    model_path: Path,
    image_path: Path,
    device: str,
    conf_accept: float,
    conf_review: float,
    imgsz: int,
) -> Path:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Не удалось прочитать изображение: {image_path}")

    detector = BillboardDetector(
        detector_mode="yolo-world",
        yolo_model_name=str(model_path),
        yolo_accept_conf=conf_accept,
        yolo_review_conf=conf_review,
        yolo_iou=0.45,
        yolo_imgsz=imgsz,
        device=device,
        max_boxes_per_image=50,
    )

    detection = detector.detect(image)
    preview = render_preview(
        image=image,
        boxes=detection.bboxes,
        scores=detection.box_scores,
        backend=detection.backend,
    )

    out_path = image_path.with_name(f"{image_path.stem}_billboards_preview.jpg")
    ok = cv2.imwrite(str(out_path), preview)
    if not ok:
        raise RuntimeError(f"Не удалось сохранить preview: {out_path}")

    if detection.bboxes:
        console.print(f"[green]Найдено биллбордов:[/green] {len(detection.bboxes)}")
    else:
        console.print("[yellow]Биллборды не найдены. Preview всё равно сохранён.[/yellow]")

    return out_path


def main() -> int:
    console = Console()
    print_header(console)

    mode = choose_mode(console)

    device = Prompt.ask("Устройство для инференса/тренировки (cpu или 0)", default="cpu")

    if mode == "train":
        dataset_yaml = ask_existing_path(
            console,
            "Путь к dataset.yaml",
            default_path=DEFAULT_DATASET_YAML,
        )
        base_model = Prompt.ask(
            "Базовая модель для обучения (YOLO11/YOLO12/YOLOv8)",
            default=DEFAULT_TRAIN_MODEL,
        ).strip()
        epochs = IntPrompt.ask("Epochs", default=40)
        imgsz = IntPrompt.ask("Image size (imgsz)", default=960)

        model_path = train_on_dataset(
            console=console,
            dataset_yaml=dataset_yaml,
            base_model=base_model,
            epochs=epochs,
            imgsz=imgsz,
            device=device,
        )
    else:
        model_path = choose_pretrained_model(console)
        imgsz = IntPrompt.ask("Image size (imgsz) для детекции", default=960)

    image_path = ask_existing_path(console, "Путь к изображению для детекции")
    conf_accept = FloatPrompt.ask("Confidence порог (accept)", default=0.16)
    conf_review = FloatPrompt.ask("Confidence порог (review)", default=0.08)

    preview_path = detect_on_image(
        console=console,
        model_path=model_path,
        image_path=image_path,
        device=device,
        conf_accept=conf_accept,
        conf_review=conf_review,
        imgsz=imgsz,
    )

    console.print(f"[bold green]Preview сохранён:[/bold green] {preview_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
