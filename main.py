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
DEFAULT_TRAIN_MODEL = "yolo11n.pt"
FALLBACK_TRAIN_MODEL = "yolov8n.pt"
DEFAULT_PRETRAINED_MODEL = Path("yolov8m-worldv2.pt")

TRAIN_PROFILES = {
    "lite": {
        "epochs": 16,
        "imgsz": 640,
        "batch": 2,
        "workers": 0,
        "patience": 8,
    },
    "balanced": {
        "epochs": 28,
        "imgsz": 768,
        "batch": 4,
        "workers": 1,
        "patience": 12,
    },
}


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


def list_local_pt_models() -> list[Path]:
    return sorted(Path.cwd().glob("*.pt"))


def choose_model_file(
    console: Console,
    title: str,
    default_model: Path | str | None = None,
) -> Path:
    models = list_local_pt_models()

    if models:
        if RICH_AVAILABLE and Table is not None:
            table = Table(title=title, header_style="bold green")
            table.add_column("№", style="cyan", no_wrap=True)
            table.add_column("File", style="white")
            for idx, model in enumerate(models, start=1):
                table.add_row(str(idx), str(model))
            console.print(table)
        else:
            console.print(title)
            for idx, model in enumerate(models, start=1):
                console.print(f"{idx}. {model}")

        use_local = Confirm.ask("Use a local .pt file from this list?", default=True)
        if use_local:
            while True:
                raw_idx = Prompt.ask(f"Model number (1..{len(models)})", default="1").strip()
                if raw_idx.isdigit():
                    idx = int(raw_idx)
                    if 1 <= idx <= len(models):
                        return models[idx - 1]
                console.print("[red]Enter a valid number.[/red]")

    default_path = Path(default_model).expanduser() if default_model else None
    return ask_existing_path(
        console=console,
        prompt_text="Path to model weights (.pt)",
        default_path=default_path,
    )


def choose_pretrained_model(console: Console) -> Path:
    models = list_local_pt_models()
    world_models = [m for m in models if "world" in m.name.lower()]
    if world_models:
        default_model: Path | str = world_models[0]
    elif models:
        default_model = models[0]
    else:
        default_model = DEFAULT_PRETRAINED_MODEL

    return choose_model_file(
        console=console,
        title="Local YOLO weights found in this project",
        default_model=default_model,
    )


def choose_training_model(console: Console) -> str:
    models = list_local_pt_models()
    preferred_local = None
    for model in models:
        if "world" not in model.name.lower():
            preferred_local = model
            break
    if preferred_local is None and models:
        preferred_local = models[0]

    use_local = Confirm.ask(
        "Use an already downloaded local model for training start?",
        default=preferred_local is not None,
    )
    if use_local:
        default_model: Path | str = preferred_local if preferred_local is not None else DEFAULT_TRAIN_MODEL
        chosen = choose_model_file(
            console=console,
            title="Choose local base weights for training",
            default_model=default_model,
        )
        return str(chosen)

    return Prompt.ask(
        "Base model name (will download if missing)",
        default=DEFAULT_TRAIN_MODEL,
    ).strip()


def find_local_train_fallback_model() -> str | None:
    for model in list_local_pt_models():
        name = model.name.lower()
        if "world" in name:
            continue
        return str(model)
    return None


def train_on_dataset(
    console: Console,
    dataset_yaml: Path,
    base_model: str,
    epochs: int,
    imgsz: int,
    device: str,
    batch: int,
    workers: int,
    patience: int,
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

    def run_train(current_model: YOLO) -> None:
        current_model.train(
            data=str(dataset_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            workers=workers,
            patience=patience,
            cache=False,
            single_cls=True,
            plots=False,
            amp=False,
            device=device,
            project=str(Path("runs") / "ooh_train"),
            name="billboard",
            exist_ok=False,
        )

    try:
        run_train(model)
    except Exception as exc:
        local_fallback = find_local_train_fallback_model()
        fallback_model = local_fallback or FALLBACK_TRAIN_MODEL
        if str(base_model) == str(fallback_model):
            raise
        console.print(
            f"[yellow]Training with '{base_model}' failed: {exc}. "
            f"Retrying with '{fallback_model}'.[/yellow]"
        )
        model = YOLO(fallback_model)
        run_train(model)

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

    device = Prompt.ask("Device (cpu, mps, or 0)", default="cpu")

    if mode == "train":
        dataset_yaml = ask_existing_path(
            console,
            "Path to dataset.yaml",
            default_path=DEFAULT_DATASET_YAML,
        )

        train_profile = Prompt.ask(
            "Training profile",
            choices=["lite", "balanced", "custom"],
            default="lite",
        )
        if train_profile in TRAIN_PROFILES:
            preset = TRAIN_PROFILES[train_profile]
            epochs = int(preset["epochs"])
            imgsz = int(preset["imgsz"])
            batch = int(preset["batch"])
            workers = int(preset["workers"])
            patience = int(preset["patience"])
        else:
            epochs = IntPrompt.ask("Epochs", default=16)
            imgsz = IntPrompt.ask("Image size (imgsz)", default=640)
            batch = IntPrompt.ask("Batch size", default=2)
            workers = IntPrompt.ask("Data loader workers", default=0)
            patience = IntPrompt.ask("Patience", default=8)

        console.print(
            f"[info] Training config: epochs={epochs}, imgsz={imgsz}, "
            f"batch={batch}, workers={workers}, patience={patience}"
        )

        base_model = choose_training_model(console)

        model_path = train_on_dataset(
            console=console,
            dataset_yaml=dataset_yaml,
            base_model=base_model,
            epochs=epochs,
            imgsz=imgsz,
            device=device,
            batch=batch,
            workers=workers,
            patience=patience,
        )
    else:
        model_path = choose_pretrained_model(console)
        imgsz = IntPrompt.ask("Image size (imgsz) for inference", default=960)

    image_path = ask_existing_path(console, "Path to image for detection")
    conf_accept = FloatPrompt.ask("Confidence threshold (accept)", default=0.16)
    conf_review = FloatPrompt.ask("Confidence threshold (review)", default=0.08)

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
