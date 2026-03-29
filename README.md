# OOH Ad Parser (Multi-source + YOLO-World)

Скрипт собирает изображения наружной рекламы из нескольких источников, детектит billboard через ML (`YOLO-World`) и готовит датасет под YOLO.

## Что теперь есть

- источники: `Bing`, `Yandex Images`, `DuckDuckGo Images`, `Openverse`, `Wikimedia Commons`, `Flickr Public Feed`;
- русские + английские запросы (расширяются автоматически);
- детектор по умолчанию: `YOLO-World` (`auto` режим);
- мульти-детекция: несколько billboard bbox на одном изображении (multi-object YOLO labels);
- fallback на `cv2`, если YOLO-World недоступен;
- поддержка своих весов `YOLO11/YOLO12` в generic-режиме (через `--yolo-world-model <ваши_веса>.pt`);
- профили запуска: `fast` / `balanced` / `quality`;
- удобный мастер запуска: `--wizard` (минимум ручных флагов);
- `review/` для сомнительных кадров (чтобы не терять полезные фото);
- автоматическая очистка уже собранного `accepted`;
- продолжение с текущего прогресса (без сброса).

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Первый запуск `YOLO-World` может скачать веса модели.
Если видите `YOLO-World недоступен` или `DuckDuckGo отключен`, просто доустановите зависимости из `requirements.txt` в активном venv.

## Рекомендуемый запуск

```bash
python3 ooh_parser.py --wizard
```

Или без мастера, через профиль:

```bash
python3 ooh_parser.py --count 1000 --profile balanced --detector auto --clean-existing --workers 12
```

## `main.py`: обучение + инференс

`main.py` — это единый интерактивный CLI:
- режим `1`: обучить модель на `dataset_ooh/accepted/dataset.yaml`;
- режим `2`: взять готовые веса YOLO (`.pt`);
- после выбора модель запускается на твоём изображении и сохраняет preview в ту же папку, что и исходный файл (`*_billboards_preview.jpg`).

Запуск:

```bash
python3 main.py
```

## Полезные режимы

Только перечистить текущий `accepted`:

```bash
python3 ooh_parser.py --only-clean --clean-existing
```

Жестко использовать только CV2 (без ML):

```bash
python3 ooh_parser.py --count 1000 --detector cv2
```

## Важные параметры

- `--profile fast|balanced|quality`
- `--wizard`
- `--detector auto|yolo-world|cv2`
- `--max-boxes-per-image`
- `--yolo-world-model` (по умолчанию для `balanced`: `yolov8m-worldv2.pt`)
- `--yolo-accept-conf` / `--yolo-review-conf`
- `--workers`
- `--pool-multiplier`
- `--max-side`
- `--jpeg-quality`
- `--clean-existing` / `--only-clean`

## Выходные папки

- `accepted/images`, `accepted/labels`, `accepted/preview`
- `accepted/metadata.csv`
- `review/images`, `review/preview`
- `trash`

## Примечание

Идеального auto-labeling не бывает, но с `YOLO-World + review` качество разметки обычно заметно выше, чем у чистого CV2.
