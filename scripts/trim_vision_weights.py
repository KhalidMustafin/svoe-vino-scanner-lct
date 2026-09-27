"""Только зрительная башня SigLIP: 4,54 ГБ чекпойнта → 1,71 ГБ для переноса на сервер.

Зачем. Сервис поднимает `SiglipVisionModel` (`app/features/embedder.py`), то есть половину
чекпойнта: текстовая башня и `logit_scale` в память не идут вовсе — transformers честно
помечает их UNEXPECTED. Но скачивается и везётся весь файл. На сервере, куда Hugging Face
отдаёт 60 КБ/с, разница между 4,5 и 1,7 ГБ — это разница между «везти нельзя» и «везти час».

Скрипт собирает готовую раскладку кэша Hugging Face, куда сервис смотрит сам:

    models--google--siglip2-so400m-patch14-384/
        refs/main                      тот же коммит, что в исходном кэше
        snapshots/<sha>/config.json
        snapshots/<sha>/preprocessor_config.json
        snapshots/<sha>/model.safetensors   только ключи vision_model.*

Точность не трогаем: тензоры копируются как есть, float32 в float32. Перевод в float16
уменьшил бы файл вдвое, но это уже другие числа — а значит другие косинусы, другой top-20 и
другие ответы стенда. Ради экономии канала подменять замер нельзя.

    python scripts/trim_vision_weights.py                 # + проверка forward-проходом
    python scripts/trim_vision_weights.py --no-verify     # без проверки (быстрее, но зря)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

MODEL_ID = "google/siglip2-so400m-patch14-384"
CACHE_NAME = "models--google--siglip2-so400m-patch14-384"
#: Что нужно самому сервису: веса зрительной башни, её настройки и настройки препроцессора.
KEEP_PREFIX = "vision_model."
SIDE_FILES = ("config.json", "preprocessor_config.json")


def find_cache(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    for candidate in (
        Path.home() / ".cache" / "huggingface" / "hub" / CACHE_NAME,
        Path.home() / ".cache" / "huggingface" / CACHE_NAME,
    ):
        if candidate.is_dir():
            return candidate
    raise SystemExit(f"не нашёл кэш {CACHE_NAME}; укажите --cache")


def trim(cache: Path, out_root: Path) -> tuple[Path, Path, int, int]:
    from safetensors import safe_open
    from safetensors.torch import save_file

    sha = (cache / "refs" / "main").read_text(encoding="utf-8").strip()
    snapshot = cache / "snapshots" / sha
    source = snapshot / "model.safetensors"
    if not source.is_file():
        raise SystemExit(f"нет весов: {source}")

    target_root = out_root / CACHE_NAME
    target_snapshot = target_root / "snapshots" / sha
    target_snapshot.mkdir(parents=True, exist_ok=True)
    (target_root / "refs").mkdir(parents=True, exist_ok=True)
    (target_root / "refs" / "main").write_text(sha, encoding="utf-8")
    for name in SIDE_FILES:
        if (snapshot / name).is_file():
            shutil.copy2(snapshot / name, target_snapshot / name)

    kept: dict = {}
    dropped = 0
    with safe_open(str(source), framework="pt") as handle:
        for key in handle.keys():  # noqa: SIM118 — у safe_open это метод, не словарь
            if key.startswith(KEEP_PREFIX):
                kept[key] = handle.get_tensor(key)
            else:
                dropped += 1
    if not kept:
        raise SystemExit(f"в чекпойнте нет ключей {KEEP_PREFIX}* — раскладка изменилась")
    save_file(kept, str(target_snapshot / "model.safetensors"), metadata={"format": "pt"})
    return target_root, target_snapshot, len(kept), dropped


def verify(cache_root: Path, trimmed_root: Path) -> bool:
    """Один и тот же кадр через обе модели: числа должны совпасть до бита."""
    import torch
    from transformers import SiglipVisionModel

    def load(hub_root: Path):
        model = SiglipVisionModel.from_pretrained(
            MODEL_ID, cache_dir=str(hub_root), local_files_only=True, dtype=torch.float32
        )
        return model.eval()

    torch.manual_seed(0)
    pixels = torch.randn(1, 3, 384, 384)
    with torch.inference_mode():
        full = load(cache_root.parent)(pixel_values=pixels).pooler_output
        thin = load(trimmed_root.parent)(pixel_values=pixels).pooler_output
    diff = (full - thin).abs().max().item()
    print(f"проверка forward-проходом: расхождение {diff:.3e}")
    return diff == 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=None, help=f"каталог {CACHE_NAME}")
    parser.add_argument(
        "--out", type=Path, default=Path.cwd() / "siglip-vision-hub", help="куда собрать"
    )
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args(argv)

    cache = find_cache(args.cache)
    before = sum(f.stat().st_size for f in cache.rglob("*") if f.is_file())
    root, snapshot, kept, dropped = trim(cache, args.out)
    after = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())

    print(f"исходный кэш : {before / 2**30:.2f} ГБ ({cache})")
    print(f"для переноса : {after / 2**30:.2f} ГБ ({root})")
    print(f"тензоров взято {kept}, отброшено {dropped}")

    if not args.no_verify:
        if not verify(cache, root):
            print("ЧИСЛА РАЗОШЛИСЬ — такие веса везти нельзя", file=sys.stderr)
            return 1
        print("числа совпали до бита: на сервере будут те же ответы")

    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    print(f"модель {config.get('model_type', '?')}, коммит {root.name}")
    print(f"\nОтправить на сервер:\n    cd {args.out} && tar cf - {CACHE_NAME} | \\")
    print("        ssh -p ПОРТ root@IP 'mkdir -p /opt/svs/hf/hub && tar xf - -C /opt/svs/hf/hub'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
