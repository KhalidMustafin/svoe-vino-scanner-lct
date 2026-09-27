"""Сервис на фейках: каталог из шести позиций, индекс из трёх координат цвета, VLM без сети.

Визуальный канал — `ColorEmbedder`: вектор кадра — его средний цвет. Кадр одного цвета режется
на виды, но у каждого вида тот же цвет, поэтому косинусы кандидатов заданы векторами индекса:

    красный кадр   alfa-muskat 0,99, beta-merlot 0,98, остальные ≤ 0,5 — CV-top1 «Мускат»,
                   но «Мерло» идёт вплотную и поднимается текстом «Бета Холмы / Мерло»;
    синий кадр     все косинусы ниже 0,75 — «картинка сама себе не верит» для правила отказа.

Модель зрения — настоящий `OllamaVlmReader` с подставным транспортом: он отвечает текстом,
падает по таймауту или недоступности, как это делал бы Ollama. Модель resolve собрана руками:
вес у счёта CV и у согласия винодельни и сорта, остальные нули.

Модуль не собирается pytest (имя не начинается с `test_`).
"""

from __future__ import annotations

import io
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from catalog import wine_record
from fakes import ColorEmbedder
from learned_synth import ALFA, BETA, GAMMA
from PIL import Image

from app.api.cards import CatalogCards
from app.api.config import ServiceSettings
from app.api.service import ScannerService, reader_identity
from app.features.contracts import IndexMeta
from app.features.index import VisualIndex
from app.reading.lexicon.build import build_from_records
from app.reading.readers.ollama_vlm import OllamaVlmReader
from app.resolve.attrs import CatalogAttrs
from app.resolve.features import FEATURE_VERSION, feature_names, feature_signs
from app.resolve.learned import LogisticRanker
from app.sommelier.settings import SommSettings

CV_MODEL = "fake/color"
READER = "vlm35"
FAKE_VLM = "fake:vlm"
#: Строка читателя, чьи чтения «видела» модель фейков: тот же `OllamaVlmReader`, что у сервиса.
FAKE_READER_KEY = reader_identity(OllamaVlmReader(FAKE_VLM))
RED = (255, 0, 0)
BLUE = (0, 0, 255)

API_RECORDS: list[dict[str, Any]] = [
    wine_record(
        "alfa-muskat", **ALFA, name="Мускат", grapes=["muscat"], grape_values=["Мускат"],
        sugar="suhoe", color="Белое",
    ),
    wine_record(
        "beta-merlot", **BETA, name="Мерло Терруар", cuvee=["терруар"], grapes=["merlot"],
        grape_values=["Мерло"], color="Красное",
    ),
    wine_record(
        "beta-shardone", **BETA, name="Шардоне Резерв", grapes=["chardonnay"],
        grape_values=["Шардоне"], serial_keywords=["reserve"], color="Белое",
    ),
    wine_record(
        "gamma-saperavi", **GAMMA, name="Саперави", grapes=["saperavi"],
        grape_values=["Саперави"], color="Красное",
    ),
    wine_record(
        "gamma-rkatsiteli", **GAMMA, name="Ркацители Янтарь", cuvee=["янтарь"],
        grapes=["rkatsiteli"], grape_values=["Ркацители"], color="Белое",
    ),
    wine_record(
        "alfa-riesling", **ALFA, name="Рислинг", grapes=["riesling"],
        grape_values=["Рислинг"], color="Белое",
    ),
]  # fmt: skip
SLUGS = [record["slug"] for record in API_RECORDS]

#: Вектор эталона: (косинус с красным, остаток в плоскости «зелёный», z — косинус с синим).
_VECTORS: dict[str, tuple[float, float]] = {
    "alfa-muskat": (0.99, 0.0),
    "beta-merlot": (0.98, 0.10),
    "beta-shardone": (0.50, 0.30),
    "gamma-saperavi": (0.45, 0.20),
    "gamma-rkatsiteli": (0.40, 0.25),
    "alfa-riesling": (0.35, 0.15),
}


def _vector(red: float, blue: float) -> list[float]:
    green = math.sqrt(max(0.0, 1.0 - red * red - blue * blue))
    return [red, green, blue]


def make_index(model: str = CV_MODEL) -> VisualIndex:
    """Индекс: по одному вектору вида `bottle` на позицию."""
    vectors = np.asarray([_vector(*_VECTORS[slug]) for slug in SLUGS], dtype=np.float32)
    meta = IndexMeta(model=model, dim=3, views=["bottle"], n_slugs=len(SLUGS), n_vectors=len(SLUGS))
    return VisualIndex(list(SLUGS), ["bottle"] * len(SLUGS), vectors, meta)


def make_model(
    top_k: int = 20,
    weights: dict[str, float] | None = None,
    reader_keys: dict[str, list[str]] | None = None,
) -> LogisticRanker:
    """Модель resolve с весами «руками»: средние 0 и σ 1 — вес равен вкладу сырого признака.

    `meta.reader_keys` — как у `bench.train_resolve`: по умолчанию читатель фейков.
    """
    names = feature_names([READER])
    weights = weights or {
        "cv_score": 5.0,
        f"{READER}.winery_match": 3.0,
        f"{READER}.grape_match": 2.0,
    }
    data = {
        "format": "svs-resolve-logistic/1",
        "feature_version": FEATURE_VERSION,
        "feature_names": names,
        "coef": [weights.get(name, 0.0) for name in names],
        "intercept": 0.0,
        "mean": [0.0] * len(names),
        "scale": [1.0] * len(names),
        "temperature": 1.0,
        "l2": 0.01,
        "loss": "listwise",
        "signs": feature_signs(names),
        "fit_info": {},
        "meta": {
            "top_k": top_k,
            "trained_at": "2026-09-18T00:00:00+00:00",
            "variant": READER,
            "reader_keys": reader_keys if reader_keys is not None else {READER: [FAKE_READER_KEY]},
        },
    }
    return LogisticRanker.from_dict(data)


class FakeOllama:
    """Транспорт `OllamaVlmReader`: отвечает заданным текстом или падает заданной ошибкой.

    `clock` и `seconds` — сколько «идёт» ответ по подставным часам сервиса.
    """

    def __init__(
        self,
        text: str = "",
        *,
        error: BaseException | None = None,
        thinking: str = "",
        clock: FakeClock | None = None,
        seconds: float = 0.0,
    ) -> None:
        self.text = text
        self.error = error
        self.thinking = thinking
        self.clock = clock
        self.seconds = seconds
        self.calls: list[dict[str, Any]] = []

    def __call__(self, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
        self.calls.append({"model": payload["model"], "timeout_s": timeout_s})
        if self.clock is not None:
            self.clock.now += self.seconds
        if self.error is not None:
            raise self.error
        return {
            "message": {"content": self.text, "thinking": self.thinking},
            "done_reason": "stop",
            "eval_count": 10,
            "prompt_eval_count": 300,
        }


class FakeClock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class SlowEmbedder(ColorEmbedder):
    """Эмбеддер, у которого считать вектор «долго» по подставным часам."""

    def __init__(self, clock: FakeClock, seconds: float, model_name: str = CV_MODEL) -> None:
        super().__init__(model_name)
        self.clock = clock
        self.seconds = seconds

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        self.clock.now += self.seconds
        return super().embed(images)


class BrokenEmbedder(ColorEmbedder):
    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        raise RuntimeError("CUDA out of memory")


def settings_for(tmp_path: Path | None = None, **overrides: Any) -> ServiceSettings:
    base = Path(tmp_path) if tmp_path is not None else Path("unused")
    values: dict[str, Any] = {
        "index_path": base / "index.npz",
        "cv_model": CV_MODEL,
        "resolve_model": base / "resolve.json",
        "vlm_model": FAKE_VLM,
        "device": "cpu",
        "lexicon_path": base / "lexicon.json",
        "attrs_path": base / "gt_tokens.jsonl",
        "catalog_csv": base / "catalog.csv",
        "photo_map": base / "photo_map.csv",
        # Данные слоя «после поиска» — тоже под `base`: иначе тесты подхватили бы настоящий
        # справочник из data/ дерева, где он лежит, и зависели бы от машины.
        "photo_dir": base / "photos",
        "wines_path": base / "wines.jsonl",
        "somm": SommSettings(data_dir=base / "somm"),
        # Индекс подделок — 3-мерные цвета: карта адаптера продукта (1152) к нему не подходит, и
        # по умолчанию подделки идут путём `off`. Адаптер в тестах — свой (test_fund_candidates.py).
        "candidate": "off",
        "cv_adapter": None,
    }
    values.update(overrides)
    return ServiceSettings(**values)


def make_service(
    transport: Callable[..., dict[str, Any]] | None = None,
    *,
    settings: ServiceSettings | None = None,
    embedder: Any = None,
    model: LogisticRanker | None = None,
    clock: Callable[[], float] | None = None,
    vlm: bool = True,
    cards: CatalogCards | None = None,
) -> ScannerService:
    settings = settings or settings_for()
    reader = (
        OllamaVlmReader(
            settings.vlm_model,
            # Адрес — из настроек, как у `ScannerService.load`: голос сомелье зеркалит его.
            settings.ollama_url,
            transport=transport or FakeOllama(""),
            probe=lambda: {"models": [{"name": settings.vlm_model}]},
        )
        if vlm
        else None
    )
    kwargs: dict[str, Any] = {}
    if clock is not None:
        kwargs["clock"] = clock
    return ScannerService(
        settings,
        index=make_index(),
        embedder=embedder or ColorEmbedder(CV_MODEL),
        lexicon=build_from_records(API_RECORDS),
        attrs=CatalogAttrs.from_records(API_RECORDS),
        model=model or make_model(settings.top_k),
        vlm=reader,
        cards=cards or CatalogCards.build(API_RECORDS),
        **kwargs,
    )


def image_bytes(
    color: Sequence[int] = RED, fmt: str = "PNG", size: tuple[int, int] = (64, 96), **save: Any
) -> bytes:
    """Одноцветный кадр в байтах заданного формата."""
    mode = "RGBA" if len(color) == 4 else "RGB"
    picture = Image.new(mode, size, tuple(color))
    buffer = io.BytesIO()
    picture.save(buffer, format=fmt, **save)
    return buffer.getvalue()


def write_files(tmp_path: Path, *, index_model: str = CV_MODEL, top_k: int = 20) -> None:
    """Файлы сборки на диске — для `ScannerService.load` и отказа старта."""
    make_index(index_model).save(tmp_path / "index.npz")
    make_model(top_k).save(tmp_path / "resolve.json")
    with (tmp_path / "gt_tokens.jsonl").open("w", encoding="utf-8") as fh:
        for record in API_RECORDS:
            row = {
                **record,
                "published": True,
                "region": "Кубань",
                "live_category": "Красное сухое",
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    build_from_records(API_RECORDS).save(tmp_path / "lexicon.json")
    (tmp_path / "catalog.csv").write_text(
        "Название вина,Категория,Цвет,Регион,Сорт винограда,Описание,Винодельня,Slug,Название фото\n"
        " Мерло Терруар,Красное,Рубиновый,Кубань,Мерло,Вишня и слива.,Бета Холмы,beta-merlot,m.webp\n"
        " Мерло Терруар,Красное,Рубиновый,Кубань,Мерло,Вишня и слива.,Бета Холмы,beta-merlot,m.webp\n",
        encoding="utf-8",
    )
    (tmp_path / "photo_map.csv").write_text(
        "slug,path\nbeta-merlot,C:/photos/beta-merlot.webp\n", encoding="utf-8"
    )


def jq_slug(body: str) -> str | None:
    """Фильтр `participant_test.sh` на Python: что скрипт запишет в `predicted_slug`.

    if type == "object" then .slug
    elif type == "array" and length > 0 then .[0].slug
    else empty end | select(type == "string" and length > 0)
    """
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict):
        value = data.get("slug")
    elif isinstance(data, list) and data:
        value = data[0].get("slug") if isinstance(data[0], dict) else None
    else:
        return None
    return value if isinstance(value, str) and value else None
