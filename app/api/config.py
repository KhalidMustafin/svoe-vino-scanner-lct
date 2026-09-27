"""Настройки сервиса из переменных окружения `SVS_*`.

Значения по умолчанию — схема, на которой мерили итог test (`bench/README.md`,
`configs/resolve/README.md`): индекс `visual-s2so400m.npz` и модель
`google/siglip2-so400m-patch14-384`, VLM `qwen3.5:4b` по целому кадру 1024 px (замер шёл с
бюджетом чтения 2 500 мс, сервис даёт 5 000 из общих 8 000), обученный resolve поверх top-20 CV. Модель resolve, агрегация CV и разбор полей
этикетки от замера отличаются. С 26.09 по умолчанию — кандидат 2 фундаментального трека
`adapter-lw-ranker` (`SVS_CANDIDATE`, ниже): адаптер поиска LW (`app/features/adapter.py`,
карта `index/cv-adapter-lw.npz` пачки данных) и `s2so400m-vlm35-lw-pool.json` — ранкер рецепта
`-goal`, переобученный на пуле по выдаче адаптера; принят по замороженному тесту kr-test
(`research/2026-09-26_fund/PREREG_final.md`). Прежний путь — `SVS_CANDIDATE=off`: поиск без
адаптера и `s2so400m-vlm35-goal.json` — модель цикла улучшений (iter 20: CV `per_slug="zmax"`,
признаки `resolve-features/3`); до неё — `s2so400m-vlm35-m2.json`, переобученная на полях после
правки родов цвета и сахара. Цифры 85,2 % / 74,6 % получены моделью `s2so400m-vlm35.json` на
полях до правки и с `per_slug="max"` (с нынешним кодом она не загружается).

Параметры замера, которые настройкой не меняются, — константы ниже: вид кропа VLM, его
размер, точность и агрегация CV. Их смена — это уже другой замер.

`SVS_DATA_DIR`, `SVS_DATASET_DIR`, `SVS_OLLAMA_URL`, `SVS_VLM_MODEL` и `SVS_DEVICE` общие
со стендами (`app.config.Settings`). Одно отличие: модель VLM сервиса по умолчанию
`qwen3.5:4b` (замер), а у стендов — `qwen3-vl:4b-instruct`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args

from app.config import REPO_ROOT, Settings
from app.sommelier.settings import FALSE_WORDS, TRUE_WORDS, SommSettings
from app.sommelier.settings import SettingsError as SommSettingsError

#: Отказ «вина нет в каталоге»: `off` — всегда slug; `ooc_only` — правило S10 из `rerank`.
AbstainPolicy = Literal["off", "ooc_only"]
ABSTAIN_POLICIES: tuple[str, ...] = get_args(AbstainPolicy)

DEFAULT_CV_MODEL = "google/siglip2-so400m-patch14-384"
DEFAULT_VLM_MODEL = "qwen3.5:4b"
DEFAULT_INDEX_NAME = "visual-s2so400m.npz"
DEFAULT_LEXICON_NAME = "lexicon.json"
DEFAULT_ATTRS_NAME = "gt_tokens.jsonl"
#: Модель resolve пути `SVS_CANDIDATE=off` (продукт до 26.09) и кандидата `adapter-lw`.
GOAL_RESOLVE_MODEL = REPO_ROOT / "configs" / "resolve" / "s2so400m-vlm35-goal.json"
#: Модель resolve кандидата `adapter-lw-ranker` — продукта по умолчанию с 26.09.
POOL_RESOLVE_MODEL = REPO_ROOT / "configs" / "resolve" / "s2so400m-vlm35-lw-pool.json"
#: Объявленные производные разметки каталога (`app/api/lineage.py`): правки карточек Э4 и
#: комплект живых карточек Э3 поверх gt, на котором обучена модель resolve.
GT_LINEAGE_PATH = REPO_ROOT / "configs" / "resolve" / "gt_lineage.json"
CATALOG_CSV_NAME = "strapi_output0709.csv"

#: Живые карточки (`SVS_LIVE_CARDS`, Э3 плана точности 25.09): второй комплект «индекс + gt +
#: словарь» — каталог CSV и 71 карточка живого портала, чьего вина в CSV нет
#: (`scripts/build_live_set.py`). Флаг переключает умолчания трёх путей вместе; нормировка
#: `zmax` в живом индексе заморожена по строкам CSV (`base_rows`), и счёт карточек CSV тот же.
#: Модель resolve обучена на gt CSV; живой gt — объявленная производная (`configs/resolve/
#: gt_lineage.json`, шаг `live71`), согласованная с моделью только при включённом флаге.
LIVE_INDEX_NAME = "visual-s2so400m-live71.npz"
LIVE_LEXICON_NAME = "lexicon-live71.json"
LIVE_ATTRS_NAME = "gt_tokens-live71.jsonl"
#: Умолчание — по воротам kr из PREREG Э3 (`research/2026-09-25_acc/E3_RESULT.md`): на 617
#: основных кадрах krasnostop ровно одна смена ответа на другое вино (K0629 → карточка портала),
#: поэтому флаг выключен и включается только по ответу организатора «сверяем по порталу».
LIVE_CARDS_DEFAULT = False

#: Путь поиска и выбора — `SVS_CANDIDATE` (фундаментальный трек 26.09,
#: `research/2026-09-26_fund/PREREG_final.md`). Кандидаты меняют только обучаемые части, правила
#: H5 / P1 / Э2 и разбор этикетки те же:
#:
#:     adapter-lw-ranker  по умолчанию: адаптер поиска LW (обученное выбеливание SigLIP,
#:                        `app/features/adapter.py`) и ранкер рецепта `-goal`, переобученный на пуле
#:                        по выдаче адаптера. Принят по одному просмотру kr-test: 292/309 = 94,5 %
#:                        «то же вино» против 277 = 89,6 % у пути `off` (18 / 3, p = 0,0015)
#:     off (none)         прежний продукт `after-search`: поиск без адаптера, ранкер `-goal` —
#:                        бит в бит (`research/2026-09-26_fund/ship/ship_gate.py`)
#:     adapter-lw         кандидат 1 — адаптер с ранкером `-goal`; не принят (R-оригиналы 61 < 62),
#:                        только для повторов замера, с предупреждением в `/v1/health`
#:
#: Карта адаптера собрана под векторы индекса CSV (`CV_ADAPTER_INDEX_SHA1`): с другим индексом, в
#: том числе с живыми карточками (`SVS_LIVE_CARDS=1`), сервис с адаптером не стартует — нужен
#: `SVS_CANDIDATE=off`.
Candidate = Literal["off", "adapter-lw", "adapter-lw-ranker"]
CANDIDATES: tuple[str, ...] = get_args(Candidate)
CANDIDATE_DEFAULT: Candidate = "adapter-lw-ranker"
#: Слова `SVS_CANDIDATE`, которые выключают кандидата: прежний путь без адаптера.
CANDIDATE_OFF_WORDS = frozenset({"off", "none"})
#: Кандидаты, не принятые по kr-test: включаются только для повторов замера, с предупреждением.
CANDIDATES_NOT_ACCEPTED = frozenset({"adapter-lw"})
#: Файл адаптера в пачке данных (рядом с индексом: карта собрана для его векторов).
CV_ADAPTER_NAME = "cv-adapter-lw.npz"
#: Паспорт карты кандидата 2 (`PREREG_final.md`, §2.1): sha1 содержимого (`LinearAdapter.sha1`;
#: его же ждёт `meta.cv_adapter_sha1` модели `-lw-pool`) и sha1 индекса, под векторы которого она
#: собрана. Сервис сверяет карту с моделью и с загруженным индексом при старте
#: (`app/api/service.py`: `load_cv_adapter`, `ScannerService._check`), тесты — эти числа с файлами.
CV_ADAPTER_SHA1 = "5d25e5c66ecd0034a018a7a6e43f092aba60daf8"
CV_ADAPTER_INDEX_SHA1 = "ccd3a01f16379a3f0a8a1e7e8dc0623c7622bda5"
#: Порог подсказки «похоже, этой бутылки нет в каталоге» (`after.suggest_not_found`) на шкале счёта
#: карты — по sha1 её содержимого. Перекрывает `meta.suggest_not_found_visual_max` файла карты: сам
#: файл (sha1 `206627c7…`, пачка данных и тесты) не меняется. Карта не из словаря берёт порог из меты.
#: Правило то же, что у 0,4299 в мете (`final/build_final.py`): наибольший порог, при котором флаг
#: стоит не больше чем на 2 % кадров каталога по выдаче карты вне фолда. Перекалибровка 26.09
#: (`research/2026-09-26_ooc/suggest_calibration.json`): кадры каталога — v2 353, kr-dev 308 и
#: R-оригиналы 65 вместе (у меты — только v2); вне каталога проверено на ooc_v2 408 и 600 студийных
#: снимках krasnostop вне каталога. На predict и top-1 порог не влияет.
SUGGEST_NOT_FOUND_VISUAL_MAX_BY_ADAPTER: dict[str, float] = {CV_ADAPTER_SHA1: 0.4379}
#: Модель resolve каждого пути: `-lw-pool` учился на выдаче адаптера, `-goal` — без него.
CANDIDATE_RESOLVE_MODELS: dict[str, Path] = {
    "off": GOAL_RESOLVE_MODEL,
    "adapter-lw": GOAL_RESOLVE_MODEL,
    "adapter-lw-ranker": POOL_RESOLVE_MODEL,
}
#: Модель resolve сервиса по умолчанию — модель пути по умолчанию.
DEFAULT_RESOLVE_MODEL = CANDIDATE_RESOLVE_MODELS[CANDIDATE_DEFAULT]


def candidate_from_text(value: str) -> str:
    """Значение `SVS_CANDIDATE` как имя пути: пусто — умолчание, `none` — `off`, регистр не важен."""
    name = value.strip().lower() or CANDIDATE_DEFAULT
    return "off" if name in CANDIDATE_OFF_WORDS else name


#: Параметры замера, которые не настраиваются (`bench.retrieval --target none --per-slug zmax`,
#: `bench.ocr_bench --crop full --crop-px 1024`). `zmax` — максимум по косинусам, выровненным
#: по парам «окно × вид» (`app.features.index.align_pairs`); модель resolve училась на таких счётах.
VLM_CROP = "full"
VLM_CROP_PX = 1024
CV_DTYPE = "float32"
CV_PER_SLUG = "zmax"

#: Читатель, чьи чтения видели модели замера: `id@модель|params_hash` из поля `reader`
#: прогонов `runs/ocr-pairs-vlm35` и `runs/ocr-pairsphone-vlm35` (и их пересборок из кэша `-m2`).
#: Хэш собран из модели, промпта, размера кадра, `num_predict`, `num_ctx`, температуры и
#: постобработки `OllamaVlmReader`: другой промпт в коде — уже другой читатель, и цифры замера к
#: нему не относятся. Сверяется со строкой читателя сервиса, если в `meta.reader_keys` модели
#: своей записи нет (так у `s2so400m-vlm35.json`; у `-m2` и `-goal` запись есть и равна этой
#: строке).
MEASURED_READER_NAME = "vlm35"
MEASURED_VLM_READER = "vlm@qwen3.5:4b|f3a017317f04"

#: Бюджет VLM в замере (`bench.ocr_bench`, `DEFAULT_BUDGET_MS` модуля чтения).
MEASURED_VLM_TIMEOUT_MS = 2500

#: Сколько оставить после чтения этикетки на разбор текста, resolve и ответ.
POST_READ_RESERVE_MS = 400
#: Меньше этого на разжатие кадра, виды и CV — и VLM получит урезанный бюджет почти на каждом
#: кадре: 12 Мп WebP разжимается 0,2–0,3 с, SigLIP so400m по видам на GPU — десятые доли секунды.
MIN_CV_ROOM_MS = 1500

#: `curl --max-time 10` в скрипте организатора: ответ позже — null, как бы он ни был хорош.
SCRIPT_MAX_TIME_MS = 10_000

_MB = 1024 * 1024


class SettingsError(ValueError):
    """Переменная окружения задана, но не разбирается или выходит за пределы."""


def _default_data_dir() -> Path:
    return REPO_ROOT / "data"


def catalog_files(data_dir: Path, live_cards: bool) -> tuple[Path, Path, Path]:
    """Индекс, gt и словарь комплекта: CSV или CSV + живые карточки (`SVS_LIVE_CARDS`)."""
    if live_cards:
        names = (LIVE_INDEX_NAME, LIVE_ATTRS_NAME, LIVE_LEXICON_NAME)
    else:
        names = (DEFAULT_INDEX_NAME, DEFAULT_ATTRS_NAME, DEFAULT_LEXICON_NAME)
    return data_dir / "index" / names[0], data_dir / "gt" / names[1], data_dir / "index" / names[2]


@dataclass(frozen=True)
class ServiceSettings:
    """Всё, что сервис читает при старте. Пути — абсолютные или от текущего каталога."""

    index_path: Path = field(
        default_factory=lambda: catalog_files(_default_data_dir(), LIVE_CARDS_DEFAULT)[0]
    )
    cv_model: str = DEFAULT_CV_MODEL
    #: Модель resolve; умолчание — модель пути по умолчанию (`CANDIDATE_RESOLVE_MODELS`).
    resolve_model: Path = DEFAULT_RESOLVE_MODEL
    vlm_model: str = DEFAULT_VLM_MODEL
    ollama_url: str = "http://127.0.0.1:11434"
    device: str = "cuda"
    #: Общий бюджет ответа: скрипт ждёт 10 с, запас — на разжатие 12 Мп и сеть.
    budget_ms: int = 8000
    #: Бюджет чтения этикетки: организатор ставит точность выше скорости (скорость — 5 баллов из
    #: 100), а обрезанное чтение — прямой проигрыш. Кадр целиком укладывается в
    #: p95 1,5 с, так что запас тратится только на редкий хвост (максимум замера — 4,0 с).
    vlm_timeout_ms: int = 5000
    abstain: AbstainPolicy = "off"
    top_k: int = 20
    host: str = "0.0.0.0"
    port: int = 8080
    lexicon_path: Path = field(
        default_factory=lambda: catalog_files(_default_data_dir(), LIVE_CARDS_DEFAULT)[2]
    )
    attrs_path: Path = field(
        default_factory=lambda: catalog_files(_default_data_dir(), LIVE_CARDS_DEFAULT)[1]
    )
    #: Комплект с живыми карточками (`SVS_LIVE_CARDS`): от него — умолчания трёх путей выше.
    live_cards: bool = LIVE_CARDS_DEFAULT
    catalog_csv: Path = field(
        default_factory=lambda: _default_data_dir() / "raw" / "dataset" / CATALOG_CSV_NAME
    )
    photo_map: Path = field(
        default_factory=lambda: _default_data_dir() / "catalog" / "slug_photo_map.csv"
    )
    #: Лёгкие копии фото каталога `<slug>.webp` (`scripts/make_photo_pack.py`). Нет каталога —
    #: фото ищется по `photo_map` и справочнику рекомендаций.
    photo_dir: Path | None = field(
        default_factory=lambda: _default_data_dir() / "catalog" / "photos_small"
    )
    #: Наш словарь групп вин (копия `field_dataset/catalog/wines_final.jsonl`); рядом с ним
    #: `wine_groups.json`. Из него берутся только группы «то же вино», ключ винодельни и коды
    #: сортов выгрузки. Нет файла — каждое вино само себе группа.
    wines_path: Path = field(
        default_factory=lambda: _default_data_dir() / "catalog" / "wines.jsonl"
    )
    #: Сомелье (`SVS_SOMM_*`, договор `docs/api-sommelier.md`, §6.7): выключатели голоса и поля
    #: вопроса, каталог данных `data/somm/*.json`, предел голоса и тихое окно ворот. Один объект на
    #: сервис: те же данные читают и «Сомелье у полки» (чип «К чему?»), и маршруты сомелье.
    somm: SommSettings = field(default_factory=SommSettings)
    max_upload_bytes: int = 25 * _MB
    #: Сколько Ollama держит модель после запроса: сервис не должен остывать между сканами.
    vlm_keep_alive: str = "24h"
    #: Путь поиска и выбора (`SVS_CANDIDATE`): по умолчанию `adapter-lw-ranker`, `off` — прежний
    #: продукт без адаптера.
    candidate: Candidate = CANDIDATE_DEFAULT
    #: Файл адаптера поиска (`SVS_CV_ADAPTER`, по умолчанию `data/index/cv-adapter-lw.npz`);
    #: `None` — поиск без адаптера, только при `candidate="off"`.
    cv_adapter: Path | None = field(
        default_factory=lambda: _default_data_dir() / "index" / CV_ADAPTER_NAME
    )

    def __post_init__(self) -> None:
        problems: list[str] = []
        if self.budget_ms <= 0:
            problems.append(f"SVS_BUDGET_MS должен быть > 0, получено {self.budget_ms}")
        if self.vlm_timeout_ms < 0:
            problems.append(f"SVS_VLM_TIMEOUT_MS не может быть < 0, получено {self.vlm_timeout_ms}")
        if self.vlm_timeout_ms > self.budget_ms:
            problems.append(
                f"SVS_VLM_TIMEOUT_MS ({self.vlm_timeout_ms}) больше общего бюджета "
                f"SVS_BUDGET_MS ({self.budget_ms})"
            )
        if self.abstain not in ABSTAIN_POLICIES:
            problems.append(
                f"SVS_ABSTAIN: {', '.join(ABSTAIN_POLICIES)}; получено {self.abstain!r}"
            )
        if self.top_k <= 0:
            problems.append(f"SVS_TOP_K должен быть > 0, получено {self.top_k}")
        if not 0 < self.port < 65536:
            problems.append(f"SVS_PORT вне 1–65535: {self.port}")
        if self.max_upload_bytes <= 0:
            problems.append(f"SVS_MAX_UPLOAD_MB должен быть > 0, получено {self.max_upload_bytes}")
        if not self.cv_model.strip():
            problems.append("SVS_CV_MODEL пуст")
        if not self.vlm_model.strip():
            problems.append("SVS_VLM_MODEL пуст")
        if self.candidate not in CANDIDATES:
            problems.append(f"SVS_CANDIDATE: {_candidate_choices()}; получено {self.candidate!r}")
        elif self.candidate == "off" and self.cv_adapter is not None:
            problems.append(
                "SVS_CV_ADAPTER задан, а SVS_CANDIDATE=off: адаптер включает только кандидат"
            )
        elif self.candidate != "off" and self.cv_adapter is None:
            problems.append(f"SVS_CANDIDATE={self.candidate}: нужен файл адаптера SVS_CV_ADAPTER")
        elif self.candidate != "off" and self.live_cards:
            problems.append(
                f"SVS_LIVE_CARDS=1 и SVS_CANDIDATE={self.candidate}: карта адаптера поиска собрана "
                "под индекс CSV, а у живых карточек свой индекс; с живыми карточками — "
                "SVS_CANDIDATE=off (прежний путь без адаптера)"
            )
        if problems:
            raise SettingsError("; ".join(problems))

    @property
    def somm_dir(self) -> Path:
        """Каталог данных сомелье (`SVS_SOMM_DIR`); нет файлов — заглушки договора или без блюд."""
        return self.somm.data_dir

    @property
    def vlm_enabled(self) -> bool:
        """`SVS_VLM_TIMEOUT_MS=0` выключает чтение этикетки: resolve решает по одной картинке."""
        return self.vlm_timeout_ms > 0

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None, *, base: Settings | None = None
    ) -> ServiceSettings:
        """Настройки из окружения. Пустая переменная — значение по умолчанию."""
        env = os.environ if environ is None else environ
        if base is None:
            base = _base_settings(env)

        def text(name: str, default: str) -> str:
            value = (env.get(name) or "").strip()
            return value or default

        def path(name: str, default: Path) -> Path:
            value = (env.get(name) or "").strip()
            return Path(value).expanduser().resolve() if value else default

        def integer(name: str, default: int) -> int:
            value = (env.get(name) or "").strip()
            if not value:
                return default
            try:
                return int(value)
            except ValueError:
                raise SettingsError(f"{name}: ожидается целое число, получено {value!r}") from None

        def flag(name: str, default: bool) -> bool:
            value = (env.get(name) or "").strip().lower()
            if not value:
                return default
            if value in TRUE_WORDS:
                return True
            if value in FALSE_WORDS:
                return False
            raise SettingsError(f"{name}: 1/true/on/yes или 0/false/off/no; получено {value!r}")

        abstain = text("SVS_ABSTAIN", "off")
        if abstain not in ABSTAIN_POLICIES:
            raise SettingsError(f"SVS_ABSTAIN: {', '.join(ABSTAIN_POLICIES)}; получено {abstain!r}")
        try:
            somm = SommSettings.from_env(env, data_dir=base.data_dir)
        except SommSettingsError as exc:
            raise SettingsError(str(exc)) from None
        candidate = candidate_from_text(env.get("SVS_CANDIDATE") or "")
        if candidate not in CANDIDATES:
            raise SettingsError(f"SVS_CANDIDATE: {_candidate_choices()}; получено {candidate!r}")
        if candidate == "off" and (env.get("SVS_CV_ADAPTER") or "").strip():
            raise SettingsError(
                "SVS_CV_ADAPTER задан, а SVS_CANDIDATE=off: адаптер включает только кандидат"
            )
        cv_adapter = (
            path("SVS_CV_ADAPTER", base.data_dir / "index" / CV_ADAPTER_NAME)
            if candidate != "off"
            else None
        )
        resolve_default = CANDIDATE_RESOLVE_MODELS[candidate]
        # Индекс, gt и словарь — одним комплектом: явный путь главнее, но только своего файла.
        live_cards = flag("SVS_LIVE_CARDS", LIVE_CARDS_DEFAULT)
        index_file, attrs_file, lexicon_file = catalog_files(base.data_dir, live_cards)
        return cls(
            index_path=path("SVS_INDEX_PATH", index_file),
            cv_model=text("SVS_CV_MODEL", DEFAULT_CV_MODEL),
            resolve_model=path("SVS_RESOLVE_MODEL", resolve_default),
            vlm_model=text("SVS_VLM_MODEL", DEFAULT_VLM_MODEL),
            ollama_url=base.ollama_url,
            device=base.device,
            # Умолчания берутся из полей класса, а не повторяются числом: 20.09 бюджеты подняли в
            # полях, а здесь остались 6000/2500, и `python -m app.api` резал чтение на 2,5 с.
            budget_ms=integer("SVS_BUDGET_MS", cls.budget_ms),
            vlm_timeout_ms=integer("SVS_VLM_TIMEOUT_MS", cls.vlm_timeout_ms),
            abstain=abstain,  # type: ignore[arg-type]
            top_k=integer("SVS_TOP_K", cls.top_k),
            host=text("SVS_HOST", cls.host),
            port=integer("SVS_PORT", cls.port),
            lexicon_path=path("SVS_LEXICON_PATH", lexicon_file),
            attrs_path=path("SVS_ATTRS_PATH", attrs_file),
            live_cards=live_cards,
            catalog_csv=path("SVS_CATALOG_CSV", base.dataset_dir / CATALOG_CSV_NAME),
            photo_map=path("SVS_PHOTO_MAP", base.data_dir / "catalog" / "slug_photo_map.csv"),
            # Каталог лёгких фото общий с полевым стендом: на VPS он задан как SVS_FIELD_PHOTO_DIR.
            photo_dir=path(
                "SVS_PHOTO_DIR",
                path("SVS_FIELD_PHOTO_DIR", base.data_dir / "catalog" / "photos_small"),
            ),
            wines_path=path("SVS_WINES_PATH", base.data_dir / "catalog" / "wines.jsonl"),
            somm=somm,
            max_upload_bytes=integer("SVS_MAX_UPLOAD_MB", 25) * _MB,
            vlm_keep_alive=text("SVS_VLM_KEEP_ALIVE", "24h"),
            candidate=candidate,  # type: ignore[arg-type]
            cv_adapter=cv_adapter,
        )

    def public(self) -> dict[str, Any]:
        """Настройки для `/v1/health` и журнала: пути строками."""
        out = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(self).items()
        }
        out["somm"] = self.somm.public()
        return out

    @property
    def cv_room_ms(self) -> int:
        """Сколько бюджета остаётся на разжатие, виды и CV, если VLM получит свой бюджет целиком."""
        return self.budget_ms - self.vlm_timeout_ms - POST_READ_RESERVE_MS

    def warnings(self) -> list[str]:
        """Допустимые, но рискованные значения: сервис стартует, а в журнале будет предупреждение.

        Каждое предупреждение — отступление от цепочки замера или риск для скрипта организатора.
        Список отдаётся и в `/v1/health` (`warnings`): отчётный прогон `run_eval.sh` без
        `--allow-degraded` с ним не идёт.
        """
        out: list[str] = []
        if self.budget_ms >= SCRIPT_MAX_TIME_MS:
            out.append(
                f"SVS_BUDGET_MS={self.budget_ms}: скрипт организатора ждёт {SCRIPT_MAX_TIME_MS} мс "
                "вместе с сетью и загрузкой файла — ответ может не успеть"
            )
        if self.vlm_timeout_ms < MEASURED_VLM_TIMEOUT_MS:
            # Больший бюджет чтение не портит: он лишь не обрывает редкий длинный кадр. Меньший —
            # обрывает, и признаки текста приходят беднее, чем при обучении resolve.
            out.append(
                f"SVS_VLM_TIMEOUT_MS={self.vlm_timeout_ms}: меньше бюджета замера "
                f"{MEASURED_VLM_TIMEOUT_MS} мс — чтение будет обрываться"
            )
        if self.vlm_enabled and self.vlm_model != DEFAULT_VLM_MODEL:
            out.append(
                f"SVS_VLM_MODEL={self.vlm_model}: resolve обучен на чтениях {DEFAULT_VLM_MODEL}, "
                "чтение другой модели пойдёт под тем же ключом — цифры замера к нему не относятся "
                "(переменная общая со стендами, у них по умолчанию другая модель)"
            )
        if self.vlm_enabled and self.cv_room_ms < MIN_CV_ROOM_MS:
            out.append(
                f"SVS_BUDGET_MS={self.budget_ms} при SVS_VLM_TIMEOUT_MS={self.vlm_timeout_ms}: на "
                f"разжатие и CV остаётся {self.cv_room_ms} мс (нужно ≥ {MIN_CV_ROOM_MS}) — VLM "
                "будет получать урезанный бюджет почти на каждом кадре (`vlm_budget_cut`)"
            )
        if self.candidate in CANDIDATES_NOT_ACCEPTED:
            out.append(
                f"SVS_CANDIDATE={self.candidate}: кандидат фундаментального трека, не принятый по "
                "research/2026-09-26_fund/PREREG_final.md (R-оригиналы 61 < 62) — только для "
                "повторов замера, цифры продукта к нему не относятся"
            )
        if self.abstain != "off":
            out.append(
                "SVS_ABSTAIN=ooc_only: отказ отдаёт slug null, а скрипт организатора засчитывает "
                "только непустой slug; порог отказа на полевых кадрах не проверен"
            )
        return out


def _candidate_choices() -> str:
    return f"{', '.join(CANDIDATES)} (none = off; пусто — {CANDIDATE_DEFAULT})"


def _base_settings(env: Mapping[str, str]) -> Settings:
    """Общие настройки стендов по тому же окружению (а не по `os.environ` процесса)."""

    def path(name: str, default: Path) -> Path:
        value = (env.get(name) or "").strip()
        return Path(value).expanduser().resolve() if value else default

    return Settings(
        dataset_dir=path("SVS_DATASET_DIR", REPO_ROOT / "data" / "raw" / "dataset"),
        data_dir=path("SVS_DATA_DIR", REPO_ROOT / "data"),
        cache_dir=path("SVS_CACHE_DIR", REPO_ROOT / "data" / "cache"),
        ollama_url=(env.get("SVS_OLLAMA_URL") or "").strip() or "http://127.0.0.1:11434",
        vlm_model=(env.get("SVS_VLM_MODEL") or "").strip() or DEFAULT_VLM_MODEL,
        device=(env.get("SVS_DEVICE") or "").strip() or "cuda",
    )
