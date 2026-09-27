"""Заглушка API страницы продукта: «после поиска» и «Сомелье» для вёрстки без бэкенда.

Страница `app/api/static/index.html` и лист сомелье верстаются раньше, чем готов бэкенд. Этот
сервер отдаёт страницу и ответы из заглушек `tests/fixtures/after/*.json` и
`tests/fixtures/somm/*` по маршрутам договоров `docs/api-after-search.md` и
`docs/api-sommelier.md`, поэтому страница ходит ровно по тем адресам, что и в бою.

Запуск из корня рабочего дерева:

    python scripts/ui_stub_server.py                          # http://127.0.0.1:8765/
    python scripts/ui_stub_server.py --data-dir data \
        --csv "…/Датасет/strapi_output0709.csv"

Сценарий скана выбирает параметр `state` (found | check | not_found | error): у самого запроса
`/v1/scan?state=…` или у адреса страницы — `http://127.0.0.1:8765/?state=check`, его сервер
берёт из заголовка Referer. Сама страница о заглушке ничего не знает. `suggest=1` там же
включает подсказку «похоже, вина нет в каталоге» (`after.suggest_not_found`): `found`
становится `check`, к причинам добавляется `visual_low`. `POST /v1/similar/by-label` понимает
`exclude` (до 20 slug): эти вина и их группы (`wine_id`) пропадают из обоих списков.

**Без портала (решение 24.09).** Карточки — по договору «после поиска», §3: полей портала
(`dishes`, `temperature`, `category_gradient`, `live_category`, `published`) нет, крепость —
только из slug выгрузки. Ссылка `portal_url` — как у сервиса: у 66 slug вне карты сайта вин
организатора (`app/recommend/portal_links.json`, список `unlisted`) её нет (`null`). Снимок
портала `data/portal/` заглушка не читает, иконок блюд портала не отдаёт, фото — только
выгрузки организатора: у всех 23 строк `wrong_photo` правки 23.09 (на фото другое вино) фото
нет, у 3 эталонов правки с тем же вином — исходное фото выгрузки.

**Сомелье.** `GET /v1/wines/{slug}/sommelier` — заглушки `sommelier_*.json` для их вин, для
остальных ответ собирается по правилам договора из фактов карточки: оси «из карточки»
(сладость, крепость, пузырьки), подача по таблице §7.3, блюда — из `pairs.json` заглушки, если
вино там есть. Осей «по сорту» у собранного ответа нет: приоров сортов заглушка не знает.
`POST /v1/sommelier/ask` отдаёт поток NDJSON одной из заглушек `ask_*.ndjson` с паузами по их
`t_ms`: вопрос о покупке и цене — отказ, «Помогите выбрать» — наводящий вопрос, чип «А к …?»
— забракованный голос, вопрос о борще — живой, `plain` — обычная сортировка, `somm=error` у
адреса страницы — ошибка сборки, остальное — «К чему подать» при занятой видеокарте.

С каталогом данных (`--data-dir` или `SVS_DATA_DIR`) отдаются настоящие фото выгрузки
(`catalog/photos_small`), карточки вин вне заглушек собираются из `gt/gt_tokens.jsonl`
(производная выгрузки организатора), а похожие считаются грубым ранжированием по фактам.
Выгрузка организатора (`--csv` или `SVS_DATASET_DIR/strapi_output0709.csv`) даёт описание
и оттенок цвета как есть; без неё описание — текст-заглушка. Без каталога данных фото
отвечают 404, и страница рисует свой силуэт бутылки. Вина «Посмотреть на примере» — позиции
выгрузки вне заглушек: их карточки есть только с каталогом данных (без него пример ведёт на
«Карточка не нашлась»), а «роза ветров» у них — лишь оси из карточки, приоров сортов заглушка
не знает. Полный профиль примеров — у живого сервиса.

`?shot=<имя>` у адреса страницы подмешивает `scripts/ui_stub_driver.js`: он проходит сценарий
за человека (18+, выбор фото, кнопки) для скриншотов `scripts/ui_screens.py`. В бою его нет.

Только стандартная библиотека и без импорта `app`: заглушка не зависит от того, какое рабочее
дерево видит окружение, не грузит модели и не трогает видеокарту.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "app" / "api" / "static"
PAGE = STATIC / "index.html"
FIXTURES = ROOT / "tests" / "fixtures" / "after"
SOMM_FIXTURES = ROOT / "tests" / "fixtures" / "somm"
DRIVER = Path(__file__).with_name("ui_stub_driver.js")
CSV_NAME = "strapi_output0709.csv"

STATES = {
    "found": "scan_found",
    "check": "scan_check",
    "not_found": "scan_not_found",
    "error": "scan_error",
}
# Скриншот → сценарий скана, если в адресе нет явного `state`.
SHOT_STATES = {
    "busy": "found",
    "found": "found",
    "taste": "found",
    "dishes": "found",
    "similar": "found",
    "suggest": "found",
}
ORDERS = ("reco", "plain")
#: Договор: в `exclude` у by-label не больше 20 slug.
MAX_EXCLUDE = 20
NOTICE = "Применяются рекомендательные технологии"
PORTAL = "https://vino-svoe.ru/wines/"
#: Поля карточки из снимка и статуса живого портала — в продукте их нет (решение 24.09).
PORTAL_KEYS = ("dishes", "temperature", "category_gradient", "live_category", "published")
#: Фото выгрузки, на котором другое вино: все 23 строки `wrong_photo` правки эталонов 23.09 —
#: тот же список, что `app.recommend.catalog.WRONG_PHOTOS` (тест сверяет). Фото нет, страница
#: рисует силуэт (договор «после поиска», §4). Заглушка `app` не импортирует — список повторён.
WRONG_PHOTO = frozenset(
    {
        "abrau-dyurso-az-abrau-bayanshira-beloe-suhoe-12",
        "agora-yachting-cabernet-sauvignon",
        "agrolayn-heritage-dg-skin-contact-rkatsiteli-rkatsiteli-krasnoe-suhoe-12",
        "agrolayn-mountain-eagle-traminer-traminer-beloe-suhoe-12",
        "belmas-winery-syrah-katya-sira-krasnoe-suhoe-125",
        "belmas-winery-vi-vione-beloe-suhoe-122",
        "chateau-le-grand-vostock-krasnostop-rezerv-krasnoe-suhoe-145",
        "derbent-vino-endemy-saperavi-krasnoe-suhoe-13",
        "derbent-vino-endemy-shardone-beloe-suhoe-13",
        "esse-demi-sec-muscat-nectar-muskat-belyy-beloe-ekstra-bryut-115",
        "fanagoriya-velvet-season-muskat-ottonel-beloe-sladkoe-13",
        "fanagoriya-velvet-season-risling-beloe-sladkoe-12",
        "legato-legato-sovinon-blan-beloe-suhoe-125",
        "method-classic-kokur",
        "novyj-svet-vyderzhannoe-bryut",
        "one-barrel-uan-barrel",
        "pinot-noir-2024-one-barrel-by-dmitry-maslov-pino-nuar-2024-uan-barrel-dmitrij-maslov",
        "skalistyy-bereg-shyopot-tsvetov-risling-beloe-suhoe-109",
        "sober-bash-kaberne-fran-krasnoe-suhoe-127",
        "sober-bash-risling-risling-reynskiy-beloe-suhoe-11",
        "valeriy-zaharin-bastardo-kefesiya-avtohtonnoe-vino-kryma-bastardo-magarachskiy-krasnoe-suhoe-115",
        "vibes-vermentino-viognier-barrel-fermented-2022",
        "vinodelnya-vedernikov-tsimlyanskiy-chernyy-rezerv-krasnoe-suhoe-145",
    }
)
#: Эталоны правки 23.09 с тем же вином, что на фото выгрузки: берётся исходное фото выгрузки.
ORIGINAL_PHOTO = frozenset(
    {
        "merlo-litavshhuk",
        "shardone-rezerv",
        "shato-pino-kaberne-fran-kaberne-sovinon-krasnoe-suhoe-135",
    }
)
#: Что отдаёт `/static` сервиса (`app.api.page.ASSET_SUFFIXES`); заглушка `app` не импортирует.
ASSET_SUFFIXES = frozenset(
    {".js", ".css", ".woff2", ".woff", ".svg", ".png", ".webp", ".jpg", ".jpeg", ".ico"}
)
SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
#: Карта сайта вин из дампа Strapi организатора (`scripts/build_portal_links.py`): у slug из
#: `unlisted` страницы на портале нет, и ссылки на неё нет — как у сервиса.
PORTAL_LINKS = ROOT / "app" / "recommend" / "portal_links.json"


def load_unlisted(path: Path = PORTAL_LINKS) -> frozenset[str]:
    """Slug выгрузки вне карты сайта вин; нет файла — пусто (ссылка у всех, как у сервиса)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return frozenset()
    return frozenset(str(slug) for slug in data.get("unlisted") or [])


UNLISTED = load_unlisted()


def portal_url(slug: str) -> str | None:
    """Ссылка на карточку портала — только у опубликованных (договор «после поиска», §1)."""
    return None if slug in UNLISTED else PORTAL + slug


SUGAR_LABEL = {
    "brut_nature": "брют натюр",
    "extra_brut": "экстра брют",
    "brut": "брют",
    "suhoe": "сухое",
    "polusuhoe": "полусухое",
    "polusladkoe": "полусладкое",
    "sladkoe": "сладкое",
}
BRUT_FAMILY = frozenset({"brut_nature", "extra_brut", "brut"})
#: Правило сахара договора (§3): класс, названный в названии вина (`build_gt_tokens.RU_SUGAR`)…
NAME_SUGAR = (
    (r"экстра[\s\-]*брют|extra[\s\-]*brut", "extra_brut"),
    (r"брют[\s\-]*натюр|brut[\s\-]*nature|zero[\s\-]*dosage|pas[\s\-]*dos", "brut_nature"),
    (r"полусух|semi[\s\-]*dry", "polusuhoe"),
    (r"полуслад|semi[\s\-]*sweet|demi[\s\-]*sec", "polusladkoe"),
    (r"(?<!полу)сухое|(?<!semi[\s\-])\bdry\b", "suhoe"),
    (r"(?<!полу)сладкое|десертн", "sladkoe"),
    (r"(?<!экстра )(?<!экстра-)брют|\bbrut\b", "brut"),
)
#: …иначе последнее слово сахара в slug.
SLUG_SUGAR = re.compile(
    r"(?:^|-)(ekstra-bryut|bryut-natyur|polusladkoe|polusuhoe|sladkoe|suhoe|bryut|brut)(?=-|$)"
)
SLUG_SUGAR_CLASS = {
    "ekstra-bryut": "extra_brut",
    "bryut-natyur": "brut_nature",
    "bryut": "brut",
    "brut": "brut",
}
#: Правило игристости договора: те же слова, что у `after_layer._SPARKLING_WORDS`, и транслит.
SPARKLING_WORDS = re.compile(
    r"игрист|шампанск|шипуч|петнат|просекко|креман"
    r"|sparkling|spumante|frizzante|prosecco|cremant|crémant|pet[\s-]?nat|p[eé]tillant"
    r"|igrist|shampansk",
    re.IGNORECASE,
)
#: Сомелье: чипы договора (§4.2), оси профиля (§1) и таблица подачи (§7.3).
CHIP_IDS = frozenset(
    {
        "what_to_eat",
        "serve",
        "softer",
        "fresher",
        "replace",
        "grape",
        "term",
        "dish_check",
        "guided",
    }
)
BASE_CHIPS = (
    ("what_to_eat", "К чему подать"),
    ("serve", "Как подать"),
    ("softer", "Помягче"),
    ("fresher", "Посвежее"),
    ("replace", "Чем заменить"),
)
AXES = (
    ("sweetness", "Сладость", "сухое", "сладкое"),
    ("acidity", "Кислотность", "мягкая", "живая"),
    ("tannin", "Танины", "мягкие", "терпкие"),
    ("body", "Тело", "лёгкое", "плотное"),
    ("alcohol", "Крепость", "лёгкое", "крепкое"),
    ("oak", "Дуб", "без дуба", "заметный"),
    ("aroma_intensity", "Аромат", "сдержанный", "яркий"),
    ("effervescence", "Пузырьки", "тихое", "игристое"),
)
#: Сладость по сахару (`app/recommend/build.py`, шкала 0–5), в долях шкалы.
SWEETNESS = {
    "brut_nature": 0.0,
    "extra_brut": 0.06,
    "brut": 0.14,
    "suhoe": 0.08,
    "polusuhoe": 0.36,
    "polusladkoe": 0.64,
    "sladkoe": 0.9,
}
SERVE_RULES = (
    ("sparkling", (6, 8)),
    ("sweet", (10, 14)),
    ("orange", (10, 12)),
    ("white", (6, 8)),
    ("rose", (8, 12)),
    ("red", (12, 14)),
)
NOTE_BASIS = "по карточке каталога, правилам подачи и сочетаний"
ALGO = "Текст и подбор — алгоритм"
PURCHASE_WORDS = re.compile(r"купи|покуп|цен[аыуе]|дешев|дорог|стоит|стоимост|доставк|магазин")
TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
}
NDJSON = "application/x-ndjson; charset=utf-8"
# Кадр для предпросмотра, когда каталога данных нет: страница показывает его на экране разбора.
SAMPLE_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 400">'
    '<rect width="300" height="400" fill="#3a3533"/>'
    '<path d="M135 40h30v70c0 8 4 14 10 20a45 45 0 0 1 14 32v190a10 10 0 0 1-10 10h-58a10 10 0 0 1'
    '-10-10V162a45 45 0 0 1 14-32c6-6 10-12 10-20z" fill="#6d2a30"/>'
    '<rect x="115" y="200" width="70" height="90" rx="6" fill="#f3ead2"/></svg>'
)


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list) and value and isinstance(value[0], (int, float)):
        return float(value[0])
    return None


def _deg(value: float) -> str:
    return (f"{value:g}").replace(".", ",") + "°"


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower().replace("ё", "е")


def sugar_of(name: str, slug: str) -> str | None:
    """Сахар по правилу договора: названный в названии вина, иначе последнее слово в slug."""
    text = name.lower()
    found = [code for pattern, code in NAME_SUGAR if re.search(pattern, text)]
    if "extra_brut" in found and "brut" in found:
        found.remove("brut")
    if found:
        return found[0]
    words = SLUG_SUGAR.findall(slug.lower())
    return SLUG_SUGAR_CLASS.get(words[-1], words[-1]) if words else None


def sparkling_of(name: str, slug: str, sugar: str | None) -> bool:
    return sugar in BRUT_FAMILY or bool(SPARKLING_WORDS.search(f"{name} {slug}"))


def style_of(color: str, sugar: str | None, sparkling: bool) -> str:
    """«Красное сухое»; сахар неизвестен — «Белое игристое» у игристого, иначе один цвет."""
    if sugar:
        return f"{color} {SUGAR_LABEL[sugar]}".strip()
    return f"{color} игристое".strip() if sparkling else color


def alcohol_of(value: Any) -> tuple[float | None, float | None]:
    """Крепость из slug (`fields.abv.value`): одно число или диапазон; вне 3–25 — нет."""
    values = [float(v) for v in value or [] if isinstance(v, (int, float))]
    if not values or not all(3 <= v <= 25 for v in values):
        return None, None
    return min(values), (max(values) if len(values) > 1 else None)


class Stub:
    """Ответы договоров: из заглушек, а при каталоге данных — ещё и из него."""

    def __init__(
        self,
        fixtures: Path = FIXTURES,
        data_dir: Path | None = None,
        delay: float = 0.0,
        default_state: str = "found",
        csv_path: Path | None = None,
        somm_fixtures: Path = SOMM_FIXTURES,
    ) -> None:
        self.fx: dict[str, Any] = {
            path.stem: json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(fixtures.glob("*.json"))
        }
        self.somm_fx: dict[str, Any] = {
            path.stem: json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(somm_fixtures.glob("*.json"))
        }
        self.streams: dict[str, list[dict[str, Any]]] = {
            path.stem: [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            for path in sorted(somm_fixtures.glob("ask_*.ndjson"))
        }
        self.placeholder = str(self.fx["wine_card"]["description"])
        self.card_keys = tuple(self.somm_fx["wine_card_noportal"])
        self.sommeliers = {
            str(body["slug"]): body
            for name, body in self.somm_fx.items()
            if name.startswith("sommelier_")
        }
        self.cards: dict[str, dict[str, Any]] = {}
        noportal = self.somm_fx["wine_card_noportal"]
        self.cards[str(noportal["slug"])] = noportal
        for name in ("wine_card", *STATES.values()):
            card = self.fx[name] if name == "wine_card" else self.fx[name].get("card")
            if card:
                self.cards.setdefault(str(card["slug"]), card)
        # Вина из плиток и кандидатов заглушек: без каталога данных их карточка собирается из
        # того, что о них известно, — иначе тап по варианту упирался бы в 404.
        self.tiles: dict[str, dict[str, Any]] = {}
        for body in self.fx.values():
            for key in ("candidates", "items", "same_winery", "similar"):
                for item in body.get(key) or [] if isinstance(body, dict) else []:
                    if isinstance(item, dict) and item.get("slug"):
                        self.tiles.setdefault(str(item["slug"]), item)
        # Якорь заглушки похожих: его карточки в заглушках нет, но маршрут известен.
        self.anchors = {str(self.fx["similar"]["slug"])} if "similar" in self.fx else set()
        self.data_dir = data_dir if data_dir and data_dir.is_dir() else None
        self.delay = delay
        self.default_state = default_state
        self.wines: dict[str, dict[str, Any]] = {}
        self.gt: dict[str, dict[str, Any]] = {}
        if self.data_dir:
            wines = self.data_dir / "catalog" / "wines.jsonl"
            if wines.is_file():
                for line in wines.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        row = json.loads(line)
                        self.wines[str(row["slug"])] = row
            gt = self.data_dir / "gt" / "gt_tokens.jsonl"
            if gt.is_file():
                for line in gt.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        row = json.loads(line)
                        self.gt[str(row["slug"])] = row
        self.csv: dict[str, dict[str, str]] = {}
        if csv_path and csv_path.is_file():
            with csv_path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    self.csv.setdefault(str(row.get("Slug") or ""), row)

    # ---------------------------------------------------------------- скан и карточка

    def scan(self, state: str, suggest: bool = False) -> dict[str, Any]:
        body = copy.deepcopy(self.fx[STATES.get(state, STATES["found"])])
        if body.get("card"):
            body["card"] = self.card(str(body["card"]["slug"])) or self.noportal(body["card"])
        body["candidates"] = self._organizer(body.get("candidates") or [])
        for cand in body["candidates"]:
            cand["photo_url"] = self.photo_url(str(cand["slug"]), cand.get("photo_url"))
        after = body.get("after")
        if isinstance(after, dict):
            # Живой сервис уже пишет вопрос в заглушку скана (`after.question`); у старой
            # заглушки без него вопрос — из случая договора с тем же прочитанным.
            after["question"] = (
                (after.get("question") or self.question(after, body))
                if after.get("state") == "check"
                else None
            )
        if suggest and isinstance(after, dict) and after.get("state") != "error":
            # Как в договоре: подсказка не делает not_found и не трогает slug — только found
            # становится check, а к причинам добавляется visual_low.
            after["suggest_not_found"] = True
            if after.get("state") == "found":
                after["state"] = "check"
                after["question"] = self.question(after, body)
            if "visual_low" not in after.setdefault("reasons", []):
                after["reasons"].append("visual_low")
        return body

    def question(self, after: dict[str, Any], body: dict[str, Any]) -> dict[str, Any] | None:
        """`after.question` (договор сомелье, §5) — из случая заглушки с тем же прочитанным."""
        slugs = [str(c.get("slug")) for c in body.get("candidates") or []]
        for case in (self.somm_fx.get("scan_check_question") or {}).get("cases", []):
            case_slugs = [str(c["slug"]) for c in case["candidates"]]
            if case["read"] == after.get("read") and case_slugs == slugs[: len(case_slugs)]:
                return copy.deepcopy(case["question"])
        return None

    def card(self, slug: str) -> dict[str, Any] | None:
        if slug in self.cards:
            return self.noportal(self.cards[slug])
        return self._card_from_data(slug) or self._card_from_tile(slug)

    def known(self, slug: str) -> bool:
        return slug in self.anchors or self.card(slug) is not None

    def has_sommelier(self, slug: str) -> bool:
        """Ответ сомелье есть у вина заглушки сомелье и у любой известной карточки."""
        return slug in self.sommeliers or self.card(slug) is not None

    def noportal(self, card: dict[str, Any]) -> dict[str, Any]:
        """Карточка без полей портала (договор «после поиска», §3)."""
        out = {k: v for k, v in copy.deepcopy(card).items() if k not in PORTAL_KEYS}
        slug = str(out["slug"])
        out["portal_url"] = portal_url(slug)
        out["photo_url"] = self.photo_url(slug, out.get("photo_url"))
        if out.get("alcohol_src") not in (None, "catalog"):
            out.update(alcohol=None, alcohol_max=None, alcohol_src=None)
        row = self.csv.get(slug)
        if row:
            out["description"] = str(row.get("Описание") or "").strip()
            out["color"] = str(row.get("Цвет") or "").strip()
        out["description_src"] = "catalog" if out.get("description") else None
        return out

    def photo_url(self, slug: str, given: Any = None) -> str | None:
        """Фото только выгрузки: у эталонов правки с чужим вином фото нет."""
        if slug in WRONG_PHOTO:
            return None
        if self.data_dir:
            return f"/v1/wines/{slug}/photo" if self.photo(slug) else None
        return str(given) if given else f"/v1/wines/{slug}/photo"

    def _card_from_tile(self, slug: str) -> dict[str, Any] | None:
        tile = self.tiles.get(slug)
        if not tile:
            return None
        style = str(tile.get("style_label") or "")
        name = str(tile.get("name") or slug)
        sugar = sugar_of(name, slug)
        sparkling = bool(tile.get("sparkling")) or sparkling_of(name, slug, sugar)
        color = style.split(" ")[0] if style else ""
        return self.noportal(
            {
                "slug": slug,
                "name": name,
                "winery": str(tile.get("winery") or ""),
                "region": str(tile.get("region") or ""),
                "grapes": list(tile.get("grapes") or []),
                "category": color,
                "color": "",
                "sugar": SUGAR_LABEL.get(sugar or "", ""),
                "sugar_class": sugar,
                "photo_name": f"{slug}.webp",
                "color_label": color,
                "sugar_label": SUGAR_LABEL.get(sugar or "", ""),
                "sparkling": sparkling,
                "style_label": style_of(color, sugar, sparkling),
                "description": self.placeholder,
                "description_src": "catalog",
                "photo_url": tile.get("photo_url") or f"/v1/wines/{slug}/photo",
                "portal_url": portal_url(slug),
                "alcohol": None,
                "alcohol_max": None,
                "alcohol_src": None,
            }
        )

    def _grapes(self, slug: str) -> list[str]:
        row = self.csv.get(slug)
        if row and row.get("Сорт винограда"):
            return [g.strip() for g in str(row["Сорт винограда"]).split(",") if g.strip()]
        gt = self.gt.get(slug) or {}
        return [str(g) for g in ((gt.get("fields") or {}).get("grape") or {}).get("values") or []]

    def _real_grapes(self, slug: str) -> set[str]:
        # «Белые сорта винограда» — не сорт: общий такой «сорт» не делает вина похожими.
        return {g for g in self._grapes(slug) if "сорта винограда" not in g.lower()}

    def _card_from_data(self, slug: str) -> dict[str, Any] | None:
        """Карточка из производной выгрузки (`gt_tokens.jsonl`) по правилам договора, §3."""
        gt = self.gt.get(slug)
        if not gt:
            return None
        row = self.csv.get(slug) or {}
        name = str(row.get("Название вина") or gt.get("name") or slug).strip()
        color = str(gt.get("category") or "")
        sugar = sugar_of(name, slug)
        sparkling = sparkling_of(name, slug, sugar)
        abv = ((gt.get("fields") or {}).get("abv") or {}).get("value")
        alcohol, alcohol_max = alcohol_of(abv)
        return self.noportal(
            {
                "slug": slug,
                "name": name,
                "winery": str(row.get("Винодельня") or gt.get("winery") or ""),
                "region": str(row.get("Регион") or gt.get("region") or ""),
                "grapes": self._grapes(slug),
                "category": color,
                "color": "",
                "sugar": SUGAR_LABEL.get(sugar or "", ""),
                "sugar_class": sugar,
                "photo_name": str(gt.get("photo_name") or ""),
                "color_label": color,
                "sugar_label": SUGAR_LABEL.get(sugar or "", ""),
                "sparkling": sparkling,
                "style_label": style_of(color, sugar, sparkling),
                "description": self.placeholder,
                "description_src": "catalog",
                "photo_url": f"/v1/wines/{slug}/photo",
                "portal_url": portal_url(slug),
                "alcohol": alcohol,
                "alcohol_max": alcohol_max,
                "alcohol_src": "catalog" if alcohol is not None else None,
            }
        )

    # ---------------------------------------------------------------- рекомендации

    def _tile(self, slug: str, reasons: list[str]) -> dict[str, Any]:
        card = self.card(slug) or {}
        return {
            "slug": slug,
            "name": str(card.get("name") or slug),
            "winery": str(card.get("winery") or ""),
            "region": str(card.get("region") or ""),
            "style_label": str(card.get("style_label") or ""),
            "sparkling": bool(card.get("sparkling")),
            "grapes": self._grapes(slug),
            "photo_url": card.get("photo_url"),
            "portal_url": portal_url(slug),
            "reasons": reasons,
        }

    def _tile_noportal(self, item: dict[str, Any]) -> dict[str, Any]:
        slug = str(item["slug"])
        item["portal_url"] = portal_url(slug)
        item["photo_url"] = self.photo_url(slug, item.get("photo_url"))
        return item

    def _organizer(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Только позиции выгрузки: 73 карточки живого портала в продукт не попадают.

        Без каталога данных заглушка выгрузки не знает и оставляет всё как есть.
        """
        return [item for item in items if not self.gt or str(item.get("slug")) in self.gt]

    def _pool(self) -> list[str]:
        """Пул подбора: канонические позиции выгрузки с фото выгрузки (договор, §5)."""
        return [
            slug
            for slug, wine in self.wines.items()
            if wine.get("in_csv") and wine.get("is_canonical") and slug in self.gt
        ]

    def _facts(self, slug: str) -> dict[str, Any]:
        card = self.card(slug) or {}
        return {
            "winery": card.get("winery"),
            "color": card.get("color_label"),
            "sparkling": bool(card.get("sparkling")),
            "sugar": card.get("sugar_class"),
            "grapes": sorted(self._real_grapes(slug)),
            "abv": card.get("alcohol"),
            "region": card.get("region"),
        }

    def _rank(
        self, anchor: dict[str, Any], limit: int, skip: frozenset[str] = frozenset()
    ) -> list[dict[str, Any]]:
        """Грубое ранжирование по фактам выгрузки — только чтобы заглушка выглядела правдой.

        Настоящее ранжирование — `app/recommend/facts.py`; здесь та же идея: тот же цвет и
        игристость, сорта по Жаккару, тот же сахар, близкая крепость, регион; одна винодельня —
        одно место в тройке.
        """
        a_grapes = {g for g in anchor.get("grapes") or [] if "сорта винограда" not in g.lower()}
        a_abv = anchor.get("abv")
        scored = []
        for slug in self._pool():
            if slug in skip or slug in WRONG_PHOTO:
                continue
            facts = self._facts(slug)
            if _norm(facts["winery"]) == _norm(anchor.get("winery")):
                continue
            if anchor.get("color") and facts["color"] != anchor["color"]:
                continue
            spark = anchor.get("sparkling")
            if spark is not None and facts["sparkling"] != bool(spark):
                continue
            grapes = set(facts["grapes"])
            union = grapes | a_grapes
            jaccard = len(grapes & a_grapes) / len(union) if union else 0.0
            same_sugar = bool(anchor.get("sugar")) and facts["sugar"] == anchor["sugar"]
            abv = _num(facts["abv"])
            gap = abs(abv - a_abv) if abv is not None and a_abv is not None else 9.0
            same_region = bool(anchor.get("region")) and facts["region"] == anchor["region"]
            scored.append(((-jaccard, not same_sugar, gap, not same_region, slug), slug))
        scored.sort()
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for _, slug in scored:
            winery = _norm(self._facts(slug)["winery"])
            if winery in seen:
                continue
            seen.add(winery)
            out.append(self._tile(slug, self._reasons(anchor, slug)))
            if len(out) >= limit:
                break
        return out

    def _reasons(self, anchor: dict[str, Any], slug: str) -> list[str]:
        facts = self._facts(slug)
        out: list[str] = []
        common = sorted(set(facts["grapes"]) & set(anchor.get("grapes") or []))
        if common:
            out.append("Тот же сорт — " + ", ".join(common[:2]))
        sugar = str(facts["sugar"] or "")
        if anchor.get("sugar") and sugar:
            if sugar == anchor["sugar"]:
                out.append("Тоже " + SUGAR_LABEL.get(sugar, sugar))
            else:
                a_sugar = SUGAR_LABEL.get(anchor["sugar"], anchor["sugar"])
                out.append(f"{SUGAR_LABEL.get(sugar, sugar).capitalize()}, а не {a_sugar}")
        region = facts["region"]
        if anchor.get("region") and region:
            if region == anchor["region"]:
                out.append(f"Тоже {region}")
            else:
                out.append(f"{region}, а не {anchor['region']}")
        abv, a_abv = _num(facts["abv"]), anchor.get("abv")
        if abv is not None and a_abv is not None:
            if abs(abv - a_abv) < 0.05:
                out.append(f"Крепость та же — {_deg(abv)}")
            else:
                out.append(f"Крепость {_deg(abv)} против {_deg(a_abv)}")
        return out

    @staticmethod
    def _plain(body: dict[str, Any], *lists: str) -> dict[str, Any]:
        body["notice"] = None
        for key in lists:
            items = body.get(key) or []
            for item in items:
                item["reasons"] = []
            items.sort(key=lambda item: _norm(item.get("name")))
        return body

    def similar(self, slug: str, order: str, limit: int) -> dict[str, Any]:
        if slug in self.gt and self.wines and slug != self.fx["similar"]["slug"]:
            items = self._rank(self._facts(slug), limit)
            card = self.card(slug) or {}
            body = {
                "slug": slug,
                "order": order,
                "limit": limit,
                "notice": NOTICE,
                "category_label": card.get("style_label"),
                "items": items,
                "notes": [],
            }
            if len(items) < 3:
                body["notes"] = [{"code": "fewer_than_three", "text": "Нашлось меньше трёх"}]
            if order == "plain":
                self._plain(body, "items")
            return body
        body = copy.deepcopy(self.fx["similar_plain" if order == "plain" else "similar"])
        body["slug"], body["limit"] = slug, limit
        body["items"] = [self._tile_noportal(t) for t in self._organizer(body["items"])][:limit]
        return body

    def groups(self, slugs: list[str]) -> frozenset[str]:
        """Сами slug и все вина их групп (`wine_id`); неизвестные slug ничего не добавляют."""
        ids = {self.wines[s].get("wine_id") for s in slugs if s in self.wines} - {None}
        return frozenset(slugs) | {s for s, w in self.wines.items() if w.get("wine_id") in ids}

    def by_label(
        self, read: dict[str, Any], order: str, limit: int, exclude: list[str] | None = None
    ) -> dict[str, Any]:
        fixture = self.fx["by_label"]
        winery = read.get("winery")
        gone = self.groups(exclude or [])
        if winery and _norm(winery) != _norm(fixture["read"]["winery"]) and self.gt:
            same = sorted(
                (s for s in self._pool() if _norm(self._facts(s)["winery"]) == _norm(winery)),
                key=lambda s: _norm((self.card(s) or {}).get("name")),
            )
            anchor = {**read, "region": None}
            body = {
                "read": read,
                "winery": {"name": winery, "in_catalog": bool(same), "count": len(same)},
                "order": order,
                "notice": NOTICE,
                "same_winery": [self._tile(s, []) for s in same if s not in gone][:6],
                "similar": self._rank(anchor, limit, gone),
                "notes": []
                if same
                else [{"code": "winery_unknown", "text": "Этой винодельни в каталоге нет"}],
            }
        elif not winery:
            body = copy.deepcopy(fixture)
            body.update(
                read=read,
                winery={"name": None, "in_catalog": False, "count": 0},
                same_winery=[],
                notes=[{"code": "winery_unknown", "text": "Винодельню на этикетке не прочитали"}],
            )
        else:
            body = copy.deepcopy(fixture)
            body["read"] = read
        body["order"] = order
        # Заглушкам группы не известны без каталога данных: там уходят только сами slug.
        body["same_winery"] = [
            self._tile_noportal(t)
            for t in self._organizer(body["same_winery"])
            if t["slug"] not in gone
        ]
        body["similar"] = [
            self._tile_noportal(t)
            for t in self._organizer(body["similar"])
            if t["slug"] not in gone
        ][:limit]
        if order == "plain":
            self._plain(body, "same_winery", "similar")
        return body

    # ---------------------------------------------------------------- сомелье

    def sommelier(self, slug: str, order: str) -> dict[str, Any]:
        """`GET …/sommelier`: заглушка договора для её вин, иначе ответ из фактов карточки."""
        if slug in self.sommeliers:
            body = copy.deepcopy(self.sommeliers[slug])
        else:
            body = self._sommelier_from_card(slug)
        body["order"] = order
        if order == "plain":
            body["notice_149"] = None
            body["dishes"].sort(key=lambda dish: _norm(dish.get("name")))
        return body

    def _sommelier_from_card(self, slug: str) -> dict[str, Any]:
        card = self.card(slug) or {"slug": slug, "name": slug, "winery": ""}
        sugar = card.get("sugar_class")
        abv = _num(card.get("alcohol"))
        sparkling = bool(card.get("sparkling"))
        values: dict[str, float | None] = {
            "sweetness": SWEETNESS.get(str(sugar)) if sugar else None,
            "alcohol": round(min(max((abv - 8.0) / 1.4, 0.0), 5.0) / 5, 2) if abv else None,
            "effervescence": 0.8 if sparkling else 0.0,
        }
        profile = [
            {
                "axis": axis,
                "label": label,
                "left": left,
                "right": right,
                "value": values.get(axis),
                "source": "catalog" if values.get(axis) is not None else None,
            }
            for axis, label, left, right in AXES
        ]
        color = str(card.get("color_label") or "")
        rule = self._serve_rule(color, sugar, sparkling)
        temperature = dict(SERVE_RULES)[rule]
        dishes = self._dishes(slug)
        return {
            "slug": slug,
            "name": card.get("name") or slug,
            "winery": card.get("winery") or "",
            "order": "reco",
            "note": {
                "text": self._note(card, temperature, dishes),
                "basis": NOTE_BASIS,
                "generated": False,
                "label": ALGO,
            },
            "profile": profile,
            "serve": {
                "temperature_c": list(temperature),
                "source": "rule",
                "rule": rule,
                "by_grape": False,
            },
            "dishes": dishes,
            "chips": [
                *({"id": cid, "text": text} for cid, text in BASE_CHIPS),
                {"id": "guided", "text": "Помогите выбрать"},
            ],
            "live": True,
            "input": True,
            "notice_149": NOTICE,
        }

    @staticmethod
    def _serve_rule(color: str, sugar: Any, sparkling: bool) -> str:
        if sparkling:
            return "sparkling"
        if sugar == "sladkoe":
            return "sweet"
        return {"Оранжевое": "orange", "Белое": "white", "Розовое": "rose"}.get(color, "red")

    def _dishes(self, slug: str) -> list[dict[str, Any]]:
        """Первые три блюда `top` из `pairs.json` заглушки с чипами правил (договор, §1)."""
        pairs = ((self.somm_fx.get("pairs") or {}).get("wines") or {}).get(slug)
        catalog = self.somm_fx.get("dishes") or {}
        dishes = {d["id"]: d for d in catalog.get("dishes") or []}
        rules = {r["id"]: r for r in catalog.get("rules") or []}
        out: list[dict[str, Any]] = []
        for dish_id in (pairs or {}).get("top", [])[:3]:
            dish = dishes.get(dish_id)
            verdict, plus, minus = (pairs["dishes"].get(dish_id) or ["neutral", [], []])[:3]
            if not dish:
                continue
            chip = [
                [{"id": r, "text": rules[r]["chip"], "source": "grape"} for r in ids if r in rules]
                for ids in (plus[:3], minus[:2])
            ]
            out.append(
                {
                    "id": dish_id,
                    "name": dish["name"],
                    "category": dish["category"],
                    "verdict": verdict,
                    "plus": chip[0],
                    "minus": chip[1],
                }
            )
        return out

    @staticmethod
    def _note(card: dict[str, Any], temperature: tuple[int, int], dishes: list[Any]) -> str:
        """Шаблон заметки договора (§2) по фактам карточки."""
        style = str(card.get("style_label") or "").lower()
        if card.get("sparkling") and "игрист" not in style:
            style = f"игристое {style}".strip()
        parts = [f"{card.get('name')} — {style}." if style else f"{card.get('name')}."]
        grapes = [g for g in card.get("grapes") or [] if "сорта винограда" not in g.lower()]
        if len(grapes) == 1:
            parts.append(f"Сорт — {grapes[0]}.")
        elif len(grapes) in (2, 3):
            parts.append(f"Сорта — {', '.join(grapes[:-1])} и {grapes[-1]}.")
        elif grapes:
            parts.append(f"Сорта — {', '.join(grapes[:3])} и другие.")
        where = ", ".join(x for x in (card.get("winery"), card.get("region")) if x)
        if where:
            parts.append(f"{where}.")
        parts.append(f"Подают при {temperature[0]}–{temperature[1]} °C.")
        names = [str(d["name"])[:1].lower() + str(d["name"])[1:] for d in dishes]
        if len(names) == 1:
            parts.append(f"По правилам сочетаний к нему подходит {names[0]}.")
        elif names:
            parts.append(
                f"По правилам сочетаний к нему подходят {', '.join(names[:-1])} и {names[-1]}."
            )
        return " ".join(parts)

    def ask_stream(self, body: dict[str, Any], order: str, fault: bool) -> list[dict[str, Any]]:
        """Поток `POST /v1/sommelier/ask` — одна из заглушек договора (§9)."""
        question = str(body.get("question") or "")
        chip = body.get("chip")
        args = body.get("args") or {}
        if fault:
            name = "ask_error"
        elif question and PURCHASE_WORDS.search(_norm(question)):
            name = "ask_refusal"
        elif chip == "guided" and not args:
            name = "ask_guided_step"
        elif order == "plain" and chip in ("softer", "fresher", "replace"):
            name = "ask_softer_plain"
        elif chip == "dish_check":
            name = "ask_dish_check_guard"
        elif question and "борщ" in _norm(question):
            name = "ask_dish_check_live"
        else:
            name = "ask_what_to_eat_busy"
        events = copy.deepcopy(self.streams[name])
        for event in events:
            if event.get("type") == "facts":
                event["slug"] = str(body["slug"])
                event["order"] = order
        return events

    # ---------------------------------------------------------------- файлы

    def photo(self, slug: str) -> Path | None:
        if not self.data_dir or slug in WRONG_PHOTO:
            return None
        catalog = self.data_dir / "catalog"
        small = "photos_small.pre-packshot-fix" if slug in ORIGINAL_PHOTO else "photos_small"
        path = catalog / small / f"{slug}.webp"
        return path if path.is_file() else None


def ask_error(body: Any) -> str | None:
    """Проверка тела `ask` по договору сомелье (§3.1); `None` — тело годное."""
    if not isinstance(body, dict):
        return "тело — объект"
    slug = body.get("slug")
    if not isinstance(slug, str) or not SLUG_RE.match(slug):
        return "нужен slug"
    question, chip = body.get("question"), body.get("chip")
    if (question is None) == (chip is None):
        return "нужно ровно одно из question и chip"
    if question is not None:
        text = re.sub(r"\s+", " ", str(question)).strip() if isinstance(question, str) else ""
        if not text or len(text) > 200:
            return "вопрос — от 1 до 200 символов"
    if chip is not None and chip not in CHIP_IDS:
        return "неизвестный чип"
    if "args" in body and not isinstance(body["args"], dict):
        return "args — объект"
    context = body.get("context")
    if context is not None and not isinstance(context, dict):
        return "context — объект"
    return None


class Handler(BaseHTTPRequestHandler):
    """Маршруты договоров поверх `Stub`; всё, чего договоры не знают, — 404."""

    stub: Stub
    verbose = False
    server_version = "svs-ui-stub"

    # ---------------------------------------------------------------- ответы

    def _send(self, status: int, body: bytes, ctype: str, cache: str = "no-store") -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _file(self, path: Path | None, missing: str) -> None:
        if path is None or not path.is_file():
            self._json(404, {"detail": missing})
            return
        ctype = TYPES.get(path.suffix.lower(), "application/octet-stream")
        self._send(200, path.read_bytes(), ctype, "public, max-age=86400")

    def _invalid(self, name: str, value: str, allowed: tuple[str, ...]) -> None:
        self._json(
            422,
            {
                "detail": [
                    {
                        "type": "enum",
                        "loc": ["query", name],
                        "msg": "Input should be " + ", ".join(repr(a) for a in allowed),
                        "input": value,
                    }
                ]
            },
        )

    def log_message(self, format: str, *args: Any) -> None:  # имя из базового класса
        if self.verbose:
            sys.stderr.write("stub: " + format % args + "\n")

    # ---------------------------------------------------------------- разбор запроса

    def _query(self) -> dict[str, str]:
        return {k: v[-1] for k, v in parse_qs(urlsplit(self.path).query).items()}

    def _page_query(self) -> dict[str, str]:
        referer = self.headers.get("Referer") or ""
        return {k: v[-1] for k, v in parse_qs(urlsplit(referer).query).items()}

    def _state(self) -> str:
        page = self._page_query()
        for value in (self._query().get("state"), page.get("state")):
            if value in STATES:
                return value
        shot = page.get("shot", "")
        if shot in STATES:
            return shot
        return SHOT_STATES.get(shot, self.stub.default_state)

    def _suggest(self) -> bool:
        """`suggest=1` у запроса скана или у адреса страницы — подсказка «нет в каталоге»."""
        values = (self._query().get("suggest"), self._page_query().get("suggest"))
        return (
            any(v in ("1", "true") for v in values) or self._page_query().get("shot") == "suggest"
        )

    def _order(self, query: dict[str, str]) -> str | None:
        order = query.get("order", "reco")
        if order not in ORDERS:
            self._invalid("order", order, ORDERS)
            return None
        return order

    def _limit(self, query: dict[str, str]) -> int | None:
        raw = query.get("limit", "3")
        if not raw.isdigit() or not 1 <= int(raw) <= 12:
            self._json(
                422, {"detail": [{"type": "range", "loc": ["query", "limit"], "input": raw}]}
            )
            return None
        return int(raw)

    def _body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length > 0 else b""

    # ---------------------------------------------------------------- маршруты

    def do_HEAD(self) -> None:  # имя из http.server
        self.do_GET()

    def do_GET(self) -> None:  # имя из http.server
        path = unquote(urlsplit(self.path).path)
        query = self._query()
        if path in ("/", "/index.html"):
            self._page(query)
        elif path.startswith("/static/"):
            self._static(path[len("/static/") :])
        elif path == "/v1/health":
            self._json(200, {"status": "ready", "stub": True})
        elif path == "/__stub/driver.js":
            self._file(DRIVER, "драйвера нет")
        elif path == "/__stub/photo":
            self._sample_photo()
        elif path == "/__stub/frame":
            self._frame(query)
        elif m := re.fullmatch(r"/v1/wines/([^/]+)/photo", path):
            self._file(self.stub.photo(m.group(1)), "фото каталога нет на этой машине")
        elif m := re.fullmatch(r"/v1/wines/([^/]+)/similar", path):
            self._similar(m.group(1), query)
        elif m := re.fullmatch(r"/v1/wines/([^/]+)/sommelier", path):
            self._sommelier(m.group(1), query)
        elif m := re.fullmatch(r"/v1/wines/([^/]+)", path):
            card = self.stub.card(m.group(1)) if SLUG_RE.match(m.group(1)) else None
            if card is None:
                self._json(404, {"detail": f"нет карточки {m.group(1)!r}"})
            else:
                self._json(200, card)
        else:
            self._json(404, {"detail": "Not Found"})

    def do_POST(self) -> None:  # имя из http.server
        path = urlsplit(self.path).path
        query = self._query()
        payload = self._body()
        if path == "/v1/scan":
            if self.stub.delay:
                time.sleep(self.stub.delay)
            self._json(200, self.stub.scan(self._state(), self._suggest()))
        elif path == "/v1/similar/by-label":
            order, limit = self._order(query), None
            if order is None or (limit := self._limit(query)) is None:
                return
            try:
                read = json.loads(payload.decode("utf-8") or "{}")
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._json(422, {"detail": [{"type": "json_invalid", "loc": ["body"]}]})
                return
            if not isinstance(read, dict):
                self._json(422, {"detail": [{"type": "dict_type", "loc": ["body"]}]})
                return
            # exclude — не часть прочитанного: в эхо `read` оно не попадает.
            exclude = read.pop("exclude", None) or []
            if not (
                isinstance(exclude, list)
                and len(exclude) <= MAX_EXCLUDE
                and all(isinstance(slug, str) for slug in exclude)
            ):
                self._json(422, {"detail": [{"type": "list_type", "loc": ["body", "exclude"]}]})
                return
            self._json(200, self.stub.by_label(read, order, limit, exclude))
        elif path == "/v1/sommelier/ask":
            self._ask(query, payload)
        else:
            self._json(404, {"detail": "Not Found"})

    def _page(self, query: dict[str, str]) -> None:
        html = PAGE.read_text(encoding="utf-8")
        if query.get("shot"):
            html = html.replace("</body>", '<script src="/__stub/driver.js"></script>\n</body>', 1)
        self._send(200, html.encode("utf-8"), TYPES[".html"])

    def _static(self, rel: str) -> None:
        # Как у сервиса (`app.api.page.AssetFiles`): шрифт, картинки и лист сомелье, HTML — нет.
        # Без долгого кэша: лист сомелье правят на этой же заглушке, и браузер не должен
        # держать вчерашний скрипт.
        target = (STATIC / rel).resolve()
        allowed = target.suffix.lower() in ASSET_SUFFIXES
        if not allowed or STATIC.resolve() not in target.parents or not target.is_file():
            self._json(404, {"detail": "Not Found"})
            return
        ctype = TYPES.get(target.suffix.lower(), "application/octet-stream")
        self._send(200, target.read_bytes(), ctype, "no-cache")

    def _frame(self, query: dict[str, str]) -> None:
        """Страница во фрейме заданного размера: безголовый Chrome не делает окно уже 500 px."""
        width = int(query["w"]) if query.get("w", "").isdigit() else 390
        height = int(query["h"]) if query.get("h", "").isdigit() else 844
        shot = re.sub(r"[^a-z_]", "", query.get("shot", ""))
        html = (
            '<!DOCTYPE html><meta charset="utf-8"><style>html,body{margin:0;background:#fff}'
            f"iframe{{display:block;width:{width}px;height:{height}px;border:0}}</style>"
            f'<iframe src="/?shot={shot}" title="страница"></iframe>'
        )
        self._send(200, html.encode("utf-8"), TYPES[".html"])

    def _sample_photo(self) -> None:
        # Без кэша: кадр зависит от сценария страницы, а адрес у него один.
        card = self.stub.scan(self._state(), self._suggest()).get("card") or {}
        path = self.stub.photo(str(card.get("slug") or ""))
        if path is not None:
            ctype = TYPES.get(path.suffix.lower(), "application/octet-stream")
            self._send(200, path.read_bytes(), ctype)
        else:
            self._send(200, SAMPLE_SVG.encode("utf-8"), TYPES[".svg"])

    def _similar(self, slug: str, query: dict[str, str]) -> None:
        order = self._order(query)
        if order is None:
            return
        limit = self._limit(query)
        if limit is None:
            return
        if not self.stub.known(slug):
            self._json(404, {"detail": f"нет карточки {slug!r}"})
            return
        self._json(200, self.stub.similar(slug, order, limit))

    def _sommelier(self, slug: str, query: dict[str, str]) -> None:
        order = self._order(query)
        if order is None:
            return
        if not SLUG_RE.match(slug) or not self.stub.has_sommelier(slug):
            self._json(404, {"detail": f"нет карточки {slug!r}"})
            return
        self._json(200, self.stub.sommelier(slug, order))

    def _ask(self, query: dict[str, str], payload: bytes) -> None:
        order = self._order(query)
        if order is None:
            return
        try:
            body = json.loads(payload.decode("utf-8") or "null")
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._json(422, {"detail": [{"type": "json_invalid", "loc": ["body"]}]})
            return
        problem = ask_error(body)
        if problem:
            self._json(422, {"detail": [{"type": "value_error", "loc": ["body"], "msg": problem}]})
            return
        if not self.stub.has_sommelier(body["slug"]):
            self._json(404, {"detail": f"нет карточки {body['slug']!r}"})
            return
        fault = self._page_query().get("somm") == "error" or query.get("somm") == "error"
        events = self.stub.ask_stream(body, order, fault)
        # Поток без длины: соединение закрывается после `done` (HTTP/1.0 сервера заглушки).
        self.send_response(200)
        self.send_header("Content-Type", NDJSON)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        last = 0
        try:
            for event in events:
                pause = min(max(int(event.get("t_ms") or 0) - last, 0), 1500) / 1000
                last = int(event.get("t_ms") or 0)
                if pause and self.stub.delay:
                    time.sleep(pause)
                line = json.dumps(event, ensure_ascii=False) + "\n"
                self.wfile.write(line.encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return  # лист закрыли — поток оборван, как в договоре (§3.4)


def make_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    data_dir: Path | None = None,
    delay: float = 0.0,
    default_state: str = "found",
    verbose: bool = False,
    csv_path: Path | None = None,
) -> ThreadingHTTPServer:
    """Сервер заглушки; `port=0` — свободный порт (для тестов и скриншотов)."""
    stub = Stub(data_dir=data_dir, delay=delay, default_state=default_state, csv_path=csv_path)
    handler = type("StubHandler", (Handler,), {"stub": stub, "verbose": verbose})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def serve_in_thread(server: ThreadingHTTPServer) -> threading.Thread:
    thread = threading.Thread(target=server.serve_forever, name="ui-stub", daemon=True)
    thread.start()
    return thread


def default_csv() -> Path | None:
    """Выгрузка организатора из `SVS_DATASET_DIR`, если переменная задана."""
    root = os.environ.get("SVS_DATASET_DIR")
    return Path(root) / CSV_NAME if root else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ["SVS_DATA_DIR"]) if os.environ.get("SVS_DATA_DIR") else None,
        help="каталог данных сервиса (фото, gt_tokens, справочник); по умолчанию SVS_DATA_DIR",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=default_csv(),
        help=f"выгрузка организатора {CSV_NAME}: описание и оттенок цвета как есть",
    )
    parser.add_argument("--delay", type=float, default=1.2, help="пауза ответа /v1/scan, с")
    parser.add_argument("--state", default="found", choices=sorted(STATES))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    server = make_server(
        args.host,
        args.port,
        data_dir=args.data_dir,
        delay=args.delay,
        default_state=args.state,
        verbose=args.verbose,
        csv_path=args.csv,
    )
    stub = server.RequestHandlerClass.stub  # type: ignore[attr-defined]
    data = f"данные: {args.data_dir}" if stub.data_dir else "без данных"
    source = f", выгрузка: {len(stub.csv)} позиций" if stub.csv else ""
    print(f"заглушка: http://{args.host}:{server.server_address[1]}/ ({data}{source})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
