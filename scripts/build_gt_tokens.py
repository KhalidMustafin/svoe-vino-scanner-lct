"""Эталонные токены этикетки по каждому slug каталога «Своё Вино» (`gt_tokens.jsonl`).

Перенос исследовательского прототипа `gt_build.py` (вне репозитория): логика та же, пути — аргументами
с умолчаниями из `Settings`. Вход (только чтение):

    --csv          strapi_output0709.csv  — название, винодельня, категория, сорт, фото
    --clusters     near_dup_clusters.json — slug_attrs (сахар, крепость, год из slug), level_B
    --live         plan_live_wines.json   — живой API: категория «Белое сухое» (сахар)
    --winery-map   strapi_winery_map.txt  — алиасы виноделен из сопоставления с «Лозой»

Таблицы прототипа `nd_common.py` (вне репозитория; транслит slug, сахар в названии) и `GRAPE_SYNONYMS` из taxonomy
«Лозы» скопированы сюда; латиница → кириллица — `app.reading.text.translit`.

Внимание: синонимы сортов — та же таблица, что возьмёт парсер,
поэтому поле «сорт» верно по построению. Часть эталона нужно проверить глазами.

Выход в `--out-dir`: `gt_tokens.jsonl` (строка на slug), `gt_summary.json` (покрытие полей,
шум, разбор пар двойников), `gt_checks.txt` (выборка для ручной проверки).

Правки атрибутов карточек (Э4, `data/gt/gt_fixes.tsv` репозитория): цвет, сахар или сорт, у
которых выгрузка спорит с упаковкой на фото карточки и с названием. Строка — slug, поле, было,
стало (`null` — поле обнуляется) и два источника. Правится только поле `fields.*` — то, из чего
берут признаки слой выбора и словарь; колонка «Категория» (`category`) остаётся как в выгрузке,
её показывают карточки и сомелье. Значение «было» сверяется со сборкой: выгрузка сменилась —
сборка падает, а не правит молча. `--no-fixes` собирает эталон без правок (sha1 bf6a8823).

    python scripts/build_gt_tokens.py --out-dir data/gt
"""

from __future__ import annotations

import argparse
import ast
import collections
import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.reading.text.translit import transcribe_variants

# ---------------------------------------------------------------- таблицы nd_common.py
_SLUG_TR = dict(zip("абвгдеёзийклмнопрстуфыэ", "abvgdeeziyklmnoprstufye", strict=True))
_SLUG_TR.update(
    {"ж": "zh", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch", "ю": "yu", "я": "ya"}
)
_SLUG_TR.update({"ь": "", "ъ": ""})

RU_SUGAR = [
    (r"экстра[\s\-]*брют|extra[\s\-]*brut", "extra_brut"),
    (r"брют[\s\-]*натюр|brut[\s\-]*nature|zero[\s\-]*dosage|pas[\s\-]*dos", "brut_nature"),
    (r"полусух|semi[\s\-]*dry", "polusuhoe"),
    (r"полуслад|semi[\s\-]*sweet|demi[\s\-]*sec", "polusladkoe"),
    (r"(?<!полу)сухое|(?<!semi[\s\-])\bdry\b", "suhoe"),
    (r"(?<!полу)сладкое|десертн", "sladkoe"),
    (r"(?<!экстра )(?<!экстра-)брют|\bbrut\b", "brut"),
]

# ---------------------------------------------------------------- GRAPE_SYNONYMS «Лозы»
# Копия `Code/backend/app/domain/taxonomy.py:GRAPE_SYNONYMS`; порядок важен — при
# совпадении нормы поздний код перекрывает ранний.
GRAPE_SYNONYMS: dict[str, list[str]] = {
    "cabernet_sauvignon": [
        "Каберне Совиньон", "Cabernet Sauvignon", "каберне-совиньон", "cab sauvignon", "каберне",
    ],
    "merlot": ["Мерло", "Merlot"],
    "cabernet_franc": ["Каберне Фран", "Cabernet Franc"],
    "pinot_noir": ["Пино Нуар", "Pinot Noir", "пино нуар"],
    "syrah": ["Сира", "Syrah", "Шираз", "Shiraz"],
    "malbec": ["Мальбек", "Malbec"],
    "sangiovese": [
        "Санджовезе", "Sangiovese", "Санджовезе Гроссо", "Sangiovese Grosso", "Брунелло",
        "Brunello",
    ],
    "tempranillo": ["Темпранильо", "Tempranillo"],
    "nebbiolo": ["Неббиоло", "Nebbiolo"],
    "petit_verdot": ["Пти Вердо", "Petit Verdot"],
    "carmenere": ["Карменер", "Carmenere", "Carménère"],
    "grenache": ["Гренаш", "Grenache", "Гарнача", "Garnacha"],
    "chardonnay": ["Шардоне", "Chardonnay"],
    "sauvignon_blanc": ["Совиньон Блан", "Sauvignon Blanc", "совиньон-блан", "совиньон"],
    "riesling": ["Рислинг", "Riesling"],
    "pinot_gris": ["Пино Гри", "Pinot Gris", "Пино Гриджо", "Pinot Grigio"],
    "pinot_blanc": ["Пино Блан", "Pinot Blanc"],
    "aligote": ["Алиготе", "Aligote", "Aligoté"],
    "muscat": ["Мускат", "Muscat", "Мускат белый", "Moscato"],
    "viognier": ["Вионье", "Viognier"],
    "gewurztraminer": ["Гевюрцтраминер", "Gewurztraminer", "Gewürztraminer", "Траминер"],
    "chenin_blanc": ["Шенен Блан", "Chenin Blanc"],
    "semillon": ["Семильон", "Semillon", "Sémillon"],
    "muscat_ottonel": ["Мускат Оттонель", "Muscat Ottonel", "Оттонель"],
    "muller_thurgau": ["Мюллер-Тургау", "Мюллер Тургау", "Muller-Thurgau"],
    "silvaner": ["Сильванер", "Silvaner", "Сильванер зеленый"],
    "sauvignon_gris": ["Совиньон Гри", "Sauvignon Gris"],
    "mtsvane": ["Мцване", "Mtsvane"],
    "kangun": ["Кангун", "Kangun"],
    "bianca": ["Бианка", "Bianca"],
    "kristall": ["Кристалл", "Kristall"],
    "platovsky": ["Платовский", "Platovsky"],
    "podarok_magaracha": ["Подарок Магарача", "Podarok Magaracha"],
    "marselan": ["Марселан", "Marselan"],
    "gamay": ["Гаме", "Gamay", "Гамэ"],
    "zweigelt": ["Цвайгельт", "Zweigelt"],
    "dornfelder": ["Дорнфельдер", "Dornfelder"],
    "regent": ["Регент", "Regent"],
    "mourvedre": ["Мурведр", "Mourvedre", "Монастрель", "Monastrell"],
    "pinotage": ["Пинотаж", "Pinotage"],
    "zinfandel": ["Зинфандель", "Zinfandel", "Примитиво", "Primitivo"],
    "albarino": ["Альбариньо", "Albarino", "Альваринью"],
    "torrontes": ["Торронтес", "Torrontes"],
    "verdejo": ["Вердехо", "Verdejo"],
    "assyrtiko": ["Ассиртико", "Assyrtiko"],
    "furmint": ["Фурминт", "Furmint"],
    "nero_davola": ["Неро д'Авола", "Неро д Авола", "Nero d'Avola"],
    "montepulciano": ["Монтепульчано", "Montepulciano"],
    "bastardo_magarachsky": ["Бастардо Магарачский", "Бастардо", "Bastardo Magarachsky"],
    "cabernet_azos": ["Каберне АЗОС", "Каберне Азос"],
    "krasnostop_azos": ["Красностоп АЗОС", "Красностоп Азос", "Красностоп анапский"],
    "livadiysky_cherny": ["Ливадийский чёрный", "Ливадийский черный"],
    "gurzufsky_rozovy": ["Гурзуфский розовый", "Гурзуфский"],
    "antey_magarachsky": ["Антей магарачский", "Антей"],
    "rubinovy_magaracha": ["Рубиновый Магарача"],
    "moldova": ["Молдова", "Moldova"],
    "isabella": ["Изабелла", "Isabella"],
    "avgustin": ["Августин", "Avgustin"],
    "danko": ["Данко"],
    "aleatiko": ["Алеатико", "Aleatico"],
    "albillo": ["Альбильо", "Albillo"],
    "amursky_potapenko": ["Амурский Потапенко", "Амурский"],
    "saperavi": ["Саперави", "Saperavi"],
    "rkatsiteli": ["Ркацители", "Rkatsiteli"],
    "kizlyarsky_chorny": ["Кизлярский чёрный", "Кизлярский черный"],
    "krasnostop": ["Красностоп Золотовский", "Красностоп", "Krasnostop Zolotovsky"],
    "tsimlyansky_cherny": ["Цимлянский чёрный", "Цимлянский черный", "Tsimlyansky Cherny"],
    "sibirkovy": ["Сибирьковый", "Сибирьковый белый", "Sibirkovy"],
    "puhlyakovsky": ["Пухляковский", "Пухляковский белый", "Puhlyakovsky"],
    "kokur": ["Кокур белый", "Кокур", "Kokur"],
    "kefesia": ["Кефесия", "Kefesia"],
    "dzhevat_kara": ["Джеват Кара", "Джеват кара", "Dzhevat Kara"],
    "ekim_kara": ["Эким Кара", "Ekim Kara"],
    "sary_pandas": ["Сары Пандас", "Sary Pandas"],
    "kefesia_krym": ["Кефесия крымская"],
    "golubok": ["Голубок", "Golubok"],
    "dostoyny": ["Достойный", "Dostoyny"],
    "pervenets_magaracha": ["Первенец Магарача", "Pervenets Magaracha"],
    "citron_magaracha": ["Цитронный Магарача", "Citronny Magaracha"],
    "narma": ["Нарма", "Narma"],
    "gimra": ["Гимра", "Gimra"],
}  # fmt: skip

# ---------------------------------------------------------------- словари этикетки
SUGAR_LABEL = {
    "suhoe": ["сухое", "dry", "sec", "secco", "seco"],
    "polusuhoe": ["полусухое", "semi dry", "off dry", "demi sec"],
    "polusladkoe": ["полусладкое", "semi sweet", "demi sec", "amabile"],
    "sladkoe": ["сладкое", "sweet", "десертное", "dolce", "doux"],
    "brut": ["брют", "brut"],
    "extra_brut": ["экстра брют", "extra brut"],
    "brut_nature": ["брют натюр", "brut nature", "zero dosage", "pas dose", "брют зеро"],
}
LIVE_SUGAR = {
    "сухое": "suhoe",
    "полусухое": "polusuhoe",
    "полусладкое": "polusladkoe",
    "сладкое": "sladkoe",
    "брют": "brut",
    "экстра брют": "extra_brut",
    "брют натюр": "brut_nature",
}
COLOR_LABEL = {
    "Белое": ["белое", "white", "blanc", "bianco", "blanco"],
    "Красное": ["красное", "red", "rouge", "rosso", "tinto"],
    "Розовое": ["розовое", "rose", "розе", "rosato", "rosado", "pink"],
    "Оранжевое": ["оранжевое", "orange", "оранж", "янтарное"],
}
SERIAL_PHRASES = [
    "семейный резерв", "гран резерв", "grand reserve", "reserve", "reserva", "резерв",
    "blanc de noirs", "blanc de noir", "блан де нуар", "blanc de blancs", "blanc de blanc",
    "блан де блан", "кюве", "cuvee", "prestige", "престиж", "premium", "премиум", "select",
    "селект", "grand cru", "гран крю", "ultra", "ультра", "pet nat", "петнат", "пет нат",
    "barrel fermented", "barrel", "баррель", "limited edition", "limited", "коллекционное",
    "collection", "терруар", "terroir", "bag in box", "бэг ин бокс", "в банке", "ice wine",
    "ледяное", "late harvest", "поздний сбор", "выдержанное", "magnum", "магнум", "classic",
    "классик", "original", "оригинал",
]  # fmt: skip
GENERIC = {
    "вино", "вина", "игристое", "шампанское", "российское", "тихое", "wine", "wines",
    "sparkling", "и", "в", "с", "&", "the", "de", "la", "le", "di", "del", "органик",
    "organic", "винодельня", "winery", "свое", "vol", "бутылка",
}  # fmt: skip
WINERY_GENERIC = {
    "винодельня", "winery", "wine", "wines", "vineyards", "vineyard", "estate", "и", "&",
    "дом", "шампанских", "вин", "имение", "усадьба", "organic", "wein", "und", "vines",
    "винодельни",
}  # fmt: skip
GRAPE_GENERIC = {"белые сорта винограда", "красные сорта винограда"}
ROMAN_RE = re.compile(r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")
YEAR_RE = re.compile(r"(?<!\d)(19[5-9]\d|20[0-3]\d)(?!\d)")

# Ручные латинские и брендовые формы с источником; остальное — автоматическая транслитерация
# («massandra», «tabiya»). Формы с публичных кадров q1–q3 сюда не пишутся: это дымовой тест.
MANUAL_WINERY: dict[str, tuple[list[str], str]] = {
    "Кубань-Вино": (
        ["chateau tamagne", "шато тамань", "aristov", "аристов"],
        (
            "Шато Тамань — комментарий translit.py «Лозы» (не проверено); "
            "Аристов — линейка в названиях 31 slug CSV («Аристов …»)"
        ),
    ),
    "Мысхако": (["myskhako"], "комментарий translit.py «Лозы»; не проверено"),
    "Новый Свет. Дом шампанских вин": (
        ["novy svet", "новый свет"],
        "комментарий translit.py «Лозы»; не проверено",
    ),
}

# Формы, прочитанные на packshot каталога: те же пиксели, что studio и synth. В эталон и
# словарь идут только с --packshot-aliases, пока не подтверждены на dev-кадрах полевого набора.
PACKSHOT_WINERY: dict[str, tuple[list[str], str]] = {
    "Фанагория": (["fanagoria"], "packshot fanagoriya-ice-wine"),
    "Абрау-Дюрсо": (
        ["abrau durso", "abrau", "abrau estates"],
        "packshot abrau-estates; кэш чтений «Лозы» на packshot (ABRAU)",
    ),
    "Валерий Захарьин": (["valery zaharin"], "packshot avtohtonnoe-vino-kryma"),
    "Золотое Поле": (["kaffa"], "packshot zolotoe-pole-kaffa (бренд линейки)"),
    "AGORA WINERY": (["agora"], "packshot agora-rosa-viva"),
}

# Порядок полей при разборе пар двойников: первое различие.
TWIN_ORDER = ["cuvee_or_grape", "year", "sugar", "serial", "color", "abv"]

# ---------------------------------------------------------------- правки карточек (Э4)
#: Таблица правок в репозитории: она часть сборки, как таблицы выше, а не данных прогона.
DEFAULT_FIXES = Path(__file__).resolve().parents[1] / "data" / "gt" / "gt_fixes.tsv"
FIX_FIELDS = ("sugar", "color", "grape")
FIX_COLUMNS = ("slug", "field", "old", "new", "rule", "source_1", "source_2")
FIX_NULL = "null"


def _fix_value(text: str) -> str | None:
    text = text.strip()
    return None if text == FIX_NULL else text


def load_fixes(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """`gt_fixes.tsv` → {(slug, поле): правка}. Кривая строка — `ValueError` с её номером."""
    fixes: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        missing = [c for c in FIX_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path.name}: нет колонок {missing}")
        for number, row in enumerate(reader, start=2):
            where = f"{path.name}:{number}"
            slug, field = row["slug"].strip(), row["field"].strip()
            old, new = _fix_value(row["old"]), _fix_value(row["new"])
            if field not in FIX_FIELDS:
                raise ValueError(f"{where}: поле {field!r} не из {FIX_FIELDS}")
            if not (row["source_1"] or "").strip() or not (row["source_2"] or "").strip():
                raise ValueError(f"{where}: у правки должно быть два источника")
            if old == new:
                raise ValueError(f"{where}: «было» и «стало» совпадают")
            known = {"sugar": SUGAR_LABEL, "color": COLOR_LABEL}.get(field)
            if known is not None and new is not None and new not in known:
                raise ValueError(f"{where}: {field} {new!r} не из {sorted(known)}")
            if (slug, field) in fixes:
                raise ValueError(f"{where}: повтор правки {slug} {field}")
            fixes[(slug, field)] = {
                "field": field,
                "old": old,
                "new": new,
                "rule": row["rule"].strip(),
            }
    return fixes


# ---------------------------------------------------------------- текст
def gt_norm(s: str | None) -> str:
    """Норма прототипа: «№» и «/» остаются, диакритика снимается только у латиницы."""
    s = unicodedata.normalize("NFKC", s or "").replace("ё", "е").replace("Ё", "Е").lower()
    out = []
    for ch in s:
        if ord(ch) < 0x0400:  # «й» не трогаем
            decomposed = unicodedata.normalize("NFKD", ch)
            ch = "".join(c for c in decomposed if not unicodedata.combining(c))
        out.append(ch)
    s = "".join(out).replace("_", " ")
    s = re.sub(r"[^\w\s/№]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def slug_translit(s: str) -> str:
    """Кириллица → латиница slug (`nd_common.translit`): «Пино Нуар» → «pino-nuar»."""
    s = s.lower()
    return re.sub(r"[^a-z0-9]+", "-", "".join(_SLUG_TR.get(c, c) for c in s)).strip("-")


def lat(s: str) -> str:
    return slug_translit(s).replace("-", " ").strip()


def cyr_variants(token: str) -> list[str]:
    """Латинский токен → кириллические прочтения (`transcribe_variants`)."""
    return [v for v in transcribe_variants(token) if v != token]


def ru_sugar(text: str) -> list[str]:
    """Классы сахара, названные в тексте; «экстра брют» поглощает «брют»."""
    text = text.lower()
    found = [cls for rx, cls in RU_SUGAR if re.search(rx, text)]
    if "extra_brut" in found and "brut" in found:
        found.remove("brut")
    return found


def is_lat(t: str) -> bool:
    return bool(re.fullmatch(r"[a-z]+", t))


def has_cyr(t: str) -> bool:
    return any("а" <= c <= "я" for c in t)


def find_phrases(text: str, phrases: Iterable[str]) -> tuple[list[str], list[tuple[int, int]]]:
    """Вхождения фраз с границами слова, длинные раньше коротких, без перекрытий."""
    found: list[tuple[int, str]] = []
    taken: list[tuple[int, int]] = []
    # Прототип сортировал множество только по длине; равные длины — по алфавиту, для
    # воспроизводимости между запусками.
    for ph in sorted(set(phrases), key=lambda p: (-len(p), p)):
        if not ph:
            continue
        for m in re.finditer(r"(?<![\w])" + re.escape(ph) + r"(?![\w])", text):
            if any(m.start() < e and m.end() > s for s, e in taken):
                continue
            taken.append((m.start(), m.end()))
            found.append((m.start(), ph))
    return [ph for _, ph in sorted(found)], taken


def cut(text: str, spans: Iterable[tuple[int, int]]) -> str:
    for s, e in sorted(spans, reverse=True):
        text = text[:s] + " " + text[e:]
    return re.sub(r"\s+", " ", text).strip()


def roman_tokens(name: str | None) -> list[str]:
    out = []
    for t in re.findall(r"(?<![\w])[IVXLC]{1,7}(?![\w])", name or ""):
        if ROMAN_RE.match(t) and (len(t) >= 2 or t in ("V", "X")):
            out.append(t)
    return out


def photo_kind(fn: str | None) -> str:
    b = Path(fn or "").stem if fn else ""
    if re.match(r"(?i)^(screenshot|снимок)", b):
        return "screenshot"
    if re.match(r"(?i)^(dsc|img|photo|p\d|_mg)", b):
        return "camera_name"
    if (
        len(b) >= 24
        and re.fullmatch(r"[A-Za-z0-9_=+\-]+", b)
        and not re.search(r"[a-z]{4,}_[a-z]{4,}", b.lower())
    ):
        return "hash"
    return "descriptive"


def variants_for_tokens(tokens: Iterable[str]) -> list[str]:
    out: set[str] = set()
    for t in tokens:
        if has_cyr(t):
            out.add(lat(t))
        elif is_lat(t):
            out.update(cyr_variants(t)[:2])
    out.discard("")
    return sorted(out)


def load_winery_aliases(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = collections.defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(.*?) -> (\[.*\])\s*$", line)
        if m:
            try:
                out[m.group(1).strip()] += ast.literal_eval(m.group(2))
            except (ValueError, SyntaxError):
                pass
    return out


# ---------------------------------------------------------------- сборка
class GtBuilder:
    def __init__(
        self,
        rows: list[dict[str, str]],
        clusters: dict[str, Any],
        live_wines: list[dict[str, Any]],
        winery_aliases: dict[str, list[str]],
        *,
        packshot_aliases: bool = False,
        fixes: dict[tuple[str, str], dict[str, Any]] | None = None,
    ) -> None:
        self.fixes = dict(fixes or {})
        self.applied: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        self.manual: list[tuple[str, dict[str, tuple[list[str], str]]]] = [
            ("manual", MANUAL_WINERY)
        ]
        if packshot_aliases:
            self.manual.append(("packshot", PACKSHOT_WINERY))
        self.first: dict[str, dict[str, str]] = {}
        for r in rows:
            self.first.setdefault(r["Slug"], r)
        self.nd = clusters
        self.slug_attrs: dict[str, dict[str, Any]] = clusters["slug_attrs"]
        self.live = {w["slug"]: w for w in live_wines}
        self.walias = winery_aliases
        # сорта: нормализованный вариант → код
        self.glook: dict[str, str] = {}
        for code, vs in GRAPE_SYNONYMS.items():
            for v in [*vs, code.replace("_", " ")]:
                self.glook[gt_norm(v)] = code
            for v in vs:
                if has_cyr(gt_norm(v)):
                    self.glook.setdefault(lat(v), code)  # «saperavi», «kaberne sovinon»
        # визуальные группы level_B
        self.vgroup: dict[str, tuple[Any, int, list[str]]] = {}
        self.cluster_of: dict[str, Any] = {}
        for cl in clusters["level_B"]["clusters"]:
            for m in cl["members"]:
                self.cluster_of[m["slug"]] = cl["cluster_id"]
            for gi, g in enumerate(cl["visual_groups"]):
                if len(g) > 1:
                    for s in g:
                        self.vgroup[s] = (cl["cluster_id"], gi, g)

    def fix(self, slug: str, field: str, current: str | None) -> dict[str, Any] | None:
        """Правка поля карточки из `gt_fixes.tsv`; «было» должно совпасть со сборкой."""
        fix = self.fixes.get((slug, field))
        if fix is None:
            return None
        if fix["old"] != current:
            raise ValueError(
                f"gt_fixes: {slug} {field}: сборка даёт {current!r}, в таблице было {fix['old']!r}"
            )
        self.applied[slug].append({"field": field, "old": fix["old"], "new": fix["new"]})
        return fix

    def winery_block(self, w: str) -> dict[str, Any]:
        n = gt_norm(w)
        key = [t for t in n.split() if t not in WINERY_GENERIC and len(t) > 1]
        variants: set[str] = set()
        src: dict[str, str] = {}
        for a in self.walias.get(w, []) + self.walias.get(w.strip(), []):
            variants.add(gt_norm(a))
            src[gt_norm(a)] = "strapi_winery_map"
        for source, table in self.manual:
            if w.strip() in table:
                forms, note = table[w.strip()]
                for v in forms:
                    variants.add(gt_norm(v))
                    src[gt_norm(v)] = f"{source}: {note}"
        k = " ".join(key)
        if has_cyr(k):
            variants.add(lat(k))
            src[lat(k)] = "translit"
        elif k:
            for t in key:
                for cv in cyr_variants(t)[:2]:
                    variants.add(cv)
                    src[cv] = "loza_transcribe"
        variants.discard(n)
        variants.discard("")
        return {
            "canonical": w.strip(),
            "norm": n,
            "key_tokens": key,
            "variants": sorted(variants),
            "variant_src": src,
        }

    def grapes_block(
        self, csv_grape: str | None, name_n: str
    ) -> tuple[dict[str, Any], list[tuple[int, int]]]:
        vals = [g.strip() for g in (csv_grape or "").split(",") if g.strip()]
        codes: list[str] = []
        unknown: list[str] = []
        generic = False
        for g in vals:
            gn = gt_norm(g)
            if gn in GRAPE_GENERIC:
                generic = True
                continue
            c = self.glook.get(gn)
            if not c:
                # «Рислинг Рейнский», «Мускат белый» — самый длинный известный вариант внутри
                hit, _ = find_phrases(gn, list(self.glook))
                c = self.glook[hit[0]] if hit else None
            if c:
                codes.append(c)
            else:
                codes.append("csv:" + gn)
                unknown.append(g)
        variants: set[str] = set()
        for c in codes:
            if c.startswith("csv:"):
                variants.add(c[4:])
                variants.add(lat(c[4:]))
            else:
                for v in GRAPE_SYNONYMS[c]:
                    variants.add(gt_norm(v))
                variants.add(lat(GRAPE_SYNONYMS[c][0]))
        in_name, spans = find_phrases(
            name_n, [v for v in variants if len(v) >= 4] + [k for k in self.glook if len(k) >= 5]
        )
        block = {
            "values": vals,
            "codes": codes,
            "variants": sorted(variants),
            "unknown": unknown,
            "generic": generic,
            "in_name": in_name,
        }
        return block, spans

    def parse_names(self) -> dict[str, dict[str, Any]]:
        """Проход 1: разбор названий на винодельню, сорт, сахар, год, серию и кюве."""
        recs: dict[str, dict[str, Any]] = {}
        for slug, r in self.first.items():
            name = (r["Название вина"] or "").strip()
            # Объём и крепость убираем до нормы: она съедает запятые, и «0,2» становилось
            # серийными «0» и «2».
            name_clean = re.sub(
                r"(?<![\w])\d+[.,]\d+\s*(?:%|л|l)?(?![\w])|(?<![\w])\d+\s*%"
                r"|(?<![\w])\d{3,4}\s*(?:мл|ml)(?![\w])",
                " ",
                name,
            )
            name_n = gt_norm(name_clean)
            w = self.winery_block(r["Винодельня"])
            csv_grape = r["Сорт винограда"]
            grape_fix = self.fix(slug, "grape", (csv_grape or "").strip() or None)
            if grape_fix is not None:
                csv_grape = grape_fix["new"] or ""
            g, _ = self.grapes_block(csv_grape, name_n)
            sa = self.slug_attrs.get(slug, {})
            # сахар
            sugar, sugar_src, sugar_conflict = None, None, False
            lv = self.live.get(slug)
            if lv:
                cat = gt_norm(lv["category"])
                for ph in sorted(LIVE_SUGAR, key=len, reverse=True):
                    if cat.endswith(ph):
                        sugar, sugar_src = LIVE_SUGAR[ph], "live_api"
                        break
            ss = sa.get("sugar_slug") or []
            slug_sugar = ("extra_brut" if "extra_brut" in ss else ss[0]) if ss else None
            if sugar is None and slug_sugar:
                sugar, sugar_src = slug_sugar, "slug"
            # на этикетке напечатано то, что в названии («Extra Brut»), даже если категория
            # каталога говорит «брют»
            catalog_sugar = sugar
            rs = ru_sugar(name)
            name_sugar = rs[0] if rs else None
            if name_sugar and name_sugar != sugar:
                if sugar is not None:
                    sugar_conflict = True
                sugar, sugar_src = name_sugar, "name" + ("_over_" + sugar_src if sugar_src else "")
            if sugar and slug_sugar and slug_sugar != sugar:
                sugar_conflict = True
            sugar_fix = self.fix(slug, "sugar", sugar)
            if sugar_fix is not None:
                sugar, sugar_src = sugar_fix["new"], "gt_fixes"
            # год
            name_years = [int(y) for y in YEAR_RE.findall(name)]
            year = sa.get("year_slug") or (name_years[0] if name_years else None)
            year_src = "slug" if sa.get("year_slug") else ("name" if name_years else None)
            year_conflict = bool(
                sa.get("year_slug") and name_years and sa["year_slug"] not in name_years
            )
            # серийные токены
            serial = roman_tokens(name)
            serial += re.findall(r"(?<![\w/])\d{1,3}/\d{1,3}(?![\w/])", name_n)
            serial += [x.replace(" ", "") for x in re.findall(r"№\s*\d+", name_n)]
            work = re.sub(r"\d+[.,]?\d*\s*%", " ", name_n)
            work = YEAR_RE.sub(" ", work)
            work = re.sub(r"(?<![\w/])\d{1,3}/\d{1,3}(?![\w/])|№\s*\d+", " ", work)
            serial += re.findall(r"(?<![\w])\d{1,3}(?![\w])", work)
            work = re.sub(r"(?<![\w])\d{1,3}(?![\w])", " ", work)
            serial_kw, kspans = find_phrases(work, SERIAL_PHRASES)
            work = cut(work, kspans)
            # вырезаем винодельню, сорта, сахар, цвет, служебные слова
            wphr = [w["norm"], *w["variants"], *w["key_tokens"]]
            _, sp = find_phrases(work, [p for p in wphr if p])
            work = cut(work, sp)
            grape_phrases = [v for v in g["variants"] if len(v) >= 3]
            _, sp = find_phrases(work, grape_phrases + [k for k in self.glook if len(k) >= 5])
            work = cut(work, sp)
            sugar_words = [v for vs in SUGAR_LABEL.values() for v in vs]
            _, sp = find_phrases(work, [*sugar_words, "экстра", "extra", "натюр", "nature"])
            work = cut(work, sp)
            _, sp = find_phrases(work, [v for vs in COLOR_LABEL.values() for v in vs])
            work = cut(work, sp)
            cuvee = [t for t in work.split() if t not in GENERIC and len(t) > 1 and not t.isdigit()]
            romans_l = {x.lower() for x in roman_tokens(name)}
            cuvee = [t for t in cuvee if t not in romans_l]
            recs[slug] = {
                "slug": slug,
                "r": r,
                "name": name,
                "name_n": name_n,
                "w": w,
                "g": g,
                "sugar": sugar,
                "sugar_src": sugar_src,
                "sugar_conflict": sugar_conflict,
                "catalog_sugar": catalog_sugar,
                "year": year,
                "year_src": year_src,
                "year_conflict": year_conflict,
                "serial": serial,
                "serial_kw": serial_kw,
                "cuvee": cuvee,
                "sa": sa,
            }
        return recs

    @staticmethod
    def winery_brands(
        recs: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """Проход 2: бренды линеек — токены кюве у ≥25 % позиций винодельни."""
        by_w: dict[str, list[str]] = collections.defaultdict(list)
        for s, x in recs.items():
            by_w[x["w"]["canonical"]].append(s)
        brands: dict[str, list[str]] = {}
        for wn, ss in by_w.items():
            if len(ss) < 4:
                continue
            cnt: collections.Counter[str] = collections.Counter()
            for s in ss:
                toks = recs[s]["cuvee"]
                grams = set(toks) | {" ".join(toks[i : i + 2]) for i in range(len(toks) - 1)}
                cnt.update(grams)
            br = [t for t, c in cnt.items() if c / len(ss) >= 0.25]  # «аристов» у Кубань-Вино
            # биграмма поглощает свои униграммы
            br = [t for t in br if not any(t != b and " " in b and t in b.split() for b in br)]
            if br:
                brands[wn] = sorted(br)
        return brands, by_w

    def assemble(
        self,
        recs: dict[str, dict[str, Any]],
        brands: dict[str, list[str]],
        flags_cnt: collections.Counter[str],
        cov: collections.Counter[str],
    ) -> dict[str, dict[str, Any]]:
        final: dict[str, dict[str, Any]] = {}
        for slug, x in recs.items():
            r, w, g = x["r"], x["w"], x["g"]
            wb = brands.get(w["canonical"], [])
            # бренд-линейку из кюве не вычитаем: внутри винодельни она и есть различитель
            cuvee = list(x["cuvee"])
            # «Категория» выгрузки остаётся в `category`; правка меняет только класс цвета поля
            category = r["Категория"].strip() or None
            color = category
            color_fix = self.fix(slug, "color", category)
            if color_fix is not None:
                color = color_fix["new"]
            pk = photo_kind(r["Название фото"])
            flags = self._flags(x, g, pk, slug)
            flags_cnt.update(flags)
            vg = self.vgroup.get(slug)
            abv = x["sa"].get("abv_slug")
            tokens: list[dict[str, Any]] = []  # плоский список для CER и поиска

            def add(
                field: str,
                text: str,
                primary: bool,
                src: str,
                tokens: list[dict[str, Any]] = tokens,
            ) -> None:
                if not text:
                    return
                if re.fullmatch(r"[a-z0-9 /№]+", text):
                    script = "lat"
                elif has_cyr(text):
                    script = "cyr"
                else:
                    script = "mixed"
                tokens.append(
                    {"field": field, "text": text, "primary": primary, "src": src, "script": script}
                )

            add("winery", " ".join(w["key_tokens"]) or w["norm"], True, "csv")
            for v in w["variants"]:
                add("winery", v, False, w["variant_src"].get(v, "alias"))
            for b in wb:
                add("winery_brand", b, False, "names_of_winery>=40%")
            for t in cuvee:
                add("cuvee", t, True, "csv_name")
            for v in variants_for_tokens(cuvee):
                add("cuvee", v, False, "translit")
            plain_values = [v for v in g["values"] if gt_norm(v) not in GRAPE_GENERIC]
            for _, val in zip(g["codes"], plain_values, strict=False):
                add("grape", gt_norm(val), True, "csv_grape")
            for v in g["variants"]:
                if not any(t["text"] == v and t["field"] == "grape" for t in tokens):
                    add("grape", v, False, "taxonomy/translit")
            if x["sugar"]:
                add("sugar", SUGAR_LABEL[x["sugar"]][0], True, x["sugar_src"])
                for v in SUGAR_LABEL[x["sugar"]][1:]:
                    add("sugar", v, False, "lexicon")
            if x["year"]:
                add("year", str(x["year"]), True, x["year_src"])
            for t in x["serial"]:
                add("serial", t.lower() if not ROMAN_RE.match(t) else t, True, "csv_name")
            for t in x["serial_kw"]:
                add("serial", t, True, "csv_name_kw")
            if color:
                add("color", COLOR_LABEL.get(color, [gt_norm(color)])[0], True, "csv_category")
            for k in ("winery", "cuvee", "grape", "sugar", "year", "serial", "color"):
                if any(t["field"] == k and t["primary"] for t in tokens):
                    cov[k] += 1
            if abv:
                cov["abv"] += 1
            final[slug] = {
                "slug": slug,
                "published": slug in self.live,
                "name": x["name"],
                "winery": w["canonical"],
                "category": category,
                "live_category": self.live[slug]["category"] if slug in self.live else None,
                "region": r["Регион"].strip(),
                "photo_name": r["Название фото"],
                "photo_kind": pk,
                "fields": {
                    "winery": {
                        "key_tokens": w["key_tokens"],
                        "variants": w["variants"],
                        "brands": wb,
                    },
                    "cuvee": {"tokens": cuvee, "variants": variants_for_tokens(cuvee)},
                    "grape": {
                        "values": g["values"],
                        "codes": g["codes"],
                        "variants": g["variants"],
                        "in_name": g["in_name"],
                    },
                    "sugar": {
                        "class": x["sugar"],
                        "src": x["sugar_src"],
                        "catalog_class": x["catalog_sugar"],
                        "variants": SUGAR_LABEL.get(x["sugar"], []),
                    },
                    "year": {"value": x["year"], "src": x["year_src"]},
                    "abv": {
                        "value": abv,
                        "src": "slug" if abv else None,
                        "front_label_expected": False,
                    },
                    "serial": {"tokens": x["serial"], "keywords": x["serial_kw"]},
                    "color": {"class": color, "variants": COLOR_LABEL.get(color, [])},
                },
                "cluster_B": self.cluster_of.get(slug),
                "visual_group": [vg[0], vg[1]] if vg else None,
                "visual_mates": [s for s in vg[2] if s != slug] if vg else [],
                "noise_flags": flags,
                "expected_tokens": tokens,
            }
            if self.applied.get(slug):
                final[slug]["fixes"] = self.applied[slug]
        return final

    def _flags(self, x: dict[str, Any], g: dict[str, Any], pk: str, slug: str) -> list[str]:
        flags = []
        if not x["cuvee"] and not x["serial"] and not x["serial_kw"]:
            flags.append("name_grape_only" if g["in_name"] else "name_generic_only")
        if (
            len(x["cuvee"]) == 1
            and has_cyr(x["cuvee"][0])
            and not g["in_name"]
            and not x["serial"]
            and len(x["name_n"].split()) == 1
        ):
            flags.append("single_word_name")  # «Олег», «Победа»: условное имя
        if g["generic"]:
            flags.append("grape_generic")
        if g["unknown"]:
            flags.append("grape_not_in_taxonomy")
        if pk in ("screenshot", "camera_name"):
            flags.append("photo_" + pk)
        if x["sugar_conflict"]:
            flags.append("sugar_conflict_name_category_slug")
        if x["year_conflict"]:
            flags.append("year_slug_vs_name_conflict")
        if x["sa"].get("dup_suffix"):
            flags.append("slug_dup_suffix")
        if not x["sa"].get("published", slug in self.live):
            flags.append("unpublished")
        if x["sugar"] is None:
            flags.append("sugar_unknown")
        # фото с сортом в имени файла, которого нет у позиции (vibes-vermentino… → 03_Silvaner_…)
        fn = gt_norm(Path(x["r"]["Название фото"] or "").stem if x["r"]["Название фото"] else "")
        lat_grapes = [k for k in self.glook if len(k) >= 6 and is_lat(k.replace(" ", ""))]
        fn_grapes = {self.glook[h] for h in find_phrases(fn, lat_grapes)[0]}
        mine = set(g["codes"]) | {self.glook[h] for h in g["in_name"] if h in self.glook}
        if fn_grapes and not (fn_grapes & mine) and not g["generic"]:
            flags.append("photo_name_grape_mismatch")
        return flags

    @staticmethod
    def keyset(rec: dict[str, Any], k: str) -> Any:
        f = rec["fields"]
        if k == "cuvee_or_grape":
            return frozenset(f["cuvee"]["tokens"]) | frozenset(f["grape"]["codes"])
        if k == "year":
            return f["year"]["value"]
        if k == "sugar":
            return f["sugar"]["class"]
        if k == "serial":
            return frozenset(f["serial"]["tokens"]) | frozenset(f["serial"]["keywords"])
        if k == "color":
            return f["color"]["class"]
        return tuple(f["abv"]["value"] or [])

    def build(self) -> tuple[dict[str, dict[str, Any]], dict[str, Any], list[Any]]:
        recs = self.parse_names()
        brands, by_w = self.winery_brands(recs)
        flags_cnt: collections.Counter[str] = collections.Counter()
        cov: collections.Counter[str] = collections.Counter()
        final = self.assemble(recs, brands, flags_cnt, cov)
        done = {(slug, fix["field"]) for slug, fixes in self.applied.items() for fix in fixes}
        unused = sorted(set(self.fixes) - done)
        if unused:
            raise ValueError(f"gt_fixes: нет в выгрузке {unused}")

        # пары двойников: чем различимы
        pairs_first: collections.Counter[str] = collections.Counter()
        pairs_any: collections.Counter[str] = collections.Counter()
        unresolved = []
        n_pairs = 0
        for cl in self.nd["level_B"]["clusters"]:
            for g in cl["visual_groups"]:
                for i in range(len(g)):
                    for j in range(i + 1, len(g)):
                        a, b = final[g[i]], final[g[j]]
                        n_pairs += 1
                        diffs = [k for k in TWIN_ORDER if self.keyset(a, k) != self.keyset(b, k)]
                        pairs_any.update(diffs)
                        pairs_first[diffs[0] if diffs else "none"] += 1
                        if not [k for k in diffs if k != "abv"]:
                            unresolved.append((g[i], g[j], "abv_only" if diffs else "identical"))
                            for s in (g[i], g[j]):
                                if "twin_text_unresolvable" not in final[s]["noise_flags"]:
                                    final[s]["noise_flags"].append("twin_text_unresolvable")
                                    flags_cnt["twin_text_unresolvable"] += 1

        n = len(final)
        top_wineries = sorted(brands, key=lambda k: -len(by_w[k]))[:25]
        summary = {
            "n_slugs": n,
            "n_published_live": sum(1 for s in final if final[s]["published"]),
            "coverage_primary_tokens": {k: [v, round(v / n, 3)] for k, v in cov.items()},
            "sugar_src": collections.Counter(final[s]["fields"]["sugar"]["src"] for s in final),
            "year_src": collections.Counter(final[s]["fields"]["year"]["src"] for s in final),
            "photo_kind": collections.Counter(final[s]["photo_kind"] for s in final),
            "noise_flags": flags_cnt,
            "wineries_with_brand_tokens": len(brands),
            "brand_examples": {k: brands[k] for k in top_wineries},
            "twin_pairs_level_B": n_pairs,
            "twin_pairs_first_distinguishing_field": pairs_first,
            "twin_pairs_any_difference": pairs_any,
            "twin_pairs_unresolved_by_front_text": len(unresolved),
            "unresolved_examples": unresolved[:40],
        }
        if self.fixes:
            summary["gt_fixes"] = {
                "applied": len(done),
                "by_field": collections.Counter(field for _, field in sorted(done)),
                "nulled": sum(
                    1 for fixes in self.applied.values() for f in fixes if f["new"] is None
                ),
            }
        return final, summary, self.checks(final)

    @staticmethod
    def checks(final: dict[str, dict[str, Any]]) -> list[Any]:
        """Выборка для ручной проверки эталона."""
        chk: list[Any] = []
        for s, rec in final.items():
            nm = rec["name"].lower()
            f = rec["fields"]
            if (
                "аристов" in nm
                or "aristov" in s
                or "аристов" in rec["winery"].lower()
                or "donum" in nm
            ):
                chk.append(("ARISTOV", s, rec["name"], rec["winery"]))
            if rec["winery"] == "Табия":
                chk.append(("TABIYA", s, rec["name"], f["cuvee"]["tokens"], rec["noise_flags"]))
            if rec["winery"] == "Массандра" and "мускат" in nm:
                chk.append(
                    ("MASS", s, rec["name"], f["cuvee"]["tokens"], f["sugar"]["class"],
                     rec["visual_mates"][:3])
                )  # fmt: skip
            if "primum" in nm:
                chk.append(
                    ("PRIMUM", s, f["cuvee"]["tokens"], f["serial"], f["year"]["value"],
                     f["sugar"]["class"])
                )  # fmt: skip
            if "photo_name_grape_mismatch" in rec["noise_flags"]:
                chk.append(
                    ("PHOTO_MISMATCH", s, rec["photo_name"], f["grape"]["values"], rec["name"])
                )
            if f["sugar"]["catalog_class"] and f["sugar"]["catalog_class"] != f["sugar"]["class"]:
                chk.append(
                    ("SUGAR_OVERRIDE", s, rec["name"], f["sugar"]["catalog_class"], "->",
                     f["sugar"]["class"])
                )  # fmt: skip
            if "single_word_name" in rec["noise_flags"] and len(chk) < 400:
                chk.append(("SINGLE", s, rec["name"], rec["winery"]))
            if f["serial"]["tokens"] and len([c for c in chk if c[0] == "SERIAL"]) < 40:
                chk.append(("SERIAL", s, rec["name"], f["serial"]))
        return chk


def write_outputs(
    out_dir: Path, final: dict[str, dict[str, Any]], summary: dict[str, Any], checks: list[Any]
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "gt_tokens.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for s in sorted(final):
            fh.write(json.dumps(final[s], ensure_ascii=False) + "\n")
    with (out_dir / "gt_summary.json").open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    with (out_dir / "gt_checks.txt").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(json.dumps(c, ensure_ascii=False) for c in checks))


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    catalog_dir = settings.data_dir / "catalog"
    parser = argparse.ArgumentParser(
        prog="python scripts/build_gt_tokens.py",
        description="Эталонные токены этикетки по slug каталога.",
    )
    parser.add_argument("--csv", type=Path, default=settings.dataset_dir / "strapi_output0709.csv")
    parser.add_argument("--clusters", type=Path, default=catalog_dir / "near_dup_clusters.json")
    parser.add_argument("--live", type=Path, default=catalog_dir / "plan_live_wines.json")
    parser.add_argument("--winery-map", type=Path, default=catalog_dir / "strapi_winery_map.txt")
    parser.add_argument("--out-dir", type=Path, default=settings.data_dir / "gt")
    parser.add_argument(
        "--packshot-aliases",
        action="store_true",
        help="добавить формы виноделен, прочитанные на packshot (утечка в studio и synth)",
    )
    parser.add_argument(
        "--fixes",
        type=Path,
        default=DEFAULT_FIXES,
        help="правки атрибутов карточек (Э4): TSV slug, поле, было, стало, два источника",
    )
    parser.add_argument(
        "--no-fixes", action="store_true", help="собрать эталон без правок карточек (как до Э4)"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = [args.csv, args.clusters, args.live, args.winery_map]
    if not args.no_fixes:
        inputs.append(args.fixes)
    for path in inputs:
        if not path.is_file():
            print(f"нет входного файла: {path}", file=sys.stderr)
            return 2
    fixes = None
    if not args.no_fixes:
        try:
            fixes = load_fixes(args.fixes)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 2
    with args.csv.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    clusters = json.loads(args.clusters.read_text(encoding="utf-8"))
    live = json.loads(args.live.read_text(encoding="utf-8"))
    builder = GtBuilder(
        rows,
        clusters,
        live,
        load_winery_aliases(args.winery_map),
        packshot_aliases=args.packshot_aliases,
        fixes=fixes,
    )
    try:
        final, summary, checks = builder.build()
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    if fixes is not None:
        summary["gt_fixes"] = {
            "file": args.fixes.name,
            # без CR: на Windows git отдаёт TSV с CRLF, а хэш должен быть один
            "sha1": hashlib.sha1(args.fixes.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            **summary.get("gt_fixes", {}),
        }
    write_outputs(args.out_dir, final, summary, checks)
    print("ok", len(final), args.out_dir, f"правок {len(fixes or {})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
