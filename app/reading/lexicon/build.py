"""Закрытый словарь каталога: что может быть напечатано на этикетке и у каких slug.

Словарь не выбирает вино. Запись говорит: «такой фрагмент текста — винодельня (кюве, сорт,
сахар, серия, цвет) и встречается у этих slug», а `idf` — насколько он редкий. Решение
принимает resolve поверх кандидатов CV.

Источники: `gt_tokens.jsonl` (`scripts/build_gt_tokens.py`) и общие термины этикетки ниже.
Запись ключуется парой (поле, норма). Одна норма у нескольких сущностей («шато» у разных
виноделен, «совиньон» у двух сортов) — одна запись с объединёнными slug, а `canonical`
тогда равна самой норме: так видно, что слово не указывает на одну сущность.

Слова многословных фраз индексируются отдельно, но запись, которая есть только как слово
фразы (`LexEntry.part`), попадания не даёт: «нуар» — обрывок «Пино Нуар» или «Блан де
Нуар», и угадывать фразу по нему нельзя. Такие записи служат поиску кандидатов, склейке
разорванных слов и тому, чтобы слово каталожной фразы не считалось «вне словаря».
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_args

from app.config import Settings, get_settings
from app.reading.contracts import Color, LexField, LexHit, SugarClass
from app.reading.text.normalize import norm, norm_token, roman_value
from app.reading.text.translit import skeleton

#: 2 — у записей появился признак `part` (слово фразы без самостоятельной формы).
LEXICON_VERSION = 2
FIELDS: tuple[LexField, ...] = get_args(LexField)
#: Доля slug, при которой норма считается принадлежащей одной сущности.
CANONICAL_SHARE = 0.8
#: Слова фраз короче этого отдельно не индексируются: «de», «ice», «пет».
MIN_PHRASE_WORD = 4

# Поля-названия: у них одиночные служебные слова не становятся записями.
_NAME_FIELDS: frozenset[LexField] = frozenset({"producer", "cuvee", "grape"})

#: Слова, которые не отличают вино: тип продукта, «винодельня», предлоги, регионы.
GENERIC_WORDS = frozenset(
    {
        # тип продукта и служебные слова названий
        "вино", "вина", "игристое", "шампанское", "российское", "тихое", "бутылка",
        "wine", "wines", "sparkling", "vol", "свое", "органик", "organic", "vino", "vina",
        # хозяйство
        "винодельня", "винодельни", "winery", "vineyard", "vineyards", "estate", "estates",
        "дом", "шампанских", "вин", "имение", "усадьба", "wein", "und", "vines",
        # предлоги и артикли
        "и", "в", "с", "на", "&", "the", "of", "and", "de", "la", "le", "les", "di", "del",
        "du", "des", "da", "d", "де", "ди", "да", "ла", "ле",
        # регионы: на этикетке это строка происхождения, а не винодельня
        "россия", "россии", "russia", "крым", "крыма", "crimea", "кубань", "kuban",
        "дагестан", "севастополь",
        # цвет в названиях сортов: «Мускат белый», «Цимлянский черный»
        "белый", "белая", "черный", "красный", "розовый", "зеленый", "belyy", "belyj",
        "chernyy", "chernyj", "cherny", "krasnyy", "rozovyy", "rozovy", "zelenyy",
    }
)  # fmt: skip

#: Классы сахара. Длинные формы и «demi-sec = полусладкое» — как в `RU_SUGAR` из `scripts/build_gt_tokens.py`.
SUGAR_TERMS: dict[SugarClass, tuple[str, ...]] = {
    SugarClass.BRUT_NATURE: (
        "брют натюр", "brut nature", "pas dosé", "zero dosage", "dosage zero", "брют зеро",
    ),
    SugarClass.EXTRA_BRUT: ("экстра брют", "extra brut"),
    SugarClass.BRUT: ("брют", "brut"),
    SugarClass.DRY: ("сухое", "dry", "sec", "secco", "seco"),
    SugarClass.SEMI_DRY: ("полусухое", "semi-dry", "off-dry"),
    SugarClass.SEMI_SWEET: ("полусладкое", "demi-sec", "semi-sweet", "amabile"),
    SugarClass.SWEET: ("сладкое", "sweet", "doux", "dolce"),
}  # fmt: skip

#: Цвет только по словам; цвет этикетки по пикселям — дело CV.
COLOR_TERMS: dict[Color, tuple[str, ...]] = {
    Color.WHITE: ("белое", "white", "blanc", "bianco", "blanco"),
    Color.RED: ("красное", "red", "rouge", "rosso", "tinto"),
    Color.ROSE: ("розовое", "rosé", "розе", "rosato", "rosado", "pink"),
    Color.ORANGE: ("оранжевое", "orange", "оранж", "янтарное"),
}

#: Серийные слова: каноническая форма → написания. Этикетка латиницей находит slug,
#: у которого в каталоге то же слово кириллицей («Blanc de Blancs» ↔ «Блан де Блан»).
SERIAL_TERMS: dict[str, tuple[str, ...]] = {
    "резерв": ("reserve", "reserva", "riserva"),
    "гран резерв": ("grand reserve", "gran reserva"),
    "семейный резерв": ("family reserve",),
    "блан де блан": ("blanc de blancs", "blanc de blanc"),
    "блан де нуар": ("blanc de noirs", "blanc de noir"),
    "кюве": ("cuvée",),
    "престиж": ("prestige",),
    "премиум": ("premium",),
    "селект": ("select", "selection"),
    "гран крю": ("grand cru",),
    "ультра": ("ultra",),
    "петнат": ("пет нат", "pét-nat", "petnat"),
    "баррель": ("barrel",),
    "barrel fermented": (),
    "limited edition": (),
    "коллекционное": (),
    "collection": (),
    "терруар": ("terroir",),
    "бэг ин бокс": ("bag in box",),
    "в банке": (),
    "ледяное": ("ice wine",),
    "поздний сбор": ("late harvest",),
    "выдержанное": (),
    "магнум": ("magnum",),
    "классик": ("classic",),
    "оригинал": ("original",),
}

_GRAPE_GENERIC = frozenset({"белые сорта винограда", "красные сорта винограда"})


def entry_norm(text: str) -> str:
    """Норма записи — та же, что у токенов этикетки: `norm_token` по словам."""
    return " ".join(word for word in (norm_token(part) for part in norm(text).split()) if word)


def idf(df: int, n_slugs: int) -> float:
    """Сглаженный idf: `ln((1 + N) / (1 + df)) + 1`; запись без slug — самая редкая."""
    return round(math.log((1 + n_slugs) / (1 + df)) + 1.0, 4)


def _entry_skeleton(text: str, canonical: str) -> str:
    # У чисел и римских номеров скелет — сама норма, как у токенов (`TokenSpan`).
    if not any(ch.isalpha() for ch in text) or (canonical.isupper() and roman_value(text)):
        return text
    return skeleton(text)


@dataclass(frozen=True, slots=True)
class LexEntry:
    """Запись словаря: норма, её скелет, сущность каталога и slug, где она встречается.

    `part=True` — норма есть в каталоге только как слово многословной фразы: попадания она
    не даёт, а служит поиску кандидатов.
    """

    field: LexField
    canonical: str
    norm: str
    skeleton: str
    slugs: frozenset[str]
    idf: float
    part: bool = False

    @property
    def n_words(self) -> int:
        return self.norm.count(" ") + 1

    def hit(self, cost: float) -> LexHit:
        return LexHit(canonical=self.canonical, field=self.field, cost=cost, slugs=self.slugs)


class Lexicon:
    """Записи словаря с точными индексами по норме и скелету.

    Поддерживает `in` по норме или скелету — это `lexicon` для `tokenize`, чтобы
    склеивать разорванные слова. Нечёткий поиск — `app.reading.lexicon.correct.lookup`.
    """

    def __init__(
        self,
        entries: Iterable[LexEntry],
        *,
        n_slugs: int,
        meta: Mapping[str, Any] | None = None,
    ) -> None:
        order = {name: i for i, name in enumerate(FIELDS)}
        self.entries: tuple[LexEntry, ...] = tuple(
            sorted(entries, key=lambda e: (order[e.field], e.norm))
        )
        self.n_slugs = n_slugs
        self.meta: dict[str, Any] = dict(meta or {})
        # Производные структуры поиска (`correct`) живут здесь и строятся лениво.
        self.cache: dict[str, Any] = {}
        by_norm: dict[str, list[int]] = defaultdict(list)
        by_skeleton: dict[str, list[int]] = defaultdict(list)
        by_key: dict[tuple[str, str], int] = {}
        for i, entry in enumerate(self.entries):
            key = (entry.field, entry.norm)
            if key in by_key:
                raise ValueError(f"Lexicon: повтор записи {key}")
            by_key[key] = i
            by_norm[entry.norm].append(i)
            by_skeleton[entry.skeleton].append(i)
        self._by_norm = {k: tuple(v) for k, v in by_norm.items()}
        self._by_skeleton = {k: tuple(v) for k, v in by_skeleton.items()}
        self._by_key = by_key
        self.max_words = max((e.n_words for e in self.entries), default=0)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[LexEntry]:
        return iter(self.entries)

    def __contains__(self, text: object) -> bool:
        return isinstance(text, str) and (text in self._by_norm or text in self._by_skeleton)

    def ids_by_norm(self, text: str) -> tuple[int, ...]:
        return self._by_norm.get(text, ())

    def ids_by_skeleton(self, text: str) -> tuple[int, ...]:
        return self._by_skeleton.get(text, ())

    def get(self, field: LexField, text: str) -> LexEntry | None:
        """Запись по полю и норме; `text` нормализуется."""
        i = self._by_key.get((field, entry_norm(text)))
        return None if i is None else self.entries[i]

    def find(self, text: str) -> list[LexEntry]:
        """Все записи с такой нормой, во всех полях."""
        return [self.entries[i] for i in self.ids_by_norm(entry_norm(text))]

    # ------------------------------------------------------------------ сохранение
    def to_json(self) -> dict[str, Any]:
        slugs = sorted({s for e in self.entries for s in e.slugs})
        index = {s: i for i, s in enumerate(slugs)}
        return {
            "version": LEXICON_VERSION,
            "n_slugs": self.n_slugs,
            "meta": self.meta,
            "slugs": slugs,
            "entries": [
                {
                    "field": e.field,
                    "canonical": e.canonical,
                    "norm": e.norm,
                    "skeleton": e.skeleton,
                    "idf": e.idf,
                    "slugs": sorted(index[s] for s in e.slugs),
                    **({"part": True} if e.part else {}),
                }
                for e in self.entries
            ],
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Lexicon:
        if data.get("version") != LEXICON_VERSION:
            raise ValueError(
                f"Lexicon: версия {data.get('version')}, ожидается {LEXICON_VERSION} — "
                "пересоберите словарь: python scripts/build_lexicon.py"
            )
        slugs = data["slugs"]
        entries = []
        for row in data["entries"]:
            if row["field"] not in FIELDS:
                raise ValueError(f"Lexicon: неизвестное поле {row['field']!r}")
            entries.append(
                LexEntry(
                    field=row["field"],
                    canonical=row["canonical"],
                    norm=row["norm"],
                    skeleton=row["skeleton"],
                    slugs=frozenset(slugs[i] for i in row["slugs"]),
                    idf=float(row["idf"]),
                    part=bool(row.get("part", False)),
                )
            )
        return cls(entries, n_slugs=int(data["n_slugs"]), meta=data.get("meta"))

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            json.dump(self.to_json(), fh, ensure_ascii=False, separators=(",", ":"))
            fh.write("\n")
        return path

    @classmethod
    def load(cls, path: Path | str) -> Lexicon:
        with Path(path).open(encoding="utf-8") as fh:
            return cls.from_json(json.load(fh))

    def stats(self, *, top: int = 5) -> dict[str, Any]:
        """Сводка для печати: записи по полям, фразы, записи без slug, самые частые нормы."""
        by_field = Counter(e.field for e in self.entries)
        common: dict[str, list[tuple[str, int]]] = {}
        for name in FIELDS:
            rows = sorted(
                (e for e in self.entries if e.field == name), key=lambda e: (-len(e.slugs), e.norm)
            )
            common[name] = [(e.norm, len(e.slugs)) for e in rows[:top]]
        return {
            "entries": len(self.entries),
            "n_slugs": self.n_slugs,
            "max_words": self.max_words,
            "by_field": {name: by_field.get(name, 0) for name in FIELDS},
            "multiword": dict(Counter(e.field for e in self.entries if e.n_words > 1)),
            "phrase_parts": dict(Counter(e.field for e in self.entries if e.part)),
            "without_slugs": dict(Counter(e.field for e in self.entries if not e.slugs)),
            "shared_norms": dict(
                Counter(
                    e.field
                    for e in self.entries
                    if e.canonical == e.norm and e.field in _NAME_FIELDS and len(e.slugs) > 1
                )
            ),
            "most_common": common,
        }


# ---------------------------------------------------------------------- сборка
@dataclass(slots=True)
class _Acc:
    slugs: set[str] = field(default_factory=set)
    votes: dict[str, set[str]] = field(default_factory=dict)
    fixed: str | None = None  # каноническая форма из таблиц терминов перекрывает голоса
    whole: bool = False  # норма добавлена сама по себе, а не только как слово фразы

    def canonical(self, text: str) -> str:
        if self.fixed is not None:
            return self.fixed
        if not self.slugs:
            return next(iter(self.votes)) if len(self.votes) == 1 else text
        best, owners = max(self.votes.items(), key=lambda kv: (len(kv[1]), kv[0]))
        return best if len(owners) >= CANONICAL_SHARE * len(self.slugs) else text


class _Builder:
    def __init__(self) -> None:
        self.acc: dict[tuple[LexField, str], _Acc] = {}
        self.slugs: set[str] = set()

    def add(
        self,
        field_name: LexField,
        text: str,
        canonical: str,
        slug: str | None,
        *,
        words: bool = False,
        fixed: bool = False,
    ) -> None:
        """Запись фразы целиком и, если `words`, её значимых слов (только как `part`)."""
        value = entry_norm(text)
        if not value:
            return
        single = " " not in value and field_name in _NAME_FIELDS
        if single and (value in GENERIC_WORDS or len(value) < 2):
            return
        self._put(field_name, value, canonical, slug, fixed=fixed, whole=True)
        if words and " " in value:
            for word in value.split():
                if len(word) >= MIN_PHRASE_WORD and word.isalpha() and word not in GENERIC_WORDS:
                    self._put(field_name, word, canonical, slug, fixed=False, whole=False)

    def _put(
        self,
        field_name: LexField,
        value: str,
        canonical: str,
        slug: str | None,
        *,
        fixed: bool,
        whole: bool,
    ) -> None:
        acc = self.acc.setdefault((field_name, value), _Acc())
        acc.whole |= whole
        if fixed and acc.fixed is None:
            acc.fixed = canonical
        owners = acc.votes.setdefault(canonical, set())
        if slug is not None:
            acc.slugs.add(slug)
            owners.add(slug)

    def finish(self, meta: Mapping[str, Any] | None = None) -> Lexicon:
        n_slugs = len(self.slugs)
        entries = []
        for (field_name, value), acc in self.acc.items():
            canonical = acc.canonical(value)
            entries.append(
                LexEntry(
                    field=field_name,
                    canonical=canonical,
                    norm=value,
                    skeleton=_entry_skeleton(value, canonical),
                    slugs=frozenset(acc.slugs),
                    idf=idf(len(acc.slugs), n_slugs),
                    part=not acc.whole,
                )
            )
        return Lexicon(entries, n_slugs=n_slugs, meta=meta)


def _distance(a: str, b: str) -> int:
    """Обычный Левенштейн — только для привязки вариантов при сборке."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _term_classes(
    builtin: Mapping[Any, Sequence[str]] | None, observed: Iterable[tuple[str, str]]
) -> dict[str, str]:
    """Норма термина → класс. Таблица важнее каталога; спорные термины каталога отбрасываются."""
    out: dict[str, str] = {}
    for cls, terms in (builtin or {}).items():
        for term in terms:
            out.setdefault(entry_norm(term), str(cls))
    seen: dict[str, set[str]] = defaultdict(set)
    for term, cls in observed:
        seen[entry_norm(term)].add(cls)
    for value, classes in seen.items():
        if value and value not in out and len(classes) == 1:
            out[value] = next(iter(classes))
    return out


def _assign(
    variant: str, refs: Mapping[str, Sequence[str]], learned: Mapping[str, Counter[str]]
) -> str | None:
    """К какой сущности записи относится вариант написания («merlot» → «merlot», а не «пино»)."""
    if not refs:
        return None
    if len(refs) == 1:
        return next(iter(refs))
    value = entry_norm(variant)
    for canonical, _ in learned.get(value, Counter()).most_common():
        if canonical in refs:
            return canonical
    key = skeleton(value)
    return min(refs, key=lambda c: min(_distance(key, skeleton(r)) for r in refs[c]))


def _fields(rec: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return (rec.get("fields") or {}).get(name) or {}


def record_color(rec: Mapping[str, Any]) -> Any:
    """Цвет записи разметки: класс поля `color`, если поле есть, иначе «Категория» выгрузки.

    Поле решает, даже когда класс пуст: так правка карточки (Э4, `data/gt/gt_fixes.tsv`)
    обнуляет цвет, не трогая колонку «Категория», которую показывают карточки. В записях без
    правок класс поля и категория совпадают.
    """
    block = _fields(rec, "color")
    return block["class"] if "class" in block else rec.get("category")


def _grape_refs(rec: Mapping[str, Any]) -> dict[str, list[str]]:
    grape = _fields(rec, "grape")
    values = [v for v in grape.get("values") or [] if norm(v) not in _GRAPE_GENERIC]
    refs: dict[str, list[str]] = {}
    for code, value in zip(grape.get("codes") or [], values, strict=False):
        canonical = code.removeprefix("csv:")
        refs.setdefault(canonical, []).append(value)
    return refs


def _cuvee_refs(rec: Mapping[str, Any]) -> dict[str, list[str]]:
    # В названиях каталога бывают смешанные алфавиты («Сabernet»): сущность — норма токена.
    refs: dict[str, list[str]] = {}
    for token in _fields(rec, "cuvee").get("tokens") or []:
        value = entry_norm(token)
        if value:
            refs.setdefault(value, []).append(token)
    return refs


def build_from_records(
    records: Iterable[Mapping[str, Any]],
    *,
    builtin_terms: bool = True,
    meta: Mapping[str, Any] | None = None,
) -> Lexicon:
    """Словарь из записей формата `gt_tokens.jsonl`.

    `builtin_terms=False` — только то, что есть в записях (таблицы терминов не добавляются
    и не перекрывают каталог); нужно для проверок сопоставления по скелету.
    """
    records = [r for r in records if r.get("slug")]
    b = _Builder()

    sugar_classes = _term_classes(
        SUGAR_TERMS if builtin_terms else None,
        (
            (v, _fields(r, "sugar")["class"])
            for r in records
            if _fields(r, "sugar").get("class")
            for v in _fields(r, "sugar").get("variants") or []
        ),
    )
    color_classes = _term_classes(
        COLOR_TERMS if builtin_terms else None,
        (
            (v, _fields(r, "color")["class"])
            for r in records
            if _fields(r, "color").get("class")
            for v in _fields(r, "color").get("variants") or []
        ),
    )
    serial_groups: dict[str, tuple[str, tuple[str, ...]]] = {}
    if builtin_terms:
        for canonical, spellings in SERIAL_TERMS.items():
            members = tuple(dict.fromkeys(entry_norm(t) for t in (canonical, *spellings)))
            for member in members:
                serial_groups.setdefault(member, (canonical, members))
    for rec in records:
        for keyword in _fields(rec, "serial").get("keywords") or []:
            value = entry_norm(keyword)
            if value and value not in serial_groups:
                serial_groups[value] = (keyword, (value,))

    # Привязка вариантов к сущности учится на записях с одной сущностью.
    learned_cuvee: dict[str, Counter[str]] = defaultdict(Counter)
    learned_grape: dict[str, Counter[str]] = defaultdict(Counter)
    for rec in records:
        for refs, variants, learned in (
            (_cuvee_refs(rec), _fields(rec, "cuvee").get("variants") or [], learned_cuvee),
            (_grape_refs(rec), _fields(rec, "grape").get("variants") or [], learned_grape),
        ):
            if len(refs) == 1:
                (canonical,) = refs
                for variant in variants:
                    learned[entry_norm(variant)][canonical] += 1

    by_sugar: dict[str, list[str]] = defaultdict(list)
    for value, cls in sugar_classes.items():
        by_sugar[cls].append(value)
    by_color: dict[str, list[str]] = defaultdict(list)
    for value, cls in color_classes.items():
        by_color[cls].append(value)

    for rec in records:
        slug = str(rec["slug"])
        b.slugs.add(slug)

        winery = str(rec.get("winery") or "").strip()
        wf = _fields(rec, "winery")
        if winery:
            key_tokens = list(wf.get("key_tokens") or [])
            b.add("producer", winery, winery, slug, words=True)
            b.add("producer", " ".join(key_tokens), winery, slug, words=True)
            for token in key_tokens:
                b.add("producer", token, winery, slug)
            for variant in [*(wf.get("variants") or []), *(wf.get("brands") or [])]:
                b.add("producer", variant, winery, slug, words=True)

        refs = _cuvee_refs(rec)
        for token in refs:
            b.add("cuvee", token, token, slug)
        for variant in _fields(rec, "cuvee").get("variants") or []:
            canonical = _assign(variant, refs, learned_cuvee)
            if canonical:
                b.add("cuvee", variant, canonical, slug, words=True)

        refs = _grape_refs(rec)
        for canonical, values in refs.items():
            for value in values:
                b.add("grape", value, canonical, slug, words=True)
        for variant in _fields(rec, "grape").get("variants") or []:
            canonical = _assign(variant, refs, learned_grape)
            if canonical:
                b.add("grape", variant, canonical, slug, words=True)

        # Сахар и цвет — по классу позиции: слова фразы («sec» в «demi-sec») не индексируются.
        sugar = _fields(rec, "sugar").get("class")
        for value in by_sugar.get(sugar, []) if sugar else []:
            b.add("sugar", value, sugar, slug, fixed=True)
        color = record_color(rec)
        for value in by_color.get(color, []) if color else []:
            b.add("color", value, color, slug, fixed=True)

        serial = _fields(rec, "serial")
        for token in serial.get("tokens") or []:
            b.add("serial", token, token, slug)
        for keyword in serial.get("keywords") or []:
            canonical, members = serial_groups[entry_norm(keyword)]
            for member in members:
                b.add("serial", member, canonical, slug, words=True, fixed=True)

    # Термины без позиций в каталоге тоже нужны: «zero dosage» — признак, даже если он чужой.
    for value, cls in sugar_classes.items():
        b.add("sugar", value, cls, None, fixed=True)
    for value, cls in color_classes.items():
        b.add("color", value, cls, None, fixed=True)
    if builtin_terms:
        for value, (canonical, _) in serial_groups.items():
            b.add("serial", value, canonical, None, words=True, fixed=True)
    return b.finish(meta)


def read_gt_tokens(path: Path | str) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_from_gt(path: Path | str, *, builtin_terms: bool = True) -> Lexicon:
    """Словарь из `gt_tokens.jsonl`; в `meta` — имя и sha1 источника."""
    path = Path(path)
    meta = {"source": path.name, "source_sha1": hashlib.sha1(path.read_bytes()).hexdigest()}
    return build_from_records(read_gt_tokens(path), builtin_terms=builtin_terms, meta=meta)


def default_gt_tokens_path(settings: Settings | None = None) -> Path:
    return (settings or get_settings()).data_dir / "gt" / "gt_tokens.jsonl"


def default_lexicon_path(settings: Settings | None = None) -> Path:
    return (settings or get_settings()).data_dir / "index" / "lexicon.json"


def load_lexicon(path: Path | str | None = None) -> Lexicon:
    return Lexicon.load(path if path is not None else default_lexicon_path())
