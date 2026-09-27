"""Данные сомелье `data/somm/*.json`: загрузка, проверка формы, запасные заглушки.

Договор — `docs/api-sommelier.md`, §6.6 (интерфейс) и §7 (форма файлов). Файлы собирает
`scripts/build_somm.py` на CPU из фактов выгрузки организатора и правил «Лозы»; сервис их только
читает — один раз при старте, без движка и без `safe_eval`:

    pairs.json      вердикт и правила «+»/«−» каждой пары «вино × блюдо», `top` — до пяти блюд
    dishes.json     82 блюда «Лозы» (дательный, группа `food`, семья, алиасы) и 39 правил:
                    34 «Лозы» и 5 своих правил сканера (`overlay.json`, раздел `added`)
    serve.json      наша таблица подачи: первое подошедшее правило по порядку файла
    vocab.json      словарь замка сущностей: винодельни, сорта, блюда, регионы, ароматика
    knowledge.json  темы справочника после чистки по 38-ФЗ и портреты сортов

Каждый файл берётся из `directory` (`SVS_SOMM_DIR`); нет его там — из `fallback` (заглушки
`tests/fixtures/somm/` для разработки и тестов, источник `fixture`); нет и там — пусто, источник
`missing` и предупреждение в журнал. Сервис стартует всегда: без пар блюда пусты, а вина
подбираются фактами. `strict=True` (тесты, `build_somm.py --check`) превращает битый файл или
форму не по §7 в `ValueError`; без него такой файл тоже `missing`.

Заглушки вместо данных — не норма, а деградация: семь вин вместо 2 103 и четыре темы вместо 32.
`SommData.degraded_reasons()` называет каждый такой файл, `/v1/health` ставит блоку `somm` статус
`degraded` и добавляет предупреждение в общий список `warnings` — его видит проверка готовности
`scripts/run_eval.sh` (финальная проверка 24.09: заглушки подменяли данные молча).

Пары занимают основной объём (2 103 вина × 82 блюда), поэтому одинаковые пары и списки правил
загружаются одним объектом: у тысяч вин «борщ — да, те же четыре правила».
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from app.recommend.profile import AXES

logger = logging.getLogger(__name__)

SOMM_FILES: tuple[str, ...] = (
    "pairs.json",
    "dishes.json",
    "serve.json",
    "vocab.json",
    "knowledge.json",
)
FIXTURE_DIR: Path = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "somm"

Verdict = Literal["yes", "caveat", "no", "neutral"]
VERDICTS: tuple[str, ...] = ("yes", "caveat", "no", "neutral")

#: Группы блюд для чипов `/shelf` и шага `guided` (§7.2); у прочих блюд группы нет.
FOOD_GROUPS: tuple[str, ...] = ("meat", "fish", "cheese", "dessert")

#: Поля вина, которые читает условие правила: оси профиля и факты выгрузки (§1, чип правила).
WINE_FIELDS = frozenset({*AXES, "region", "color", "grapes", "serve", "descriptors"})

#: Знаки правил: «+» и «−» (U+2212), как на чипах.
SIGNS: tuple[str, ...] = ("+", "−")

#: Правила о рыбе, которые при равном весе идут первыми (`rule_order`): к рыбе и морепродуктам
#: главная причина — сахар или танины рядом с рыбой, а не общее «Вино мощнее блюда» (третий
#: круг проверки 25.09: у терпкого красного к устрицам оба правила весят −3, и по `id` первым было
#: «Вино мощнее блюда» — подборка шла «полегче», а не к винам почти без танинов).
FIRST_AT_EQUAL_WEIGHT: tuple[str, ...] = ("sweet_wine_on_fish", "no_tannin_with_oily_fish")


def rule_order(rule_id: str, weight: float) -> tuple[float, int, str]:
    """Порядок правил пары (§7.1): по убыванию модуля веса, при равенстве — правила о рыбе
    (`FIRST_AT_EQUAL_WEIGHT`), затем по `id`. Им же сервис выбирает сильнейшее правило."""
    first = (
        FIRST_AT_EQUAL_WEIGHT.index(rule_id)
        if rule_id in FIRST_AT_EQUAL_WEIGHT
        else len(FIRST_AT_EQUAL_WEIGHT)
    )
    return (-abs(weight), first, rule_id)


#: Ключи словаря замка сущностей (§7.4).
VOCAB_KEYS: tuple[str, ...] = (
    "wineries",
    "winery_words",
    "grapes",
    "dishes",
    "cooking",
    "regions",
    "descriptors",
    "generic",
    "taste_stems",
)


# ------------------------------------------------------------------ записи
@dataclass(frozen=True, slots=True)
class PairRule:
    """Правило сочетаний «Лозы»: знак, вес, подпись чипа, фраза шаблона, поля вина."""

    id: str
    sign: str
    weight: float
    name: str
    chip: str
    text: str
    wine_fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DishInfo:
    """Блюдо «Лозы»: дательный падеж для шаблонов, группа для чипов, семья, алиасы."""

    id: str
    name: str
    dative: str
    category: str
    food: str | None
    family: str
    aliases: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Pair:
    """Вердикт пары и id сработавших правил «+» и «−» по убыванию модуля веса."""

    verdict: Verdict
    plus: tuple[str, ...]
    minus: tuple[str, ...]


NEUTRAL = Pair("neutral", (), ())


@dataclass(frozen=True, slots=True)
class WinePairs:
    """Пары одного вина: `top` — блюда «к чему подать», `dishes` — все записанные пары."""

    top: tuple[str, ...]
    dishes: Mapping[str, Pair]

    def pair(self, dish_id: str) -> Pair:
        """Пара с блюдом; нет записи — `neutral`: правила об этой паре молчат."""
        return self.dishes.get(dish_id, NEUTRAL)


@dataclass(frozen=True, slots=True)
class ServeRule:
    """Правило подачи: условие (все заданные ключи сразу), градусы и фраза-основание."""

    id: str
    temperature_c: tuple[int, int]
    text: str
    sparkling: bool | None
    sugar: frozenset[str] | None
    color: frozenset[str] | None
    body_min: float | None
    #: Сахар неизвестен, а название — креплёного, десертного или мускатного вина
    #: (`catalog.sweet_name`): «Портвейн Крымский» без сахара в выгрузке — не сухое белое.
    sweet_name: bool | None = None


@dataclass(frozen=True, slots=True)
class ServeMatch:
    """Подошедшее правило подачи и `by_grape`: сверялось с телом, известным только по сорту."""

    rule: ServeRule
    by_grape: bool

    def public(self) -> dict[str, Any]:
        """Объект `serve` ответа (§1): `{"temperature_c": [16, 18], "source": "rule", …}`."""
        low, high = self.rule.temperature_c
        return {
            "temperature_c": [low, high],
            "source": "rule",
            "rule": self.rule.id,
            "by_grape": self.by_grape,
        }


@dataclass(frozen=True, slots=True)
class Vocab:
    """Словарь замка сущностей (§7.4): что модель не вправе назвать без пакета фактов."""

    wineries: frozenset[str]
    winery_words: frozenset[str]
    grapes: frozenset[str]
    dishes: frozenset[str]
    cooking: frozenset[str]
    regions: frozenset[str]
    descriptors: frozenset[str]
    generic: frozenset[str]
    taste_stems: frozenset[str]


EMPTY_VOCAB = Vocab(*(frozenset() for _ in VOCAB_KEYS))


@dataclass(frozen=True, slots=True)
class Topic:
    """Тема справочника: триггеры и контекст для разбора, ответ, подробность, чипы."""

    id: str
    name: str
    triggers: tuple[str, ...]
    context: tuple[str, ...]
    answer: str
    detail: str
    chips: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class GrapeNote:
    """Портрет сорта: подпись, текст, счёт вин выгрузки и главные регионы."""

    code: str
    label: str
    text: str
    count: int
    regions: tuple[str, ...]


@dataclass(frozen=True)
class SommData:
    """Все данные сомелье; поля пусты, если файла нет (см. `sources`)."""

    pairs: Mapping[str, WinePairs] = field(default_factory=dict)
    dishes: Mapping[str, DishInfo] = field(default_factory=dict)
    rules: Mapping[str, PairRule] = field(default_factory=dict)
    categories: Mapping[str, str] = field(default_factory=dict)
    serve: tuple[ServeRule, ...] = ()
    vocab: Vocab = EMPTY_VOCAB
    topics: Mapping[str, Topic] = field(default_factory=dict)
    grapes: Mapping[str, GrapeNote] = field(default_factory=dict)
    #: Файл → откуда взят: `data`, `fixture` или `missing`.
    sources: Mapping[str, str] = field(default_factory=dict)

    def stats(self) -> dict[str, Any]:
        """Источник и счёт каждого файла — для `/v1/health.somm.data`."""
        counts: dict[str, dict[str, int]] = {
            "pairs.json": {"wines": len(self.pairs)},
            "dishes.json": {"dishes": len(self.dishes), "rules": len(self.rules)},
            "serve.json": {"rules": len(self.serve)},
            "vocab.json": {
                key: len(getattr(self.vocab, key)) for key in ("wineries", "grapes", "dishes")
            },
            "knowledge.json": {"topics": len(self.topics), "grapes": len(self.grapes)},
        }
        return {
            name: {"source": self.sources.get(name, "missing"), **counts[name]}
            for name in SOMM_FILES
        }

    def degraded_reasons(self) -> list[str]:
        """Файлы не из данных: «data:pairs.json=fixture», «data:vocab.json=missing»."""
        out = []
        for name in SOMM_FILES:
            source = self.sources.get(name, "missing")
            if source != "data":
                out.append(f"data:{name}={source}")
        return out

    def wine(self, slug: str) -> WinePairs | None:
        """Пары вина; `None` — вина нет в `pairs.json`."""
        return self.pairs.get(slug)

    def food_dishes(self, food: str) -> tuple[str, ...]:
        """Блюда группы `food` (§7.2) в порядке `dishes.json`."""
        return tuple(dish.id for dish in self.dishes.values() if dish.food == food)

    def match_serve(
        self,
        *,
        color: str | None,
        sugar: str | None,
        sparkling: bool,
        body: float | None,
        body_source: str | None,
        sweet_name: bool = False,
    ) -> ServeMatch | None:
        """Подача вина по таблице `serve.json` — см. `match_serve`."""
        return match_serve(
            self.serve,
            color=color,
            sugar=sugar,
            sparkling=sparkling,
            body=body,
            body_source=body_source,
            sweet_name=sweet_name,
        )


# ------------------------------------------------------------------ подача
def match_serve(
    rules: Sequence[ServeRule],
    *,
    color: str | None,
    sugar: str | None,
    sparkling: bool,
    body: float | None,
    body_source: str | None,
    sweet_name: bool = False,
) -> ServeMatch | None:
    """Первое подошедшее правило подачи (§7.3) — одна функция для сборки и сервиса.

    `color` — цвет выгрузки («Красное»), `sugar` — код `SugarClass` или `None`, `body` — ось
    `body` профиля (0–5), `body_source` — её источник. Тело без источника («типичное для
    цвета») условию `body_min` не отвечает: неизвестное тело не делает вино плотным.
    `sweet_name` — сахар неизвестен, а название креплёного, десертного или мускатного вина
    (`catalog.sweet_name`). `by_grape` — по пути к правилу сверялось `body_min`, а тело
    известно по сорту.
    """
    known_body = body if body_source is not None else None
    consulted = False
    for rule in rules:
        if rule.sparkling is not None and sparkling != rule.sparkling:
            continue
        if rule.sweet_name is not None and sweet_name != rule.sweet_name:
            continue
        if rule.sugar is not None and sugar not in rule.sugar:
            continue
        if rule.color is not None and color not in rule.color:
            continue
        if rule.body_min is not None:
            consulted = True
            if known_body is None or known_body < rule.body_min:
                continue
        return ServeMatch(rule=rule, by_grape=consulted and body_source == "grape")
    return None


# ------------------------------------------------------------------ разбор файлов
class _Shape(ValueError):
    """Файл не по форме §7."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise _Shape(message)


def _obj(value: Any, where: str) -> Mapping[str, Any]:
    _need(isinstance(value, Mapping), f"{where}: ожидался объект")
    return value


def _str(value: Any, where: str, *, empty: bool = False) -> str:
    _need(isinstance(value, str) and (empty or bool(value)), f"{where}: ожидалась строка")
    return value


def _strs(value: Any, where: str) -> tuple[str, ...]:
    _need(isinstance(value, list), f"{where}: ожидался список строк")
    return tuple(_str(item, f"{where}[]") for item in value)


def _keys(value: Mapping[str, Any], expected: set[str], where: str) -> None:
    _need(set(value) == expected, f"{where}: ключи {sorted(value)} вместо {sorted(expected)}")


def _header(raw: Any, name: str, keys: set[str]) -> Mapping[str, Any]:
    data = _obj(raw, name)
    _keys(data, keys, name)
    _need(data["version"] == 1, f"{name}: version {data['version']!r}, ожидалась 1")
    return data


def _parse_dishes(
    raw: Any,
) -> tuple[dict[str, DishInfo], dict[str, PairRule], dict[str, str]]:
    data = _header(raw, "dishes.json", {"version", "source", "categories", "dishes", "rules"})
    categories = {
        _str(code, "categories"): _str(label, f"categories.{code}")
        for code, label in _obj(data["categories"], "categories").items()
    }
    dishes: dict[str, DishInfo] = {}
    _need(isinstance(data["dishes"], list), "dishes: ожидался список")
    for entry in data["dishes"]:
        item = _obj(entry, "dishes[]")
        _keys(item, {"id", "name", "dative", "category", "food", "family", "aliases"}, "dish")
        dish_id = _str(item["id"], "dish.id")
        _need(dish_id not in dishes, f"dish {dish_id}: повтор")
        dative = _str(item["dative"], f"{dish_id}.dative")
        _need(dative.startswith("к "), f"{dish_id}.dative: не «к …»")
        category = _str(item["category"], f"{dish_id}.category")
        _need(category in categories, f"{dish_id}.category {category!r}: нет в categories")
        food = item["food"]
        _need(food is None or food in FOOD_GROUPS, f"{dish_id}.food {food!r}")
        aliases = _strs(item["aliases"], f"{dish_id}.aliases")
        _need(bool(aliases), f"{dish_id}.aliases: пусто")
        dishes[dish_id] = DishInfo(
            id=dish_id,
            name=_str(item["name"], f"{dish_id}.name"),
            dative=dative,
            category=category,
            food=food,
            family=_str(item["family"], f"{dish_id}.family"),
            aliases=aliases,
        )
    for dish in dishes.values():
        _need(dish.family in dishes, f"{dish.id}.family {dish.family!r}: нет такого блюда")

    rules: dict[str, PairRule] = {}
    _need(isinstance(data["rules"], list), "rules: ожидался список")
    for entry in data["rules"]:
        item = _obj(entry, "rules[]")
        _keys(item, {"id", "sign", "weight", "name", "chip", "text", "wine_fields"}, "rule")
        rule_id = _str(item["id"], "rule.id")
        _need(rule_id not in rules, f"rule {rule_id}: повтор")
        weight = item["weight"]
        _need(
            isinstance(weight, int | float) and not isinstance(weight, bool) and weight != 0,
            f"{rule_id}.weight",
        )
        sign = _str(item["sign"], f"{rule_id}.sign")
        _need(sign == ("+" if weight > 0 else SIGNS[1]), f"{rule_id}.sign {sign!r} к весу {weight}")
        chip = _str(item["chip"], f"{rule_id}.chip")
        _need(len(chip) <= 30, f"{rule_id}.chip длиннее 30 знаков")
        fields = _strs(item["wine_fields"], f"{rule_id}.wine_fields")
        _need(bool(fields) and set(fields) <= WINE_FIELDS, f"{rule_id}.wine_fields {fields}")
        rules[rule_id] = PairRule(
            id=rule_id,
            sign=sign,
            weight=float(weight),
            name=_str(item["name"], f"{rule_id}.name"),
            chip=chip,
            text=_str(item["text"], f"{rule_id}.text"),
            wine_fields=fields,
        )
    return dishes, rules, categories


def _new_pair(value: Any, where: str) -> Pair:
    """Пара из `[вердикт, [«+»…], [«−»…]]` с проверкой формы §7.1."""
    _need(
        isinstance(value, list) and len(value) == 3,
        f"{where}: ожидалось [вердикт, «+», «−»]",
    )
    verdict, plus, minus = value
    _need(verdict in VERDICTS, f"{where}: вердикт {verdict!r}")
    plus, minus = _strs(plus, where), _strs(minus, where)
    _need(
        (verdict == "neutral") == (not plus and not minus),
        f"{where}: neutral тогда и только тогда, когда правил нет",
    )
    _need(verdict != "yes" or not minus, f"{where}: yes с правилом «−»")
    return Pair(verdict, tuple(sys.intern(rule) for rule in plus), tuple(map(sys.intern, minus)))


def _parse_pairs(raw: Any) -> dict[str, WinePairs]:
    """Пары всех вин. 172 тысячи записей повторяют несколько тысяч различных пар, поэтому
    каждая различная пара проверяется и создаётся один раз, а дальше берётся из словаря —
    так файл читается за доли секунды и занимает единицы мегабайт."""
    data = _header(raw, "pairs.json", {"version", "built", "wines"})
    _str(data["built"], "built")
    seen: dict[tuple[Any, ...], Pair] = {}
    out: dict[str, WinePairs] = {}
    for slug, entry in _obj(data["wines"], "wines").items():
        item = _obj(entry, slug)
        _keys(item, {"top", "dishes"}, slug)
        top = tuple(sys.intern(_str(dish, f"{slug}.top[]")) for dish in item["top"])
        _need(len(top) <= 5 and len(set(top)) == len(top), f"{slug}.top: {list(top)}")
        dishes: dict[str, Pair] = {}
        for dish_id, value in _obj(item["dishes"], f"{slug}.dishes").items():
            try:
                verdict, plus, minus = value
                key = (verdict, type(plus), type(minus), *plus, None, *minus)
                pair = seen.get(key)
            except (TypeError, ValueError):
                pair, key = None, None
            if pair is None:
                pair = _new_pair(value, f"{slug}.{dish_id}")
                if key is not None:
                    seen[key] = pair
            dishes[sys.intern(dish_id)] = pair
        for dish_id in top:
            verdict = dishes.get(dish_id, NEUTRAL).verdict
            _need(verdict in ("yes", "caveat"), f"{slug}.top: {dish_id} с вердиктом {verdict}")
        out[slug] = WinePairs(top=top, dishes=dishes)
    return out


def _parse_serve(raw: Any) -> tuple[ServeRule, ...]:
    data = _header(raw, "serve.json", {"version", "source", "rules"})
    _need(isinstance(data["rules"], list) and bool(data["rules"]), "serve.rules: пусто")
    rules: list[ServeRule] = []
    for entry in data["rules"]:
        item = _obj(entry, "serve.rules[]")
        _keys(item, {"id", "when", "temperature_c", "text"}, "serve rule")
        rule_id = _str(item["id"], "serve.id")
        when = _obj(item["when"], f"{rule_id}.when")
        _need(
            bool(when) and set(when) <= {"sparkling", "sugar", "color", "body_min", "sweet_name"},
            f"{rule_id}.when {sorted(when)}",
        )
        temperature = item["temperature_c"]
        _need(
            isinstance(temperature, list)
            and len(temperature) == 2
            and all(isinstance(t, int) and not isinstance(t, bool) for t in temperature)
            and 0 < temperature[0] < temperature[1] <= 25,
            f"{rule_id}.temperature_c {temperature!r}",
        )
        sparkling = when.get("sparkling")
        _need(sparkling is None or isinstance(sparkling, bool), f"{rule_id}.when.sparkling")
        sweet = when.get("sweet_name")
        _need(sweet is None or isinstance(sweet, bool), f"{rule_id}.when.sweet_name")
        body_min = when.get("body_min")
        _need(
            body_min is None or (isinstance(body_min, int | float) and 0 <= body_min <= 5),
            f"{rule_id}.when.body_min",
        )
        rules.append(
            ServeRule(
                id=rule_id,
                temperature_c=(temperature[0], temperature[1]),
                text=_str(item["text"], f"{rule_id}.text"),
                sparkling=sparkling,
                sugar=frozenset(_strs(when["sugar"], "sugar")) if "sugar" in when else None,
                color=frozenset(_strs(when["color"], "color")) if "color" in when else None,
                body_min=float(body_min) if body_min is not None else None,
                sweet_name=sweet,
            )
        )
    _need(len({rule.id for rule in rules}) == len(rules), "serve.rules: повтор id")
    return tuple(rules)


def _parse_vocab(raw: Any) -> Vocab:
    data = _header(raw, "vocab.json", {"version", *VOCAB_KEYS})
    words = {key: frozenset(_strs(data[key], f"vocab.{key}")) for key in VOCAB_KEYS}
    for key, values in words.items():
        _need(bool(values), f"vocab.{key}: пусто")
    _need(
        all(word == word.lower() for word in words["winery_words"]),
        "vocab.winery_words: только строчные",
    )
    return Vocab(**words)


def _parse_knowledge(raw: Any) -> tuple[dict[str, Topic], dict[str, GrapeNote]]:
    data = _header(raw, "knowledge.json", {"version", "source", "topics", "grapes"})
    topics: dict[str, Topic] = {}
    _need(isinstance(data["topics"], list), "topics: ожидался список")
    for entry in data["topics"]:
        item = _obj(entry, "topics[]")
        _keys(item, {"id", "name", "triggers", "context", "answer", "detail", "chips"}, "topic")
        topic_id = _str(item["id"], "topic.id")
        _need(topic_id not in topics, f"topic {topic_id}: повтор")
        _need(isinstance(item["chips"], list), f"{topic_id}.chips")
        chips = []
        for chip in item["chips"]:
            chip_item = _obj(chip, f"{topic_id}.chips[]")
            _need(set(chip_item) <= {"id", "text", "args"}, f"{topic_id}.chips[]: ключи")
            _str(chip_item.get("id"), f"{topic_id}.chip.id")
            _str(chip_item.get("text"), f"{topic_id}.chip.text")
            chips.append(MappingProxyType(dict(chip_item)))
        topics[topic_id] = Topic(
            id=topic_id,
            name=_str(item["name"], f"{topic_id}.name"),
            triggers=_strs(item["triggers"], f"{topic_id}.triggers"),
            context=_strs(item["context"], f"{topic_id}.context"),
            answer=_str(item["answer"], f"{topic_id}.answer"),
            detail=_str(item["detail"], f"{topic_id}.detail", empty=True),
            chips=tuple(chips),
        )
    grapes: dict[str, GrapeNote] = {}
    for code, entry in _obj(data["grapes"], "grapes").items():
        item = _obj(entry, f"grapes.{code}")
        _keys(item, {"label", "text", "count", "regions"}, f"grapes.{code}")
        count = item["count"]
        _need(isinstance(count, int) and not isinstance(count, bool) and count > 0, f"{code}.count")
        regions = _strs(item["regions"], f"{code}.regions")
        _need(bool(regions), f"{code}.regions: пусто")
        grapes[code] = GrapeNote(
            code=code,
            label=_str(item["label"], f"{code}.label"),
            text=_str(item["text"], f"{code}.text"),
            count=count,
            regions=regions,
        )
    return topics, grapes


def _check_links(
    pairs: Mapping[str, WinePairs],
    dishes: Mapping[str, DishInfo],
    rules: Mapping[str, PairRule],
    strict: bool,
) -> dict[str, WinePairs]:
    """Пары ссылаются только на блюда и правила `dishes.json`, знак правила — по своей стороне.

    В строгом режиме чужая ссылка — `ValueError`. Иначе (файлы из разных источников, например
    пары из данных, а блюда из заглушки) пары с чужими блюдами и правилами отбрасываются с одним
    предупреждением: сервис не должен показать блюдо без названия.
    """
    if not pairs or not dishes or not rules:
        return dict(pairs)
    plus_ok = {rule_id for rule_id, rule in rules.items() if rule.sign == "+"}
    minus_ok = {rule_id for rule_id, rule in rules.items() if rule.sign != "+"}
    dropped = 0
    out: dict[str, WinePairs] = {}
    for slug, entry in pairs.items():
        kept: dict[str, Pair] = {}
        for dish_id, pair in entry.dishes.items():
            ok = dish_id in dishes and set(pair.plus) <= plus_ok and set(pair.minus) <= minus_ok
            if ok:
                kept[dish_id] = pair
                continue
            if strict:
                raise ValueError(f"pairs.json: {slug}.{dish_id} ссылается не на dishes.json")
            dropped += 1
        top = tuple(dish_id for dish_id in entry.top if dish_id in kept)
        if strict and len(top) != len(entry.top):
            raise ValueError(f"pairs.json: {slug}.top ссылается не на dishes.json")
        out[slug] = WinePairs(top=top, dishes=kept)
    if dropped:
        logger.warning("Сомелье: %d пар ссылаются не на dishes.json и отброшены", dropped)
    return out


# ------------------------------------------------------------------ загрузка
def _read(
    name: str, directory: Path | None, fallback: Path | None, strict: bool
) -> tuple[Any, str]:
    """Сырой JSON файла и его источник: `data`, `fixture` или `missing`."""
    for base, source in ((directory, "data"), (fallback, "fixture")):
        if base is None:
            continue
        path = base / name
        if not path.is_file():
            continue
        try:
            return json.loads(path.read_text(encoding="utf-8")), source
        except (OSError, ValueError) as exc:
            if strict:
                raise ValueError(f"{path}: {exc}") from exc
            logger.warning("Сомелье: %s не прочитан (%s) — файла нет", path, exc)
            return None, "missing"
    logger.warning("Сомелье: %s нет ни в данных, ни в заглушках", name)
    return None, "missing"


def load_somm_data(
    directory: Path | None, *, fallback: Path | None = FIXTURE_DIR, strict: bool = False
) -> SommData:
    """Данные сомелье из `directory` с запасом `fallback`; см. docstring модуля."""
    raw: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for name in SOMM_FILES:
        raw[name], sources[name] = _read(name, directory, fallback, strict)

    parsers = {
        "pairs.json": _parse_pairs,
        "dishes.json": _parse_dishes,
        "serve.json": _parse_serve,
        "vocab.json": _parse_vocab,
        "knowledge.json": _parse_knowledge,
    }
    parsed: dict[str, Any] = {}
    for name, parse in parsers.items():
        if raw[name] is None:
            continue
        try:
            parsed[name] = parse(raw[name])
        # Форма не по §7: `_Shape` из проверок, остальное — от формы, которую проверки не ждали.
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            if strict:
                raise ValueError(f"{name}: {exc!r}") from exc
            logger.warning("Сомелье: %s не по договору (%s) — файла нет", name, exc)
            sources[name] = "missing"
        raw[name] = None  # сырой JSON пар — десятки мегабайт, держать его незачем

    dishes, rules, categories = parsed.get("dishes.json", ({}, {}, {}))
    topics, grapes = parsed.get("knowledge.json", ({}, {}))
    pairs = _check_links(parsed.get("pairs.json", {}), dishes, rules, strict)
    stand_ins = [name for name, source in sources.items() if source == "fixture"]
    if stand_ins and directory is not None:
        logger.warning(
            "Сомелье: в %s нет %s — взяты заглушки %s (семь вин): соберите data/somm "
            "(scripts/build_somm.py) или задайте SVS_SOMM_DIR",
            directory,
            ", ".join(stand_ins),
            fallback,
        )
    return SommData(
        pairs=pairs,
        dishes=dishes,
        rules=rules,
        categories=categories,
        serve=parsed.get("serve.json", ()),
        vocab=parsed.get("vocab.json", EMPTY_VOCAB),
        topics=topics,
        grapes=grapes,
        sources=MappingProxyType(sources),
    )
