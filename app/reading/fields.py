"""Поля этикетки из токенов чтений и попаданий словаря каталога.

OCR не выбирает вино: здесь только признаки с доказательствами, решение — за
resolve. Ловушки описаны общими правилами, а не строками конкретных кадров:
- год — с якорем («урожай», «vintage») или отдельным токеном 1990..(год+1), но не
  рядом со словами основания, розлива и медалей, не внутри даты и не в номере
  стандарта («ГОСТ 32030-2013»);
- крепость — только число с «%», «об.» или «vol» и не доля сорта в купаже на той же строке;
- цвет — только словом и не из более длинного названия сорта, серии, винодельни или кюве
  («Rosso Antico», «Мускат белый»); название из одних слов таксономии («Белое сухое») цвет
  не отнимает. Цвет и сахар — в любом роде и числе («Мускатель белый», «Херес сухой»), но
  форма не среднего рода перед словом имени — начало имени, а не признак вина («Красная
  Горка», «КРАСНАЯ» / «СТРЕЛКА»; `_starts_proper_name`). Строка урожая, объёма или крепости
  после цвета именем не бывает («БЕЛЫЙ» / «ГОД УРОЖАЯ» — белое, Э7);
- сахар и цвет из словаря каталога — только опечатка OCR слова из списков таксономии и
  только там, где таксономия не заняла слова более длинной фразой («Extra Dry», «Brut
  Nature», «Traminer Rose»). Скелет короткого слова совпадает случайно: «Свет» — не «sweet».
  Опечатка формы не среднего рода проходит ту же проверку имени, что и само слово;
- кюве — не из слов, которые сорт, сахар, цвет или серия объясняют не хуже: более длинной
  фразой или попаданием не дороже («Пино Нуар» — сорт, а не кюве «нуар»; «Сира» — сорт, а не
  кюве «сира»; «красное» — цвет, а не нечёткое кюве «красные»). Точное кюве против нечёткой
  серии («Классика» и «классик» за 1,0) остаётся: выбор за resolve. Кюве, написанное на
  этикетке так же, как в каталоге, остаётся рядом с цветом и сахаром: каталог называет этим
  словом саму позицию («Chateau de Talu Блан», «Два Сердца Мускат Сухой»). Нечёткое кюве и
  транслит («Red» → кюве «Ред») цвет и сахар отнимают (`ADJECTIVE_CLAIM_COST`). Не берутся
  и общие слова этикетки (`LABEL_GENERIC`: «выдержка», «месяцев», «столовое»). Кюве со своим
  словом рядом с занятым («Кюве Александр») остаётся;
- фразы таксономии не склеиваются через разрыв строк на кадре (`token_segments`);
- одно слово — одно поле (Э6): сорт внутри более длинного сорта словаря сортом не считается
  («Каберне Совиньон» — без «совиньон»), винодельня из слов, занятых другим полем фразой
  длиннее («пино» из «Пино Нуар»), — тоже; слова категории и хозяйства («шато», «виноград»)
  и регионы латиницей (`crimea`) — не кюве.

Одиночное поле (год, крепость, цвет) с двумя равносильными значениями внутри одного
чтения остаётся пустым: пустое поле — «не прочитано», а неверное уводит resolve к
соседнему урожаю. Если равносильные значения дали разные чтения, берётся значение чтения,
стоящего раньше: порядок читателей задаёт вызывающий (VLM первым, классика — второй голос).
Альтернатива при этом теряется: контракт держит одно значение (запрос на списки в контракте).
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from app.reading.contracts import (
    Color,
    Evidence,
    LabelFields,
    LexField,
    LexHit,
    Reading,
    SugarClass,
    TokenSpan,
)
from app.reading.lexicon.correct import SHORT_WORD, cost_budget, weighted_distance
from app.reading.taxonomy import (
    AGREEING_FORMS,
    BLEND_WORDS,
    COLOR_TERMS,
    LABEL_GENERIC,
    LABEL_TERMS,
    MASCULINE_FORMS,
    NAME_WINE_WORDS,
    READ_TERMS,
    REGION_WORDS,
    REGION_WORDS_EXTRA,
    STYLE_WORDS,
    SUGAR_TERMS,
    VINTAGE_LINE_WORDS,
    VOLUME_LINE_WORDS,
    WINE_WORDS,
    Term,
    TermKind,
    color_of,
    phrase_words,
    sugar_class,
)
from app.reading.text.layout import token_segments
from app.reading.text.normalize import norm, norm_token
from app.reading.text.tokenize import ABV_UNITS

#: Урожай без якоря: раньше — скорее год основания, чем урожай.
BARE_VINTAGE_MIN = 1990
#: Урожай рядом со словом «урожай»: коллекционные вина бывают старше.
ANCHORED_VINTAGE_MIN = 1950
#: Правдоподобная крепость вина на этикетке.
ABV_RANGE = (4.0, 22.0)

# Сколько токенов от года смотреть в поисках якоря.
_ANCHOR_BEFORE = 3
_ANCHOR_AFTER = 2
_NUMERIC_KINDS = frozenset({"number", "year", "abv", "ratio"})


def _words(text: str) -> frozenset[str]:
    return frozenset(norm_token(word) for word in text.split())


_VINTAGE_ANCHORS = _words(
    "урожай урожая урожаи vintage millesime millesimato vendemmia cosecha harvest annata jahrgang"
)
# Основание, розлив, сроки, стандарты и награды: год рядом с ними — не урожай.
_NOT_VINTAGE_ANCHORS = _words(
    """
    основан основана основано основания основатель since est established founded fondee
    fonde fondata depuis seit desde gegrundet розлив розлива разлив разлито bottled bottling
    embouteille imbottigliato дата изготовления изготовлено производства годен годности гост ту
    лицензия license партия lot batch медаль medal award конкурс competition copyright
    """
)
# «с 1995 года», «dal 1907»: предлог — якорь, только если стоит прямо перед годом.
_SINCE_PREPOSITIONS = _words("с c dal")

_ABV_ANCHORS_BEFORE = _words("alc alcohol алк алкоголь крепость спирта спирт этилового abv")
_ABV_ANCHORS_AFTER = _words("vol об alc алк")

_MONTHS = (
    "января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря|"
    "январь|февраль|март|апрель|май|июнь|июль|август|сентябрь|октябрь|ноябрь|декабрь|"
    "янв|фев|мар|апр|июн|июл|авг|сен|сент|окт|ноя|дек|"
    "january|february|march|april|may|june|july|august|september|october|november|december|"
    "jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
# Что может стоять в строке прямо перед годом, если это дата или номер документа.
_DATE_BEFORE_RE = re.compile(
    r"(?:"
    r"(?<!\d)\d{1,2}\s*[./\-]\s*\d{1,2}\s*[./\-]\s*"  # 12.03.2024, 12 / 03 / 2024
    r"|(?<!\d)\d{2}\s+\d{2}\s+"  # 12 03 2024
    r"|(?<!\d)\d{1,2}\s*[./\-\s]\s*[ivx]{1,4}\s*[./\-\s]\s*"  # 12.IX.2021
    rf"|(?<![^\W\d_])(?:{_MONTHS})\.?\s*"  # 15 марта 2024, March 2024
    r"|\d\s*[-‐‑–—]\s*"  # ГОСТ 32030-2013
    r"|(?:©|\([cс]\))\s*"  # © 2021
    r")$",
    re.IGNORECASE,
)
# И прямо после: 2024-03-12, 2013-001.
_DATE_AFTER_RE = re.compile(r"^[./\-‐‑–—]\d")


@dataclass(frozen=True, slots=True)
class _Found:
    term: Term
    phrase: str
    ids: tuple[int, ...]


@dataclass(slots=True)
class _Context:
    tokens: list[TokenSpan]
    order: dict[str, int]  # ключ чтения → место в `readings`
    sequences: list[list[int]]  # индексы токенов каждого чтения по порядку
    position: dict[int, tuple[int, int]]  # индекс токена → (последовательность, место)
    raw: dict[int, tuple[str, int, int]]  # индекс токена → (текст строки, начало, конец)
    segments: list[int]  # отрезок текста каждого токена: фраза не выходит за него

    @classmethod
    def build(cls, tokens: list[TokenSpan], readings: Sequence[Reading]) -> _Context:
        order: dict[str, int] = {}
        lines: dict[tuple[str, int], str] = {}
        for reading in readings:
            key = reading.key
            order.setdefault(key, len(order))
            for line in reading.lines:
                lines[(key, line.id)] = unicodedata.normalize("NFKC", line.text)

        groups: dict[str, list[int]] = {}
        for i, token in enumerate(tokens):
            groups.setdefault(token.reading, []).append(i)
        sequences = list(groups.values())
        position = {i: (s, p) for s, seq in enumerate(sequences) for p, i in enumerate(seq)}

        # Токены строки идут по порядку, и их `text` — подстрока строки после NFKC.
        raw: dict[int, tuple[str, int, int]] = {}
        cursor: dict[tuple[str, int], int] = {}
        for i, token in enumerate(tokens):
            line_key = (token.reading, token.line_id)
            text = lines.get(line_key)
            if text is None:
                continue
            start = text.find(token.text, cursor.get(line_key, 0))
            if start < 0:
                continue
            cursor[line_key] = start + len(token.text)
            raw[i] = (text, start, start + len(token.text))
        return cls(tokens, order, sequences, position, raw, token_segments(tokens, readings))

    def neighbour(self, i: int, offset: int) -> int | None:
        seq_index, place = self.position[i]
        seq = self.sequences[seq_index]
        target = place + offset
        return seq[target] if 0 <= target < len(seq) else None

    def terms(self) -> list[_Found]:
        """Фразы таксономии (сорта, сахар, цвет, серии) по отрезкам текста, через строки."""
        found: list[_Found] = []
        pieces: dict[int, list[int]] = {}
        for i in range(len(self.tokens)):
            pieces.setdefault(self.segments[i], []).append(i)
        for seq in pieces.values():
            tokens = [self.tokens[i] for i in seq]
            norms = [t.norm for t in tokens]
            skeletons = [t.skeleton if t.kind == "word" else "" for t in tokens]
            for match in READ_TERMS.find(norms, skeletons):
                ids = tuple(seq[match.start : match.end])
                found.append(_Found(match.value, match.phrase, ids))
        return found


@dataclass(slots=True)
class _Vote[T]:
    value: T
    first: int
    readings: dict[str, None] = field(default_factory=dict)  # упорядоченное множество
    confs: list[float] = field(default_factory=list)
    matched: dict[str, int] = field(default_factory=dict)  # написание → число голосов
    strong: bool = False
    cost: float | None = None
    ids: set[int] = field(default_factory=set)  # токены, давшие значение
    # Окна, давшие значение, с ценой доказательства: чем слово занято для кюве.
    spans: list[tuple[tuple[int, ...], float]] = field(default_factory=list)

    @property
    def support(self) -> int:
        return len(self.readings)


class _Ballot[T]:
    """Голоса за значения поля: одно значение из нескольких чтений — один голос."""

    def __init__(self, ctx: _Context) -> None:
        self._ctx = ctx
        self._votes: dict[Hashable, _Vote[T]] = {}

    def add(
        self,
        key: Hashable,
        value: T,
        ids: Sequence[int],
        *,
        matched: str | None = None,
        strong: bool = False,
        cost: float | None = None,
        claim_cost: float = 0.0,
    ) -> None:
        """`cost` участвует в ранжировании полей словаря; `claim_cost` — только в споре за слово
        с кюве (текст и таксономия — 0, попадание словаря — его цена)."""
        vote = self._votes.get(key)
        if vote is None:
            vote = self._votes[key] = _Vote(value=value, first=min(ids))
        vote.first = min(vote.first, *ids)
        vote.ids.update(ids)
        vote.spans.append((tuple(ids), claim_cost))
        for i in ids:
            token = self._ctx.tokens[i]
            vote.readings.setdefault(token.reading, None)
            if token.conf is not None:
                vote.confs.append(token.conf)
        if matched is not None:
            vote.matched[matched] = vote.matched.get(matched, 0) + 1
        vote.strong |= strong
        if cost is not None:
            vote.cost = cost if vote.cost is None else min(vote.cost, cost)

    def ranked(self) -> list[_Vote[T]]:
        return sorted(
            self._votes.values(),
            key=lambda v: (-v.support, v.cost if v.cost is not None else 0.0, v.first),
        )

    def single(self) -> _Vote[T] | None:
        """Лучшее значение.

        Равносильные значения внутри одного чтения — отказ. Равносильные значения из разных
        чтений — значение чтения, стоящего раньше: слабый второй читатель не стирает первого.
        """
        votes = sorted(self._votes.values(), key=lambda v: (not v.strong, -v.support, v.first))
        if not votes:
            return None
        tied = [v for v in votes if (v.strong, v.support) == (votes[0].strong, votes[0].support)]
        if len(tied) == 1:
            return tied[0]
        seen: set[str] = set()
        for vote in tied:
            if not seen.isdisjoint(vote.readings):
                return None
            seen.update(vote.readings)
        rank = self._ctx.order
        return min(tied, key=lambda v: min(rank.get(key, len(rank)) for key in v.readings))


def _evidence[T](
    model: type[Evidence[T]], vote: _Vote[T], ctx: _Context, *, lexical: bool = False
) -> Evidence[T]:
    """Для полей словаря `conf` — близость попадания 1/(1+cost), иначе — лучшая уверенность строк."""
    sources = sorted(vote.readings, key=lambda key: ctx.order.get(key, len(ctx.order)))
    if lexical and vote.cost is not None:
        conf: float | None = 1.0 / (1.0 + vote.cost)
    else:
        conf = max(vote.confs) if vote.confs else None
    matched = max(vote.matched, key=vote.matched.__getitem__) if vote.matched else None
    return model(
        value=vote.value, sources=sources, support=len(sources), conf=conf, matched=matched
    )


def _lexicon_field(
    ctx: _Context, hits: Sequence[tuple[list[int], LexHit]], lex_field: LexField
) -> list[Evidence[str]]:
    ballot: _Ballot[str] = _Ballot(ctx)
    for ids, hit in hits:
        if hit.field == lex_field:
            ballot.add(hit.canonical, hit.canonical, ids, matched=hit.canonical, cost=hit.cost)
    return [_evidence(Evidence[str], vote, ctx, lexical=True) for vote in ballot.ranked()]


#: Слово занято другим полем: токен → окна (число токенов, цена доказательства).
Claims = Mapping[int, Sequence[tuple[int, float]]]


#: Цена, с которой слово цвета или сахара занято для кюве. Дороже точного попадания словаря
#: (0) и дешевле любого неточного (скелет — 0,2, путаница OCR — от 0,3). Кюве, написанное
#: на этикетке так же, как в каталоге («Блан», «Руж», «Сухой», «Десертный»), остаётся рядом с
#: цветом и сахаром: каталог называет этим словом саму позицию, и без кюве её не отличить от
#: соседей по линейке («Chateau de Talu Блан» и «… Руж»). Нечёткое кюве («красные» на слове
#: «красное») и транслит («Red» → «Ред», «Rosso» → «Россо»: на этикетке это чаще цвет, чем
#: кюве «Шато Тамань Резерв Ред Бленд») слово теряют (`_cuvee_cost`).
ADJECTIVE_CLAIM_COST = 0.1


def _taken(i: int, ids: Sequence[int], cost: float, claims: Claims) -> bool:
    """Токен `i` кюве объяснён другим полем не хуже: фразой длиннее кюве или не дороже его."""
    return any(size > len(ids) or other <= cost + 1e-9 for size, other in claims.get(i, ()))


def _cuvee_cost(ctx: _Context, ids: Sequence[int], hit: LexHit) -> float:
    """Цена кюве в споре за слово: точное попадание не в написании каталога — как цвет.

    Точное попадание (0) по норме, которая не совпадает с названием кюве в каталоге, —
    транслит («red» у кюве «ред»): против цвета и сахара (`ADJECTIVE_CLAIM_COST`) оно не
    сильнее их. Против сорта и серии ничего не меняется: их цена 0 или не меньше 0,2.
    """
    if hit.cost > 0.0:
        return hit.cost
    written = " ".join(ctx.tokens[i].norm for i in ids)
    literal = written == " ".join(phrase_words(hit.canonical))
    return 0.0 if literal else ADJECTIVE_CLAIM_COST


#: Слова, которые сами по себе не кюве: общие слова этикетки, слова региона (и латиницей) и
#: слова винной этикетки кроме слов стиля (Э6-Р, Ш).
_NOT_CUVEE = LABEL_GENERIC | REGION_WORDS | REGION_WORDS_EXTRA | NAME_WINE_WORDS

#: Э6-Г: поля словаря, где попадание строго внутри более длинного попадания того же поля в поле
#: не идёт: «Каберне Совиньон» — `cabernet_sauvignon`, без «каберне» и «совиньон» (а
#: «совиньон» resolve читает как Совиньон Блан); «Muscat Ottonel» — без `muscat`.
NESTED_LEX_FIELDS: frozenset[LexField] = frozenset({"grape"})
#: Э6-В: поля словаря, где попадание в поле не идёт, если каждое его слово занято другим полем
#: фразой длиннее самого попадания: «пино» из «Пино Нуар» — сорт, а не винодельня «Шато Пино»,
#: «Grand» из «Grand Reserve» — серия. Сравнивается только длина, цена — нет.
LONGER_CLAIM_FIELDS: frozenset[LexField] = frozenset({"producer"})


def _cuvee(
    ctx: _Context, hits: Sequence[tuple[list[int], LexHit]], claims: Claims
) -> list[Evidence[str]]:
    """Кюве из словаря без слов, занятых сортом, сахаром, цветом или серией, и без общих слов.

    Попадание отбрасывается, если каждый его токен занят (`_taken`), общий (`LABEL_GENERIC`)
    или слово региона (`REGION_WORDS`: «КРЫМ» — строка происхождения), либо если сама форма
    словаря состоит из таких слов: опечатка «выдерска» → «выдержка» — тоже не кюве. Занятость
    зависит от силы доказательства: сорт «Пино Нуар» длиннее кюве «нуар», точный сорт «Сира»
    не дороже кюве «сира» — слово занято; нечёткая серия «классик» за 1,0 точное кюве
    «классика» не отнимает; цвет и сахар отнимают только кюве не в написании каталога
    (`ADJECTIVE_CLAIM_COST`). Как у цвета внутри названия: кюве со своим словом («Кюве
    Александр» при серии «кюве») остаётся.
    """
    kept = []
    for ids, hit in hits:
        if hit.field != "cuvee":
            continue
        cost = _cuvee_cost(ctx, ids, hit)
        if all(_taken(i, ids, cost, claims) or ctx.tokens[i].norm in _NOT_CUVEE for i in ids):
            continue
        words = set(phrase_words(hit.canonical))
        if words and words <= _NOT_CUVEE:
            continue
        kept.append((ids, hit))
    return _lexicon_field(ctx, kept, "cuvee")


def _outermost(
    hits: Sequence[tuple[list[int], LexHit]], lex_field: LexField
) -> list[tuple[list[int], LexHit]]:
    """Попадания без тех, чьи токены — строгое подмножество другого попадания поля `lex_field`."""
    spans = [frozenset(ids) for ids, hit in hits if hit.field == lex_field]
    return [
        (ids, hit)
        for ids, hit in hits
        if hit.field != lex_field or not any(frozenset(ids) < span for span in spans)
    ]


def _claimed_by_longer(ids: Sequence[int], claims: Claims) -> bool:
    """Каждый токен окна занят другим полем фразой длиннее самого окна."""
    return all(any(size > len(ids) for size, _ in claims.get(i, ())) for i in ids)


def _lexical(
    ctx: _Context, hits: Sequence[tuple[list[int], LexHit]], lex_field: LexField, claims: Claims
) -> list[Evidence[str]]:
    """Поле словаря (винодельня, сорт) с правилами Э6-Г и Э6-В для полей из их таблиц."""
    selected = list(hits)
    if lex_field in NESTED_LEX_FIELDS:
        selected = _outermost(selected, lex_field)
    if lex_field in LONGER_CLAIM_FIELDS:
        selected = [
            (ids, hit)
            for ids, hit in selected
            if hit.field != lex_field or not _claimed_by_longer(ids, claims)
        ]
    return _lexicon_field(ctx, selected, lex_field)


def _vintage_anchor(ctx: _Context, i: int) -> str | None:
    """«vintage», «not» или None — по ближайшему якорю; при равенстве побеждает «not»."""
    before = ctx.neighbour(i, -1)
    if before is not None and ctx.tokens[before].norm in _SINCE_PREPOSITIONS:
        return "not"
    nearest: list[tuple[int, bool, str]] = []
    for direction, limit in ((-1, _ANCHOR_BEFORE), (1, _ANCHOR_AFTER)):
        for distance in range(1, limit + 1):
            j = ctx.neighbour(i, direction * distance)
            if j is None:
                break
            token = ctx.tokens[j]
            if token.kind in _NUMERIC_KINDS:
                break  # за другим числом якорь уже не наш: «EST. 1907 / 2020»
            if token.norm in _NOT_VINTAGE_ANCHORS:
                nearest.append((distance, False, "not"))
                break
            if token.norm in _VINTAGE_ANCHORS:
                nearest.append((distance, True, "vintage"))
                break
    return min(nearest)[2] if nearest else None


def _in_date(ctx: _Context, i: int) -> bool:
    located = ctx.raw.get(i)
    if located is None:
        return False
    text, start, end = located
    return bool(_DATE_BEFORE_RE.search(text[:start]) or _DATE_AFTER_RE.match(text[end:]))


def _vintage(ctx: _Context, year_now: int) -> Evidence[int] | None:
    ballot: _Ballot[int] = _Ballot(ctx)
    for i, token in enumerate(ctx.tokens):
        if token.kind != "year":
            continue
        year = int(token.norm)
        if year > year_now + 1 or _in_date(ctx, i):
            continue
        anchor = _vintage_anchor(ctx, i)
        if anchor == "not":
            continue
        low = ANCHORED_VINTAGE_MIN if anchor == "vintage" else BARE_VINTAGE_MIN
        if year >= low:
            ballot.add(year, year, [i], strong=anchor == "vintage")
    vote = ballot.single()
    return _evidence(Evidence[int], vote, ctx) if vote else None


def _spellings(terms: Mapping[str, Sequence[str]]) -> dict[str, tuple[str, ...]]:
    return {
        str(value): tuple(" ".join(phrase_words(text)) for text in texts)
        for value, texts in terms.items()
    }


_SUGAR_SPELLINGS = _spellings(SUGAR_TERMS)
_COLOR_SPELLINGS = _spellings(COLOR_TERMS)


def _spelling_cost(text: str, spelling: str) -> float:
    """Цена опечатки OCR от написания в пределах бюджета словаря, без скелета; иначе inf."""
    length = sum(ch.isalnum() for ch in text)
    shortest = min(length, sum(ch.isalnum() for ch in spelling))
    budget = 0.0 if shortest <= SHORT_WORD else cost_budget(shortest)
    cost = weighted_distance(text, spelling, cutoff=budget)
    return cost if cost <= budget + 1e-9 else math.inf


def _misspelled(text: str, spellings: Sequence[str]) -> bool:
    """`text` — написание из списка с опечатками OCR в пределах бюджета словаря, без скелета."""
    return any(_spelling_cost(text, spelling) < math.inf for spelling in spellings)


def _nearest_agreeing_form(text: str, spellings: Sequence[str]) -> str | None:
    """Форма не среднего рода, к которой опечатка ближе, чем к любому другому написанию.

    «красвая» → «красная»; «kpachoe» (гомоглифы) — это «красное», а не «красные» за одну
    правку; при равной цене форма не выбирается: опечатка остаётся признаком вина.
    """
    agreeing: tuple[float, str | None] = (math.inf, None)
    other = math.inf
    for spelling in spellings:
        cost = _spelling_cost(text, spelling)
        if spelling in AGREEING_FORMS:
            agreeing = min(agreeing, (cost, spelling), key=lambda item: item[0])
        else:
            other = min(other, cost)
    return agreeing[1] if agreeing[0] < other else None


@dataclass(frozen=True, slots=True)
class _Phrase:
    ids: tuple[int, ...]
    value: str
    matched: str
    cost: float  # 0 — фраза таксономии, иначе цена попадания словаря


#: Слова, перед которыми прилагательное не среднего рода — ещё признак вина, а не начало
#: имени: «белый купаж», «сухой херес», «белые вина», «красный брют» (и любые фразы таксономии).
#: Э7: и слова строк урожая, объёма и крепости — «БЕЛЫЙ» / «ГОД УРОЖАЯ», «РОЗОВЫЙ» / «Алк. 12%».
_WINE_VOCABULARY = WINE_WORDS | LABEL_GENERIC | BLEND_WORDS | VINTAGE_LINE_WORDS | VOLUME_LINE_WORDS
#: Слова, после которых прилагательное стоит при них: стиль и купаж — это и есть вино.
_WINE_HEADS = STYLE_WORDS | BLEND_WORDS


def _segment_neighbour(ctx: _Context, i: int, offset: int) -> int | None:
    """Соседний токен того же чтения и того же отрезка текста (`token_segments`)."""
    j = ctx.neighbour(i, offset)
    return j if j is not None and ctx.segments[j] == ctx.segments[i] else None


def _starts_proper_name(
    ctx: _Context, i: int, form: str, term_ids: set[int], grape_ids: set[int]
) -> bool:
    """«Красная Горка», «Белая Львица», «Сухой Лог»: прилагательное не среднего рода `form`
    перед словом не из словаря этикетки — начало имени собственного, а не цвет или сахар вина.

    Сосед ищется в том же отрезке текста, через перенос строки: имя на этикетке бывает в две
    строки («КРАСНАЯ» / «СТРЕЛКА»), а VLM пишет строки без рамок, одним отрезком. Признак вина
    остаются прилагательное в конце отрезка и перед числом («МУСКАТЕЛЬ» / «БЕЛЫЙ» / «0,75»),
    перед фразой таксономии или словом этикетки («Белый купаж», «Красный брют», «Вина белые
    сухие»), перед строкой урожая, объёма или крепости (Э7: «БЕЛЫЙ» / «ГОД УРОЖАЯ», «СУХАЯ» /
    «Алк. 12% об.») и прилагательное, которое согласуется со словом перед ним:
    - после стиля или купажа — всегда: они стоят вместо слова «вино» («Портвейн белый Алушта»,
      «Херес сухой», «Мадера белая», «Купаж красный»);
    - после сорта — только мужской род: сорта почти все мужского рода («Кокур десертный
      Сурож», «МУСКАТ ДЕСЕРТНЫЙ» / …). Женский род и множественное число с сортом не
      согласуются: «Шардоне Красная Горка», «Мерло Белая Скала» — сорт и имя виноградника.
      Мужской род после сорта перед именем («Саперави Белый Колодец») от «Кокур десертный
      Сурож» по словам не отличить: это остаётся признаком вина.
    """
    before = _segment_neighbour(ctx, i, -1)
    if before is not None:
        if ctx.tokens[before].norm in _WINE_HEADS:
            return False
        if before in grape_ids and form in MASCULINE_FORMS:
            return False
    after = _segment_neighbour(ctx, i, 1)
    if after is None:
        return False
    token = ctx.tokens[after]
    return token.kind == "word" and after not in term_ids and token.norm not in _WINE_VOCABULARY


def _term_phrases(
    ctx: _Context,
    found: Sequence[_Found],
    hits: Sequence[tuple[list[int], LexHit]],
    kind: TermKind,
    value_of: Callable[[str], str | None],
    spellings: Mapping[str, Sequence[str]],
    names: Sequence[tuple[tuple[int, ...], float]] = (),
    proper_name: Callable[[int, str], bool] | None = None,
) -> list[_Phrase]:
    """Фразы сахара или цвета из таксономии и словаря каталога: длинные первыми, без перекрытий.

    Фразы таксономии любого вида занимают слова (заглушка «Extra Dry» тоже), как и названия
    `names` (окно и цена попадания). Попадание словаря — только опечатка написания из списка.
    При равной длине побеждает таксономия, затем более дешёвое попадание. Прилагательное не
    среднего рода в начале имени (`proper_name`: «Красная Горка») занимает слово без значения,
    и опечатка словаря, которая ближе всего к такой форме («Красвая Горка»), — тоже.
    """

    def names_a_proper_name(ids: Sequence[int], form: str | None) -> bool:
        if proper_name is None or len(ids) != 1 or form not in AGREEING_FORMS:
            return False
        return proper_name(ids[0], form)

    def value_of_found(item: _Found) -> str | None:
        if item.term.kind != kind:
            return None
        if names_a_proper_name(item.ids, item.phrase):
            return None
        return item.term.value

    candidates: list[tuple[tuple[int, ...], bool, float, str | None, str]] = [
        (item.ids, False, 0.0, value_of_found(item), item.phrase) for item in found
    ]
    candidates += [(ids, True, cost, None, "") for ids, cost in names]
    lexical = "sugar" if kind == "sugar" else "color"
    for ids, hit in hits:
        value = value_of(hit.canonical) if hit.field == lexical else None
        if value is None:
            continue
        text = " ".join(ctx.tokens[i].norm for i in ids)
        written = spellings.get(str(value), ())
        if not _misspelled(text, written):
            continue
        # Опечатка формы не среднего рода проходит ту же проверку имени, что и сама форма.
        named = names_a_proper_name(ids, _nearest_agreeing_form(text, written))
        candidates.append(
            (tuple(ids), True, hit.cost, None if named else str(value), hit.canonical)
        )
    candidates.sort(key=lambda c: (-len(c[0]), c[1], c[2], min(c[0])))

    taken: set[int] = set()
    phrases: list[_Phrase] = []
    for ids, _, cost, value, matched in candidates:
        if not taken.isdisjoint(ids):
            continue
        taken.update(ids)
        if value is not None:
            phrases.append(_Phrase(ids, value, matched, cost))
    return phrases


def _sugar(
    ctx: _Context,
    found: Sequence[_Found],
    hits: Sequence[tuple[list[int], LexHit]],
    proper_name: Callable[[int, str], bool],
) -> list[_Vote[SugarClass]]:
    ballot: _Ballot[SugarClass] = _Ballot(ctx)
    phrases = _term_phrases(
        ctx, found, hits, "sugar", sugar_class, _SUGAR_SPELLINGS, proper_name=proper_name
    )
    # Фраза таксономии с двумя классами («extra brut zero dosage») голосует за оба.
    also = {(item.ids, item.phrase): item.term.also for item in found if item.term.kind == "sugar"}
    for phrase in phrases:
        claim_cost = max(phrase.cost, ADJECTIVE_CLAIM_COST)
        for code in (phrase.value, *also.get((phrase.ids, phrase.matched), ())):
            value = SugarClass(code)
            ballot.add(value, value, phrase.ids, matched=phrase.matched, claim_cost=claim_cost)
    return ballot.ranked()


# Номер серии — малое римское число: L и C на этикетке чаще мусор OCR или «cl».
_SERIAL_ROMAN_RE = re.compile(r"[ivx]+")
_ROMAN_OPENERS = frozenset("\"'«“„(№#")
_ROMAN_CLOSERS = frozenset("\"'»”).,;:!?")


def _is_roman_serial(ctx: _Context, i: int) -> bool:
    """Римский номер серии, а не палочки мусора («[II(», «Ю0 II h») и не «75 CL»."""
    token = ctx.tokens[i]
    if not _SERIAL_ROMAN_RE.fullmatch(token.norm):
        return False
    for offset in (-1, 1):
        j = ctx.neighbour(i, offset)
        if j is None or ctx.tokens[j].line_id != token.line_id:
            continue
        neighbour = ctx.tokens[j]
        if offset == -1 and neighbour.kind in _NUMERIC_KINDS:
            return False  # единица или счётчик после числа
        if len(neighbour.norm) == 1:
            return False  # палочки среди одиночных знаков
    located = ctx.raw.get(i)
    if located is not None:
        text, start, end = located
        if start > 0 and not (text[start - 1].isspace() or text[start - 1] in _ROMAN_OPENERS):
            return False
        if end < len(text) and not (text[end].isspace() or text[end] in _ROMAN_CLOSERS):
            return False
    return True


def _serial_canonical(text: str) -> str:
    """Каноническая серия: «Reserve» и «Riserva» → «резерв», как у словаря каталога."""
    term = LABEL_TERMS.lookup(text)
    if term is not None and term.kind == "serial" and term.value is not None:
        return term.value
    return text


def _serial(
    ctx: _Context, found: Sequence[_Found], hits: Sequence[tuple[list[int], LexHit]]
) -> list[_Vote[str]]:
    ballot: _Ballot[str] = _Ballot(ctx)
    for i, token in enumerate(ctx.tokens):
        if token.kind == "roman":
            if _is_roman_serial(ctx, i):
                value = token.norm.upper()
                ballot.add(("roman", value), value, [i])
        elif token.kind == "ratio":
            ballot.add(("ratio", token.norm), token.norm, [i])
    # Слова серии голосуют канонической формой: «Reserve» из текста и попадание словаря
    # «резерв» на тех же токенах — одно значение, а не два.
    for item in found:
        if item.term.kind == "serial" and item.term.value is not None:
            value = item.term.value
            key = ("word", " ".join(phrase_words(value)))
            ballot.add(key, value, item.ids, matched=value)
    for ids, hit in hits:
        if hit.field != "serial":
            continue
        token = ctx.tokens[ids[0]]
        if len(ids) == 1 and token.kind in ("roman", "ratio"):
            # Римский номер и доля из словаря проходят те же проверки, что и из текста.
            if token.kind == "roman" and not _is_roman_serial(ctx, ids[0]):
                continue
            value = token.norm.upper() if token.kind == "roman" else token.norm
            ballot.add((token.kind, value), value, ids, matched=hit.canonical, claim_cost=hit.cost)
            continue
        if _SERIAL_ROMAN_RE.fullmatch(norm_token(hit.canonical)):
            continue  # римский номер словаря на строчном слове («ii», «vi») — не серия
        value = _serial_canonical(hit.canonical)
        key = ("word", " ".join(phrase_words(value)))
        ballot.add(key, value, ids, matched=value, claim_cost=hit.cost)
    return ballot.ranked()


def _next_to_grape(ctx: _Context, i: int, grape_ids: set[int]) -> bool:
    line_id = ctx.tokens[i].line_id
    for offset in (-1, 1):
        j = ctx.neighbour(i, offset)
        if j is not None and j in grape_ids and ctx.tokens[j].line_id == line_id:
            return True
    return False


def _abv(ctx: _Context, grape_ids: set[int]) -> Evidence[float] | None:
    ballot: _Ballot[float] = _Ballot(ctx)
    for i, token in enumerate(ctx.tokens):
        if token.kind != "abv":
            continue
        value = float(token.norm.rstrip("%"))
        if not ABV_RANGE[0] <= value <= ABV_RANGE[1]:
            continue
        strong = any(unit in ABV_UNITS for unit in norm(token.text).split())
        for offset, anchors in ((-1, _ABV_ANCHORS_BEFORE), (-2, _ABV_ANCHORS_BEFORE)):
            j = ctx.neighbour(i, offset)
            strong |= j is not None and ctx.tokens[j].norm in anchors
        for offset in (1, 2):
            j = ctx.neighbour(i, offset)
            strong |= j is not None and ctx.tokens[j].norm in _ABV_ANCHORS_AFTER
        # «Каберне Совиньон 85%», «15% Мерло» — доля сорта, а не крепость. Сорт строкой выше
        # («Саперави» / «13,5%») — не купаж: VLM пишет крепость отдельной строкой.
        if not strong and _next_to_grape(ctx, i, grape_ids):
            continue
        ballot.add(value, value, [i], strong=strong)
    vote = ballot.single()
    return _evidence(Evidence[float], vote, ctx) if vote else None


def _color(
    ctx: _Context,
    found: Sequence[_Found],
    hits: Sequence[tuple[list[int], LexHit]],
    names: Sequence[tuple[tuple[int, ...], float]],
    proper_name: Callable[[int, str], bool],
) -> _Ballot[Color]:
    ballot: _Ballot[Color] = _Ballot(ctx)
    # «Rosso» внутри более длинного названия кюве или винодельни — не цвет вина.
    phrases = _term_phrases(
        ctx, found, hits, "color", color_of, _COLOR_SPELLINGS, names, proper_name=proper_name
    )
    for phrase in phrases:
        value = Color(phrase.value)
        claim_cost = max(phrase.cost, ADJECTIVE_CLAIM_COST)
        ballot.add(value, value, phrase.ids, matched=phrase.matched, claim_cost=claim_cost)
    return ballot


def extract_fields(
    tokens: list[TokenSpan],
    hits: list[tuple[list[int], LexHit]],
    *,
    readings: list[Reading],
    year_now: int | None = None,
) -> LabelFields:
    """Признаки этикетки из токенов всех чтений кадра.

    `hits` — попадания словаря каталога: индексы токенов в `tokens` и сам LexHit.
    `readings` нужны для текста строк (даты, номера стандартов) и порядка `sources`.
    `unmatched` не заполняется: это делает pipeline.
    """
    for ids, hit in hits:
        if not ids or any(not 0 <= i < len(tokens) for i in ids):
            raise ValueError(f"extract_fields: неверные индексы токенов {ids} у {hit.canonical!r}")

    ctx = _Context.build(tokens, readings)
    found = ctx.terms()
    grape_ids = {i for item in found if item.term.kind == "grape" for i in item.ids}
    term_ids = {i for item in found for i in item.ids}
    # Название занимает слова цвета, только если в нём есть слово не из таксономии: нечёткое
    # кюве «красные» на слове «красное» или кюве «Белое сухое» — не повод терять цвет.
    names = [
        (tuple(ids), hit.cost)
        for ids, hit in hits
        if hit.field in ("producer", "cuvee", "grape") and not term_ids.issuperset(ids)
    ]
    wine_words = any(token.norm in WINE_WORDS for token in tokens)
    wine_terms = any(
        item.term.kind == "grape" or (item.term.kind == "sugar" and item.term.value is not None)
        for item in found
    )
    lexicon_grapes = {i for ids, hit in hits if hit.field == "grape" for i in ids}

    def proper_name(i: int, form: str) -> bool:
        return _starts_proper_name(ctx, i, form, term_ids, grape_ids | lexicon_grapes)

    sugar = _sugar(ctx, found, hits, proper_name)
    serial = _serial(ctx, found, hits)
    color = _color(ctx, found, hits, names, proper_name)
    # Одно слово — одно поле: слово занимает то поле, которое дало из него значение. Сорт —
    # попадания словаря (они же поле `grapes`), сахар, серия и цвет — принятые голоса,
    # заглушки таксономии («Extra Dry») занимают слова без значения. Сорт таксономии без
    # попадания словаря слово не занимает: иначе пропали бы и сорт, и кюве. Цвет занимает
    # слово, даже если поле цвета осталось пустым (два цвета в одном чтении): «красное» —
    # цвет, а не нечёткое кюве «красные». Цвет и сахар занимают слово ценой
    # `ADJECTIVE_CLAIM_COST`: кюве в написании каталога («Блан», «Сухой») остаётся рядом с ними.
    claims: dict[int, list[tuple[int, float]]] = {}

    def claim(ids: Sequence[int], cost: float) -> None:
        for i in ids:
            claims.setdefault(i, []).append((len(ids), cost))

    for ids, hit in hits:
        if hit.field == "grape":
            claim(ids, hit.cost)
    for item in found:
        if item.term.kind == "sugar" and item.term.value is None:
            claim(item.ids, 0.0)
    for vote in (*sugar, *serial, *color.ranked()):
        for ids, cost in vote.spans:
            claim(ids, cost)
    color_vote = color.single()

    return LabelFields(
        producer=_lexical(ctx, hits, "producer", claims),
        cuvee=_cuvee(ctx, hits, claims),
        grapes=_lexical(ctx, hits, "grape", claims),
        sugar=[_evidence(Evidence[SugarClass], vote, ctx) for vote in sugar],
        vintage=_vintage(ctx, year_now if year_now is not None else datetime.now(UTC).year),
        serial=[_evidence(Evidence[str], vote, ctx) for vote in serial],
        abv=_abv(ctx, grape_ids | lexicon_grapes),
        color=_evidence(Evidence[Color], color_vote, ctx) if color_vote else None,
        unmatched=[],
        is_wine_label=bool(hits) or wine_words or wine_terms,
    )
