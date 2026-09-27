"""Поиск фрагментов этикетки в закрытом словаре каталога.

Для каждого окна до четырёх соседних токенов одного чтения:
1. точная норма — цена 0;
2. совпал скелет транслитерации («Chardonnay» и «Шардоне») — `SKELETON_COST`;
3. взвешенный Левенштейн по норме и по скелету: гомоглифы стоят 0, типичные путаницы
   OCR — 0,3–0,5, пробел — `SPACE_COST`, прочие правки — 1.

Порог цены растёт с длиной (`cost_budget`), слова до трёх букв находятся только точно
(с точностью до гомоглифов). Словарь не декодирует текст: сырой токен остаётся в
`TokenSpan`, до трёх лучших попаданий на окно возвращаются с ценой, а отсекает и выбирает
вызывающий слой.

Окно и фраза не выходят за отрезок текста: чтение, в котором строки идут подряд на кадре
(`app.reading.text.layout.token_segments`). Слова одного ряда, порезанные читателем на
отдельные строки, и название в две строки — один отрезок; слова с разных мест кадра — нет.
Без рамок отрезок — всё чтение.

Многословная запись (фраза) засчитывается только целиком:
- окно совпало со всей фразой по правилам выше;
- или по порядку совпали все её значимые слова, каждое — своим токеном. Между соседними
  значимыми словами допускается не больше одного служебного токена (`PHRASE_SERVICE_WORDS`):
  «Блан Нуар» и «Блан ле Нуар» — это «Блан де Нуар». Пропущенное или другое служебное слово
  стоит `SERVICE_SKIP_COST`.
Одно слово фразы попадания не даёт: запись `part` («нуар» из «Пино Нуар») служит только
поиску кандидатов и не делает слово «вне словаря».

Веса путаниц заданы вручную и не обучались: dev-разбиения чтений пока нет. Матрицу
путаниц учим позже только на dev (чтения против `gt_tokens.jsonl`), не на test и не на
публичных кадрах.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Collection, Hashable, Sequence
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any

import numpy as np

from app.reading.contracts import LexField, LexHit, TokenSpan
from app.reading.lexicon.build import FIELDS, GENERIC_WORDS
from app.reading.text.normalize import norm_token
from app.reading.text.translit import skeleton

if TYPE_CHECKING:
    from app.reading.lexicon.build import Lexicon

#: Норма разная, скелет совпал: транслитерация, а не ошибка чтения.
SKELETON_COST = 0.2
#: Лишний или пропущенный пробел: «ПИНОНУАР», «ABRAU DURSO».
SPACE_COST = 0.3
#: Служебное слово фразы пропущено или заменено другим: «Блан Нуар» вместо «Блан де Нуар».
SERVICE_SKIP_COST = 0.3
MAX_COST = 2.0
MAX_WORDS = 4
PER_SPAN = 3
#: Слова не длиннее — только точное совпадение.
SHORT_WORD = 3

#: Пары кириллица—латиница, неотличимые на глаз: замена бесплатна.
_HOMOGLYPHS = ("аa", "вb", "еe", "кk", "мm", "нh", "оo", "рp", "сc", "тt", "хx", "уy", "іi")

#: Типичные путаницы OCR. Веса ручные; обучение — только на dev-разбиении.
CONFUSIONS: tuple[tuple[str, str, float], ...] = (
    ("0", "о", 0.3),
    ("0", "o", 0.3),
    ("1", "l", 0.3),
    ("1", "i", 0.3),
    ("1", "і", 0.3),
    ("l", "i", 0.3),
    ("l", "і", 0.3),
    ("ш", "щ", 0.3),
    ("и", "й", 0.3),
    ("ь", "ъ", 0.3),
    ("б", "6", 0.3),
    ("з", "3", 0.4),
    ("в", "8", 0.5),
    ("c", "e", 0.5),
    ("с", "е", 0.5),
    ("u", "v", 0.5),
    ("п", "н", 0.5),
    ("rn", "m", 0.4),
    ("cl", "d", 0.4),
    ("vv", "w", 0.4),
)

#: Служебные слова этикетки: вне словаря они не признак «чужого» вина.
LABEL_STOP_WORDS = GENERIC_WORDS | frozenset(
    {
        "урожай", "урожая", "год", "года", "выдержка", "выдержки", "месяц", "месяца",
        "месяцев", "лет", "объем", "крепость", "алк", "спирт", "содержание", "сахар",
        "сахара", "изготовитель", "производитель", "произведено", "разлито", "розлив",
        "адрес", "область", "край", "район", "республика", "поселок", "улица", "тел",
        "ооо", "зао", "гост", "защищенного", "географического", "указания",
        "наименования", "места", "происхождения", "категории", "столовое", "сортовое",
        "газированное", "бутылке", "годен", "хранить", "температуре", "беременным",
        "детям", "противопоказано", "чрезмерное", "употребление", "вредит", "здоровью",
        "более", "менее", "дм3", "литр", "литра", "срок", "дата", "партия", "кормящим",
        "лицам", "моложе", "содержит", "диоксид", "серы", "консервант",
        "агрофирма", "агрокомплекс", "компания", "завод", "предприятие", "хозяйство",
        "краснодарский", "ставропольский", "ростовская", "крымская", "республики",
        "product", "produced", "bottled", "vintage", "alc", "contains", "sulfites",
        "sulphites", "grown", "made", "red", "white", "dry", "wine", "russia", "region",
    }
)  # fmt: skip

#: Служебные слова фраз: артикли и предлоги названий, в том числе в русской записи
#: («Блан де Блан», «Шато ле Гран»). Их можно пропустить или заменить другим служебным.
_PHRASE_SERVICE_TEXT = "де de la le di del du и & ла ле ди дель дю"
PHRASE_SERVICE_WORDS = frozenset(
    word for word in (norm_token(w) for w in _PHRASE_SERVICE_TEXT.split()) if word
)

#: Включатель rapidfuzz (тесты сравнивают его с запасным путём на чистом Python).
USE_RAPIDFUZZ = True


def _pairs() -> tuple[dict[tuple[str, str], float], list[tuple[str, str, float]]]:
    single: dict[tuple[str, str], float] = {}
    multi: list[tuple[str, str, float]] = []
    pairs = [(cyr, lat, 0.0) for cyr, lat in _HOMOGLYPHS] + list(CONFUSIONS)
    for a, b, w in pairs:
        if len(a) == 1 and len(b) == 1:
            single[(a, b)] = single[(b, a)] = w
        else:
            multi += [(a, b, w), (b, a, w)]
    return single, multi


_SUB, _MULTI = _pairs()
# Нижняя граница цены на единицу разницы длин: пробел или «rn»→«m».
_MIN_INDEL = min([SPACE_COST, *(w for _, _, w in _MULTI)])


def _fold_table() -> tuple[dict[int, int], tuple[tuple[str, str], ...]]:
    """Свёртка для предотбора: всё, что стоит меньше 1, сводится к одному символу."""
    parent: dict[str, str] = {}

    def find(ch: str) -> str:
        while parent.get(ch, ch) != ch:
            ch = parent[ch]
        return ch

    for a, b in _SUB:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    table = {ord(ch): ord(find(ch)) for ch in {c for pair in _SUB for c in pair}}
    multi = tuple((a, b) for a, b, _ in _MULTI if len(a) > len(b))
    return table, multi


_FOLD, _FOLD_MULTI = _fold_table()


def fold_key(text: str) -> str:
    """Ключ предотбора: без пробелов, гомоглифы и путаницы сведены.

    Обычный Левенштейн между ключами не больше взвешенного между строками, поэтому
    отбор по ключам с порогом `floor(бюджет)` не теряет кандидатов (кроме редких
    перекрытий многосимвольных замен).
    """
    text = text.replace(" ", "")
    for src, dst in _FOLD_MULTI:
        text = text.replace(src, dst)
    return text.translate(_FOLD)


def _indel(ch: str) -> float:
    return SPACE_COST if ch == " " else 1.0


def weighted_distance(a: str, b: str, *, cutoff: float = math.inf) -> float:
    """Взвешенный Левенштейн: гомоглифы 0, путаницы OCR по `CONFUSIONS`, пробел `SPACE_COST`.

    Возвращает `math.inf`, если цена заведомо больше `cutoff`.
    """
    if a == b:
        return 0.0
    n, m = len(a), len(b)
    if abs(n - m) * _MIN_INDEL > cutoff + 1e-9:
        return math.inf
    first = [0.0] * (m + 1)
    for j in range(1, m + 1):
        first[j] = first[j - 1] + _indel(b[j - 1])
    rows = [first]
    for i in range(1, n + 1):
        ai = a[i - 1]
        up = rows[i - 1]
        row = [up[0] + _indel(ai)] + [0.0] * m
        for j in range(1, m + 1):
            bj = b[j - 1]
            best = up[j - 1] + (0.0 if ai == bj else _SUB.get((ai, bj), 1.0))
            best = min(best, up[j] + _indel(ai), row[j - 1] + _indel(bj))
            for src, dst, w in _MULTI:
                ls, ld = len(src), len(dst)
                if ls <= i and ld <= j and a[i - ls : i] == src and b[j - ld : j] == dst:
                    best = min(best, rows[i - ls][j - ld] + w)
            row[j] = best
        rows.append(row)
        # Многосимвольная замена перескакивает не больше одной строки.
        if min(row) > cutoff + 1e-9 and min(up) > cutoff + 1e-9:
            return math.inf
    return rows[n][m]


def cost_budget(length: int, max_cost: float = MAX_COST) -> float:
    """Допустимая цена по длине (буквы и цифры без пробелов)."""
    if length <= SHORT_WORD:
        budget = 0.0
    elif length <= 5:
        budget = 0.5
    elif length <= 8:
        budget = 1.0
    elif length <= 12:
        budget = 1.5
    else:
        budget = 2.0
    return min(budget, max_cost)


def _compact_len(text: str) -> int:
    return sum(ch.isalnum() for ch in text)


def bounded_levenshtein(a: str, b: str, k: int) -> int:
    """Обычный Левенштейн с отсечкой: больше `k` — возвращается `k + 1`."""
    if abs(len(a) - len(b)) > k:
        return k + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > k:
            return k + 1
        prev = cur
    return min(prev[-1], k + 1)


@cache
def _load_rapidfuzz() -> tuple[Any, Any] | None:
    try:
        from rapidfuzz.distance import Levenshtein
        from rapidfuzz.process import cdist
    except ImportError:
        return None
    return cdist, Levenshtein.distance


# ---------------------------------------------------------------------- индекс
@dataclass(slots=True)
class _FuzzyIndex:
    keys: list[str]
    entry_ids: list[tuple[int, ...]]
    position: dict[str, int]
    by_len: dict[int, list[int]]  # длина ключа → номера ключей
    multi_by_len: dict[int, list[int]]  # то же, только ключи фраз
    lengths: list[int]  # длина нормы записи без пробелов


def _fuzzy_index(lexicon: Lexicon) -> _FuzzyIndex:
    cached = lexicon.cache.get("fuzzy")
    if cached is not None:
        return cached
    ids: dict[str, set[int]] = defaultdict(set)
    for i, entry in enumerate(lexicon.entries):
        ids[fold_key(entry.norm)].add(i)
        ids[fold_key(entry.skeleton)].add(i)
    keys = sorted(ids)
    entry_ids = [tuple(sorted(ids[k])) for k in keys]
    by_len: dict[int, list[int]] = defaultdict(list)
    multi_by_len: dict[int, list[int]] = defaultdict(list)
    for pos, key in enumerate(keys):
        by_len[len(key)].append(pos)
        if any(lexicon.entries[i].n_words > 1 for i in entry_ids[pos]):
            multi_by_len[len(key)].append(pos)
    index = _FuzzyIndex(
        keys=keys,
        entry_ids=entry_ids,
        position={k: i for i, k in enumerate(keys)},
        by_len=dict(by_len),
        multi_by_len=dict(multi_by_len),
        lengths=[_compact_len(e.norm) for e in lexicon.entries],
    )
    lexicon.cache["fuzzy"] = index
    return index


@dataclass(frozen=True, slots=True)
class _Phrase:
    """Многословная запись по значимым словам."""

    entry: int
    words: tuple[str, ...]  # значимые слова по порядку
    skeletons: tuple[str, ...]
    gaps: tuple[tuple[str, ...], ...]  # служебные слова между значимыми словами j и j+1


@dataclass(slots=True)
class _PhraseIndex:
    phrases: list[_Phrase]
    by_norm: dict[str, list[int]]  # первое значимое слово → номера фраз
    by_skeleton: dict[str, list[int]]


def _word_skeleton(word: str) -> str:
    return skeleton(word) if any(ch.isalpha() for ch in word) else word


def _phrase_index(lexicon: Lexicon) -> _PhraseIndex:
    cached = lexicon.cache.get("phrases")
    if cached is not None:
        return cached
    phrases: list[_Phrase] = []
    by_norm: dict[str, list[int]] = defaultdict(list)
    by_skeleton: dict[str, list[int]] = defaultdict(list)
    for eid, entry in enumerate(lexicon.entries):
        if entry.n_words < 2 or entry.part:
            continue
        words: list[str] = []
        gaps: list[tuple[str, ...]] = []
        pending: list[str] = []
        for word in entry.norm.split():
            if word in PHRASE_SERVICE_WORDS:
                if words:
                    pending.append(word)
                continue
            if words:
                gaps.append(tuple(pending))
            pending = []
            words.append(word)
        if len(words) < 2:
            continue  # фраза с одним значимым словом совпадает только целиком
        phrase = _Phrase(eid, tuple(words), tuple(_word_skeleton(w) for w in words), tuple(gaps))
        by_norm[phrase.words[0]].append(len(phrases))
        by_skeleton[phrase.skeletons[0]].append(len(phrases))
        phrases.append(phrase)
    index = _PhraseIndex(phrases, dict(by_norm), dict(by_skeleton))
    lexicon.cache["phrases"] = index
    return index


# ---------------------------------------------------------------------- поиск
@dataclass(slots=True)
class _Span:
    ids: tuple[int, ...]
    norm: str
    skeleton: str
    length: int
    fuzzy: bool  # только слова: числа, годы и доли ищутся лишь точно


def _groups(
    tokens: Sequence[TokenSpan], segments: Sequence[int] | None, cross_lines: bool
) -> list[tuple[str, int, int]]:
    """Ключ отрезка каждого токена: чтение, отрезок, а при `cross_lines=False` — и строка."""
    if segments is not None and len(segments) != len(tokens):
        raise ValueError(f"search: {len(segments)} номеров отрезков на {len(tokens)} токенов")
    return [
        (
            token.reading,
            segments[i] if segments is not None else 0,
            -1 if cross_lines else token.line_id,
        )
        for i, token in enumerate(tokens)
    ]


def _spans(tokens: Sequence[TokenSpan], max_words: int, groups: Sequence[Hashable]) -> list[_Span]:
    out: list[_Span] = []
    for start, first in enumerate(tokens):
        if not first.norm:
            continue
        norms: list[str] = []
        skeletons: list[str] = []
        fuzzy = True
        for end in range(start, min(len(tokens), start + max_words)):
            token = tokens[end]
            if not token.norm or groups[end] != groups[start]:
                break
            norms.append(token.norm)
            skeletons.append(token.skeleton or token.norm)
            fuzzy = fuzzy and token.kind == "word"
            text = " ".join(norms)
            out.append(
                _Span(
                    ids=tuple(range(start, end + 1)),
                    norm=text,
                    skeleton=" ".join(skeletons),
                    length=_compact_len(text),
                    fuzzy=fuzzy,
                )
            )
    return out


def _group_ends(groups: Sequence[Hashable]) -> list[int]:
    """Для каждого токена — индекс последнего токена его отрезка."""
    ends = [0] * len(groups)
    end = len(groups) - 1
    for i in range(len(groups) - 1, -1, -1):
        if i < len(groups) - 1 and groups[i] != groups[i + 1]:
            end = i
        ends[i] = end
    return ends


def _candidate_keys(
    queries: dict[str, list[_Span]], index: _FuzzyIndex, max_cost: float
) -> dict[str, list[int]]:
    """Ключ запроса → номера ключей словаря, у которых обычное расстояние ≤ floor(бюджет)."""
    groups: dict[tuple[int, int, bool], list[str]] = defaultdict(list)
    for query, spans in queries.items():
        length = max(span.length for span in spans)
        k = math.floor(cost_budget(length, max_cost) + 1e-9)
        multi = all(len(span.ids) > 1 for span in spans)
        groups[(len(query), k, multi)].append(query)

    rapid = _load_rapidfuzz() if USE_RAPIDFUZZ else None
    found: dict[str, list[int]] = defaultdict(list)
    for (length, k, multi), group in groups.items():
        if k == 0:
            for query in group:
                pos = index.position.get(query)
                if pos is not None:
                    found[query].append(pos)
            continue
        table = index.multi_by_len if multi else index.by_len
        choice_ids = [p for n in range(length - k, length + k + 1) for p in table.get(n, ())]
        if not choice_ids:
            continue
        choices = [index.keys[p] for p in choice_ids]
        if rapid is not None:
            cdist, distance = rapid
            matrix = cdist(group, choices, scorer=distance, score_cutoff=k, workers=1)
            for qi, ci in zip(*np.nonzero(matrix <= k), strict=True):
                found[group[qi]].append(choice_ids[ci])
        else:
            for query in group:
                for pos, choice in zip(choice_ids, choices, strict=True):
                    if bounded_levenshtein(query, choice, k) <= k:
                        found[query].append(pos)
    return found


def _text_cost(
    norm: str, skel: str, length: int, other: str, other_skel: str, other_len: int, max_cost: float
) -> float:
    """Цена совпадения текста с нормой словаря: по норме или по скелету, в бюджете длины."""
    shortest = min(length, other_len)
    if shortest <= SHORT_WORD:
        cost = weighted_distance(norm, other, cutoff=0.0)
        return cost if cost == 0.0 else math.inf
    budget = cost_budget(shortest, max_cost)
    by_norm = weighted_distance(norm, other, cutoff=budget)
    by_skeleton = SKELETON_COST + weighted_distance(skel, other_skel, cutoff=budget - SKELETON_COST)
    cost = min(by_norm, by_skeleton)
    return cost if cost <= budget + 1e-9 else math.inf


def _span_cost(span: _Span, norm: str, skel: str, entry_len: int, max_cost: float) -> float:
    return _text_cost(span.norm, span.skeleton, span.length, norm, skel, entry_len, max_cost)


def _match_phrase(
    tokens: Sequence[TokenSpan],
    start: int,
    line_end: int,
    phrase: _Phrase,
    word_cost: Callable[[int, _Phrase, int], float],
    max_cost: float,
) -> tuple[tuple[int, ...], float] | None:
    """Значимые слова фразы по порядку с токена `start` в пределах отрезка (до `line_end`).

    Между соседними значимыми словами — не больше одного служебного токена. Возвращает
    индексы токенов от первого до последнего значимого слова и цену: сумма цен слов плюс
    `SERVICE_SKIP_COST` за каждое пропущенное или заменённое служебное слово.
    """
    first = word_cost(start, phrase, 0)
    if first == math.inf:
        return None
    best: tuple[float, int] | None = None
    # (номер следующего значимого слова, позиция последнего совпавшего токена, цена)
    stack: list[tuple[int, int, float]] = [(1, start, first)]
    while stack:
        j, pos, cost = stack.pop()
        if j == len(phrase.words):
            if best is None or (cost, pos) < best:
                best = (cost, pos)
            continue
        expected = phrase.gaps[j - 1]
        for gap in (0, 1):
            nxt = pos + 1 + gap
            if nxt > line_end:
                break
            if gap:
                between = tokens[pos + 1].norm
                if between not in PHRASE_SERVICE_WORDS:
                    break
                penalty = 0.0 if expected == (between,) else SERVICE_SKIP_COST
            else:
                penalty = SERVICE_SKIP_COST if expected else 0.0
            step = word_cost(nxt, phrase, j)
            if step < math.inf and cost + penalty + step <= max_cost + 1e-9:
                stack.append((j + 1, nxt, cost + penalty + step))
    if best is None:
        return None
    return tuple(range(start, best[1] + 1)), best[0]


@dataclass(frozen=True, slots=True)
class LookupResult:
    """Попадания словаря и токены, узнанные только как слово фразы каталога."""

    hits: list[tuple[tuple[int, ...], LexHit]]
    #: Токены, совпавшие лишь с записями `part`: попадания нет, но и «вне словаря» они не
    #: считаются — это слово каталожной фразы, прочитанное без остальных слов.
    parts: frozenset[int]


def lookup(
    tokens: Sequence[TokenSpan],
    lexicon: Lexicon,
    *,
    max_cost: float = MAX_COST,
    max_words: int = MAX_WORDS,
    per_span: int = PER_SPAN,
    fields: Collection[LexField] | None = None,
    cross_lines: bool = True,
    segments: Sequence[int] | None = None,
) -> list[tuple[tuple[int, ...], LexHit]]:
    """Попадания токенов и фраз в словарь — `search(...).hits`."""
    return search(
        tokens,
        lexicon,
        max_cost=max_cost,
        max_words=max_words,
        per_span=per_span,
        fields=fields,
        cross_lines=cross_lines,
        segments=segments,
    ).hits


def search(
    tokens: Sequence[TokenSpan],
    lexicon: Lexicon,
    *,
    max_cost: float = MAX_COST,
    max_words: int = MAX_WORDS,
    per_span: int = PER_SPAN,
    fields: Collection[LexField] | None = None,
    cross_lines: bool = True,
    segments: Sequence[int] | None = None,
) -> LookupResult:
    """Попадания токенов и фраз до `max_words` значимых слов в словарь.

    `hits` — пары (номера токенов в `tokens`, `LexHit`): до `per_span` лучших на окно
    по цене, при равной цене — более редкие (idf). Записи одной сущности в одном поле
    сливаются в одно попадание. Порядок: по первому токену, длинные окна раньше. Записи
    `part` попаданий не дают, их токены — в `parts`.

    Окно не пересекает границу чтения и отрезка: `segments` — номер отрезка каждого токена
    (`token_segments` по рамкам строк); без них отрезок — всё чтение. `cross_lines=False`
    запрещает окну переходить строку.
    """
    if not tokens or not len(lexicon):
        return LookupResult([], frozenset())
    entries = lexicon.entries
    allowed = set(fields) if fields is not None else None
    groups = _groups(tokens, segments, cross_lines)
    spans = _spans(tokens, max(1, min(max_words, lexicon.max_words)), groups)
    costs: dict[tuple[int, ...], dict[int, float]] = defaultdict(dict)

    def keep(ids: tuple[int, ...], eid: int, cost: float) -> None:
        bucket = costs[ids]
        if cost < bucket.get(eid, math.inf):
            bucket[eid] = cost

    index = _fuzzy_index(lexicon)
    queries: dict[str, list[_Span]] = defaultdict(list)
    for span in spans:
        for eid in lexicon.ids_by_norm(span.norm):
            keep(span.ids, eid, 0.0)
        if span.length > SHORT_WORD:
            for eid in lexicon.ids_by_skeleton(span.skeleton):
                if index.lengths[eid] > SHORT_WORD and entries[eid].norm != span.norm:
                    keep(span.ids, eid, SKELETON_COST)
        if span.fuzzy:
            queries[fold_key(span.norm)].append(span)
            queries[fold_key(span.skeleton)].append(span)

    # Повторы строк (петли VLM, два масштаба) не пересчитывают одну и ту же цену.
    known: dict[tuple[str, str, int], float] = {}
    for query, positions in _candidate_keys(queries, index, max_cost).items():
        for span in queries[query]:
            done = costs.get(span.ids, {})
            for pos in positions:
                for eid in index.entry_ids[pos]:
                    if done.get(eid) == 0.0:
                        continue
                    key = (span.norm, span.skeleton, eid)
                    cost = known.get(key)
                    if cost is None:
                        entry = entries[eid]
                        cost = _span_cost(
                            span, entry.norm, entry.skeleton, index.lengths[eid], max_cost
                        )
                        known[key] = cost
                    if cost < math.inf:
                        keep(span.ids, eid, cost)

    _phrases_by_words(
        tokens, lexicon, costs, keep, _group_ends(groups), max_cost=max_cost, max_words=max_words
    )

    order = {name: i for i, name in enumerate(FIELDS)}
    out: list[tuple[tuple[int, ...], LexHit]] = []
    parts: set[int] = set()
    for ids, bucket in costs.items():
        merged: dict[tuple[str, str], list[Any]] = {}
        for eid, cost in bucket.items():
            entry = entries[eid]
            if entry.part:
                parts.update(ids)
                continue
            if allowed is not None and entry.field not in allowed:
                continue
            slot = merged.get((entry.field, entry.canonical))
            if slot is None:
                merged[(entry.field, entry.canonical)] = [cost, entry.idf, set(entry.slugs)]
            else:
                slot[0] = min(slot[0], cost)
                slot[1] = max(slot[1], entry.idf)
                slot[2] |= entry.slugs
        # Написания общего слова («каберне», «kaberne») — одно доказательство: в одном поле
        # при одинаковых slug остаётся самое дешёвое. Записи без slug не сливаются.
        unique: dict[tuple[str, Any], tuple[str, str, list[Any]]] = {}
        for (field_name, canonical), slot in merged.items():
            key = (field_name, frozenset(slot[2]) or canonical)
            prev = unique.get(key)
            if prev is None or (slot[0], canonical) < (prev[2][0], prev[1]):
                unique[key] = (field_name, canonical, slot)
        ranked = sorted(
            unique.values(),
            key=lambda item: (item[2][0], -item[2][1], order[item[0]], item[1]),
        )
        for field_name, canonical, (cost, _, slugs) in ranked[:per_span]:
            hit = LexHit(
                canonical=canonical,
                field=field_name,  # type: ignore[arg-type]
                cost=round(cost, 3),
                slugs=frozenset(slugs),
            )
            out.append((ids, hit))
    out.sort(key=lambda item: (item[0][0], -len(item[0]), item[1].cost))
    return LookupResult(out, frozenset(parts))


def _phrases_by_words(
    tokens: Sequence[TokenSpan],
    lexicon: Lexicon,
    costs: dict[tuple[int, ...], dict[int, float]],
    keep: Callable[[tuple[int, ...], int, float], None],
    ends: Sequence[int],
    *,
    max_cost: float,
    max_words: int,
) -> None:
    """Фразы по значимым словам: «Блан Нуар» — «Блан де Нуар», но не одно «Нуар».

    Кандидаты — фразы, чьё первое значимое слово совпало с токеном точно, по скелету или
    нечётко (через однословные записи, в том числе `part`, найденные для этого токена).
    `ends` — последний токен отрезка для каждого токена.
    """
    pindex = _phrase_index(lexicon)
    if not pindex.phrases:
        return
    entries = lexicon.entries
    memo: dict[tuple[str, str, str], float] = {}

    def word_cost(i: int, phrase: _Phrase, j: int) -> float:
        token, word = tokens[i], phrase.words[j]
        if token.norm == word:
            return 0.0
        if token.kind != "word" or not token.norm:
            return math.inf
        token_skeleton = token.skeleton or token.norm
        key = (token.norm, token_skeleton, word)
        cost = memo.get(key)
        if cost is None:
            cost = _text_cost(
                token.norm,
                token_skeleton,
                _compact_len(token.norm),
                word,
                phrase.skeletons[j],
                _compact_len(word),
                max_cost,
            )
            memo[key] = cost
        return cost

    for i, token in enumerate(tokens):
        if not token.norm:
            continue
        found = set(pindex.by_norm.get(token.norm, ()))
        if token.kind == "word":
            found.update(pindex.by_skeleton.get(token.skeleton or token.norm, ()))
            for eid in costs.get((i,), {}):
                if entries[eid].n_words == 1:
                    found.update(pindex.by_norm.get(entries[eid].norm, ()))
        for number in sorted(found):
            phrase = pindex.phrases[number]
            if len(phrase.words) > max_words:
                continue
            match = _match_phrase(tokens, i, ends[i], phrase, word_cost, max_cost)
            if match is not None:
                ids, cost = match
                keep(ids, phrase.entry, cost)


def unmatched(
    tokens: Sequence[TokenSpan],
    hits: Sequence[tuple[tuple[int, ...], LexHit]],
    *,
    max_cost: float | None = None,
    min_len: int = 4,
    stop_words: Collection[str] = LABEL_STOP_WORDS,
    known: Collection[int] = (),
) -> list[TokenSpan]:
    """Сильные токены вне словаря — признак «вне каталога».

    Сильный токен: слово из букв не короче `min_len`, не служебное, не покрытое ни одним
    попаданием (с `max_cost` — попаданием не дороже порога) и не из `known` (слова
    каталожных фраз без попадания, `LookupResult.parts`). Числа, годы, крепость и римские
    номера сюда не попадают.
    """
    covered = {i for ids, hit in hits if max_cost is None or hit.cost <= max_cost for i in ids}
    covered.update(known)
    return [
        token
        for i, token in enumerate(tokens)
        if i not in covered
        and token.kind == "word"
        and len(token.norm) >= min_len
        and token.norm.isalpha()
        and token.norm not in stop_words
    ]
