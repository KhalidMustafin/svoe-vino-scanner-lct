"""Маршрут вопроса сомелье: чип или текст → намерение со слотами.

Маршрут детерминированный, как у «Лозы» (`api/intent.py`): намерение решает, какой движок
посчитает ответ, и видно, какое слово к какому ответу привело. Модель в маршрут не входит,
а сырой вопрос никуда дальше разбора не идёт — ни в модель, ни в журнал (договор
`docs/api-sommelier.md`, §1, §4).

Порядок разбора текста (договор, §4):

1. липкий отказ из контекста (`context.refusal_topic`) — тот же отказ на любой вопрос;
2. светская беседа — фраза целиком (`smalltalk.py`);
3. барьер (`barrier.py`) — ему заранее сообщают, названо ли блюдо: «хочу купить вино к
   борщу» — вопрос «какое», а не «где»; промолчал барьер — смысловой слой (`safety.py`, если он
   есть): он может только добавить отказ по окольной фразе, но не снять отказ правил;
4. намерение со слотами: ответ на наводящий вопрос прошлого шага, «помоги выбрать»,
   «помягче / посвежее» (регулярные выражения `preferences.py` «Лозы»), блюдо по имени и
   алиасам `dishes.json` (поиск `DishRegistry` «Лозы» со словоформами и отрицанием: «нет у
   меня борща» — не борщ), подача, тема справочника, факт карточки («насколько оно сладкое?»,
   «оно крепкое?», «сколько его можно хранить?» — `fact`), сорт, замена, «к чему подать» (и «с
   какой едой оно дружит», «к новогоднему столу»), группа блюд («а к мясу?»), эллипсис из
   контекста («а помягче?» после «а к борщу?»);
5. `unknown`: «к хинкали» — блюда нет в правилах (`unknown_dish`), и так же блюдо с другим
   уточнением, чем в правилах: «к пасте с грибами» при блюде «Паста с томатным соусом»
   (`DishMatch.variant`), блюдо, у которого слово правил — только начинка или состав («рыба с
   овощами», «пирог с грибами»), родовое слово с чужим составом («салат с курицей» при
   «Греческом салате») и другой способ готовки, чем в названии («жареная картошка» при
   «Запечённой картошке»); остальное — «вот что я умею».

Чип сразу даёт намерение; перед ним стоит только липкий отказ.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.reading.taxonomy import GRAPE_SYNONYMS, find_grapes
from app.sommelier import barrier, smalltalk
from app.sommelier.barrier import Refusal
from app.sommelier.text import (
    contains_any,
    covered_words,
    negated_at,
    normalize,
    refused_after,
    with_boundaries,
    words_match,
)

INTENTS: tuple[str, ...] = (
    "what_to_eat",
    "serve",
    "softer",
    "fresher",
    "replace",
    "grape",
    "term",
    "dish_check",
    "guided",
    "fact",
    "refuse",
    "smalltalk",
    "unknown",
)
#: Факты карточки, о которых спрашивают словами (`fact`): сахар, крепость, срок хранения.
FACTS: tuple[str, ...] = ("sugar", "alcohol", "storage")
#: Чипы договора (§4.2) и их аргументы.
CHIP_ARGS: dict[str, frozenset[str]] = {
    "what_to_eat": frozenset(),
    "serve": frozenset(),
    "softer": frozenset({"dish"}),
    "fresher": frozenset({"dish"}),
    "replace": frozenset(),
    "grape": frozenset({"grape"}),
    "term": frozenset({"topic"}),
    "dish_check": frozenset({"dish"}),
    "guided": frozenset({"food", "want"}),
}
REQUIRED_ARGS: dict[str, frozenset[str]] = {
    "grape": frozenset({"grape"}),
    "term": frozenset({"topic"}),
    "dish_check": frozenset({"dish"}),
}
GUIDED_FOOD = ("meat", "fish", "cheese", "dessert", "none")
GUIDED_WANT = ("softer", "fresher", "none")
MAX_SHOWN = 12
MAX_CONTEXT_STRING = 200


# ------------------------------------------------------------------ результат
@dataclass(frozen=True, slots=True)
class Route:
    """Намерение и слоты. `dish` — id блюда, `grape` — код сорта, `topic` — тема справочника,
    `fact` — факт карточки (`FACTS`); у «есть послаще?» ещё `want="sweeter"`."""

    intent: str
    dish: str | None = None
    grape: str | None = None
    topic: str | None = None
    food: str | None = None
    want: str | None = None
    refusal: Refusal | None = None
    smalltalk: str | None = None
    unknown_dish: bool = False
    source: str = "chip"  # chip | text
    chip: str | None = None
    fact: str | None = None


@dataclass(frozen=True, slots=True)
class Context:
    """Контекст разговора (договор, §4.3): страница возвращает `facts.context` как есть."""

    intent: str | None = None
    dish: str | None = None
    food: str | None = None
    want: str | None = None
    step: str | None = None
    refusal_topic: str | None = None
    shown: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "dish": self.dish,
            "food": self.food,
            "want": self.want,
            "step": self.step,
            "refusal_topic": self.refusal_topic,
            "shown": list(self.shown),
        }


class ContextError(ValueError):
    """Контекст не по договору: `shown` длиннее 12, строка длиннее 200, не те типы."""


def context_of(raw: Mapping[str, Any] | None, dishes: Mapping[str, Any]) -> Context:
    """Контекст из тела запроса: лишние ключи игнорируются, чужие значения — `None`.

    Неверная форма (`shown` не список строк или длиннее 12, строка длиннее 200) —
    `ContextError`: маршрут отвечает 422. Неизвестные id блюда, группы, направления и темы
    отказа не ломают разговор — они просто забываются.
    """
    if raw is None:
        return Context()
    if not isinstance(raw, Mapping):
        raise ContextError("context — не объект")
    values: dict[str, str | None] = {}
    for key in ("intent", "dish", "food", "want", "step", "refusal_topic"):
        value = raw.get(key)
        if value is not None and not isinstance(value, str):
            raise ContextError(f"context.{key} — не строка")
        if isinstance(value, str) and len(value) > MAX_CONTEXT_STRING:
            raise ContextError(f"context.{key} длиннее {MAX_CONTEXT_STRING}")
        values[key] = value or None
    shown = raw.get("shown") or []
    if not isinstance(shown, list) or not all(isinstance(slug, str) for slug in shown):
        raise ContextError("context.shown — не список строк")
    if len(shown) > MAX_SHOWN or any(len(slug) > MAX_CONTEXT_STRING for slug in shown):
        raise ContextError(f"context.shown длиннее {MAX_SHOWN} или строка длиннее 200")
    sticky = values["refusal_topic"]
    return Context(
        intent=values["intent"] if values["intent"] in INTENTS else None,
        dish=values["dish"] if values["dish"] in dishes else None,
        food=values["food"] if values["food"] in GUIDED_FOOD else None,
        want=values["want"] if values["want"] in GUIDED_WANT else None,
        step=values["step"] if values["step"] in ("food", "want") else None,
        refusal_topic=sticky if sticky in barrier.STICKY else None,
        shown=tuple(dict.fromkeys(shown)),
    )


# ------------------------------------------------------------------ блюда
@dataclass(frozen=True, slots=True)
class _DishEntry:
    key: str
    words: tuple[str, ...]
    dish_id: str


#: Служебные слова, через которые отрицание перешагивает к блюду («нет у меня шашлыка»).
_NEGATION_BRIDGE = frozenset(
    [
        "у",
        "меня",
        "нас",
        "вас",
        "него",
        "нее",
        "них",
        "тебя",
        "себя",
        "есть",
        "было",
        "будет",
        "будем",
        "станем",
        "хочу",
        "хотим",
        "дома",
        "сейчас",
        "сегодня",
        "завтра",
        "вечером",
        "вообще",
        "совсем",
        "пока",
        "никакого",
        "то",
        "и",
        "уже",
        "так",
        "вот",
        "разве",
    ]
)


#: Алиасы блюд, которые в вопросе о вине почти всегда значат само вино: «из какого
#: винограда» — не вопрос о фруктовой тарелке (алиас «виноград» у «Фруктов и ягод»).
_WINE_ALIASES = frozenset({"виноград", "винограду", "винограда"})
#: Предлоги уточнения в названии блюда: «Паста с томатным соусом», «Блины со сметаной».
_WITH = frozenset({"с", "со"})
#: Слова после «с», которые не уточняют блюдо: само вино, местоимения, компания за столом —
#: «буженина с этим вином», «к пасте с друзьями».
_NOT_VARIANT = frozenset(
    {
        "этим", "этой", "таким", "такой", "ним", "ней", "ними", "собой", "вином", "винами",
        "друзьями", "гостями", "семьей", "женой", "мужем", "детьми", "коллегами",
    }
)  # fmt: skip
#: Слова блюд, которые определяет начинка или состав (проверка 25.09, вечер): «салат с курицей»,
#: «пирог с грибами», «рыба с овощами», «картошка с мясом» — другое блюдо, чем «Греческий
#: салат» (алиас «салат»), «Грибы в сметане», «Овощи на гриле» и «Запечённая картошка». Слово
#: целиком, в любом падеже: «рис» — не «рислинг».
_FILLED_HEAD = re.compile(
    r"(?:салат|пирог|пирож|бутерброд|рулет|лаваш|тарталетк|запеканк|блюд|рыб|мяс|картош|картоф"
    r"|капуст|овощ)\w*|суп(?:а|у|ом|е|ы|ов)?|каш(?:а|и|е|у|ей)|рис(?:а|у|ом|е)?|пюре"
)
#: Слова вопроса, дальше которых к началу фразы блюдо не ищется: «к рыбе, а с грибами?».
_HEAD_STOP = frozenset({"к", "под", "а", "и", "или", "для", "но"})
#: Способы готовки в названиях блюд: «Запечённая картошка» — не «жареная картошка» и не
#: «картошка фри», «Гусь запечённый» — не «варёный гусь». Сверяются только у блюд, в названии
#: которых способ назван.
_COOKING = (
    ("жарка", re.compile(r"(?:об)?жарен\w*|фри")),
    ("запекание", re.compile(r"запечен\w*|печен(?:ый|ая|ое|ые|ой|ую|ого|ом|ых|ому|ым)|духовк\w*")),
    ("варка", re.compile(r"варен(?:ый|ая|ое|ые|ой|ую|ого|ом|ых|ому|ым)|отварн\w*")),
    ("тушение", re.compile(r"тушен\w*")),
)


def _cooking_of(word: str) -> str | None:
    """Способ готовки, который называет слово: «жареной» — жарка, «печёная» — запекание."""
    for method, pattern in _COOKING:
        if pattern.fullmatch(word):
            return method
    return None


@dataclass(frozen=True, slots=True)
class DishMatch:
    """Блюдо вопроса (`DishIndex.find`): id, было ли блюдо под отказом («борща нет») и названо
    ли оно с другим уточнением, чем в правилах.

    `variant` — у блюда правил уточнение в самом названии («Паста с томатным соусом»,
    «Вареники с картошкой»), а гость назвал другое: «к пасте с грибами», «вареники с вишней».
    Это другое блюдо, и ответ о блюде правил был бы ответом не на тот вопрос (прогон на
    видеокарте 25.09: «к пасте с грибами» отвечали про томатный соус), поэтому `dish` тогда
    `None`, а маршрут отвечает честным «такого блюда в правилах нет». Так же — проверка 25.09,
    вечер (`DishIndex._other_dish`): «рыба с овощами» (слово правил — начинка чужого блюда),
    «салат с курицей» (родовое слово «салат» у «Греческого салата» с чужим составом) и «жареная
    картошка» (у «Запечённой картошки» способ готовки назван в названии).
    """

    dish: str | None
    refused: bool = False
    variant: bool = False


class DishIndex:
    """Блюда `dishes.json` по имени и алиасам — поиск `DishRegistry.find_all_in_text` «Лозы»."""

    def __init__(self, dishes: Mapping[str, Any]) -> None:
        entries = []
        #: Уточнения блюд, у которых уточнение есть в самом названии: слова после «с» в названии
        #: и алиасах («томатным соусом», «сыром», «овощами» у пасты).
        self.qualifiers: dict[str, frozenset[tuple[str, ...]]] = {}
        #: Уточнения всех блюд — слова после «с» в названии и алиасах: «фетой» у «Греческого
        #: салата» («салат с фетой»), «мясом» у «Тушёной капусты» («тушёная капуста с мясом»).
        self.own_qualifiers: dict[str, frozenset[tuple[str, ...]]] = {}
        #: Способы готовки блюд, в названии которых способ назван, — из названия и алиасов:
        #: у «Запечённой картошки» — запекание («печёная картошка», «картошка в духовке»).
        self.cooking: dict[str, frozenset[str]] = {}
        for dish_id, dish in dishes.items():
            name_words = normalize(dish.name, fold_homoglyphs=True).split()
            qualified = None if _WITH.isdisjoint(name_words) else set()
            own: set[tuple[str, ...]] = set()
            methods: set[str] = set()
            for phrase in (dish.name, *dish.aliases):
                key = normalize(phrase, fold_homoglyphs=True)
                after = _after_with(key.split())
                if after:
                    own.add(after)
                if qualified is not None and after:
                    qualified.add(after)
                methods.update(m for m in map(_cooking_of, key.split()) if m)
                if key in _WINE_ALIASES:
                    continue
                if len(key) >= 3:
                    entries.append(_DishEntry(key, tuple(key.split()), dish_id))
            if qualified:
                self.qualifiers[dish_id] = frozenset(qualified)
            self.own_qualifiers[dish_id] = frozenset(own)
            if any(_cooking_of(word) for word in name_words):
                self.cooking[dish_id] = frozenset(methods)
        entries.sort(key=lambda entry: len(entry.key), reverse=True)
        self.entries = tuple(entries)

    def find(self, question: str) -> DishMatch:
        """Блюдо вопроса, было ли названо блюдо под отказом и не другое ли это уточнение.

        Точное вхождение фразы сильнее словоформ, длинная фраза — короткой («борщ с
        пампушками» раньше «борща»), среди словоформ ближе по длине — вернее («паста», а
        не «пастила»).
        """
        haystack = normalize(question, fold_homoglyphs=True)
        query = haystack.split()
        if not query:
            return DishMatch(None)
        bounded = with_boundaries(question)
        found: list[_Found] = []
        negated: list[_Found] = []
        seen: set[str] = set()
        for entry in self.entries:
            if entry.dish_id in seen:
                continue
            covered = covered_words(entry.words, query)
            if f" {entry.key} " in f" {haystack} ":
                score = (len(entry.words), 2, 0)
            elif len(covered) == len(entry.words):
                score = (len(covered), 1, -_length_gap(entry.words, covered))
            else:
                continue
            seen.add(entry.dish_id)
            positions = [index for index, word in enumerate(query) if word in covered]
            item = _Found(score, entry, min(positions), max(positions))
            if covered and _phrase_negated(bounded, covered):
                negated.append(item)
                continue
            found.append(item)
        refused = bool(negated)
        # Блюдо за «с» после другого блюда — уточнение, а не блюдо вопроса: «паста с белыми
        # грибами» — о пасте, хотя «белые грибы» — фраза длиннее «пасты»; «нет у меня пасты с
        # грибами» — ни о пасте, ни о грибах.
        heads = [item for item in found if not _qualifies(item, [*found, *negated], query)]
        best = max(heads, key=lambda item: item.score, default=None)
        if best is None:
            return DishMatch(None, refused)
        if self._variant(best, query) or self._other_dish(best, query):
            return DishMatch(None, refused, variant=True)
        return DishMatch(best.entry.dish_id, refused)

    def _variant(self, found: _Found, query: Sequence[str]) -> bool:
        """Гость уточнил блюдо иначе, чем правила: «паста с грибами» при «Пасте с томатным соусом».

        Только у блюд с уточнением в названии и только когда найденная фраза уточнения не
        содержит: «паста» (алиас) — да, «утка с яблоками» — уже названо. Уточнение вопроса — одно
        или два слова сразу после «с» за словами блюда; совпало с уточнением блюда
        (`_same_qualifier`) — то же блюдо.
        """
        qualifiers = self.qualifiers.get(found.entry.dish_id)
        if not qualifiers or not _WITH.isdisjoint(found.entry.words):
            return False
        said = query[found.last + 1 : found.last + 4]
        if len(said) < 2 or said[0] not in _WITH or said[1] in _NOT_VARIANT:
            return False
        return not any(_same_qualifier(said[1:], phrase) for phrase in qualifiers)

    def _other_dish(self, found: _Found, query: Sequence[str]) -> bool:
        """Слова правил есть в вопросе, но гость назвал другое блюдо (проверка 25.09, вечер).

        1. Блюдо правил — начинка или гарнир чужого блюда: оно стоит за «с», а перед «с» —
           слово блюда, которое определяет начинка (`_FILLED_HEAD`), и оно не из найденной
           фразы правил: «рыба с овощами» — не «Овощи на гриле», «пирог с грибами» и «мясо с
           грибами» — не «Грибы в сметане». «А с грибами?» и «вино с грибами» — те же грибы, а
           «пюре с котлетой» и «картошка с селёдкой» — алиасы «котлеты с пюре» и «селёдка с
           картошкой» задом наперёд: котлеты и сельдь. Место блюда — первое слово его названия:
           слова фразы в вопросе могут стоять вразбивку («рыба на гриле с овощами» покрывает
           «Овощи на гриле»).
        2. Родовое слово блюда правил с чужим составом: найден один алиас-родовое слово
           («салат», «картошка», «капуста», «овощи»), а за ним «с …», которого нет в названии и
           алиасах блюда: «салат с курицей» — не «Греческий салат». «Салат с фетой» — алиас.
        3. Другой способ готовки: у блюда он назван в названии («Запечённая картошка»), а рядом
           со словами блюда в вопросе — другой: «жареная картошка», «картошка фри», «картошка
           варёная».
        """
        head = found.entry.words[0]
        at = next((i for i, word in enumerate(query) if words_match(word, head)), found.first)
        before = at - 1
        if before >= 1 and query[before] not in _WITH:
            before -= 1  # «с белыми грибами»: определение между «с» и блюдом
        if before >= 1 and query[before] in _WITH:
            for word in reversed(query[max(0, before - 3) : before]):
                if word in _HEAD_STOP:
                    break
                if any(words_match(word, own) for own in found.entry.words):
                    continue  # «пюре с котлетой»: «пюре» — из самой фразы «котлеты с пюре»
                if _FILLED_HEAD.fullmatch(word):
                    return True
        words = found.entry.words
        said = query[found.last + 1 : found.last + 4]
        if (
            len(words) == 1
            and _FILLED_HEAD.fullmatch(words[0])
            and len(said) >= 2
            and said[0] in _WITH
            and said[1] not in _NOT_VARIANT
            and not any(
                _same_qualifier(said[1:], phrase)
                for phrase in self.own_qualifiers.get(found.entry.dish_id, ())
            )
        ):
            return True
        methods = self.cooking.get(found.entry.dish_id)
        if methods:
            near = [query[i] for i in (found.first - 1, found.last + 1) if 0 <= i < len(query)]
            said_methods = {m for m in map(_cooking_of, near) if m}
            if said_methods - methods:
                return True
        return False


@dataclass(frozen=True, slots=True)
class _Found:
    """Найденное в вопросе блюдо: счёт фразы и где в вопросе её первое и последнее слово."""

    score: tuple[int, int, int]
    entry: _DishEntry
    first: int
    last: int


def _qualifies(item: _Found, found: Sequence[_Found], query: Sequence[str]) -> bool:
    """Стоит ли блюдо за «с» сразу после другого найденного блюда («паста с [белыми] грибами»)."""
    for gap in (1, 2):  # «с грибами», «с белыми грибами»
        at = item.first - gap
        if at >= 1 and query[at] in _WITH:
            return any(other.last == at - 1 for other in found if other is not item)
    return False


def _after_with(words: Sequence[str]) -> tuple[str, ...]:
    """Слова после первого «с» / «со»: «паста с томатным соусом» → «томатным», «соусом»."""
    for index, word in enumerate(words):
        if word in _WITH:
            return tuple(words[index + 1 :])
    return ()


def _same_word(left: str, right: str) -> bool:
    """Одно слово уточнения: словоформа или общие первые пять букв («томатами» — «томатным»)."""
    return words_match(left, right) or (len(left) >= 5 and left[:5] == right[:5])


def _same_qualifier(said: Sequence[str], phrase: Sequence[str]) -> bool:
    """Уточнение вопроса (одно-два слова после «с») — уточнение блюда правил `phrase`.

    Первое слово совпало с первым словом уточнения: «с томатами» — «томатным соусом», «с
    сыром». У однословного уточнения перед ним может стоять определение: «с белыми грибами» —
    «грибами». Составное уточнение узнаётся и по одному главному слову: «с соусом». Главного слова
    после чужого определения мало: «со сливочным соусом» и «с грибным соусом» — не «с томатным
    соусом», иначе паста с грибами снова получила бы ответ о томатном соусе.
    """
    if _same_word(said[0], phrase[0]):
        return True
    if len(phrase) == 1:
        return len(said) > 1 and _same_word(said[1], phrase[0])
    return _same_word(said[0], phrase[-1])


def _length_gap(phrase_words: Sequence[str], covered: set[str]) -> int:
    gap = 0
    for word in phrase_words:
        gap += min((abs(len(word) - len(other)) for other in covered), default=0)
    return gap


def _phrase_negated(words: Sequence[str], covered: set[str]) -> bool:
    positions = [i for i, word in enumerate(words) if word in covered]
    if not positions:
        return False
    left, right = min(positions), max(positions)
    return negated_at(
        words, left, window=3, transparent=lambda w: w in _NEGATION_BRIDGE
    ) or refused_after(words, right, window=5, bridge=lambda w: w in _NEGATION_BRIDGE)


# ------------------------------------------------------------------ словари намерений
#: «Помоги выбрать» — наводящие вопросы ТЗ.
_GUIDED = (
    "помоги выбрать", "помогите выбрать", "помоги подобрать", "помогите подобрать",
    "подбери", "подберите", "не знаю что взять", "не знаю что выбрать", "не знаю какое",
    "помоги с выбором", "выбрать вино", "с выбором вина", "что выбрать",
)  # fmt: skip
#: «Помягче / посвежее» — регулярные выражения `preferences.py` «Лозы» как слоты `want`.
_SOFTER_RE = re.compile(
    r"\b(?:помягче|мягче|мягк(?:ое|ого|ие|их)|полегче|легче|легк(?:ое|ого|ие|их)|нежнее|"
    r"понежнее|менее\s+терпк\w*|не\s+(?:такое|такой|так|очень)\s+терпк\w*|без\s+танин\w*|"
    r"поменьше\s+танин\w*|меньше\s+танин\w*|танин\w*\s+(?:помягче|мягче|поменьше|меньше)|"
    r"не\s+такое\s+плотн\w*|попроще)\b"
)
_FRESHER_RE = re.compile(
    r"\b(?:посвежее|свежее|свежего|покислее|кислее|поживее|живее|бодрее|с\s+кислинкой|"
    r"кислотн\w*\s+(?:выше|побольше|повыше))\b"
)
#: Отрицание направления: «не такое лёгкое» — не просьба о лёгком.
_NOT_WANT = re.compile(r"\bне\s+(?:нужно\s+|надо\s+)?(?:помягче|мягче|легче|полегче|свежее)\b")
#: Подача этого вина.
_SERVE = (
    "как подать", "как подавать", "как правильно подать", "как правильно подавать",
    "как его подать", "как его подавать", "как лучше подать", "подача", "подачи",
    "температур", "охлажд", "охлади", "охлаждать", "холодильник", "со льдом", "греть",
    "комнатной", "декантер", "в графин", "перелить", "какой бокал", "из какого бокала",
    "в каком бокале", "как пить", "за сколько открыть", "открыть заранее", "подышать",
    "продышаться", "дать подышать",
)  # fmt: skip
#: «Как его правильно подавать», «как лучше подать» — до двух слов между «как» и глаголом.
_SERVE_RE = re.compile(r"\bкак\b(?:\s+\S+){0,2}\s+пода(?:ть|вать)\b")
#: «Нужен ли декантер» — подача; «зачем декантировать» — справочник.
_SERVE_NEED = ("нужен ли", "нужно ли", "надо ли", "стоит ли", "нужна ли", "обязательно ли")
#: Вопрос «что это такое» — о теме справочника или сорте.
_ASK_WHAT = (
    "что такое", "что значит", "что означает", "что за", "это что", "что это", "зачем",
    "почему", "чем отлича", "в чем разница", "разница", "объясни", "расскажи про",
    "расскажи о", "расскажите про", "что представляет", "для чего", "что вообще",
)  # fmt: skip
#: О сорте.
_GRAPE_WORDS = ("сорт", "о сорте", "про сорт", "виноград")
#: Замена этого вина — «похожие из других виноделен».
_REPLACE = (
    "чем заменить", "замен", "похож", "аналог", "такое же", "такие же", "таких же",
    "такого же", "такому же", "то же самое", "рядом с",
    "вместо", "в том же духе", "в этом же духе", "что то вроде", "нечто подобное",
    "близк", "соседей по вкусу", "соседн", "в ту же сторону", "того же плана", "двойник",
    "дублер", "родственн", "что еще попробовать", "что дальше пробовать", "другие вина",
    "той же оперы", "в характере", "в том же стиле", "в этом стиле", "той же линейки",
    "что взять вместо", "закончил", "разобрали", "нет в магазине", "импортозамещ",
    "отечествен", "российск", "из наших", "наше похожее", "что то такое же", "в таком же",
    "по вкусу как", "продолжить в том же", "рядом по вкусу", "стоит рядом", "перебиться",
)  # fmt: skip
#: «К чему подать» — еда к этому вину (`_FOOD_FOR_WINE_MARKERS` «Лозы» без «что взять к»:
#: у сомелье на карточке «что взять к шашлыку» — вопрос о блюде, а не о еде к вину).
_WHAT_TO_EAT = (
    "подать", "закус", "с чем", "что съесть", "к чему", "на стол", "поесть", "покушать",
    "приготов", "под что", "какая еда", "что едят", "какое блюдо", "что готовить",
    "заедать", "нарезать", "гастропар", "что подойдет", "что подходит", "к чему взять",
    "еду", "еды", "кормить", "накормить", "чем угощать", "угостить", "на ужин", "к ужину",
    "на обед", "к обеду", "на закуску", "что сочетается", "сочетается с", "идут с",
    "идет с", "что идет", "какой сыр", "какое мясо", "какую рыбу", "какой десерт",
    "что из еды", "еда к", "еда под", "блюда к", "блюдо к", "что к нему", "что к ней",
    "чем сопроводить", "какие блюда", "какие закуски", "что есть с", "что есть под",
    "еду под", "чем дополнить", "что поставить", "с чем едят", "с чем подают", "с чем пить",
    "с чем пьют", "меню", "пожевать", "снеки", "под это вино", "к этому вину",
    "сочетание", "сочетания", "пара к", "что к этому", "к нему что",
)  # fmt: skip
#: «С какой едой оно дружит», «к новогоднему столу», «на праздничный стол» — тоже «к чему подать»
#: (без этого — «Вот что я умею», а «к новогоднему» — ещё и «такого блюда нет»).
_WHAT_TO_EAT_RE = re.compile(
    r"\b(?:с\s+как\w+\s+(?:едой|блюдами|закусками)|дружит|дружат)\b|"
    r"\b(?:к|на|для)\s+(?:\S+\s+){0,2}?(?:стол|столу|застолью|застолье)\b"
)
#: Факты карточки словами (`fact`): «насколько оно сладкое», «оно крепкое?», «есть послаще?».
_SUGAR_ASK_RE = re.compile(
    r"\b(?:сладк(?:ое|ий|ая|о)|сладост\w*|сахар\w*|послаще|слаще|сух(?:ое|ой|ая)|сухост\w*)\b"
)
_SWEETER_RE = re.compile(r"\b(?:послаще|слаще)\b")
_ALCOHOL_ASK_RE = re.compile(r"\b(?:крепк(?:ое|ий|ая|о)|крепост\w*|градус\w*|покрепче|крепче)\b")
#: Срок хранения: слово хранения и срока вместе, «пить или хранить», «ещё пить или уже поздно»,
#: «когда его открыть». Открытая бутылка — тема справочника, «как хранить» — тоже.
_STORAGE_WORD_RE = re.compile(r"\b(?:хран\w*|пролеж\w*|простоит|состар\w*)\b")
_DURATION_RE = re.compile(r"\b(?:скольк\w*|долго|до\s+какого|лет|годами|срок\w*)\b")
_STORAGE_WHEN_RE = re.compile(
    r"\bпить\s+или\s+(?:хранить|ждать|подождать)\b|\b(?:еще|уже)\s+(?:пить|поздно)\b|"
    r"\bкогда\s+(?:его\s+|лучше\s+|же\s+)*(?:открыть|открывать|пить|выпить)\b"
)
#: Открытая бутылка — тема справочника, а не срок хранения закрытой: «сколько живёт открытое».
_OPEN_BOTTLE_RE = re.compile(r"\bоткрыт(?:ая|ое|ую|ой|ом|ых|ые|ый|ия|ии)\b")
#: Вопрос о самом вине карточки: тогда факт карточки сильнее общей темы справочника.
_THIS_WINE_RE = re.compile(
    r"\b(?:оно|у\s+него|в\s+нем|его|это\s+вино|этого\s+вина|этом\s+вине|у\s+этого)\b"
)
#: Группы блюд словом: «а к мясу?», «полегче к рыбе» — чипы наводящих вопросов.
_FOOD_GROUPS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("meat", re.compile(r"\b(?:мяс\w*|мясн\w*|шашлык\w*|барбекю|гриль|стейк\w*)\b")),
    ("fish", re.compile(r"\b(?:рыб\w*|морепродукт\w*|морск\w*)\b")),
    ("cheese", re.compile(r"\b(?:сыр|сыра|сыру|сыром|сыре|сыры|сыров|сырам|сырами|сырн\w*)\b")),
    ("dessert", re.compile(r"\b(?:десерт\w*|сладкому|сладкого|сладост\w*|выпечк\w*)\b")),
)
_FOOD_NONE = ("без блюда", "без еды", "ни к чему", "просто так", "само по себе", "без закуски")
_WANT_NONE = ("как это", "без разницы", "все равно", "любое", "такое же", "не важно", "неважно")
#: После предлога — не блюдо: местоимения, поводы, само вино.
_NOT_DISH = frozenset(
    [
        "чему",
        "нему",
        "ней",
        "ним",
        "нему",
        "нему",
        "этому",
        "этой",
        "тому",
        "той",
        "тем",
        "чем",
        "что",
        "вину",
        "вину",
        "вине",
        "нему",
        "столу",
        "ужину",
        "обеду",
        "завтраку",
        "празднику",
        "новому",
        "году",
        "юбилею",
        "свадьбе",
        "дню",
        "вечеру",
        "вечеринке",
        "пикнику",
        "закуске",
        "еде",
        "блюду",
        "блюдам",
        "друзьям",
        "гостям",
        "мне",
        "нам",
        "вам",
        "тебе",
        "вам",
        "себе",
        "ним",
        "сомелье",
        "красному",
        "белому",
        "розовому",
        "игристому",
        "сухому",
        "сладкому",
        "брюту",
        "вина",
        "вином",
        "нем",
        "мясу",
        "рыбе",
        "сыру",
    ]
)


def _asks_serve(text: str) -> bool:
    return contains_any(text, _SERVE) or _SERVE_RE.search(text) is not None


def _fact_of(text: str, *, general: bool) -> str | None:
    """Факт карточки, о котором спрашивают: сахар, крепость или срок хранения.

    `general` — вопрос попал и в тему справочника («экстра брют — это совсем без сахара?»,
    «сколько градусов в игристом»): тогда сахар и крепость — факт карточки, только если вопрос о
    самом вине («оно», «у него», «это вино»); так же с «что такое сухое вино». Открытая бутылка
    — тема справочника.
    """
    this_wine = _THIS_WINE_RE.search(text) is not None
    about_wine = this_wine or not (general or contains_any(text, _ASK_WHAT))
    if _SUGAR_ASK_RE.search(text) and about_wine:
        return "sugar"
    if _ALCOHOL_ASK_RE.search(text) and about_wine:
        return "alcohol"
    if _OPEN_BOTTLE_RE.search(text):
        return None
    if _STORAGE_WHEN_RE.search(text):
        return "storage"
    if _STORAGE_WORD_RE.search(text) and _DURATION_RE.search(text):
        return "storage"
    return None


def _want_of(text: str) -> str | None:
    if _NOT_WANT.search(text):
        return None
    if _SOFTER_RE.search(text):
        return "softer"
    if _FRESHER_RE.search(text):
        return "fresher"
    return None


def _food_of(text: str) -> str | None:
    for food, pattern in _FOOD_GROUPS:
        if pattern.search(text):
            return food
    if contains_any(text, _FOOD_NONE):
        return "none"
    return None


#: Слова «вино», «вина» есть почти в каждом вопросе: тема, у которой триггеры только они
#: («Что такое вино»), — общая, как `broad` в справочнике «Лозы».
_WINE_WORDS = frozenset({"вино", "вина"})
#: Слова, которые ничего не сужают (`_EMPTY_WORDS` «Лозы», `recommend/knowledge.py`): любое
#: другое слово рядом с «вином» делает вопрос вопросом о частном — «что такое оранжевое вино»,
#: «графин для вина вообще для чего нужен», — и общая тема на него не отвечает.
_EMPTY_WORDS = frozenset(
    (
        "что", "это", "такое", "таком", "значит", "означает", "вообще",
        "определение", "а", "и", "же", "ли", "по", "в", "на", "с", "у",
        "мне", "нам", "вы", "ты", "я", "он", "она", "оно", "они",
        "расскажи", "объясни", "скажи", "подскажи", "пожалуйста",
        "простыми", "словами", "коротко", "кратко", "если", "то",
    )
)  # fmt: skip


def _broad(triggers: Iterable[str]) -> bool:
    """Общая тема: все её триггеры — само слово «вино»."""
    stems = {normalize(stem) for stem in triggers}
    return bool(stems) and stems <= _WINE_WORDS


def _narrowing(words: Sequence[str], triggers: Iterable[str], context: Iterable[str]) -> bool:
    """Есть ли в вопросе слово, сужающее предмет, — `_narrowing_words` «Лозы»."""
    stems = [normalize(stem) for stem in (*triggers, *context) if normalize(stem)]
    return any(
        word not in _EMPTY_WORDS and not any(word.startswith(stem) for stem in stems)
        for word in words
    )


def _topic_hits(words: Sequence[str], text: str, stems: Iterable[str]) -> int:
    hits = 0
    for stem in stems:
        stem = normalize(stem)
        if not stem:
            continue
        if " " in stem:
            hits += stem in text
        else:
            hits += any(word.startswith(stem) for word in words)
    return hits


def _unknown_dish(words: Sequence[str]) -> bool:
    """«А к хинкали?» — предлог и слово, которое не блюдо справочника и не само вино."""
    for index, word in enumerate(words[:-1]):
        if word not in ("к", "под", "с", "со"):
            continue
        follower = words[index + 1]
        if follower in _NOT_DISH or len(follower) < 3:
            continue
        if find_grapes(follower) or any(words_match(follower, g) for g in _GRAPE_WORDS_RU):
            continue  # «к рислингу» — о вине, а не о блюде
        return True
    return False


#: Первые слова русских написаний сортов: «рислингу», «саперави», «каберне» в падежах.
_GRAPE_WORDS_RU = frozenset(
    word
    for variants in GRAPE_SYNONYMS.values()
    for variant in variants
    for word in normalize(variant).split()[:1]
    if len(word) >= 4 and word.isalpha() and not word.isascii()
)


# ------------------------------------------------------------------ маршрут
@dataclass
class Router:
    """Маршрут на данных сомелье: блюда, темы и портреты сортов из `SommData`."""

    data: Any
    #: Смысловой слой барьера (`SafetyLayer`) или `None` — отказы только по правилам.
    safety: Any = None
    dish_index: DishIndex = field(init=False)

    def __post_init__(self) -> None:
        self.dish_index = DishIndex(self.data.dishes)

    # -------------------------------------------------------------- чип
    def chip(self, chip_id: str, args: Mapping[str, str], context: Context) -> Route:
        """Чип сразу даёт намерение; липкий отказ контекста сильнее."""
        if context.refusal_topic:
            return Route("refuse", refusal=barrier.refusal(context.refusal_topic), chip=chip_id)
        if chip_id == "guided":
            food = args.get("food")
            want = args.get("want") if food else None
            return Route("guided", food=food, want=want, chip=chip_id)
        if chip_id in ("softer", "fresher"):
            return Route(chip_id, dish=args.get("dish") or context.dish, want=chip_id, chip=chip_id)
        return Route(
            chip_id,
            dish=args.get("dish"),
            grape=args.get("grape"),
            topic=args.get("topic"),
            chip=chip_id,
        )

    # -------------------------------------------------------------- текст
    def text(self, question: str, context: Context, *, grapes: Sequence[str] = ()) -> Route:
        """Вопрос текстом → намерение. `grapes` — коды сортов вина карточки."""
        route = self._text(question, context, grapes)
        return Route(
            route.intent,
            dish=route.dish,
            grape=route.grape,
            topic=route.topic,
            food=route.food,
            want=route.want,
            refusal=route.refusal,
            smalltalk=route.smalltalk,
            unknown_dish=route.unknown_dish,
            source="text",
            fact=route.fact,
        )

    def _text(self, question: str, context: Context, grapes: Sequence[str]) -> Route:
        if context.refusal_topic:
            return Route("refuse", refusal=barrier.refusal(context.refusal_topic))
        kind = smalltalk.detect(question)
        if kind is not None:
            return Route("smalltalk", smalltalk=kind)
        match = self.dish_index.find(question)
        dish = match.dish
        refusal = barrier.check(question, dish=dish is not None or match.variant)
        if refusal is not None:
            return Route("refuse", refusal=refusal)
        if self.safety is not None:
            # Односторонний слой: только добавляет отказ там, где правила промолчали.
            topic = self.safety.refusal_topic(question)
            if topic is not None:
                return Route("refuse", refusal=barrier.refusal(topic))
        text = normalize(question)
        words = text.split()

        # Ответ на наводящий вопрос прошлого шага: «к мясу», «помягче», «без разницы».
        if context.step == "food":
            food = self._food(dish, text)
            if food is not None:
                return Route("guided", food=food)
        if context.step == "want" and context.food:
            want = _want_of(text) or ("none" if contains_any(text, _WANT_NONE) else None)
            if want is not None:
                return Route("guided", food=context.food, want=want)

        want = _want_of(text)
        if want is not None:
            if dish is not None:
                return Route(want, dish=dish, want=want)
            food = _food_of(text)
            if food is not None and food != "none":
                return Route("guided", food=food, want=want)
            if context.food and context.intent == "guided":
                return Route("guided", food=context.food, want=want)
            return Route(want, dish=context.dish, want=want)

        if dish is not None:
            return Route("dish_check", dish=dish)
        if match.variant:
            # «К пасте с грибами»: блюдо названо, но в правилах другое — паста с томатным соусом.
            return Route("unknown", unknown_dish=True)

        topic = self._topic(words, text)
        asks_what = contains_any(text, _ASK_WHAT)
        if topic == "decanting" and contains_any(text, _SERVE_NEED):
            return Route("serve")
        if topic == "serve_temp" or (_asks_serve(text) and not asks_what):
            return Route("serve")
        if topic is not None and asks_what:
            return Route("term", topic=topic)

        grape = self._grape(text, grapes)
        if grape is not None:
            return Route("grape", grape=grape)
        if contains_any(text, _REPLACE):
            return Route("replace")
        food = _food_of(text)
        if contains_any(text, _GUIDED):
            return Route("guided", food=food) if food is not None else Route("guided")
        if _asks_serve(text):
            return Route("serve")
        if contains_any(text, _WHAT_TO_EAT) or _WHAT_TO_EAT_RE.search(text):
            return Route("what_to_eat")
        # Факт карточки — после еды: «чем заедать сухое белое» — о еде, а не о сахаре.
        fact = _fact_of(text, general=topic is not None)
        if fact is not None:
            sweeter = fact == "sugar" and _SWEETER_RE.search(text) is not None
            return Route("fact", fact=fact, want="sweeter" if sweeter else None)
        near_food = food is not None and food != "none"
        if near_food and not {"к", "под", "с", "со"}.isdisjoint(words):
            return Route("guided", food=food)
        if topic is not None:
            return Route("term", topic=topic)
        if _unknown_dish(words):
            return Route("unknown", unknown_dish=True)
        return Route("unknown")

    # -------------------------------------------------------------- слоты
    def _food(self, dish: str | None, text: str) -> str | None:
        """Группа блюд шага `food`: названное блюдо даёт свою группу («к устрицам» — рыба)."""
        if dish is not None:
            return self.data.dishes[dish].food or "none"
        return _food_of(text)

    def _topic(self, words: Sequence[str], text: str) -> str | None:
        """Тема справочника — как `KnowledgeBase.find_in_text` «Лозы»: триггер и контекст.

        Счёт — два очка за триггер и очко за слово контекста. Тема без списка контекста —
        узкая: её триггер решает сам («ЗГУ», «графин»), поэтому он засчитывается и за контекст,
        а при равном счёте узкая тема сильнее той, которой нужен контекст: «что значит ЗГУ на
        этикетке» — про ЗГУ, а не про этикетку вообще. Общая тема («Что такое вино») не
        отвечает на вопрос о частном (`_narrowing`): в полном справочнике на 32 темы слово
        «вино» иначе уводило к ней «что такое оранжевое вино» и «графин для вина».
        """
        best: str | None = None
        best_key = (0, False)
        for topic in self.data.topics.values():
            trigger = _topic_hits(words, text, topic.triggers)
            if not trigger:
                continue
            context = _topic_hits(words, text, topic.context)
            if topic.context and not context:
                continue
            if _broad(topic.triggers) and _narrowing(words, topic.triggers, topic.context):
                continue
            key = (trigger * 2 + (context if topic.context else 1), not topic.context)
            if key > best_key:
                best, best_key = topic.id, key
        return best

    def _grape(self, text: str, grapes: Sequence[str]) -> str | None:
        """Портрет сорта: названный в вопросе с «что за / расскажи про» или «о сорте» вина."""
        portraits = self.data.grapes
        named = [code for code in find_grapes(text) if code in portraits]
        asks = contains_any(text, _ASK_WHAT) or contains_any(text, _GRAPE_WORDS)
        if named and asks:
            return named[0]
        asks_sort = "сорт" in text.split() and contains_any(text, ("какой", "какие", "что за"))
        if asks_sort or contains_any(
            text, ("о сорте", "про сорт", "что за сорт", "какой сорт", "расскажи о сорте")
        ):
            own = [code for code in grapes if code in portraits]
            return own[0] if own else None
        return None
