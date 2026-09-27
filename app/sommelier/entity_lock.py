"""Замок сущностей и дескрипторов: модель не вправе назвать то, чего нет в пакете фактов.

Перенос `Code/backend/app/llm/entity_lock.py` «Лозы». Логика — как есть: бракуется только
пересечение «слово похоже на имя из словаря, но к этому ответу отношения не имеет». Проверять
принадлежность каталогу бесполезно: «краб» в словаре есть, и выдуманный «салат из краба» прошёл
бы такую проверку насквозь. Родовые слова («мясо», «рыба», «салат») и вкусовые слова
(«свежая», «холодное», «плотное») не бракуются: модель вправе обобщать. Способы готовки ловят
дописанное к честному блюду — «сёмга» превращается в «сёмгу на гриле», и рекомендация
становится другой.

Отличия от «Лозы»:

- **словарь — только `vocab.json`** (договор, §7.4): винодельни и сорта выгрузки организатора,
  регионы, 82 блюда «Лозы» с алиасами, способы готовки. Реестр «Лозы» (её каталог, стили, блюда
  портала) сюда не идёт. Из виноделен, как и у «Лозы», берутся только однословные имена и
  отличительные слова (`winery_words`): «Хорошая компания» рассыпается на обиходные слова;
- **родовые и вкусовые слова** — списки «Лозы» плюс `vocab.generic` и `vocab.taste_stems`,
  а к вкусовым основам добавлены цвета вина («розовое» — стиль, а не роза) и «горький»,
  «острый», «жирный»: они сидят в именах блюд («горький шоколад»), но описывают вкус;
- **регионы** — новые: «вино из Крыма» к кубанскому вину — выдумка. Слова вроде «долина» и
  «зона» из имён регионов под подозрение не попадают;
- **латиница** — новая: слово с латинскими буквами, которого нет в пакете, чужое всегда, даже без
  словаря. Так ловятся имена, которых в словаре нет («Lefkadia»), и имена словаря с латинскими
  двойниками букв («Мeрло»), которые сравнение словоформ не узнаёт;
- **замок дескрипторов** — новый (`descriptors`): ароматические слова `vocab.descriptors` и
  слова дегустационной заметки («ноты», «оттенок», «послевкусие», «букет»), которых нет в
  пакете. Дескрипторы приоров — это сорт, а не это вино, и в модель они не идут; значит, и из
  модели выйти не могут. «Ноты вишни» ловятся в любой словоформе: «вишнёвые», «вишни». Слово
  заметки рядом со словом пакета в том же предложении — пересказ правила: «дополняет морские
  ноты» при «Минеральность к морскому» не брак (замер 25.09). Сосед-аромат («ноты шоколада» при
  блюде «Шоколад») пересказом не делает.

Сравнение словоформ и нормализация — общие с разбором вопроса (`app/sommelier/text.py`:
`normalize`, `words_match` и закрытый список окончаний `ENDINGS` — перенос `core/text_match.py` и
`domain/taxonomy.py:normalize` «Лозы»): замок и маршрут сверяют слова одними правилами.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from app.sommelier.text import ENDINGS, normalize, words_match

__all__ = ["EntityLock", "PlainVocab", "VocabLike", "normalize", "words_match"]

logger = logging.getLogger(__name__)

#: Короче четырёх букв словоформы не различаются, да и предлоги с союзами сюда не должны попасть.
_MIN_WORD = 4

#: Родовые слова «Лозы»: модель вправе обобщить подборку, не называя каждое блюдо. Падежи
#: коротких слов дописаны: словоформы сравниваются по общему началу, и «рыба» с «рыбе» так не
#: сводятся.
_GENERIC = frozenset(
    (
        "блюдо", "блюда", "еда", "закуска", "закуски", "перекус", "ужин", "обед", "стол",
        "трапеза", "вино", "вина", "винишко", "напиток", "бокал", "бутылка", "мясо", "мясу",
        "мяса", "рыба", "рыбе", "рыбу", "рыбы", "птица", "птице", "дичь", "морепродукты",
        "овощи", "овощ", "овощам", "фрукты", "фруктам", "ягоды", "ягодам", "грибы", "грибам",
        "зелень", "салат", "салаты", "салату", "суп", "супы", "супу", "гарнир", "соус",
        "соусу", "специи", "пряности", "сыр", "сыру", "сыра", "сыры", "десерт", "десерту",
        "сладкое", "выпечка", "хлеб", "хлебу", "каша", "каше", "пара", "пару", "часа",
        "часов", "бокал", "бокала", "бокалов", "красное", "белое", "розовое", "игристое", "сухое",
        "полусухое", "сладкое", "полусладкое",
    )
)  # fmt: skip
#: Родовые слова точной формы. «Соль» и «виноград» — слова вкуса и сорта («уравновешивает соль
#: блюда», «сорт винограда»), а не блюда «Лосось слабой соли» и «Виноград» (замер 25.09,
#: договор §6.5). Сверяются словом целиком, а не словоформой: «к винограду» и «с виноградом» —
#: блюдо, и замок его ловит.
_GENERIC_EXACT = frozenset(("соль", "соли", "солью", "виноград", "винограда"))

#: Способы приготовления «Лозы»: модель охотно дописывает их к честному блюду.
_COOKING = frozenset(
    (
        "гриль", "гриле", "гриля", "жареный", "жареная", "жареное", "жаренный",
        "копчёный", "копченый", "копчёная", "копченая",
        "запечённый", "запеченный", "запечённая", "запеченная",
        "варёный", "вареный", "варёная", "вареная", "тушёный", "тушеный", "тушёная", "тушеная",
        "маринованный", "маринованная", "солёный", "соленый", "солёная", "соленая",
        "вяленый", "вяленая", "панировке", "кляре", "фритюре", "мангале", "углях", "вертеле",
    )
)  # fmt: skip

#: «На пару» — способ приготовления, а одинокое «пару» — это «пару бокалов» и «за пару часов».
_COOKING_PAIRS: tuple[tuple[str, str], ...] = (("на", "пару"),)

#: Вкусовые основы «Лозы» и добавки сканера (цвета вина и вкусы из имён блюд). Сверяются
#: точным словом «основа + окончание»: «холодное» — про подачу, «холодец» — блюдо.
_TASTE_STEMS = (
    "свеж", "холодн", "дубов", "лёгк", "легк", "плотн", "мягк", "ярк",
    "спел", "зрел", "молод", "тёпл", "тепл", "сочн", "терпк", "кисл",
    # добавки сканера
    "горьк", "остр", "жирн", "красн", "бел", "розов", "черн", "тёмн", "темн", "оранжев",
    "янтарн", "золотист", "соломенн",
)  # fmt: skip
_TASTE_ENDINGS = (
    "ый", "ий", "ое", "ее", "ая", "яя", "ые", "ие", "ым", "им", "ом", "ем",
    "ой", "ей", "ого", "его", "ых", "их", "ому", "ему", "ую", "юю", "ыми", "ими",
)  # fmt: skip

#: Слова имён регионов, которые сами по себе не имя: «долина», «зона», «северная».
_REGION_GENERIC = frozenset(
    (
        "долина", "долины", "зона", "зоны", "северная", "южная", "западная", "восточная",
        "центральная", "нижняя", "верхняя", "берег", "край", "область", "республика", "район",
        "полуостров",
    )
)  # fmt: skip

#: Слова дегустационной заметки: в пакете фактов их нет, значит, модель описывает вкус сама.
_NOTE_WORDS = ("нота", "нотка", "оттенок", "послевкусие", "букет")
#: Сосед слова заметки не короче стольких букв, совпавший со словом пакета, делает его
#: пересказом правила: «морские ноты» при «Минеральность к морскому». «Ноты вина» — нет.
_NOTE_NEIGHBOUR = 5
#: Граница предложения: сосед слова заметки ищется только в своём предложении.
_SENTENCE_END_RE = re.compile(r"[.!?…\n]+")

#: Словообразовательные суффиксы после основы дескриптора: «вишня» — «вишнёвый», «мёд» —
#: «медовый», «дым» — «дымный», «ваниль» — «ванильный», «кофе» — «кофейный», «кожа» —
#: «кожаный», «хвоя» — «хвойный». Хвост без такого суффикса и не окончание — другое слово:
#: «сливаются» не слива, «мангал» не манго.
_DESCRIPTOR_SUFFIXES = ("ов", "ев", "н", "ьн", "ан", "ян", "ок", "оч", "ич", "ист", "ейн", "йн")
#: Сколько букв может идти за основой дескриптора: «вишн» + «евыми».
_DESCRIPTOR_TAIL = 6


class VocabLike(Protocol):
    """Словарь замка (`vocab.json`, договор §7.4) — `Vocab` загрузчика данных или его подобие."""

    @property
    def wineries(self) -> Collection[str]: ...
    @property
    def winery_words(self) -> Collection[str]: ...
    @property
    def grapes(self) -> Collection[str]: ...
    @property
    def dishes(self) -> Collection[str]: ...
    @property
    def cooking(self) -> Collection[str]: ...
    @property
    def regions(self) -> Collection[str]: ...
    @property
    def descriptors(self) -> Collection[str]: ...
    @property
    def generic(self) -> Collection[str]: ...
    @property
    def taste_stems(self) -> Collection[str]: ...


@dataclass(frozen=True, slots=True)
class PlainVocab:
    """Словарь из разобранного `vocab.json` — для зонда и тестов без загрузчика данных."""

    wineries: frozenset[str] = frozenset()
    winery_words: frozenset[str] = frozenset()
    grapes: frozenset[str] = frozenset()
    dishes: frozenset[str] = frozenset()
    cooking: frozenset[str] = frozenset()
    regions: frozenset[str] = frozenset()
    descriptors: frozenset[str] = frozenset()
    generic: frozenset[str] = frozenset()
    taste_stems: frozenset[str] = frozenset()

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> PlainVocab:
        def words(key: str) -> frozenset[str]:
            value = data.get(key) or ()
            if not isinstance(value, list | tuple | set | frozenset):
                raise TypeError(f"vocab.{key}: ожидается список строк")
            return frozenset(str(item) for item in value if str(item).strip())

        return cls(**{name: words(name) for name in cls.__dataclass_fields__})


# ---------------------------------------------------------------------- слова
#: Латинская буква, в том числе с диакритикой («Château»).
_LATIN_RE = re.compile(r"[a-zÀ-ɏ]")


def _words(text: str) -> list[str]:
    return normalize(text or "").split()


#: Ключ корзины — первые три буквы. У «Лозы» было четыре, но `words_match` сводит слова из
#: четырёх букв по трём первым («гусь» — «гусю»), и такие пары попадали в разные корзины: честное
#: «к гусю» при «Гусь запечённый» в пакете уходило в брак.
_BUCKET = 3


def _index(words: Iterable[str]) -> dict[str, list[str]]:
    """Слова по первым буквам: перебирать весь словарь дорого."""
    buckets: dict[str, list[str]] = {}
    for word in words:
        if len(word) >= _MIN_WORD:
            buckets.setdefault(word[:_BUCKET], []).append(word)
    return buckets


def _matches(word: str, buckets: Mapping[str, list[str]]) -> bool:
    return any(words_match(word, candidate) for candidate in buckets.get(word[:_BUCKET], ()))


# ---------------------------------------------------------------------- дескрипторы
@dataclass(frozen=True, slots=True)
class _Descriptor:
    label: str
    base: str
    stems: tuple[str, ...]


def _descriptor(label: str) -> _Descriptor | None:
    words = _words(label)
    if len(words) != 1:
        return None  # дескриптор — одно слово; «чёрная смородина» ловится по «смородине»
    base = words[0]
    stem = base[:-1] if base[-1] in "аяоеиыйьу" and len(base) > 3 else base
    stems = {stem}
    # Чередования основ: «яблоко» — «яблочный», «перец» — «перца», «перечный».
    if stem.endswith("к") and len(stem) >= 4:
        stems.add(stem[:-1] + "ч")
    if stem.endswith("ец"):
        stems.update({stem[:-2] + "ц", stem[:-2] + "еч"})
    if stem.endswith("ок"):
        stems.update({stem[:-2] + "к", stem[:-2] + "оч"})
    if stem.endswith("ц") and not stem.endswith("ец"):
        stems.add(stem[:-1] + "ч")
    return _Descriptor(label=label, base=base, stems=tuple(sorted(stems)))


def _is_descriptor_form(word: str, descriptor: _Descriptor) -> bool:
    """Словоформа или прилагательное от дескриптора: «вишни», «вишнёвые», «перца», «перечный»."""
    if words_match(word, descriptor.base):
        return True
    for stem in descriptor.stems:
        if not word.startswith(stem):
            continue
        tail = word[len(stem) :]
        if tail in ENDINGS or (
            len(tail) <= _DESCRIPTOR_TAIL and tail.startswith(_DESCRIPTOR_SUFFIXES)
        ):
            return True
    return False


def _note_retells(
    text: str, descriptor: _Descriptor, allowed: Iterable[str], aromas: Sequence[_Descriptor]
) -> bool:
    """У каждого вхождения слова заметки сосед слева или справа — слово пакета, а не аромат.

    Сосед — из того же предложения и не короче `_NOTE_NEIGHBOUR` букв, слово пакета — не короче
    `_MIN_WORD`, сверка словоформ — `words_match`: «морские ноты» при «Минеральность к
    морскому» — пересказ. «…Рислинг. Ноты вина» — нет: «Рислинг» в другом предложении. «Ноты
    шоколада» при блюде «Шоколад» — тоже нет: сосед — аромат, это описание вкуса вина.
    """
    trusted = [word for word in allowed if len(word) >= _MIN_WORD]

    def retold(neighbour: str) -> bool:
        return (
            len(neighbour) >= _NOTE_NEIGHBOUR
            and not any(_is_descriptor_form(neighbour, aroma) for aroma in aromas)
            and any(words_match(neighbour, other) for other in trusted)
        )

    for sentence in _SENTENCE_END_RE.split(text or ""):
        words = _words(sentence)
        for position, word in enumerate(words):
            if not _is_descriptor_form(word, descriptor):
                continue
            neighbours = [words[i] for i in (position - 1, position + 1) if 0 <= i < len(words)]
            if not any(retold(neighbour) for neighbour in neighbours):
                return False
    return True


# ---------------------------------------------------------------------- замок
class EntityLock:
    """Словарь подозрений по `vocab.json` и две проверки: сущности и дескрипторы."""

    def __init__(self, vocab: VocabLike) -> None:
        generic = set(_GENERIC)
        for term in vocab.generic:
            generic.update(_words(term))
        self._generic = _index(generic)
        stems = {*_TASTE_STEMS, *(normalize(stem) for stem in vocab.taste_stems)}
        self._taste = frozenset(
            normalize(stem + ending) for stem in stems if stem for ending in _TASTE_ENDINGS
        )

        suspects: set[str] = set()
        for name in vocab.wineries:
            words = _words(name)
            if len(words) == 1:
                suspects.update(words)
        for term in vocab.winery_words:
            suspects.update(_words(term))
        for term in (*vocab.grapes, *vocab.dishes):
            suspects.update(_words(term))
        for term in vocab.regions:
            suspects.update(word for word in _words(term) if word not in _REGION_GENERIC)
        pairs = set(_COOKING_PAIRS)
        cooking = {normalize(word) for word in _COOKING}
        for term in vocab.cooking:
            words = _words(term)
            long_words = [word for word in words if len(word) >= _MIN_WORD and word not in generic]
            if long_words:
                cooking.update(long_words)
            elif len(words) == 2:
                pairs.add((words[0], words[1]))
        # Без словаря замка нет вовсе: иначе в индексе остались бы одни способы готовки, и он
        # браковал бы по ним в одиночку (как у «Лозы» при пустых справочниках).
        if suspects:
            suspects |= cooking
        suspects -= generic
        self._suspects = _index(suspects)
        self.size = len(suspects)
        self._pairs = tuple(sorted(pairs)) if suspects else ()
        if not suspects:
            logger.warning("замок сущностей без словаря: vocab.json пуст — имена не сверяются")

        descriptors = [_descriptor(label) for label in (*vocab.descriptors, *_NOTE_WORDS)]
        self._descriptors = tuple(
            sorted({d for d in descriptors if d is not None}, key=lambda d: d.base)
        )
        self._aromas = tuple(d for d in self._descriptors if d.label not in _NOTE_WORDS)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> EntityLock:
        """Замок по разобранному `vocab.json`."""
        return cls(PlainVocab.from_mapping(data))

    def violations(self, text: str, allowed: Collection[str]) -> list[str]:
        """Слова текста, похожие на имена из словаря, но чужие этому ответу (по алфавиту).

        Плюс любое слово с латинскими буквами, которого нет в пакете. Пакет — русский текст, а
        латиницей в нём пишутся только имена вин и виноделен выгрузки. Значит, «Lefkadia» или
        «Merlot» вне пакета — имя, которое модель принесла сама, есть оно в словаре или нет, а
        «Мeрло» с латинской «e» — имя из словаря, спрятанное от сравнения словоформ.
        """
        allowed_words: set[str] = set()
        for term in allowed:
            allowed_words.update(_words(term))
        words = _words(text)
        foreign: set[str] = {
            word
            for word in words
            if len(word) >= 2 and _LATIN_RE.search(word) and word not in allowed_words
        }
        if not self._suspects:
            return sorted(foreign)
        allowed_index = _index(allowed_words)
        for word in words:
            if len(word) < _MIN_WORD or word in self._taste or word in _GENERIC_EXACT:
                continue
            if _matches(word, self._generic) or _matches(word, allowed_index):
                continue
            if _matches(word, self._suspects):
                foreign.add(word)
        for first, second in self._pairs:
            for position in range(len(words) - 1):
                if (words[position], words[position + 1]) == (first, second) and (
                    second not in allowed_words
                ):
                    foreign.add(f"{first} {second}")
        return sorted(foreign)

    def descriptors(self, text: str, allowed_text: str) -> list[str]:
        """Ароматические слова текста, которых нет в пакете фактов (подписи словаря).

        Слово заметки («ноты», «оттенок»…) не брак, если у каждого его вхождения сосед в том же
        предложении — слово пакета и не аромат (`_note_retells`): так модель пересказывает
        правило, а не описывает вкус.
        """
        # Цвет вина — не аромат: «розовое» не роза, «чёрная» не черника.
        words = {word for word in _words(text) if word not in self._taste}
        if not words:
            return []
        allowed_words = set(_words(allowed_text))
        found: set[str] = set()
        for descriptor in self._descriptors:
            if not any(_is_descriptor_form(word, descriptor) for word in words):
                continue
            if any(_is_descriptor_form(word, descriptor) for word in allowed_words):
                continue
            if descriptor.label in _NOTE_WORDS and _note_retells(
                text, descriptor, allowed_words, self._aromas
            ):
                continue
            found.add(descriptor.label)
        return sorted(found)
