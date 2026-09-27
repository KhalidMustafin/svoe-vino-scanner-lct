"""Проверки живого текста сомелье: первая непройденная — и наружу уходит шаблон.

Разделение труда жёсткое: движки считают, модель говорит. Запрет добавлять факты держится не
на просьбе в промпте, а на проверках после генерации (договор, §6.5). Проверки идут по порядку
`GUARDS`, `check` возвращает код первой непройденной или `None`:

| код | что ловит |
|---|---|
| `think` | ответ был только рассуждением: `<think>` без текста после, незакрытый тег |
| `empty` | после вырезки рассуждения пусто |
| `length` | короче 20 или длиннее 700 символов: обрубок или сочинение |
| `cutoff` | упёрся в `num_predict` (`done_reason == "length"`) или фраза без точки в конце |
| `preamble` | мета-преамбула в начале: «Конечно», «Вот пересказ», «Как сомелье», «Хорошо.»… |
| `dish_as_wine` | блюдо ответа объявлено вином: «Борщ — это красное вино» |
| `false_refusal` | «к сожалению», «извините», «не могу» при блюдах или винах в ответе |
| `numbers` | число, которого нет в пакете, цифрами или словами; «процент», «рубль» |
| `entities` | винодельня, сорт, блюдо, регион, способ готовки не из пакета (`EntityLock`) |
| `descriptors` | ароматы и «ноты», которых нет в пакете |
| `stoplist` | оценки, «хит», медали, призывы, цены, покупка, здоровье, `%` не в «% об.» |
| `legal` | `content_filter.check` (38-ФЗ: польза, покупка, цена, руль, дозы) |

Проверки `sommelier.py` «Лозы» (`:44-139`, `:198-267`) перенесены с тремя отличиями. Первое:
замок сверяет только с пакетом фактов — вопроса гостя модель не видит, и его слова ничего не
оправдывают. Второе: нарушение фильтра контента бракует текст целиком, а не вычёркивает фразу
(`sanitize` пишет фразу в журнал, а текст модели журналу не отдаётся). Третье: новые стоп-лист,
замок дескрипторов и цифровой замок с десятичными числами и числами прописью от «два».

Ни одна проверка не пишет в журнал текст модели или его отрывки: только код.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Collection, Mapping, Sequence
from typing import Any, Protocol

from app.recommend import content_filter
from app.sommelier.entity_lock import EntityLock

GUARDS: tuple[str, ...] = (
    "think",
    "empty",
    "length",
    "cutoff",
    "preamble",
    "dish_as_wine",
    "false_refusal",
    "numbers",
    "entities",
    "descriptors",
    "stoplist",
    "legal",
)

#: Рамки вменяемого ответа: короче — обрубок, длиннее — сочинение (верх снижен под 160 токенов).
MIN_LENGTH = 20
MAX_LENGTH = 700
#: Чем может кончаться законченная фраза.
_SENTENCE_END = ".!?…»)"


class FactsLike(Protocol):
    """Что проверкам и голосу нужно от пакета фактов (`app.sommelier.answers.FactsPackage`)."""

    @property
    def public(self) -> Mapping[str, Any]: ...
    @property
    def verdict_template(self) -> str: ...
    @property
    def voiced(self) -> bool: ...
    @property
    def intent_label(self) -> str: ...
    @property
    def voice_input(self) -> Sequence[str]: ...
    @property
    def allowed_entities(self) -> Collection[str]: ...
    @property
    def allowed_numbers(self) -> Collection[str]: ...
    @property
    def allowed_text(self) -> str: ...


# ---------------------------------------------------------------------- рассуждения
#: Незакрытый тег (обрыв по лимиту токенов) вычищается до конца текста.
_THINK_RE = re.compile(r"<think>.*?(?:</think>|\Z)", re.DOTALL | re.IGNORECASE)
_THINK_CLOSE = "</think>"
_SPACES_RE = re.compile(r"\s+")


def strip_think(raw: str) -> str:
    """Текст без рассуждений: `<think>…</think>`, незакрытый `<think>` и всё до лишнего `</think>`.

    Шаблон чата qwen3 иногда открывает рассуждение сам, и в ответ приходит только закрывающий
    тег: всё до него — рассуждение, а не ответ.
    """
    text = _THINK_RE.sub("", raw or "")
    lowered = text.lower()
    if _THINK_CLOSE in lowered:
        text = text[lowered.rfind(_THINK_CLOSE) + len(_THINK_CLOSE) :]
    return text.strip()


def clean(raw: str) -> str:
    """Текст для показа: без рассуждений, пробелы схлопнуты."""
    return _SPACES_RE.sub(" ", strip_think(raw)).strip()


# ---------------------------------------------------------------------- преамбулы и отказы
#: Мета-преамбулы: модель рассказывает о задаче или обращается к автору промпта, а не к гостю.
#: Список «Лозы» (`sommelier.py:98-105`) плюс коды договора и формулы «по фактам».
_META_MARKERS = (
    "конечно", "вот пересказ", "пересказ", "переформулир", "разговорн", "своими словами",
    "вот текст", "готовый ответ", "готовом ответе", "в вашем", "в предоставленн",
    "как сомелье", "как ии", "как искусственн", "как языков", "намерени", "факты:",
    "по фактам", "согласно фактам", "ответ:", "с удовольствием", "давайте разбер",
    "разумеется", "отвечаю", "мой ответ", "итак",
)  # fmt: skip
#: Междометие-отклик в самом начале: «Хорошо.», «Ладно,», «Понял:» — ответ автору промпта.
#: Только с знаком после слова: «Хорошо сочетается с борщом» — уже ответ гостю.
_OPENER_RE = re.compile(r"^(?:хорошо|ладно|окей|ок|понял|поняла|принято)\s*[,.!:;—–-]")

#: Слова сожаления и отказа (`sommelier.py:88-94` «Лозы»): при блюдах или винах в ответе они
#: переворачивают смысл — гость читает отказ и не смотрит на подборку.
_REFUSAL_MARKERS = (
    "к сожалению", "увы", "не нашлось", "не нашел", "нет подходящ", "ничего не под",
    "не могу", "не удалось подобрать", "извините", "прошу прощения",
)  # fmt: skip


def _plain(text: str) -> str:
    """NFKC, нижний регистр, «ё» → «е»: форма для словарных проверок."""
    return unicodedata.normalize("NFKC", text or "").lower().replace("ё", "е")


# ---------------------------------------------------------------------- блюдо не вино
#: Связка «это», без которой фраза не утверждение о предмете: «Борщу подойдёт красное вино» —
#: правда, «Борщ — это красное вино» — ложь (`sommelier.py:111` «Лозы»).
_IS_WINE_TAIL = r"\s*(?:[—–:]|-\s|это)\s*(?:это\s+)?(?:[а-яе]+\s+){0,2}вин[оа]\b"


def calls_dish_a_wine(text: str, dish_names: Sequence[str]) -> bool:
    """Объявила ли модель блюдо этого ответа вином (`sommelier.py:114-135` «Лозы»).

    Первое слово блюда — только в именительном падеже, как в имени. У «Лозы» после него шёл
    любой хвост (`борщ\\w*`), и честное «к борщу это вино подходит» считалось ложью: «борщу»
    + «это вино». Остальные слова имени склоняются как угодно.
    """
    lowered = _plain(text)
    for name in dish_names:
        words = _plain(name).split()
        if not words or len(words[0]) < 4:
            continue
        first = r"(?<!\w)" + re.escape(words[0])
        heads = [first + "".join(r"\s+" + re.escape(word) + r"\w*" for word in words[1:])]
        if len(words) > 1 and len(words[0]) >= 5:
            heads.append(first)
        if any(re.search(head + r"(?!\w)" + _IS_WINE_TAIL, lowered) for head in heads):
            return True
    return False


# ---------------------------------------------------------------------- числа
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")

#: Вторая часть сложного слова с числом: «пятилетняя выдержка», «двухчасовая аэрация».
_COMPOUND = r"(?:лет|месяч|недел|днев|час|минут|градус|кратн)\w*"

#: Числа прописью от «два» (договор: «один» и «одно» — обычная речь). Ключ — само число: «три»
#: и «трёх» — одно и то же, и «3» в пакете оправдывает «три» в тексте. У «-надцати» и десятков
#: ключ тоже точный: «16–18 °C» в пакете оправдывает «шестнадцати–восемнадцати градусах».
#: Границы — только буквы: дефис числа не прячет, «два-три часа» — два выдуманных числа.
_NUMBER_WORDS: tuple[tuple[str, str], ...] = (
    ("2", rf"дв(?:а|е|ух|ум|умя|ое|оих)|двух?{_COMPOUND}"),
    ("3", rf"тр(?:и|ех|ем|емя|ое|оих)|трех{_COMPOUND}"),
    ("4", rf"четыр(?:е|ех|ем|ьмя)|четвер(?:о|ых)|четырех{_COMPOUND}"),
    ("5", rf"пят(?:ь|и|ью|еро)|пяти{_COMPOUND}"),
    ("6", rf"шест(?:ь|и|ью|еро)|шести{_COMPOUND}"),
    ("7", rf"сем(?:ь|и|ью|еро)|семи{_COMPOUND}"),
    ("8", rf"восем(?:ь|и|ью)|восьм(?:и|ью)|восьми{_COMPOUND}"),
    ("9", rf"девят(?:ь|и|ью)|девяти{_COMPOUND}"),
    ("10", rf"десят(?:ь|и|ью|ок|ка)|десяти{_COMPOUND}"),
    ("11", r"одиннадцат\w*"),
    ("12", r"двенадцат\w*"),
    ("13", r"тринадцат\w*"),
    ("14", r"четырнадцат\w*"),
    ("15", r"пятнадцат\w*"),
    ("16", r"шестнадцат\w*"),
    ("17", r"семнадцат\w*"),
    ("18", r"восемнадцат\w*"),
    ("19", r"девятнадцат\w*"),
    ("20", r"двадцат\w*"),
    ("30", r"тридцат\w*"),
    ("40", rf"сорока?|сорока{_COMPOUND}|сороков\w*"),
    ("50", r"пят(?:ьдесят|идесят)\w*"),
    ("60", r"шест(?:ьдесят|идесят)\w*"),
    ("70", r"сем(?:ьдесят|идесят)\w*"),
    ("80", r"восем(?:ьдесят)\w*|восьмидесят\w*"),
    ("90", r"девяност\w*"),
    ("100", rf"сто|ста|сотн\w*|сто{_COMPOUND}"),
    (
        "200-900",
        (
            r"(?:двест|двухсот|трист|трехсот|четырест|четырехсот|пят[ьи]сот|шест[ьи]сот"
            r"|сем[ьи]сот|восем[ьи]сот|восьмисот|девят[ьи]сот)\w*"
        ),
    ),
    ("1000", r"тысяч\w*"),
    ("1e6", r"миллион\w*|миллиард\w*"),
    ("1,5", r"полтор\w*"),
)
_NUMBER_WORD_RES = tuple(
    (key, re.compile(rf"(?<!\w)(?:{pattern})(?!\w)")) for key, pattern in _NUMBER_WORDS
)
#: Денежные и процентные слова — всегда брак: в пакете их нет и быть не может.
#: Рубль — только словоформы «рубля»: «рубленые котлеты» и «порубленная зелень» — не деньги.
_RUBLE = r"рубл(?:ь|я|ю|ем|е|и|ей|ям|ями|ях)\b"
_MONEY_WORDS_RE = re.compile(rf"процент\w*|\b{_RUBLE}")


def _numbers(text: str) -> set[str]:
    return {match.replace(".", ",") for match in _NUMBER_RE.findall(text or "")}


def _number_words(text: str) -> set[str]:
    plain = _plain(text)
    return {key for key, pattern in _NUMBER_WORD_RES if pattern.search(plain)}


def invented_numbers(text: str, trusted_numbers: Collection[str], trusted_text: str) -> bool:
    """Есть ли в тексте число не из пакета: цифрами, прописью или денежным словом."""
    allowed = {number for value in trusted_numbers for number in _numbers(str(value))}
    allowed |= _numbers(trusted_text)
    if _numbers(text) - allowed:
        return True
    if _MONEY_WORDS_RE.search(_plain(text)):
        return True
    allowed_words = _number_words(trusted_text) | {
        key for key, _ in _NUMBER_WORDS if key in allowed
    }
    return bool(_number_words(text) - allowed_words)


# ---------------------------------------------------------------------- стоп-лист
#: Невидимые символы, которыми прячут слова от фильтров.
_INVISIBLE_RE = re.compile(r"[\u00ad\u200b\u200c\u200d\u2060\ufeff]")
#: Латинские двойники кириллических букв — как у `content_filter`.
_HOMOGLYPHS = str.maketrans("aoecxypkbmtnAOECXYPKBMTH", "аоесхурквмтпАОЕСХУРКВМТН")

#: Оценки, превосходные степени, призывы, цены и покупка (договор, §6.5, и 38-ФЗ). Проверяются
#: после NFKC, вычистки невидимых символов и схлопывания латинских двойников в кириллицу.
_STOPLIST: tuple[tuple[str, str], ...] = (
    # превосходные степени и оценки; «лучше (всего) подавать / охлаждать» и «подавать (их)
    # лучше» с температурой следом («при…», «до…», «охлаждённым») — совет подачи, а не оценка
    # вина (замер 25.09, договор §6.5). Без температуры это выбор: «к рыбе лучше подавать
    # Рислинг» — то же, что «лучше выбрать»; «лучше подаваемых» — сравнение.
    (
        "лучш",
        (
            r"лучш(?!е\s+(?:всего\s+)?(?:подавать|охлаждать)(?:\s+(?:его|ее|их))?"
            r"(?:\s+(?:при|до)\b|\s+охлажд))"
            r"(?!(?<=подавать\sлучш)е\s+(?:при\b|охлажд))"
            r"(?!(?<=подавать\sих\sлучш)е\s+(?:при\b|охлажд))"
        ),
    ),
    ("идеальн", r"идеальн"),
    ("превосходн", r"превосходн"),
    ("шедевр", r"шедевр"),
    ("безупречн", r"безупречн"),
    ("непревзойд", r"непревзойд"),
    ("уникальн", r"уникальн"),
    ("вино недели", r"вин\w*\s+недели"),
    ("номер один", r"номер\s+(?:один|1)\b"),
    ("рейтинг", r"рейтинг"),
    ("балл", r"\bбалл"),
    ("звезд", r"\bзвезд"),
    # «то же самое», «тот же самый», «те же самые» — не превосходная степень; «это же самое
    # лёгкое», «уже самый свежий» — она
    (
        "самый",
        (
            r"(?<!\bт[оаеу]\sже\s)(?<!\bт(?:от|ой|ем|ех)\sже\s)(?<!\bт(?:ого|ому|еми)\sже\s)"
            r"\bсам(?:ый|ая|ое|ые|ого|ой|ому|ым|ых|ую|ыми)\b"
        ),
    ),
    ("-ейший", r"\w{2,}(?<!ближ)(?<!дальн)(?<!мал)(?:ейш|айш)\w*"),
    # «наименее», но не «наименование» (ЗНМП в справке — «наименование места происхождения»)
    ("наиболее", r"\bнаибол|\bнаимене|\bвысш|\bвне\s+конкуренц|\bнесравн"),
    (
        "эпитет",
        (
            r"отличн(?!\w*\s+от(?!\w))|прекрасн|великолепн|восхитительн|изумительн|потрясающ"
            r"|роскошн|выдающ|шикарн|чудесн|волшебн|бесподобн|эксклюзивн|элитн|премиальн|легендарн"
            r"|первоклассн|отменн|блестящ|\bтопов|изыскан|божествен|безукоризн|\bклассн"
            r"|фантастич|невероятн|феноменальн"
            r"|\bсовершенн(?:ый|ая|ое|ые|ого|ой|ому|ым|ых|ую|ыми)\b"
        ),
    ),
    (
        "популярность",
        (
            r"\bхит(?:а|ом|ы|ов)?\b|бестселлер|фаворит|популярн|знаменит|\bмедал|\bнаград"
            r"|\bпризер|\bпобедител|\bлюбят\s+все\b|\bвсе\s+любят\b"
        ),
    ),
    # цены и покупка
    ("скидк", r"скидк|распродаж|промокод"),
    ("акци", r"\bакци(?:я|и|ю|ей|ях|ями)\b"),
    ("купи", r"\bкупи|\bзакупит|\bзакупайте\b|\bприобре[тс]"),
    ("покуп", r"покуп"),
    ("закаж", r"закаж|\bзаказ"),
    ("цена", r"\bцен(?:а|ы|у|е|ой|ам|ами|ах)\b"),
    ("стоимост", r"стоимост"),
    (
        "дешев",
        r"дешев|недорог|\bдорог(?:ой|ая|ое|ие|ого|ую|им|их|ими|о)?\b|\bдороже\b|бюджетн|выгодн",
    ),
    ("магазин", r"магазин|доставк|супермаркет|гипермаркет|винотек|маркетплейс|\bв\s+продаже\b"),
    ("руб", rf"\bруб\b|\b{_RUBLE}"),
    ("₽", r"₽"),
    ("%", r"%(?!\s*об\b)"),
    # призывы
    ("пейте", r"\w*пейте\b"),
    ("попробуйте", r"\bпопроб\w*|\bпробуйте\b"),
    ("наслажд", r"наслажд"),
    ("обязательно", r"\bобязательн"),
    (
        "возьмите",
        (
            r"\bвозьмите\b|\bберите\b|\bвыбирайте\b|\bналейте\b|\bпобалуйте\b|\bугостите\b"
            r"|\bотведайте\b|\bопробуйте\b"
        ),
    ),
    (
        "советую",
        (
            r"\bсовету(?:ю|ем)\b|\bрекоменду(?:ю|ем)\b|\bне\s+пожалеете\b"
            r"|\bоткройте\s+для\s+себя\b"
        ),
    ),
    # здоровье и вред: промпт голоса о них молчать велит, а фильтр закона ловит только пары
    # «вино + польза», и «без вреда» или «легче для желудка» проходили бы его
    ("здоровье", r"\bвред|\bбезвред|\bздоров|\bорганизм|пищеварен|желудк|похмел|опьян"),
)
_STOPLIST_RES = tuple((code, re.compile(pattern)) for code, pattern in _STOPLIST)


def _folded(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = _INVISIBLE_RE.sub("", text)
    return text.translate(_HOMOGLYPHS).lower().replace("ё", "е")


def stoplist_hits(text: str) -> list[str]:
    """Коды стоп-листа, найденные в тексте."""
    folded = _folded(text)
    return [code for code, pattern in _STOPLIST_RES if pattern.search(folded)]


# ---------------------------------------------------------------------- все проверки
def _dish_names(facts: FactsLike) -> list[str]:
    names = []
    for dish in facts.public.get("dishes") or ():
        if isinstance(dish, Mapping) and dish.get("name"):
            names.append(str(dish["name"]))
    return names


def _answer_is_not_empty(facts: FactsLike) -> bool:
    return bool(facts.public.get("dishes") or facts.public.get("wines"))


def trusted_text(facts: FactsLike) -> str:
    """Всё, что модель видела из пакета, одной строкой: факты, шаблон и общий текст пакета."""
    return "\n".join([facts.allowed_text, facts.verdict_template, *facts.voice_input])


def check(
    text: str,
    facts: FactsLike,
    lock: EntityLock,
    *,
    done_reason: str | None = None,
    thinking_only: bool = False,
) -> str | None:
    """Код первой непройденной проверки `GUARDS` или `None` — текст можно показать.

    `text` — сырой `content` модели. `done_reason` и `thinking_only` — из ответа Ollama: первое
    ловит обрыв по `num_predict`, второе — ответ, целиком ушедший в рассуждение.
    """
    raw = text or ""
    body = clean(raw)
    had_think = "<think" in raw.lower() or _THINK_CLOSE in raw.lower()
    if thinking_only or (had_think and not body) or "<think" in body.lower():
        return "think"
    if not body:
        return "empty"
    if not MIN_LENGTH <= len(body) <= MAX_LENGTH:
        return "length"
    if done_reason == "length" or body[-1] not in _SENTENCE_END:
        return "cutoff"
    head = _plain(body[:80])
    if any(marker in head for marker in _META_MARKERS) or _OPENER_RE.match(head):
        return "preamble"
    if calls_dish_a_wine(body, _dish_names(facts)):
        return "dish_as_wine"
    plain = _plain(body)
    if _answer_is_not_empty(facts) and any(marker in plain for marker in _REFUSAL_MARKERS):
        return "false_refusal"
    trusted = trusted_text(facts)
    if invented_numbers(body, facts.allowed_numbers, trusted):
        return "numbers"
    if lock.violations(body, [trusted, *facts.allowed_entities]):
        return "entities"
    if lock.descriptors(body, trusted):
        return "descriptors"
    if stoplist_hits(body):
        return "stoplist"
    if content_filter.check(body).violations:
        return "legal"
    return None
