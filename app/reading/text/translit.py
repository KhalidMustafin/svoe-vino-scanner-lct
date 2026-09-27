"""Практическая транскрипция латиницы винных этикеток в кириллицу.

Российские винодельни печатают этикетки латиницей: Chateau Tamagne — это
«Шато Тамань», а не иностранное вино. Каталог при этом хранит названия
кириллицей, и без транскрипции снимок настоящей бутылки не находил ничего —
хуже того, слово «chateau» цеплялось за иностранные стили, и своё вино
объявлялось импортом.

Здесь не научная транслитерация, а транскрипция винной номенклатуры:
она французско-немецкая по духу (ch — «ш», gn — «нь», немое конечное
«et» — «е»), потому что такова традиция названий на этикетках. Точность
до буквы не требуется: сопоставление дальше идёт по общим началам слов,
и «шардонне» находит «шардоне». Важно попасть в начало слова.

Таблица правил намеренно плоская и видимая целиком: каждое правило —
строка, которую можно проверить тестом и объяснить жюри.

Перенесено из «Лозы» как есть (`transcribe`, `transcribe_gost`,
`transcribe_variants`, `_strip_diacritics`). Добавлены обратное направление
(`romanize`, как в slug каталога) и общий скелет двух алфавитов (`skeleton`).
"""

from __future__ import annotations

import re
import unicodedata

from app.reading.text.normalize import fold_homoglyphs, norm

#: Токен, целиком состоящий из базовой латиницы.
LATIN_RE = re.compile(r"[a-z]+")

#: Правила конца слова — применяются первыми, пока хвост ещё на месте.
#: Немые французские окончания: Cabernet — «каберне», Tamagne — «тамань»,
#: Merlot — «мерло», Blanc — «блан». Латинизированные русские хвосты:
#: Fanagoria — «фанагория».
_FINAL_RULES = (
    ("gnes", "нь"),
    ("gne", "нь"),
    ("et", "е"),
    ("ots", "о"),
    ("ot", "о"),
    ("ncs", "н"),
    ("nc", "н"),
    ("ia", "ия"),
)

#: Многобуквенные сочетания — раньше однобуквенных, длинные раньше коротких.
_DIGRAPHS = (
    ("tsch", "ч"),
    ("shch", "щ"),
    ("eaux", "о"),
    ("sch", "ш"),
    ("tch", "ч"),
    ("eau", "о"),
    ("ch", "ш"),
    ("sh", "ш"),
    ("zh", "ж"),
    ("kh", "х"),
    ("ph", "ф"),
    ("th", "т"),
    ("qu", "к"),
    ("gn", "нь"),
    ("ck", "к"),
    ("ts", "ц"),
    ("au", "о"),
    ("ou", "у"),
    ("ie", "и"),
    ("ay", "е"),
    ("ey", "е"),
)

#: Одиночные буквы. «h» вне сочетаний на этикетках нема (Syrah — «сира»).
_SINGLE = {
    "a": "а",
    "b": "б",
    "c": "к",
    "d": "д",
    "e": "е",
    "f": "ф",
    "g": "г",
    "h": "",
    "i": "и",
    "j": "ж",
    "k": "к",
    "l": "л",
    "m": "м",
    "n": "н",
    "o": "о",
    "p": "п",
    "q": "к",
    "r": "р",
    "s": "с",
    "t": "т",
    "u": "у",
    "v": "в",
    "w": "в",
    "x": "кс",
    "y": "и",
    "z": "з",
}

#: «c» читается как «с» перед мягкими гласными: Franciacorta — «франчакорта»
#: было бы точнее, но «с» даёт то же начало слова, а правило — проще.
_SOFT_AFTER_C = frozenset("eiy")


def transcribe(token: str, *, soft_u: bool = False, gost_au: bool = False) -> str:
    """Кириллическое прочтение латинского токена.

    Токен должен быть уже нормализован: нижний регистр, без диакритики
    и пунктуации. Не-латинские строки возвращаются как есть.

    :param soft_u: читать «u» как «ю» (французское Durso — «дюрсо»).
    :param gost_au: читать «au» как «ау» (гостовская транслитерация
        русских имён: Abrau — «абрау», а не французское «абро»).
    """
    if not LATIN_RE.fullmatch(token):
        return token

    for suffix, replacement in _FINAL_RULES:
        if token.endswith(suffix) and len(token) > len(suffix) + 1:
            token = token[: -len(suffix)] + "\0" + replacement
            break

    result = []
    i = 0
    while i < len(token):
        if token[i] == "\0":
            # Уже переведённый хвост копируется без повторной обработки.
            result.append(token[i + 1 :])
            break
        matched = False
        for digraph, replacement in _DIGRAPHS:
            if token.startswith(digraph, i):
                if digraph == "au" and gost_au:
                    replacement = "ау"
                result.append(replacement)
                i += len(digraph)
                matched = True
                break
        if matched:
            continue
        char = token[i]
        if char == "c" and i + 1 < len(token) and token[i + 1] in _SOFT_AFTER_C:
            result.append("с")
        elif char == "u" and soft_u:
            result.append("ю")
        else:
            result.append(_SINGLE.get(char, char))
        i += 1

    return "".join(result)


#: Диграфы гостовской транслитерации русских имён. Отличаются от
#: французских радикально: ch здесь «ч», y — «ы»/«й».
_GOST_DIGRAPHS = (
    ("shch", "щ"),
    ("tsch", "ч"),
    ("sch", "щ"),
    ("kh", "х"),
    ("zh", "ж"),
    ("ch", "ч"),
    ("sh", "ш"),
    ("ts", "ц"),
    ("ya", "я"),
    ("yu", "ю"),
    ("yo", "ё"),
    ("ye", "е"),
    ("ck", "к"),
)

_VOWELS = frozenset("aeiou")


def transcribe_gost(token: str) -> str:
    """Гостовское прочтение: Novy Svet — «новы свет», Myskhako — «мысхако».

    Русские имена латиницей — не французские слова: конечное «et» в Svet
    не немое, «y» — это «ы» (или «й» после гласной), «ch» — «ч».
    """
    if not LATIN_RE.fullmatch(token):
        return token
    result = []
    i = 0
    while i < len(token):
        matched = False
        for digraph, replacement in _GOST_DIGRAPHS:
            if token.startswith(digraph, i):
                result.append(replacement)
                i += len(digraph)
                matched = True
                break
        if matched:
            continue
        char = token[i]
        if char == "y":
            result.append("й" if i > 0 and token[i - 1] in _VOWELS else "ы")
        else:
            result.append(_SINGLE.get(char, char))
        i += 1
    return "".join(result)


def _strip_diacritics(token: str) -> str:
    """Снимает диакритику: Satèn — saten. NFKC этого не делает."""
    decomposed = unicodedata.normalize("NFKD", token)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def transcribe_variants(token: str) -> tuple[str, ...]:
    """Все разумные кириллические прочтения токена, без повторов.

    Одного чтения мало: латиница этикеток пишется то по-французски
    (Durso — «дюрсо», Brut — «брют»), то гостовской транслитерацией
    русского имени (Novy Svet — «новы свет», Myskhako — «мысхако»).
    Какое чтение верное — знает только каталог, поэтому предлагаются
    все, а решает совпадение.
    """
    token = _strip_diacritics(token)
    if not LATIN_RE.fullmatch(token):
        return (token,)
    variants = (
        transcribe(token),
        transcribe(token, soft_u=True),
        transcribe(token, gost_au=True),
        transcribe_gost(token),
    )
    return tuple(dict.fromkeys(variants))


# ---------------------------------------------------------------------------
# Обратное направление: кириллица → латиница
# ---------------------------------------------------------------------------

#: Та же таблица, что строит slug каталога (`slug_translit` в `scripts/build_gt_tokens.py`):
#: «Сухое» — suhoe, «Брют» — bryut, «Пино Нуар» — pino nuar.
_ROMANIZE = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "e",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "й": "y",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "h",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "sch",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
}


def romanize(text: str) -> str:
    """Латинское написание кириллицы по правилам slug каталога.

    Не-кириллические символы остаются как есть; регистр — нижний.
    """
    return "".join(_ROMANIZE.get(ch, ch) for ch in text.lower())


# ---------------------------------------------------------------------------
# Скелет: общий ключ кириллицы и латиницы
# ---------------------------------------------------------------------------
#
# Скелет — грубая фонетическая свёртка в строчную латиницу, одна буква на класс
# звуков: «x» — ш/щ/ч (sh, ch, sch), «c» — ц (ts, tz), «z» — з и ж, «h» — х (kh),
# «i» — и/й/ы/y/j. Удвоения схлопываются, «ь» и «ъ» выпадают. Цель — чтобы
# «Шардоне» и «Chardonnay», «Блан де Блан» и «Blanc de Blancs» давали одинаковые
# или отличающиеся на одну правку ключи. Это ключ сопоставления, а не чтение.

_SKELETON_CYR = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "e",
    "ж": "z",
    "з": "z",
    "и": "i",
    "й": "i",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "h",
    "ц": "c",
    "ч": "x",
    "ш": "x",
    "щ": "x",
    "ъ": "",
    "ы": "i",
    "ь": "",
    "э": "e",
    "ю": "iu",
    "я": "ia",
    "і": "i",
    "ї": "i",
    "є": "e",
    "ґ": "g",
}

#: Немые французские хвосты латиницы: Blancs — «блан», Viognier — «вионье».
_SKELETON_LATIN_FINAL = (
    ("gnes", "n"),
    ("eaux", "o"),
    ("gne", "n"),
    ("aux", "o"),
    ("ncs", "n"),
    ("ier", "e"),
    ("ots", "o"),
    ("nc", "n"),
    ("ot", "o"),
)

#: Сочетания латиницы — длинные раньше коротких.
_SKELETON_LATIN_DIGRAPHS = (
    ("shch", "x"),
    ("tsch", "x"),
    ("sch", "x"),
    ("tch", "x"),
    ("ch", "x"),
    ("sh", "x"),
    ("zh", "z"),
    ("kh", "h"),
    ("ph", "f"),
    ("th", "t"),
    ("ck", "k"),
    ("qu", "k"),
    ("ts", "c"),
    ("tz", "c"),
    ("gn", "n"),
    ("oi", "ua"),
)

_SKELETON_LATIN = {
    "a": "a",
    "b": "b",
    "d": "d",
    "e": "e",
    "f": "f",
    "g": "g",
    "i": "i",
    "j": "i",
    "k": "k",
    "l": "l",
    "m": "m",
    "n": "n",
    "o": "o",
    "p": "p",
    "q": "k",
    "r": "r",
    "s": "s",
    "t": "t",
    "u": "u",
    "v": "v",
    "w": "v",
    "x": "ks",
    "y": "i",
    "z": "z",
}

_SKELETON_VOWELS = frozenset("aeiou")
_SKELETON_SOFT_AFTER_C = frozenset("eiy")
#: Гласные сочетания сводятся одинаково для обоих алфавитов: «ай» и «ay» — «e».
_SKELETON_VOWEL_FOLDS = {"eau": "o", "au": "o", "ou": "u", "ai": "e", "ei": "e"}
_SKELETON_VOWEL_RE = re.compile("|".join(_SKELETON_VOWEL_FOLDS))
_SKELETON_DOUBLE_RE = re.compile(r"([a-z])\1+")
_CYRILLIC_CLASS = f"{chr(0x0400)}-{chr(0x052F)}"
_SKELETON_RUN_RE = re.compile(f"[a-z]+|[{_CYRILLIC_CLASS}]+|[^a-z{_CYRILLIC_CLASS}]+")


def _latin_skeleton(word: str) -> str:
    stop, tail = len(word), ""
    for suffix, replacement in _SKELETON_LATIN_FINAL:
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            stop, tail = len(word) - len(suffix), replacement
            break
    out: list[str] = []
    i = 0
    while i < stop:
        for digraph, replacement in _SKELETON_LATIN_DIGRAPHS:
            if word.startswith(digraph, i) and i + len(digraph) <= stop:
                out.append(replacement)
                i += len(digraph)
                break
        else:
            char = word[i]
            if char == "c":
                nxt = word[i + 1] if i + 1 < len(word) else ""
                out.append("s" if nxt in _SKELETON_SOFT_AFTER_C else "k")
            elif char == "h":
                # «h» после согласной нема: Syrah, Rhone, Ghiaccio.
                prev = next((piece[-1] for piece in reversed(out) if piece), "")
                out.append("h" if not prev or prev in _SKELETON_VOWELS else "")
            else:
                out.append(_SKELETON_LATIN.get(char, char))
            i += 1
    return "".join(out) + tail


def _word_skeleton(word: str) -> str:
    pieces = []
    for match in _SKELETON_RUN_RE.finditer(word):
        run = match.group()
        if "a" <= run[0] <= "z":
            pieces.append(_latin_skeleton(run))
        elif 0x0400 <= ord(run[0]) <= 0x052F:
            pieces.append("".join(_SKELETON_CYR.get(ch, ch) for ch in run))
        else:
            pieces.append(run)
    result = _SKELETON_DOUBLE_RE.sub(r"\1", "".join(pieces))
    result = _SKELETON_VOWEL_RE.sub(lambda m: _SKELETON_VOWEL_FOLDS[m.group()], result)
    return _SKELETON_DOUBLE_RE.sub(r"\1", result)


def skeleton(text: str) -> str:
    """Общий скелет кириллицы и латиницы для сопоставления транслитераций.

    «Шардоне» и «Chardonnay» — «xardone», «Блан Де Блан» и «Blanc de Blancs» —
    «blan de blan», «Мускатель» и «Muscatel» — «muskatel». Текст проходит `norm`
    и `fold_homoglyphs` по словам; слова разделяются одним пробелом, цифры
    и знаки внутри чисел остаются как есть.
    """
    words = (_word_skeleton(fold_homoglyphs(word)) for word in norm(text).split())
    return " ".join(word for word in words if word)
