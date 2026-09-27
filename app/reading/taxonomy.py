"""Справочники этикетки: сорта с синонимами, сахар, цвет, серии и винная лексика.

Урезанный перенос `domain/taxonomy.py` «Лозы»: без регионов, типов вина и шкалы
сладости. Поиск идёт по целым словам после `norm_token`, а не подстрокой:
«МУСКАТЕЛЬ» — не «мускат», «PROSECCO» — не «rose».

Названия сортов совпадают и по скелету транслитерации: «Shardone» — это «Шардоне»,
«Pino Nuar» — «Пино Нуар». Сахар, цвет и серии — только словами из списков: у
коротких служебных слов скелеты совпадают случайно («Свет» и «sweet» — оба «svet»).

Все фразы разбираются одним индексом, длинные раньше коротких и без перекрытий:
«Sauvignon Blanc» — сорт, а не белый цвет, «Blanc de Blancs» — серия, «экстра
брют» не даёт заодно «брют», «Мускат белый» — сорт, а не белый цвет.

Цвет и сахар по-русски — прилагательные, и род у них от слова, к которому они стоят:
«вино белое», «мускатель белый», «мадера белая», «вина белые». Поэтому списки хранят все
четыре формы именительного падежа (`adjective_forms`). Формы не среднего рода бывают и
началом имени собственного («Красная Горка», «Белая Львица») — это решает
`app.reading.fields`, а не список.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import chain
from typing import Literal

from app.reading.contracts import Color, SugarClass
from app.reading.text.normalize import norm, norm_token
from app.reading.text.translit import romanize, skeleton

#: Короче этого слово словаря сравнивается только по норме.
SKELETON_MIN_LEN = 4

# ---------------------------------------------------------------------------
# Формы прилагательных
# ---------------------------------------------------------------------------

#: После г, к, х окончание м. р. без ударения — «-ий» («сладкий»), а не «-ый».
_VELAR = frozenset("гкх")
#: После г, к, х, ж, ш, ч, щ во множественном числе пишется «и»: «сухие», «сладкие».
_PLURAL_I = frozenset("гкхжшчщ")


def adjective_forms(masculine: str) -> tuple[str, ...]:
    """Именительный падеж прилагательного во всех родах и числах: ср., м., ж., мн.

    «белый» → белое, белый, белая, белые; «сухой» → сухое, сухой, сухая, сухие;
    «сладкий» → сладкое, сладкий, сладкая, сладкие. Средний род первым: так подписан
    каталог («Белое», «сухое»), и первое написание сахара — подпись карточки. Только
    твёрдая основа: мягкой («синий» → «синее») и «-ий» после шипящих («свежий» → «свежее»)
    в списках этикетки нет, и такое слово — ошибка, а не неверные формы.
    """
    word = masculine.lower().replace("ё", "е")
    stem, ending = word[:-2], word[-2:]
    if not stem or ending not in ("ый", "ой", "ий"):
        raise ValueError(f"adjective_forms: {masculine!r} — не полное прилагательное м. р.")
    if ending == "ий" and stem[-1] not in _VELAR:
        raise ValueError(f"adjective_forms: {masculine!r} — основа не твёрдая")
    plural = "ие" if stem[-1] in _PLURAL_I else "ые"
    return (stem + "ое", word, stem + "ая", stem + plural)


def adjective_spellings(masculine: str) -> tuple[str, ...]:
    """`adjective_forms` и их латинское написание по правилам slug каталога.

    «сухой» → сухое, сухой, сухая, сухие, suhoe, suhoy, suhaya, suhie.
    """
    forms = adjective_forms(masculine)
    return tuple(dict.fromkeys((*forms, *(romanize(form) for form in forms))))


# ---------------------------------------------------------------------------
# Сорта
# ---------------------------------------------------------------------------

#: Код сорта → написания. Первое — подпись как в колонке «Сорт винограда» CSV.
#: Одинокие «каберне» и «пино» не сорт: на этикетке это обрывок «Каберне Фран»,
#: «Каберне АЗОС» или «Пино Нуар», и угадывать нельзя.
GRAPE_SYNONYMS: dict[str, tuple[str, ...]] = {
    # Международные красные
    "cabernet_sauvignon": ("Каберне Совиньон", "Cabernet Sauvignon", "Cab Sauvignon"),
    "cabernet_franc": ("Каберне Фран", "Cabernet Franc"),
    "merlot": ("Мерло", "Merlot"),
    "pinot_noir": ("Пино Нуар", "Pinot Noir", "Пино Черный", "Pinot Nero", "Spatburgunder"),
    "pinot_franc": ("Пино Фран", "Pinot Franc"),
    "meunier": ("Менье", "Пино Менье", "Pinot Meunier", "Meunier"),
    "syrah": ("Сира", "Syrah", "Шираз", "Shiraz"),
    "malbec": ("Мальбек", "Malbec"),
    "petit_verdot": ("Пти Вердо", "Petit Verdot"),
    "carmenere": ("Карменер", "Carmenere"),
    "grenache": ("Гренаш", "Grenache", "Гарнача", "Garnacha"),
    "mourvedre": ("Мурведр", "Mourvedre", "Монастрель", "Monastrell"),
    "cinsault": ("Сенсо", "Cinsault", "Cinsaut"),
    "sangiovese": ("Санджовезе", "Sangiovese"),
    "nebbiolo": ("Неббиоло", "Nebbiolo"),
    "montepulciano": ("Монтепульчано", "Montepulciano"),
    "corvina": ("Корвина", "Corvina"),
    "tempranillo": ("Темпранильо", "Tempranillo"),
    "zinfandel": ("Зинфандель", "Zinfandel", "Примитиво", "Primitivo"),
    "blaufrankisch": ("Блауфранкиш", "Blaufrankisch", "Lemberger", "Kekfrankos"),
    "zweigelt": ("Цвайгельт", "Zweigelt"),
    "marselan": ("Марселан", "Marselan"),
    "areni": ("Арени", "Areni"),
    "cabernet_cortis": ("Каберне Кортис", "Cabernet Cortis"),
    # Международные белые
    "chardonnay": ("Шардоне", "Chardonnay"),
    "sauvignon_blanc": ("Совиньон Блан", "Sauvignon Blanc", "Совиньон", "Sauvignon"),
    "sauvignon_vert": ("Совиньон Зеленый", "Sauvignon Vert", "Sauvignonasse"),
    "riesling": ("Рислинг", "Riesling", "Рислинг Рейнский", "Rhine Riesling"),
    "rieslaner": ("Рисланер", "Rieslaner"),
    "pinot_gris": ("Пино Гри", "Pinot Gris", "Пино Гриджио", "Пино Гриджо", "Pinot Grigio"),
    "pinot_blanc": ("Пино Блан", "Pinot Blanc", "Pinot Bianco"),
    "aligote": ("Алиготе", "Aligote"),
    "viognier": ("Вионье", "Viognier"),
    "chenin_blanc": ("Шенен Блан", "Chenin Blanc"),
    "semillon": ("Семильон", "Semillon"),
    "gewurztraminer": ("Гевюрцтраминер", "Gewurztraminer"),
    "traminer": ("Траминер", "Traminer"),
    "traminer_rose": ("Траминер Розовый", "Traminer Rose", "Roter Traminer"),
    "muller_thurgau": ("Мюллер-Тургау", "Muller-Thurgau", "Rivaner"),
    "silvaner": ("Сильванер", "Silvaner", "Sylvaner"),
    "gruner_veltliner": ("Грюнер", "Грюнер Вельтлинер", "Gruner Veltliner", "Gruner"),
    "vermentino": ("Верментино", "Vermentino"),
    "glera": ("Глера", "Glera"),
    "malvasia": ("Мальвазия", "Malvasia"),
    "colombard": ("Коломбар", "Colombard"),
    "ugni_blanc": ("Уньи Блан", "Ugni Blanc", "Trebbiano"),
    "marsanne": ("Марсан", "Marsanne"),
    "roussanne": ("Русан", "Roussanne"),
    "petit_manseng": ("Пти Мансен", "Petit Manseng"),
    "petite_arvine": ("Петит Арвин", "Пти Арвин", "Petite Arvine"),
    "verdelho": ("Вердельо", "Verdelho"),
    "sercial": ("Серсиаль", "Sercial"),
    "pedro_ximenez": ("Педро Хименес", "Pedro Ximenez"),
    "albillo": ("Альбильо", "Albillo"),
    "aleatiko": ("Алеатико", "Aleatico"),
    "solaris": ("Солярис", "Solaris"),
    # Мускаты: в каталоге это разные сорта, «мускат» внутри них уже занят
    "muscat": (
        "Мускат",
        "Muscat",
        "Moscato",
        "Мускат Белый",
        "Мускат Блан",
        "Мускат Мелкозернистый",
        "Muscat Blanc",
        "Moscato Bianco",
    ),
    "muscat_rose": ("Мускат Розовый", "Muscat Rose", "Moscato Rosa"),
    "muscat_amber": ("Мускат Янтарный",),
    "muscat_hamburg": ("Мускат Гамбургский", "Muscat Hamburg", "Muscat of Hamburg"),
    "muscat_alexandria": ("Мускат Александрийский", "Muscat of Alexandria"),
    "moscato_giallo": ("Мускат Джалло", "Москато Джалло", "Мускат Желтый", "Moscato Giallo"),
    "muscat_ottonel": ("Мускат Оттонель", "Muscat Ottonel", "Оттонель", "Ottonel"),
    # Кавказ, Крым, Дон и отечественная селекция
    "saperavi": ("Саперави", "Saperavi"),
    "saperavi_severny": ("Саперави Северный", "Saperavi Severny"),
    "rkatsiteli": ("Ркацители", "Rkatsiteli"),
    "mtsvane": ("Мцване", "Mtsvane", "Мцване Кахетинский"),
    "kangun": ("Кангун", "Kangun"),
    "tavkveri": ("Тавквери", "Tavkveri"),
    "khikhvi": ("Хихви", "Khikhvi"),
    "krasnostop": ("Красностоп Золотовский", "Красностоп", "Krasnostop Zolotovsky", "Krasnostop"),
    "krasnostop_azos": ("Красностоп Анапский", "Красностоп АЗОС", "Krasnostop AZOS"),
    "tsimlyansky_cherny": ("Цимлянский Черный", "Tsimlyansky Cherny"),
    "sibirkovy": ("Сибирьковый", "Sibirkovy"),
    "puhlyakovsky": ("Пухляковский", "Puhlyakovsky"),
    "plechistik": ("Плечистик", "Plechistik"),
    "kumshatsky": ("Кумшацкий Белый", "Кумшацкий", "Kumshatsky"),
    "levokumsky": ("Левокумский", "Levokumsky"),
    "marinovsky": ("Мариновский", "Marinovsky"),
    "stanichny": ("Станичный", "Stanichny"),
    "kokur": ("Кокур", "Кокур Белый", "Kokur"),
    "kefesia": ("Кефесия", "Kefesia"),
    "sary_pandas": ("Сары Пандас", "Sary Pandas"),
    "ekim_kara": ("Эким Кара", "Ekim Kara"),
    "dzhevat_kara": ("Джеват Кара", "Dzhevat Kara"),
    "bastardo_magarachsky": (
        "Бастардо Магарачский",
        "Бастардо",
        "Bastardo Magarachsky",
        "Bastardo",
    ),
    "citron_magaracha": ("Цитронный Магарача", "Citronny Magaracha"),
    "pervenets_magaracha": ("Первенец Магарача", "Pervenets Magaracha"),
    "podarok_magaracha": ("Подарок Магарача", "Podarok Magaracha"),
    "rubinovy_magaracha": ("Рубиновый Магарача", "Rubinovy Magaracha"),
    "antey_magarachsky": ("Антей Магарачский", "Antey Magarachsky"),
    "cabernet_azos": ("Каберне АЗОС", "Cabernet AZOS"),
    "rubin_azos": ("Рубин АЗОС", "Rubin AZOS"),
    "rubin_golodrigi": ("Рубин Голодриги", "Rubin Golodrigi"),
    "odessky_cherny": ("Одесский Черный", "Odessky Cherny"),
    "golubok": ("Голубок", "Golubok"),
    "dostoyny": ("Достойный", "Dostoyny"),
    "platovsky": ("Платовский", "Platovsky"),
    "bianca": ("Бианка", "Бьянка", "Bianca"),
    "moldova": ("Молдова", "Moldova"),
    "isabella": ("Изабелла", "Isabella"),
    "avgustin": ("Августин", "Avgustin"),
    "amursky_potapenko": ("Амурский Потапенко", "Amursky Potapenko"),
    "bukovinka": ("Буковинка", "Bukovinka"),
    "vostorg": ("Восторг", "Vostorg"),
    "dekabrsky": ("Декабрьский", "Dekabrsky"),
    "fioletovy_ranny": ("Фиолетовый Ранний", "Fioletovy Ranny"),
    "tsvetochny": ("Цветочный", "Tsvetochny"),
    "rebo": ("Ребо", "Rebo"),
    "shabash": ("Шабаш", "Shabash"),
    "bayanshira": ("Баяншира", "Bayanshira"),
    "madrasa": ("Мадраса", "Матраса", "Madrasa", "Matrassa"),
    "risus": ("Рисус", "Risus"),
    "gechei_zamatosh": ("Гечеи Заматош",),
}

# ---------------------------------------------------------------------------
# Сахар, цвет, серии
# ---------------------------------------------------------------------------

#: Сахар и цвет по-русски — прилагательные (м. р.); на этикетке они во всех родах и числах:
#: «Херес сухой», «Мадера сладкая», «Вина белые». Латиницей в списки этикетки идёт только
#: средний род, как в slug каталога («vino beloe suhoe»): «Krasnaya», «Belaya» латиницей —
#: транслит имени собственного («Krasnaya Gorka»), а не цвет вина.
SUGAR_ADJECTIVES: dict[SugarClass, tuple[str, ...]] = {
    SugarClass.DRY: ("сухой",),
    SugarClass.SEMI_DRY: ("полусухой",),
    SugarClass.SEMI_SWEET: ("полусладкий",),
    SugarClass.SWEET: ("сладкий", "десертный"),
}
#: «Чёрный» — не цвет вина: «Чёрный принц» в каталоге белое, «Цимлянский чёрный» бывает
#: розовым. Его, как «зелёный» и «серый», сравнивает словом названия resolve.
COLOR_ADJECTIVES: dict[Color, tuple[str, ...]] = {
    Color.WHITE: ("белый",),
    Color.RED: ("красный",),
    Color.ROSE: ("розовый",),
    Color.ORANGE: ("оранжевый", "янтарный"),
}
#: Цвет ягоды в названиях сортов и позиций, которого нет среди цветов вина: «Мускат
#: чёрный», «Совиньон зелёный», «Пино серый». Полем цвета он не становится.
GRAPE_COLOR_ADJECTIVES: tuple[str, ...] = ("черный", "зеленый", "серый")


def _spellings_of(adjectives: Iterable[str]) -> tuple[str, ...]:
    """Все русские формы и латиница среднего рода: сухое, сухой, сухая, сухие, suhoe."""
    out: list[str] = []
    for adjective in adjectives:
        forms = adjective_forms(adjective)
        out += (*forms, romanize(forms[0]))
    return tuple(out)


#: Формы не среднего рода: они согласуются не с «вином», а со своим словом — «Мускатель
#: белый», но и «Красная Горка», «Белая Львица». Перед словом не из словаря этикетки такое
#: прилагательное — часть имени (`app.reading.fields`).
AGREEING_FORMS: frozenset[str] = frozenset(
    form
    for table in (SUGAR_ADJECTIVES, COLOR_ADJECTIVES)
    for adjectives in table.values()
    for adjective in adjectives
    for form in adjective_forms(adjective)[1:]
)
#: Мужской род единственного числа: так прилагательное согласуется с названием сорта
#: (почти все сорта мужского рода) — «Кокур десертный Сурож», «МУСКАТ ДЕСЕРТНЫЙ». Женский род
#: и множественное число после сорта с ним не согласуются: «Шардоне Красная Горка».
MASCULINE_FORMS: frozenset[str] = frozenset(
    adjective_forms(adjective)[1]
    for table in (SUGAR_ADJECTIVES, COLOR_ADJECTIVES)
    for adjectives in table.values()
    for adjective in adjectives
)


def _with_prefix(prefix: str, adjective: str) -> tuple[str, ...]:
    """«полу-» + формы: «полу-сухое», «полу-сухой» (дефис — отдельное слово фразы)."""
    return tuple(prefix + form for form in adjective_forms(adjective))


#: Класс сахара → написания; классы и «demi-sec = полусладкое» — как в `scripts/build_gt_tokens.py`.
#: Первое написание — подпись карточки (`app.api.cards.sugar_label`): «сухое», «брют».
SUGAR_TERMS: dict[SugarClass, tuple[str, ...]] = {
    SugarClass.BRUT_NATURE: (
        "брют натюр",
        "brut nature",
        "bryut natyur",
        "pas dosé",
        "non dosé",
        "zero dosage",
        "dosage zero",
        "brut zero dosage",
        "брют зеро",
        "зеро дозаж",
        "брют зеро дозаж",
    ),
    SugarClass.EXTRA_BRUT: ("экстра брют", "extra brut", "ekstra bryut"),
    SugarClass.BRUT: ("брют", "brut", "bryut"),
    SugarClass.DRY: (
        *_spellings_of(SUGAR_ADJECTIVES[SugarClass.DRY]),
        "dry",
        "sec",
        "сек",
        "secco",
        "seco",
        "trocken",
    ),
    SugarClass.SEMI_DRY: (
        *_spellings_of(SUGAR_ADJECTIVES[SugarClass.SEMI_DRY]),
        *_with_prefix("полу-", "сухой"),
        "semi-dry",
        "semidry",
        "off-dry",
        "medium dry",
        "semi-secco",
        "semi-seco",
        "halbtrocken",
        "abboccato",
    ),
    SugarClass.SEMI_SWEET: (
        *_spellings_of(SUGAR_ADJECTIVES[SugarClass.SEMI_SWEET]),
        *_with_prefix("полу-", "сладкий"),
        "semi-sweet",
        "semisweet",
        "demi-sec",
        "medium sweet",
        "amabile",
        "lieblich",
    ),
    SugarClass.SWEET: (
        *_spellings_of(SUGAR_ADJECTIVES[SugarClass.SWEET]),
        "sweet",
        "doux",
        "dolce",
        "dulce",
    ),
}

#: Фразы, которые занимают слова сахара, но класса не дают. «Extra Dry» и «Extra Sec» у
#: игристых слаще брюта, и «dry» из них — неверный сахар.
SUGAR_BLOCKERS: tuple[str, ...] = (
    "extra dry",
    "extra sec",
    *_with_prefix("экстра ", "сухой"),
    "экстра драй",
    "экстра сек",
)

#: Цвет → написания. Цвет берётся только из слов, пиксели этикетки — дело CV. Цвет внутри
#: названия сорта отнимает более длинная фраза: «Мускат белый», «Траминер розовый» (вино из
#: него белое), «Sauvignon Blanc».
COLOR_TERMS: dict[Color, tuple[str, ...]] = {
    Color.WHITE: (
        *_spellings_of(COLOR_ADJECTIVES[Color.WHITE]),
        "white",
        "blanc",
        "блан",
        "bianco",
        "blanco",
        "branco",
        "weisswein",
    ),
    Color.RED: (
        *_spellings_of(COLOR_ADJECTIVES[Color.RED]),
        "red",
        "rouge",
        "руж",
        "rosso",
        "tinto",
        "rotwein",
    ),
    Color.ROSE: (
        *_spellings_of(COLOR_ADJECTIVES[Color.ROSE]),
        "rosé",
        "розе",
        "rosato",
        "rosado",
        "pink",
    ),
    Color.ORANGE: (*_spellings_of(COLOR_ADJECTIVES[Color.ORANGE]), "orange", "оранж"),
}

# ---------------------------------------------------------------------------
# Полевые написания (Э1, пункт R1 плана точности 25.09; списки закрыты PREREG)
# ---------------------------------------------------------------------------
# Таблицы ниже идут только в индекс фраз разбора чтения (`READ_TERMS`) — не в `SUGAR_TERMS`,
# `COLOR_TERMS` и `LABEL_TERMS`, из которых строятся признаки названий каталога
# (`app.resolve.features`), атрибуты карточек и подписи: их эти написания не меняют. Словарь
# каталога (`app.reading.lexicon.build`) держит свои таблицы и тоже не меняется. Замер Э1 (25.09):
# опечатки цвета чинят R014 и M455, слова региона — M543; ни один пункт не ломает ни одного
# кадра catalog_v2 и krasnostop, точность сахара и цвета на ooc_v2 не падает.

#: Опечатки OCR цвета с полевых кадров («МУСКАТЕЛЬ / БЕЛЬЙ», «СУКАЕ ВЕЛОЕ»). Это не форма рода:
#: проверки имени собственного у них нет. «белоe» с латинской «e» `fold_homoglyphs` и так
#: сводит к «белое».
COLOR_TYPOS: dict[Color, tuple[str, ...]] = {
    Color.WHITE: ("бельй", "бельи", "велое", "белоe"),
}
#: Сахар на итальянских и французских этикетках: слово «dolce» / «doux» в них — не «сладкое».
#: «Полу сладкое» раздельно уже даёт `_with_prefix("полу-", …)`: слова те же.
SUGAR_VARIANTS: dict[SugarClass, tuple[str, ...]] = {
    SugarClass.SEMI_SWEET: ("semi-dolce", "demi-doux"),
}
#: Фраза с двумя классами сахара. «EXTRA BRUT • ZERO DOSAGE» иначе даёт только brut_nature:
#: «brut zero dosage» длиннее и отнимает «brut» у «extra brut». Каталог же зовёт такие вина
#: «экстра брют», так что прочитанное должно не спорить ни с одним из двух классов.
SUGAR_COMBINED: dict[str, tuple[SugarClass, ...]] = {
    "extra brut zero dosage": (SugarClass.EXTRA_BRUT, SugarClass.BRUT_NATURE),
}
#: Имена, в которых слово цвета — не цвет вина: бренды и топонимы. Фраза занимает слова, как
#: заглушки сахара, и цвета не даёт; кюве у неё слова не отнимает.
COLOR_BLOCKERS: tuple[str, ...] = (
    "красная стрелка",
    "red cat",
    "красная горка",
    "красная поляна",
    "кубань красная",
)

#: Каноническая серия → написания.
SERIAL_TERMS: dict[str, tuple[str, ...]] = {
    "гран резерв": ("гран резерв", "grand reserve", "gran reserva", "gran riserva"),
    "семейный резерв": ("семейный резерв", "family reserve"),
    "резерв": ("резерв", "reserve", "reserva", "riserva"),
    "блан де нуар": ("блан де нуар", "blanc de noirs", "blanc de noir"),
    "блан де блан": ("блан де блан", "blanc de blancs", "blanc de blanc"),
    "кюве": ("кюве", "cuvée"),
}

#: Названия стилей: на этикетке они стоят вместо слова «вино», а цвет сразу после них
#: отличает позицию от соседей по линейке («Портвейн белый», «Мускатель розовый»).
_STYLE_WORDS = (
    "херес мадера портвейн кагор мускатель марсала токай "
    "sherry jerez porto madeira marsala tokaji moscatel muscatel"
)
STYLE_WORDS: frozenset[str] = frozenset(norm_token(word) for word in _STYLE_WORDS.split())
#: Купаж — тоже вино: «белый купаж», «Красный бленд» — цвет, а не имя.
_BLEND_WORDS = "купаж купажное бленд blend ассамбляж assemblage"
BLEND_WORDS: frozenset[str] = frozenset(norm_token(word) for word in _BLEND_WORDS.split())

#: Слова, по которым видно, что это винная этикетка. Цвет и серия сюда не входят:
#: «red» и «reserve» бывают на чём угодно. Названия стилей (`STYLE_WORDS`) входят.
_WINE_WORDS_RU = (
    "вино вина вин винодельня винодельни винодельческое виноградник виноградники виноград "
    "винограда виноградное игристое игристые шампанское шампанских шато урожай урожая згу знмп "
    "петнат"
)
_WINE_WORDS_LATIN = (
    "wine wines winery vineyard vineyards vino vin vins vinho wein weingut spumante frizzante "
    "sparkling champagne cremant prosecco chateau domaine cantina bodega tenuta vintage "
    "millesime millesimato petnat"
)
WINE_WORDS: frozenset[str] = STYLE_WORDS | frozenset(
    norm_token(word) for text in (_WINE_WORDS_RU, _WINE_WORDS_LATIN) for word in text.split()
)

#: Общие слова этикетки: категория продукта, выдержка и обязательная информация. По правилам
#: маркировки вина (ГОСТ, ТР ЕАЭС) они стоят почти на каждой бутылке: «вино столовое сухое»,
#: «с защищённым географическим указанием», «выдержка 18 месяцев в дубовых бочках»,
#: «производитель», «объём», «крепость», «сахар». В названиях каталога такие слова бывают
#: только описанием («Arie. Выдержка сталь»), поэтому кюве по ним не отличить: попадание
#: уводит resolve к случайным позициям. Поле кюве их не берёт, даже если слово есть в словаре.
#: Серии («резерв», «выдержанное» как серия) этот список не трогает. Категории-прилагательные
#: — во всех родах и числах, как цвет и сахар: «Шампанское российское», «Херес креплёный».
_LABEL_GENERIC_RU = (
    "выдержка выдержки выдержкой "
    "урожай урожая месяц месяца месяцев "
    "дуб дуба дубе дубовых дубовой дубовые бочка бочки бочке бочках "
    "вино вина россии защищённым защищённого географическим географического "
    "указанием указания наименованием наименования "
    "производитель изготовитель объём крепость сахар сахара "
    "винодельня винодельни винный дом завод"
)
_LABEL_GENERIC_ADJECTIVES = (
    "выдержанный игристый тихий столовый ординарный марочный коллекционный российский "
    "креплёный ликёрный"
)
LABEL_GENERIC: frozenset[str] = frozenset(
    norm_token(word)
    for word in (
        *_LABEL_GENERIC_RU.split(),
        *(form for lemma in _LABEL_GENERIC_ADJECTIVES.split() for form in adjective_forms(lemma)),
    )
)
#: Слова региона — строка происхождения («КРЫМ», «Долина Дона»), а не название позиции: поле
#: кюве их не берёт, как и общие слова (Э1, R1д; список закрыт PREREG, формы — ровно эти).
REGION_WORDS: frozenset[str] = frozenset(
    norm_token(word)
    for word in (
        "долина", "дона", "долины", "крым", "кубань", "кубани", "тамань", "анапа", "севастополь",
    )
)  # fmt: skip

# ---------------------------------------------------------------------------
# Слова «не имя» для поля кюве (Э6, план точности 26.09; списки закрыты PREREG
# research/2026-09-26_e6/PREREG.md). Идут только в разбор чтения (`app.reading.fields`): признаки
# названий каталога и словарь их не видят. Замер Э6: ни одного сменившегося ответа на
# catalog_v2 и krasnostop, точность сахара и цвета на ooc_v2 та же — это гигиена полей.
# ---------------------------------------------------------------------------
#: Э6-Р: регионы списка R1(д) латиницей и родительный «Крыма»: «THE TASTE OF CRIMEA», «ЗГУ
#: KUBAN», «вина Крыма» — строка происхождения, а не имя позиции.
REGION_WORDS_EXTRA: frozenset[str] = frozenset(
    norm_token(word)
    for word in ("crimea", "krym", "kuban", "taman", "anapa", "sevastopol", "крыма")
)
#: Э6-Ш: слова винной этикетки (`WINE_WORDS`: «виноград», «шато», `chateau`, `spumante`,
#: `millesimato`…) — категория продукта, хозяйство или урожай, а не имя позиции. Словарь каталога
#: одиночными словами «винодельня», «усадьба», `estate` и так не берёт. Слова стиля
#: (`STYLE_WORDS`: «портвейн», «мускатель») кюве остаются: пункт Э6-С замер отклонил — без
#: «мускатель» ломается R014 («МУСКАТЕЛЬ / БЕЛЬЙ»).
NAME_WINE_WORDS: frozenset[str] = WINE_WORDS - STYLE_WORDS

# ---------------------------------------------------------------------------
# Слова строк урожая, объёма и крепости (Э7, план точности 26.09; списки закрыты PREREG
# research/2026-09-26_e7/PREREG.md). Идут только в правило имени собственного
# (`app.reading.fields._starts_proper_name`): прилагательное цвета или сахара перед ними —
# признак вина, а не начало имени («БЕЛЫЙ» / «ГОД УРОЖАЯ», «РОЗОВЫЙ» / «Алк. 12% об.»). Кюве,
# винодельня, словарь и признаки названий каталога их не видят. Годы и числа объёма и крепости
# («2023», «0,75», «12%») — не слова, и имя перед ними не начинается и без таблиц.
# ---------------------------------------------------------------------------
#: Э7-У: строка урожая и возраста — слова задачи, их формы и якоря урожая разбора.
_VINTAGE_LINE_WORDS = (
    "год года году годы лет урожай урожая урожаи выдержка выдержки выдержкой выпуска "
    "vintage millesime millesimato harvest vendemmia annata cosecha jahrgang"
)
VINTAGE_LINE_WORDS: frozenset[str] = frozenset(
    norm_token(word) for word in _VINTAGE_LINE_WORDS.split()
)
#: Э7-О: строка объёма и крепости — якоря крепости разбора и единицы объёма.
_VOLUME_LINE_WORDS = (
    "объем объема литр литра литров л мл ml cl alc alcohol алк алкоголь крепость спирта "
    "спирт этилового abv vol об"
)
VOLUME_LINE_WORDS: frozenset[str] = frozenset(
    norm_token(word) for word in _VOLUME_LINE_WORDS.split()
)

# ---------------------------------------------------------------------------
# Индекс фраз
# ---------------------------------------------------------------------------

TermKind = Literal["grape", "sugar", "color", "serial"]


@dataclass(frozen=True, slots=True)
class Term:
    """Значение фразы словаря. `value=None` — заглушка: слова заняты, значения нет.

    `also` — ещё значения того же вида на тех же словах (`SUGAR_COMBINED`).
    """

    kind: TermKind
    value: str | None  # код сорта, SugarClass, Color или каноническая серия
    also: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PhraseMatch[V]:
    """Фраза словаря на словах `start:end` последовательности."""

    start: int
    end: int
    value: V
    phrase: str  # фраза словаря в нормализованном виде
    exact: int  # сколько слов совпало по норме, а не по скелету


@dataclass(frozen=True, slots=True)
class _Entry[V]:
    words: tuple[str, ...]
    skeletons: tuple[str | None, ...]  # None — слово сравнивается только по норме
    value: V


def phrase_words(text: str) -> tuple[str, ...]:
    """Фраза в том виде, в каком с ней сравниваются токены: слова после `norm_token`."""
    return tuple(word for word in (norm_token(part) for part in norm(text).split()) if word)


def words_of(text: str) -> tuple[list[str], list[str]]:
    """Нормы и скелеты слов свободного текста — вход для `PhraseIndex.find`."""
    words = list(phrase_words(text))
    return words, [skeleton(word) for word in words]


class PhraseIndex[V]:
    """Поиск фраз словаря по последовательности слов: целыми словами, длинные первыми.

    `fuzzy(value)` решает, можно ли фразе совпадать по скелету; без него — всем.
    """

    def __init__(
        self,
        entries: Iterable[tuple[str, V]],
        *,
        skeleton_min_len: int = SKELETON_MIN_LEN,
        fuzzy: Callable[[V], bool] | None = None,
    ) -> None:
        self._phrases: dict[tuple[str, ...], V] = {}
        self._by_norm: dict[str, list[_Entry[V]]] = {}
        self._by_skeleton: dict[str, list[_Entry[V]]] = {}
        self.entries: list[_Entry[V]] = []
        for text, value in entries:
            words = phrase_words(text)
            if not words:
                raise ValueError(f"PhraseIndex: пустая фраза {text!r}")
            if words in self._phrases:
                if self._phrases[words] != value:
                    raise ValueError(f"PhraseIndex: фраза {text!r} ведёт к двум значениям")
                continue
            self._phrases[words] = value
            allow = fuzzy is None or fuzzy(value)
            skeletons = tuple(
                (skeleton(word) or None) if allow and len(word) >= skeleton_min_len else None
                for word in words
            )
            entry = _Entry(words, skeletons, value)
            self.entries.append(entry)
            self._by_norm.setdefault(words[0], []).append(entry)
            if skeletons[0]:
                self._by_skeleton.setdefault(skeletons[0], []).append(entry)

    def lookup(self, text: str) -> V | None:
        """Значение фразы, записанной в словаре ровно так (после нормализации)."""
        return self._phrases.get(phrase_words(text))

    def candidates(
        self, norms: Sequence[str], skeletons: Sequence[str] | None = None
    ) -> list[PhraseMatch[V]]:
        """Все вхождения, в том числе перекрывающиеся."""
        found: list[PhraseMatch[V]] = []
        n = len(norms)
        for i in range(n):
            entries = list(self._by_norm.get(norms[i], ()))
            if skeletons is not None and skeletons[i]:
                entries += [e for e in self._by_skeleton.get(skeletons[i], ()) if e not in entries]
            for entry in entries:
                size = len(entry.words)
                if i + size > n:
                    continue
                exact = 0
                for j, word in enumerate(entry.words):
                    if norms[i + j] == word:
                        exact += 1
                    elif not (
                        skeletons is not None
                        and entry.skeletons[j] is not None
                        and skeletons[i + j] == entry.skeletons[j]
                    ):
                        break
                else:
                    phrase = " ".join(entry.words)
                    found.append(PhraseMatch(i, i + size, entry.value, phrase, exact))
        return found

    def find(
        self, norms: Sequence[str], skeletons: Sequence[str] | None = None
    ) -> list[PhraseMatch[V]]:
        """Вхождения без перекрытий: больше слов, больше точных слов, длиннее фраза."""
        chosen: list[PhraseMatch[V]] = []
        taken: set[int] = set()
        ranked = sorted(
            self.candidates(norms, skeletons),
            key=lambda m: (-(m.end - m.start), -m.exact, -len(m.phrase), m.start),
        )
        for match in ranked:
            span = range(match.start, match.end)
            if taken.isdisjoint(span):
                chosen.append(match)
                taken.update(span)
        return sorted(chosen, key=lambda m: m.start)


def _label_entries() -> Iterable[tuple[str, Term]]:
    for code, variants in GRAPE_SYNONYMS.items():
        for text in (*variants, code.replace("_", " ")):
            yield text, Term("grape", code)
    for sugar, variants in SUGAR_TERMS.items():
        for text in variants:
            yield text, Term("sugar", sugar)
    for text in SUGAR_BLOCKERS:
        yield text, Term("sugar", None)
    for color, variants in COLOR_TERMS.items():
        for text in variants:
            yield text, Term("color", color)
    for canonical, variants in SERIAL_TERMS.items():
        for text in variants:
            yield text, Term("serial", canonical)


#: Общий индекс сортов, сахара, цвета и серий: перекрытия разрешаются между видами.
LABEL_TERMS: PhraseIndex[Term] = PhraseIndex(
    _label_entries(), fuzzy=lambda term: term.kind == "grape"
)


def _read_entries() -> Iterable[tuple[str, Term]]:
    """Полевые написания (Э1, R1): их видит только разбор чтения этикетки."""
    for sugar, variants in SUGAR_VARIANTS.items():
        for text in variants:
            yield text, Term("sugar", sugar)
    for text, (first, *rest) in SUGAR_COMBINED.items():
        yield text, Term("sugar", first, tuple(rest))
    for color, variants in COLOR_TYPOS.items():
        for text in variants:
            yield text, Term("color", color)
    for text in COLOR_BLOCKERS:
        yield text, Term("color", None)


#: Индекс фраз разбора чтения (`app.reading.fields`): `LABEL_TERMS` и полевые написания. Названия
#: каталога (признаки выбора, `app.resolve.features`) и атрибуты карточек разбираются только
#: `LABEL_TERMS`: полевые написания признаков каталога не меняют.
READ_TERMS: PhraseIndex[Term] = PhraseIndex(
    chain(_label_entries(), _read_entries()), fuzzy=lambda term: term.kind == "grape"
)


def _found(text: str, kind: TermKind) -> list[str]:
    norms, skeletons = words_of(text)
    out: list[str] = []
    for match in LABEL_TERMS.find(norms, skeletons):
        value = match.value.value
        if match.value.kind == kind and value is not None and value not in out:
            out.append(value)
    return out


def find_grapes(text: str) -> list[str]:
    """Коды сортов в порядке появления: «Пино чёрный (Пино нуар) и Мерло» → pinot_noir, merlot."""
    return _found(text, "grape")


def find_sugar(text: str) -> list[SugarClass]:
    """Все классы сахара в порядке появления; «экстра брют» не даёт заодно «брют»."""
    return [SugarClass(value) for value in _found(text, "sugar")]


def find_colors(text: str) -> list[Color]:
    """Цвета, названные словом вне названий сортов и серий."""
    return [Color(value) for value in _found(text, "color")]


def find_serial(text: str) -> list[str]:
    """Канонические серии: «Grand Reserve» → «гран резерв»."""
    return _found(text, "serial")


def canonical_grape(name: str) -> str | None:
    """Любое написание сорта или сам код → код; «Рислинг Рейнский» → riesling."""
    if name in GRAPE_SYNONYMS:
        return name
    term = LABEL_TERMS.lookup(name)
    if term is not None and term.kind == "grape":
        return term.value
    found = find_grapes(name)
    return found[0] if found else None


def grape_label(code: str) -> str:
    """Код → подпись как в каталоге."""
    variants = GRAPE_SYNONYMS.get(code)
    return variants[0] if variants else code.replace("_", " ")


def sugar_class(text: str) -> SugarClass | None:
    """Класс сахара по коду («extra_brut») или написанию («Экстра брют»)."""
    if text in SugarClass._value2member_map_:
        return SugarClass(text)
    term = LABEL_TERMS.lookup(text)
    if term is not None and term.kind == "sugar" and term.value is not None:
        return SugarClass(term.value)
    return None


def color_of(text: str) -> Color | None:
    """Цвет по значению каталога («Белое») или написанию («rosé»)."""
    if text in Color._value2member_map_:
        return Color(text)
    term = LABEL_TERMS.lookup(text)
    if term is not None and term.kind == "color" and term.value is not None:
        return Color(term.value)
    return None
