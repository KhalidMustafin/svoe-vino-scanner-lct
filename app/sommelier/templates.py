"""Тексты сомелье: метки, этапы, чипы, шаблоны ответов и вопрос экрана `check`.

Всё, что сомелье говорит без модели, собрано здесь и детерминировано: один и тот же пакет
фактов даёт один и тот же текст. Формулировки — договор `docs/api-sommelier.md` (§1, §2,
§3.3, §4.2, §4.5, §5). Шаблоны не содержат чисел, кроме температуры подачи и крепости
«% об.», и оценочных слов: вердикт — это правило сочетаний или факт карточки, пересказанный
одной фразой. Каждую строку для человека тесты прогоняют через `content_filter.check`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from app.reading.taxonomy import canonical_grape, find_grapes
from app.recommend.catalog import SUGAR_WORDS, RecoWine, number_label
from app.recommend.shelf import not_evaluable_text
from app.sommelier.text import count_wines, join_words, lower_first, upper_first

# ------------------------------------------------------------------ метки (договор, §1)
LABEL_AI = "Текст — ИИ, подбор — алгоритм"
LABEL_ALGO = "Текст и подбор — алгоритм"
NOTICE_149 = "Применяются рекомендательные технологии"
NOTE_BASIS = "по карточке каталога, правилам подачи и сочетаний"

# ------------------------------------------------------------------ основания ответа
BASIS_CARD = "карточка каталога"
BASIS_PAIRS = "правила сочетаний"
BASIS_GRAPE = "профиль по сорту"
BASIS_SERVE = "правила подачи"
BASIS_KNOWLEDGE = "справочник"
BASIS_PICK = "подбор из каталога"

# ------------------------------------------------------------------ этапы (договор, §3.3)
STAGES: dict[str, tuple[str, str]] = {
    "card": ("Открываю карточку «{name}»", "Открыл карточку"),
    "rules": ("Проверяю правила сочетаний", "Проверил правила сочетаний"),
    "catalog": ("Ищу в каталоге вина других виноделен", "Посмотрел каталог"),
    "voice": ("Сомелье формулирует ответ", "Сомелье сформулировал"),
    "verify": ("Сверяю имена и числа", "Сверил имена и числа"),
}


def stage(stage_id: str, name: str = "") -> dict[str, str]:
    """Событие этапа без `t_ms`: `{type, id, text, done}`."""
    text, done = STAGES[stage_id]
    return {"type": "stage", "id": stage_id, "text": text.format(name=name), "done": done}


ERROR_TEXTS = {
    "internal": "Не получилось собрать ответ — повторите вопрос",
    "not_ready": "Сомелье запускается — повторите через минуту",
}
INPUT_OFF = "вопрос текстом выключен"

# ------------------------------------------------------------------ чипы (договор, §4.2)
CHIP_TEXTS: dict[str, str] = {
    "what_to_eat": "К чему подать",
    "serve": "Как подать",
    "softer": "Помягче",
    "fresher": "Посвежее",
    "replace": "Чем заменить",
    "guided": "Помогите выбрать",
}
#: Подписи чипов справочника, которые сомелье ставит сам (у тем из `knowledge.json` — свои).
TERM_CHIP_TEXTS: dict[str, str] = {
    "brut_scale": "Что значит «брют»?",
    "decanting": "Зачем декантировать",
    "tannins": "Что такое танины",
    "serve_temp": "Температура подачи",
}
#: Наводящие вопросы (договор, §4.2).
GUIDED_FOOD_QUESTION = "К чему подбираете вино?"
GUIDED_WANT_QUESTION = "Какое хочется?"
GUIDED_FOODS: tuple[tuple[str, str], ...] = (
    ("meat", "К мясу"),
    ("fish", "К рыбе"),
    ("cheese", "К сыру"),
    ("dessert", "К десерту"),
    ("none", "Без блюда"),
)
GUIDED_WANTS: tuple[tuple[str, str], ...] = (
    ("softer", "Помягче"),
    ("fresher", "Посвежее"),
    ("none", "Как это"),
)
#: Группа блюд в дательном: «К мясу это вино по правилам сочетаний подходит».
FOOD_DATIVE = {"meat": "к мясу", "fish": "к рыбе", "cheese": "к сыру", "dessert": "к десерту"}
WANT_WORDS = {"softer": "помягче", "fresher": "посвежее"}
#: Подборка по причине «−» пары (`answers.REASONS`, проверка 25.09, вечер; третий круг — и после
#: «скорее нет»): начало хвоста оговорки «Если хочется …», заголовок подборки «Полегче к борщу» и
#: слово хвоста «скорее нет» «Вот вина других виноделен полегче, …».
CAVEAT_WORDS: dict[str, tuple[str, str, str]] = {
    "softer": ("мягче", "Помягче", "помягче"),
    "fresher": ("свежее", "Посвежее", "посвежее"),
    "lighter": ("вино полегче", "Полегче", "полегче"),
    "fuller": ("вино помощнее", "Помощнее", "помощнее"),
    "drier": ("суше", "Суше", "суше"),
    "sweeter": ("слаще", "Послаще", "послаще"),
}


def chip(chip_id: str, text: str | None = None, **args: str) -> dict[str, Any]:
    """Чип вопроса: `{id, text[, args]}`; `args` без пустых значений."""
    body: dict[str, Any] = {"id": chip_id, "text": text or CHIP_TEXTS[chip_id]}
    clean = {key: value for key, value in args.items() if value is not None}
    if clean:
        body["args"] = clean
    return body


def dish_chip(dish_id: str, dative: str) -> dict[str, Any]:
    """«А к борщу?» — чип вопроса о блюде."""
    return chip("dish_check", f"А {dative}?", dish=dish_id)


def grape_chip(code: str, label: str) -> dict[str, Any]:
    return chip("grape", f"О сорте {label}", grape=code)


def term_chip(topic_id: str, name: str) -> dict[str, Any]:
    return chip("term", TERM_CHIP_TEXTS.get(topic_id, name), topic=topic_id)


# ------------------------------------------------------------------ заметка карточки (§2)
def style_words(style_label: str, sparkling: bool) -> str:
    """Стиль строчными: «красное сухое»; игристому без слова «игристое» — «игристое белое брют»."""
    style = style_label.strip().lower()
    if sparkling and "игрист" not in style:
        style = f"игристое {style}".strip()
    return style


def _plain_words(text: str) -> str:
    """Для сравнения названий: строчные, «ё» → «е», дефисы и знаки — пробел."""
    return " ".join(re.sub(r"[^\w]+", " ", text.casefold().replace("ё", "е")).split())


def grapes_not_in_name(grapes: Sequence[str], name: str) -> list[str]:
    """Сорта, которых нет в названии: «Саперави — красное сухое. Сорт — Саперави» — повтор.

    Сорт в названии — теми же словами («Саперави», «Каберне Фран») или другим написанием того же
    сорта («Cru Lermont Saperavi», «Cru Lermont Рислинг» при сорте «Рислинг Рейнский»).
    """
    plain = f" {_plain_words(name)} "
    named = set(find_grapes(name))
    out = []
    for grape in grapes:
        if f" {_plain_words(grape)} " in plain:
            continue
        code = canonical_grape(grape)
        if code is not None and code in named:
            continue
        out.append(grape)
    return out


def grapes_sentence(grapes: Sequence[str]) -> str:
    """«Сорт — Саперави.», «Сорта — Рислинг и Шардоне.», «…, В и другие.»; нет — пусто."""
    if not grapes:
        return ""
    if len(grapes) == 1:
        return f"Сорт — {grapes[0]}."
    if len(grapes) <= 3:
        return f"Сорта — {join_words(grapes)}."
    return f"Сорта — {', '.join(grapes[:3])} и другие."


def serve_range(temperature: Sequence[int]) -> str:
    low, high = temperature
    return f"{low}–{high} °C"


def dishes_clause(names: Sequence[str]) -> str:
    """«подходит гусь», «подходят гусь и борщ», «подходят А, Б и В» — названия строчными."""
    names = [lower_first(name) for name in names]
    verb = "подходит" if len(names) == 1 else "подходят"
    return f"{verb} {join_words(names)}"


def note_text(
    *,
    name: str,
    style: str,
    grapes: Sequence[str],
    winery: str,
    region: str,
    temperature: Sequence[int] | None,
    dishes: Sequence[str],
) -> str:
    """Шаблон «Сомелье · коротко» (договор, §2) — только факты выгрузки и наши правила."""
    parts = [f"{name} — {style}." if style else f"{name}."]
    sentence = grapes_sentence(grapes)
    if sentence:
        parts.append(sentence)
    place = ", ".join(part for part in (winery, region) if part)
    if place:
        parts.append(f"{place}.")
    if temperature is not None:
        parts.append(f"Подают при {serve_range(temperature)}.")
    if dishes:
        parts.append(f"По правилам сочетаний к нему {dishes_clause(dishes[:3])}.")
    return " ".join(parts)


# ------------------------------------------------------------------ шаблоны ответов (§4.5)
UNKNOWN = (
    "Вот что я умею: подсказать, к чему подать это вино, как его подать и чем заменить. "
    "Выберите вопрос ниже."
)
UNKNOWN_DISH = "Такого блюда в правилах сочетаний нет. Выберите из списка или спросите иначе."
UNKNOWN_GRAPE = "Об этом сорте в справочнике пока ничего нет. Могу рассказать, как подать это вино."
NO_PAIRS = "Правила сочетаний об этом вине пока ничего не говорят."
NEUTRAL = "Правила сочетаний об этой паре ничего не говорят."
NO_SERVE = "Правила подачи для этого вина не подходят: в карточке нет цвета."
NO_SIMILAR = "Похожих в каталоге не нашлось."


def what_to_eat(names: Sequence[str], first_plus_text: str | None) -> str:
    """«По правилам сочетаний к этому вину подходят А, Б и В. {правило первого блюда}»."""
    if not names:
        return NO_PAIRS
    head = f"По правилам сочетаний к этому вину {dishes_clause(names)}."
    return f"{head} {first_plus_text}" if first_plus_text else head


def dish_verdict(verdict: str, dative: str, rule_text: str | None, tail: str | None) -> str:
    """Вердикт пары: «К борщу — да, с оговоркой. {правило «−»}[ {хвост про вина}]»."""
    head = upper_first(dative)
    if verdict == "yes":
        text = f"{head} — да."
    elif verdict == "caveat":
        text = f"{head} — да, с оговоркой."
    elif verdict == "no":
        text = f"{head} — скорее нет."
    else:
        return NEUTRAL
    if rule_text:
        text = f"{text} {rule_text}"
    return f"{text} {tail}" if tail else text


def sweet_dish_sugar_unknown(dative: str) -> str:
    """Десерт, варенье или мёд к вину без сахара в карточке: правила о сахаре здесь не судят.

    Профиль такого вина считал бы его сухим, и шаблон «скорее нет» говорил бы «сухое вино
    теряет фруктовость» — допущение вместо факта (проверка на видеокарте 25.09).
    """
    return (
        f"{upper_first(dative)} правила сочетаний не подскажут: сахар этого вина в карточке "
        "каталога не указан, а к сладкому решает именно он."
    )


def caveat_tail(want: str, dative: str) -> str:
    """Хвост оговорки о подборке по её причине: «Если хочется вино полегче — вот вина других
    виноделен, которые подходят к борщу.»"""
    return (
        f"Если хочется {CAVEAT_WORDS[want][0]} — вот вина других виноделен, которые подходят "
        f"{dative}."
    )


def no_tail(want: str, dative: str) -> str:
    """Хвост «скорее нет» о подборке по причине: «Вот вина других виноделен помягче, которые
    подходят к устрицам.» — не «если хочется»: без сдвига пара не складывается."""
    return f"Вот вина других виноделен {CAVEAT_WORDS[want][2]}, которые подходят {dative}."


def caveat_title(want: str, dative: str) -> str:
    """Заголовок подборки по причине — и у оговорки, и у «скорее нет»: «Полегче к борщу»,
    «Суше к пицце», «Помягче к устрицам»."""
    return f"{CAVEAT_WORDS[want][1]} {dative}"


def matching_tail(dative: str) -> str:
    return f"Вот вина других виноделен, которые подходят {dative}."


def serve_text(name: str, temperature: Sequence[int], rule_text: str) -> str:
    """«{name} подают при 16–18 °C.» и объяснение правила подачи.

    Правило, которое само называет ту же температуру («Плотные красные подают при 16–18 °C:
    тепло раскрывает аромат…»), её второй раз не пишет: к имени вина идёт только объяснение —
    «Саперави подают при 16–18 °C: тепло раскрывает аромат…».
    """
    head = f"{name} подают при {serve_range(temperature)}"
    said = f"подают при {serve_range(temperature)}"
    if said in rule_text:
        why = rule_text.split(said, 1)[1].strip().lstrip(":—–-,").strip()
        return f"{head}: {why}" if why.rstrip(".") else f"{head}."
    return f"{head}. {rule_text}".strip()


#: Цвет правила подачи во множественном числе: «общая для красных вин».
_COLOR_PLURAL = {
    "Красное": "красных",
    "Белое": "белых",
    "Розовое": "розовых",
    "Оранжевое": "оранжевых",
}


def serve_neutral(color: str | None) -> str:
    """Текст подачи, когда правило сверяло тело, а тело вина неизвестно: без «лёгкие красные».

    Правило `red` / `white` тогда выбрано не по телу, а за его неизвестностью, и его текст про
    лёгкие вина был бы утверждением, которого в данных нет (финальная проверка 24.09).
    """
    group = _COLOR_PLURAL.get(color or "")
    tail = f"общая для {group} вин" if group else "общая для вин этого цвета"
    return f"Тело этого вина по сорту не оценить, поэтому температура — {tail}."


def direction_title(want: str, dative: str | None) -> str:
    """«Помягче», «Помягче к борщу», «Посвежее»."""
    word = upper_first(WANT_WORDS[want])
    return f"{word} {dative}" if dative else word


def direction_text(want: str, dative: str | None, count: int, *, expanded: bool) -> str:
    """«Помягче — три вина других виноделен того же стиля.»"""
    head = direction_title(want, dative)
    where = "из всего каталога" if expanded else "того же стиля"
    return f"{head} — {count_wines(count)} других виноделен {where}."


def direction_empty(want: str, dative: str | None, *, no_difference: bool) -> str:
    """Честная фраза, когда вин нет: как у «Сомелье у полки» (`no_difference`, `fewer_than_three`)."""
    word = WANT_WORDS[want]
    if no_difference:
        return f"По описаниям разницы нет — {word} в каталоге не найти."
    if dative:
        return f"{upper_first(dative)} {word} в каталоге не нашлось."
    return f"{upper_first(word)} в каталоге не нашлось."


def direction_unknown(want: str, wine: RecoWine) -> str:
    """Направление не оценить — у вина нет ни сахара, ни оси «по сорту» (как `not_evaluable`
    «Сомелье у полки»): «Сахар этого вина в каталоге не указан, а по сорту мягкость не оценить —
    помягче подобрать не по чему.» Это не «разницы нет»: сравнивать не с чем."""
    return f"{not_evaluable_text(want, wine)}."


#: Похожие совпали с вином карточки на всех общих осях «по сорту» и из карточки: «роза ветров»
#: наложением ничего бы не показала (`compare.overlay` — `null`).
PROFILE_SAME = "Профиль вкуса по сорту и карточке у них такой же, как у этого вина."


def replace_text(style: str, grape: str | None, *, same_profile: bool = False) -> str:
    """«Похожие из других виноделен: красное сухое, тот же сорт — Саперави.»"""
    tail = f", тот же сорт — {grape}" if grape else ""
    text = f"Похожие из других виноделен: {style}{tail}."
    return f"{text} {PROFILE_SAME}" if same_profile else text


def guided_food_verdict(food: str, verdict: str | None) -> str:
    """«К мясу это вино по правилам сочетаний подходит.» — перед вопросом «Какое хочется?»."""
    head = upper_first(FOOD_DATIVE[food])
    if verdict == "yes":
        return f"{head} это вино по правилам сочетаний подходит."
    if verdict == "caveat":
        return f"{head} это вино по правилам сочетаний подходит с оговоркой."
    if verdict == "no":
        return f"{head} это вино по правилам сочетаний скорее не подходит."
    return f"{head} правила сочетаний об этом вине ничего не говорят."


def guided_title(food: str | None, want: str | None) -> str:
    """«К мясу — помягче», «К мясу», «Помягче», «Вина других виноделен»."""
    group = upper_first(FOOD_DATIVE[food]) if food in FOOD_DATIVE else None
    word = WANT_WORDS.get(want or "")
    if group and word:
        return f"{group} — {word}"
    if group:
        return group
    if word:
        return upper_first(word)
    return "Вина других виноделен"


def guided_text(food: str | None, want: str | None, count: int) -> str:
    """«К мясу — три вина других виноделен, помягче.»"""
    group = upper_first(FOOD_DATIVE[food]) if food in FOOD_DATIVE else None
    word = WANT_WORDS.get(want or "")
    body = f"{count_wines(count)} других виноделен" + (f", {word}" if word else "")
    return f"{group} — {body}." if group else f"{upper_first(body)}."


def guided_empty(food: str | None, want: str | None) -> str:
    what = " ".join(p for p in (FOOD_DATIVE.get(food or ""), WANT_WORDS.get(want or "")) if p)
    if not what:
        return "Вин других виноделен в каталоге не нашлось."
    return f"{upper_first(what)} в каталоге не нашлось."


# ------------------------------------------------------------------ факт карточки (§4.5, `fact`)
SUGAR_UNKNOWN = "Сахар этого вина в карточке каталога не указан."
SWEETER_ALREADY = "Это вино уже сладкое."
SWEETER_NONE = "Подбора «послаще» в листе нет — могу подобрать помягче или посвежее."
SWEETER_UNKNOWN = "Послаще подобрать не по чему."
ALCOHOL_UNKNOWN = "Крепость этого вина в карточке каталога не указана."
#: Крепость словом по шкале WSET: до 11 % об. — невысокая, от 14 — высокая.
ALCOHOL_BANDS: tuple[tuple[float, str], ...] = ((11.0, "невысокая"), (14.0, "средняя"))
STORAGE_HEAD = "Срока хранения в карточке каталога нет."
#: Наше правило хранения по стилю (договор, §4.5): первое подошедшее. Чисел в тексте нет, слов
#: стоп-листа голоса тоже: «после покупки» у 297 игристых ловил наш же стоп-лист (`покуп`).
STORAGE_TEXTS: dict[str, str] = {
    "sparkling": "Игристые дома долго не хранят: их обычно открывают в первые год-два.",
    "sweet": "Сладкие и креплёные вина хранятся дольше сухих: сахар и спирт их берегут.",
    "white": "Белые, розовые и оранжевые вина обычно пьют молодыми — в первые годы после урожая.",
    "red_full": "Плотные красные могут храниться годами: танины и кислотность держат вино.",
    "red": "Лёгкие красные обычно пьют молодыми — в первые годы после урожая.",
    "red_unknown": (
        "Красные с плотным телом хранятся дольше лёгких, а тело этого вина по сорту не оценить."
    ),
    "unknown": "Цвета в карточке нет, поэтому и правила хранения для этого вина не подобрать.",
}


def sugar_fact(style: str, sugar: str | None, *, sweeter: bool) -> str:
    """«По карточке каталога это красное сухое.»; «послаще» — честная фраза вслед."""
    if sugar is None:
        return f"{SUGAR_UNKNOWN} {SWEETER_UNKNOWN}" if sweeter else SUGAR_UNKNOWN
    head = f"По карточке каталога это {style}." if style else f"По карточке каталога это {sugar}."
    if not sweeter:
        return head
    return f"{head} {SWEETER_ALREADY if sugar == 'sladkoe' else SWEETER_NONE}"


def alcohol_fact(value: float | None, high: float | None = None) -> str:
    """«Крепость по карточке каталога — 13,5 % об. Для вина это средняя крепость.»"""
    if value is None:
        return ALCOHOL_UNKNOWN
    shown = number_label(value) + (f"–{number_label(high)}" if high else "")
    band = next((word for limit, word in ALCOHOL_BANDS if value < limit), "высокая")
    return f"Крепость по карточке каталога — {shown} % об. Для вина это {band} крепость."


def storage_rule(
    *,
    color: str | None,
    sugar: str | None,
    sparkling: bool,
    sweet_name: bool,
    body: float | None,
) -> str:
    """Правило хранения по стилю: игристое → сладкое и креплёное → белое, розовое и оранжевое →
    красное по телу (тело «по сорту»; неизвестно — нейтральная фраза) → нет цвета."""
    if sparkling:
        return "sparkling"
    if sugar == "sladkoe" or sweet_name:
        return "sweet"
    if color in ("Белое", "Розовое", "Оранжевое"):
        return "white"
    if color == "Красное":
        if body is None:
            return "red_unknown"
        return "red_full" if body >= 3.5 else "red"
    return "unknown"


def storage_fact(rule: str) -> str:
    return f"{STORAGE_HEAD} {STORAGE_TEXTS[rule]}"


# ------------------------------------------------------------------ обычная сортировка
def plural_style(style_label: str) -> str:
    """«Красное сухое» → «красные сухие»; «Белое брют» → «белые брют»."""
    out = []
    for word in style_label.lower().split():
        if word.endswith("ое"):
            stem = word[:-2]
            out.append(stem + ("ие" if stem[-1:] in "гкхжшщч" else "ые"))
        else:
            out.append(word)
    return " ".join(out)


def plain_text(style_label: str) -> str:
    """«Обычная сортировка: красные сухие других виноделен по названию.»"""
    style = plural_style(style_label) or "вина"
    return f"Обычная сортировка: {style} других виноделен по названию."


def plain_title(style_label: str) -> str:
    return f"{style_label or 'Вина'} — по названию"


# ------------------------------------------------------------------ вопрос экрана check (§5)
QUESTION_FIELDS = ("sugar", "color", "sparkling", "grapes", "abv", "name")
#: Поля прочитанного, по которым кандидат «согласуется» с этикеткой.
AGREE_FIELDS = ("sugar", "color", "sparkling", "abv")
MAX_OPTIONS = 4


def _alternatives(labels: Sequence[str]) -> str:
    """«А или Б», «А, Б или В»."""
    return join_words(labels, conjunction="или")


def check_question(
    read: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]
) -> dict[str, Any] | None:
    """Один вопрос экрана `check` по полю, которым различаются варианты (договор, §5).

    `candidates` — в порядке ранга: `{slug, name, facts: {sugar, color, sparkling, grapes,
    abv}}`; `None` в фактах — «неизвестно», пустой список сортов — тоже. Меньше двух
    кандидатов или различать нечем — `None`.
    """
    if len(candidates) < 2:
        return None

    def agrees(candidate: Mapping[str, Any]) -> bool:
        facts = candidate.get("facts") or {}
        for key in AGREE_FIELDS:
            ours, theirs = read.get(key), facts.get(key)
            if ours is not None and theirs is not None and ours != theirs:
                return False
        return True

    agreeing = [c for c in candidates if agrees(c)]
    pool = agreeing if len(agreeing) >= 2 else list(candidates)
    read_fields = {key for key in AGREE_FIELDS if read.get(key) is not None}
    for field in QUESTION_FIELDS:
        if field in read_fields:
            continue
        if field == "grapes" and any(
            len((c.get("facts") or {}).get("grapes") or ()) > 2 for c in pool
        ):
            continue
        options: dict[Any, str] = {}
        for candidate in pool:
            facts = candidate.get("facts") or {}
            value = candidate.get("name") or None if field == "name" else facts.get(field)
            if isinstance(value, list | tuple):
                value = tuple(value) or None
            if value is not None and value not in options:
                options[value] = str(candidate["slug"])
        if len(options) < 2:
            continue
        values = list(options)[:MAX_OPTIONS]
        labels, text = _question_text(field, values)
        return {
            "field": field,
            "text": text,
            "options": [
                {"label": label, "slug": options[value]}
                for label, value in zip(labels, values, strict=True)
            ],
        }
    return None


def _question_text(field: str, values: Sequence[Any]) -> tuple[list[str], str]:
    if field == "sugar":
        labels = [SUGAR_WORDS.get(str(value), str(value)) for value in values]
        return labels, f"Что на этикетке: {_alternatives(labels)}?"
    if field == "color":
        labels = [str(value).lower() for value in values]
        return labels, f"Какого цвета вино: {_alternatives(labels)}?"
    if field == "sparkling":
        labels = ["игристое" if value else "тихое" for value in values]
        return labels, "Вино игристое или тихое?"
    if field == "grapes":
        labels = [" и ".join(value) for value in values]
        return labels, f"Какой сорт на этикетке: {_alternatives(labels)}?"
    if field == "abv":
        numbers = [number_label(float(value)) for value in values]
        labels = [f"{number} % об." for number in numbers]
        return labels, f"Какая крепость на этикетке: {_alternatives(numbers)} % об.?"
    labels = [str(value) for value in values]
    quoted = [f"«{label}»" for label in labels]
    return labels, f"Какое название на этикетке: {_alternatives(quoted)}?"
