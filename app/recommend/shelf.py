"""«Сомелье у полки»: к вину со скана — до трёх вин других виноделен под блюдо и направление.

От пользователя приходят только значения двух чипов:

    food  meat | fish | cheese | none         «К чему?»
    want  fresher | softer | sweeter | none   «Какое хочется?»

**Блюда.** Решение 24.09: блюд портала нет, чип идёт по нашим правилам сочетаний
(`data/somm/`, `app.recommend.somm_data`). Вино подходит чипу, если хотя бы одна его пара с
блюдом группы чипа (`dishes.json`, поле `food`) имеет вердикт `yes` (договор, §7). Объяснение
так и пишется — «К рыбе и морепродуктам — по правилам сочетаний». Данных пар нет вовсе — при
выбранном блюде выдача пуста с нотой `no_pairs`.

**Направления.** Сдвиг в нужную сторону подтверждается фактом каталога или осью профиля «по
сорту» (`build.py`). Факт каталога сильнее: такие вина идут первыми.

    want      каталог (want_source = "catalog")        по сорту (want_source = "grape")
    fresher   сахар ниже; крепость ниже на 0,5° и более  кислотность выше
    softer    сахар на ступень выше, не слаще            танины ниже у красного и оранжевого,
              полусухого: сухое → полусухое, экстра      кислотность ниже у белого и розового
              брют → брют
    sweeter   сахар выше                                 — (сладость по сорту не оценить)

Ось «по сорту» считается, только если у обоих вин профиль опирается на сорт (`basis` не
`low`), сорта у них разные и разница не меньше 0,3 (порог зонда «Лозы»). Иначе разница — от
региона или года, а объяснение называет сорта. Против направления нельзя идти ни по одному
признаку: вино «посвежее» не бывает слаще и не крепче на 0,5° и больше, «помягче» — суше, а по
сорту ни одно из них не уходит в обратную сторону на 0,3 и больше.

**Подборка от причины** (`reason_move`) — только для сомелье: после «скорее нет» и оговорки вина
идут в сторону причины сильнейшего «−» пары (договор сомелье, §4.5; третий круг проверки 25.09).
Чипами `/shelf` эти направления не приходят. Доказательство сдвига — только то, что называет
причина, и только заметная разница, а не шум порога:

    want      доказательства (by)       что считается сдвигом
    lighter   strength, body, oak       крепость ниже на 0,5° и более (у диапазона — весь
                                        диапазон), тело или дуб ниже по сорту на 0,5 и более
    fuller    strength, body            крепость выше на 0,5° и более, тело выше на 0,5 и более
    softer    colour, tannin            белое или розовое вместо красного и оранжевого (почти
                                        без танинов), танины ниже по сорту на 0,5 и более
    fresher   acidity                   кислотность выше по сорту на 0,5 и более
    drier     sugar                     сахар ниже по карточке
    sweeter   sugar                     сахар выше по карточке

Какие доказательства в ходу, решает причина: у «Крепость перекрывает блюдо» — только крепость
(«полегче» по причине крепости — правда слабее), у «Мягкому вину тяжело с жирным» — только
кислотность, у «Дуб спорит с лёгким» — только дуб. Против направления нельзя ни по крепости, ни
по оси причины: «полегче» не крепче на 0,5° и не плотнее по сорту на 0,3, «помощнее» — наоборот,
«помягче» не терпче, «посвежее» не мягче по кислотности.

«Помягче» никогда не становится «послаще». Потолок сахара — полусухое (`SOFTER_MAX_STEP`), а
у якоря слаще полусухого — его собственный сахар; у якоря без сахара — полусухое. Потолок
держит любой путь, и «по сорту» тоже: вино с известным сахаром выше потолка не идёт даже с
танинами ниже. Поэтому якорю до полусухого — ступень сахара вверх, но не выше полусухого, или
вина мягче по сорту не слаще полусухого; полусладкому и сладкому — только вина того же сахара,
мягче по сорту: ступень сахара вверх для них уже «послаще». Неизвестный сахар кандидата
потолок не проверяет: сахар берётся только из названия и slug выгрузки, и у 376 из 2 103
позиций его нет.

**Пул.** Кандидаты — вина пула справочника (`RecoCatalog.pool`) без своей винодельни и своей
группы `wine_id` (`facts.excluded`), с блюдом чипа и сдвигом в нужную сторону; при
`want=none` — сахар в пределах ступени, как у похожих. Сначала берутся 60 ближайших по фактам
среди вин того же цвета и игристости. Ключ близости — как в `facts.py`: мера Жаккара по сортам
→ разница сахара в ступенях → разница крепости → регион → slug. Внутри 60 первыми идут вина,
где направление подтверждено и каталогом, и сортом, затем только каталогом, затем только сортом.
Если таких вин меньше трёх, берётся весь пул любого цвета и игристости; вина того же стиля
остаются первыми. `pool="catalog"` и нота `pool_expanded` — только если расширение добавило в
выдачу вино другого стиля. Одна винодельня — одно место (`one_per_winery`).

**Ответ** — три вина или честная фраза:

* `not_evaluable` — у самого вина нет признака, от которого считается сдвиг
  (`direction_evaluable`): «послаще» без сахара в выгрузке (376 позиций) — «Сахар этого вина в
  каталоге не указан — послаще подобрать не по чему»; «помягче» и «посвежее» — когда нет ещё и
  оси «по сорту» (и крепости у «посвежее»). Это не «разницы нет»: сравнивать не с чем;
* `no_difference` — ни одно вино пула (даже без учёта блюда) не сдвинуто в нужную сторону:
  «послаще» сладкому якорю — «Это вино уже сладкое: послаще в каталоге по описаниям не найти»;
  иначе — «По описаниям разницы нет — помягче в каталоге не найти» (или посвежее, послаще);
* `fewer_than_three` — направление в каталоге есть, но с этим блюдом вин меньше трёх;
* `no_pairs` — блюдо выбрано, а данных правил сочетаний нет.

**Обычная сортировка** (`plain`): направление не применяется, вина той же категории (цвет,
игристость, сахар) с блюдом чипа идут по названию, без объяснений.

Процентов похожести нет, крепость — градусами, все строки проходят `content_filter`
(это проверяют тесты и зонд `research/2026-09-24_after/probe_shelf.py`).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

from app.reading.taxonomy import grape_label
from app.recommend.build import load_priors, profiles_of
from app.recommend.catalog import (
    SUGAR_STEPS,
    SUGAR_WORDS,
    Alcohol,
    RecoCatalog,
    RecoWine,
    degrees_label,
    name_key,
)
from app.recommend.facts import (
    ABV_DIFFERENCE,
    DEFAULT_LIMIT,
    Facts,
    Note,
    abv_gap,
    excluded,
    explain,
    fewer_note,
    jaccard,
    one_per_winery,
    sugar_close,
)
from app.recommend.profile import AXIS_LABELS, CATALOG, GRAPE, StyleProfile
from app.recommend.somm_data import DishInfo, SommData, WinePairs

Food = Literal["meat", "fish", "cheese", "none"]
Want = Literal["fresher", "softer", "sweeter", "none"]
FOODS: tuple[str, ...] = ("meat", "fish", "cheese", "none")
WANTS: tuple[str, ...] = ("fresher", "softer", "sweeter", "none")
#: Направления только для подборки сомелье от причины «−» (`reason_move`, `answers.REASONS`):
#: чипами `/shelf` они не приходят.
REASON_WANTS: tuple[str, ...] = ("lighter", "fuller", "drier")

#: Первая строка объяснения при выбранном блюде: откуда это известно. Группы блюд чипа —
#: поле `food` блюд правил сочетаний (`dishes.json`, договор сомелье, §7.2); птица в «Мясо»
#: не входит.
FOOD_REASONS: dict[str, str] = {
    "meat": "К мясу — по правилам сочетаний",
    "fish": "К рыбе и морепродуктам — по правилам сочетаний",
    "cheese": "К сырам — по правилам сочетаний",
}
#: Сколько блюд `top` правил сочетаний показывает плитка.
TILE_DISHES = 3
#: «К рыбе посвежее в каталоге не нашлось».
FOOD_TO: dict[str, str] = {"meat": "к мясу", "fish": "к рыбе", "cheese": "к сыру"}
#: Подписи чипов направления — они же начало строки объяснения.
WANT_WORDS: dict[str, str] = {
    "fresher": "Посвежее",
    "softer": "Помягче",
    "sweeter": "Послаще",
    "lighter": "Полегче",
    "fuller": "Помощнее",
    "drier": "Суше",
}

#: Сколько ближайших по фактам вин смотреть до расширения на весь каталог.
NEAR_POOL = 60
#: Шаг по оси «по сорту»: как в зонде «Лозы» (`probe_shelf.py`).
GRAPE_STEP = 0.3
#: Разница сахара, если он неизвестен у одного из вин: дальше любой известной.
UNKNOWN_SUGAR_GAP = 9
#: Потолок сахара «помягче» — полусухое, а у якоря слаще полусухого — его собственный сахар
#: (`move_of`): и по сахару, и по сорту. Выше потолка это уже «послаще».
SOFTER_MAX_STEP = SUGAR_STEPS["polusuhoe"]
#: Строк объяснения в плитке: блюдо, направление и факты — не больше четырёх.
MAX_REASONS = 4
#: Шаг «по сорту» подборки от причины (`reason_move`): разница, которую видно, а не шум порога
#: (третий круг проверки 25.09: «Мускат 3,0 → Мускат 3,3» — не «тело плотнее»). У чипов `/shelf` —
#: прежний `GRAPE_STEP`.
REASON_STEP = 0.5
#: Цвета почти без танинов — белое и розовое, тихие и игристые: подборка «помягче» к рыбе от
#: терпкого красного идёт к ним (`reason_move`, доказательство `colour`).
LOW_TANNIN_COLOURS: tuple[str, ...] = ("Белое", "Розовое")
#: Цвета, у которых терпкость бывает: красное и оранжевое (мацерация на кожице).
TANNIC_COLOURS: tuple[str, ...] = ("Красное", "Оранжевое")
#: Оси «по сорту» подборки от причины и их знак: полегче — тело и дуб вниз, помощнее — тело
#: вверх, помягче — танины вниз, посвежее — кислотность вверх.
REASON_AXES: dict[tuple[str, str], int] = {
    ("lighter", "body"): -1,
    ("lighter", "oak"): -1,
    ("fuller", "body"): +1,
    ("softer", "tannin"): -1,
    ("fresher", "acidity"): +1,
}
#: Оси, по которым направление не должно уходить в обратную сторону (`reason_move`).
REASON_GUARDS: dict[str, tuple[tuple[str, int], ...]] = {
    "lighter": (("body", -1),),
    "fuller": (("body", +1),),
    "softer": (("tannin", -1),),
    "fresher": (("acidity", +1),),
}

NO_DIFFERENCE = "no_difference"
#: Направление не оценить: у самого вина нет ни одного признака, по которому считается сдвиг
#: (`direction_evaluable`). Это не «разницы нет»: сравнивать просто не по чему.
NOT_EVALUABLE = "not_evaluable"
POOL_EXPANDED = Note(
    "pool_expanded", "Среди близких по стилю не нашлось — подобрали из всего каталога"
)
NO_PAIRS = Note("no_pairs", "Подбор к блюдам недоступен: нет данных правил сочетаний")


# ------------------------------------------------------------------ сдвиг в нужную сторону
@dataclass(frozen=True, slots=True)
class Move:
    """Чем подтверждён сдвиг кандидата в нужную сторону: строки объяснения по источникам.

    `catalog` — фактом карточки (сахар, крепость, цвет), `grape` — осью профиля «по сорту».
    `sugar`, `strength` и `colour` — какие факты уже названы, чтобы не повторять их в
    объяснении; `axis` — ось «по сорту», которой подтверждён сдвиг подборки от причины.
    """

    catalog: str | None = None
    grape: str | None = None
    sugar: bool = False
    strength: bool = False
    colour: bool = False
    axis: str | None = None

    @property
    def source(self) -> str | None:
        if self.catalog:
            return CATALOG
        return GRAPE if self.grape else None

    @property
    def tier(self) -> int:
        """0 — и каталог, и сорт; 1 — только каталог; 2 — только сорт."""
        if self.catalog and self.grape:
            return 0
        return 1 if self.catalog else 2

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(text for text in (self.catalog, self.grape) if text)


NO_MOVE = Move()


def _step(sugar: str | None) -> int | None:
    return SUGAR_STEPS.get(sugar) if sugar else None


def _grape_names(codes: Sequence[str], limit: int | None = 2) -> str:
    """«Рислинг», «Сира и Мурведр» — два первых сорта, как в объяснениях похожих.

    `limit=None` — все сорта: «Каберне Совиньон, Мерло и Пино Нуар».
    """
    names = [grape_label(code) for code in codes[:limit]]
    if len(names) <= 1:
        return "".join(names)
    return f"{', '.join(names[:-1])} и {names[-1]}"


def _grape_pair(wine: RecoWine, anchor: RecoWine) -> str:
    """Сорта в объяснении «по сорту»: «Мерло, а не Сира».

    Если первые два сорта у вин общие, разница дальше — и называются все сорта: иначе вышло
    бы «Каберне Совиньон и Мерло, а не Каберне Совиньон и Мерло».
    """
    limit = None if set(wine.grapes[:2]) == set(anchor.grapes[:2]) else 2
    return f"{_grape_names(wine.grapes, limit)}, а не {_grape_names(anchor.grapes, limit)}"


def _grape_axis(want: str, anchor: RecoWine) -> tuple[str, int] | None:
    """Ось «по сорту» направления чипа и знак: посвежее — кислотность вверх, помягче — танины
    вниз (у белого и розового — кислотность вниз)."""
    if want == "fresher":
        return "acidity", +1
    if want == "softer":
        if anchor.color in ("Белое", "Розовое"):
            return "acidity", -1
        return "tannin", -1
    return None


def move_of(
    want: str,
    anchor: RecoWine,
    wine: RecoWine,
    *,
    alcohol: tuple[Alcohol, Alcohol],
    profiles: tuple[StyleProfile | None, StyleProfile | None],
) -> Move | None:
    """Сдвиг `wine` относительно `anchor` в сторону чипа `want`; `None` — не туда или никак.

    `alcohol` и `profiles` — пары (исходное вино, кандидат). Направления подборки сомелье от
    причины «−» — `reason_move`.
    """
    word = WANT_WORDS[want]
    ours, theirs = _step(anchor.sugar), _step(wine.sugar)
    a_alc, w_alc = alcohol
    a_prof, w_prof = profiles
    sugar_known = ours is not None and theirs is not None
    abv_known = a_alc.value is not None and w_alc.value is not None

    # Против направления по фактам каталога идти нельзя.
    if want == "fresher" and (
        (sugar_known and theirs > ours)
        or (abv_known and w_alc.value - a_alc.value >= ABV_DIFFERENCE)
    ):
        return None
    # «Помягче» не бывает суше и никогда не бывает «послаще» — ни по каталогу, ни по сорту.
    # Потолок — полусухое или сахар самого якоря, если он слаще; у якоря без сахара —
    # полусухое. Неизвестный сахар кандидата потолок не проверяет.
    ceiling = SOFTER_MAX_STEP if ours is None else max(SOFTER_MAX_STEP, ours)
    if want == "softer" and (
        (sugar_known and theirs < ours) or (theirs is not None and theirs > ceiling)
    ):
        return None
    weaker = abv_known and a_alc.value - w_alc.value >= ABV_DIFFERENCE

    catalog: str | None = None
    sugar = strength = False
    sugar_text = (
        f"{word}: {SUGAR_WORDS[wine.sugar]}, а не {SUGAR_WORDS[anchor.sugar]}"
        if sugar_known
        else ""
    )
    sweeter = want == "sweeter" and sugar_known and theirs > ours
    softer = want == "softer" and sugar_known and theirs == ours + 1
    strength_text = f"{word}: крепость {degrees_label(w_alc)} против {degrees_label(a_alc)}"
    if sweeter or softer:
        catalog, sugar = sugar_text, True
    elif want == "fresher":
        if sugar_known and theirs < ours:
            catalog, sugar = sugar_text, True
        elif weaker:
            catalog, strength = strength_text, True

    grape: str | None = None
    axis_sign = _grape_axis(want, anchor)
    if axis_sign is not None and a_prof is not None and w_prof is not None:
        axis, sign = axis_sign
        grape_based = a_prof.source(axis) == GRAPE and w_prof.source(axis) == GRAPE
        # Округление: 3,4 − 3,1 во float — 0,2999…, а по шкале это ровно шаг.
        delta = round(sign * (w_prof.axis(axis) - a_prof.axis(axis)), 6)
        if grape_based and delta <= -GRAPE_STEP:
            return None  # по сорту — в обратную сторону
        if grape_based and delta >= GRAPE_STEP and set(wine.grapes) != set(anchor.grapes):
            more = "выше" if sign > 0 else "ниже"
            grape = f"{word} по сорту: {AXIS_LABELS[axis]} {more} — {_grape_pair(wine, anchor)}"
    if catalog is None and grape is None:
        return None
    return Move(catalog, grape, sugar, strength)


def _abv_shift(anchor: Alcohol, wine: Alcohol) -> tuple[float, float] | None:
    """Насколько `wine` слабее и крепче `anchor` с запасом диапазонов: (ниже, выше) в градусах.

    «Ниже» — от нижней границы якоря до верхней кандидата, «выше» — от верхней якоря до нижней
    кандидата: у «10–12,5°» против «12°» ни то ни другое не полградуса, и это не «крепость ниже».
    Разница округляется: 12,7 − 12,2 во float — 0,4999…, а на этикетке это ровно полградуса.
    """
    if anchor.value is None or wine.value is None:
        return None
    a_hi = anchor.max if anchor.max is not None else anchor.value
    w_hi = wine.max if wine.max is not None else wine.value
    return round(anchor.value - w_hi, 6), round(wine.value - a_hi, 6)


def reason_move(
    want: str,
    by: Sequence[str],
    anchor: RecoWine,
    wine: RecoWine,
    *,
    alcohol: tuple[Alcohol, Alcohol],
    profiles: tuple[StyleProfile | None, StyleProfile | None],
    anchor_sugar: str | None = None,
) -> Move | None:
    """Сдвиг `wine` в сторону причины «−» пары у вина `anchor` (подборка сомелье, §4.5).

    `want` — `lighter`, `fuller`, `softer`, `fresher`, `drier` или `sweeter`; `by` — какие
    доказательства причина принимает (`strength`, `sugar`, `colour`, `body`, `tannin`,
    `acidity`, `oak`), по порядку. Сдвиг засчитывается, только если он заметен: крепость — на
    0,5° и больше с запасом диапазонов (`_abv_shift`), ось «по сорту» — на `REASON_STEP` при
    разных сортах, сахар — по карточке кандидата. `anchor_sugar` — сахар вина карточки для
    правил, если в карточке его нет: «сладкое» по креплёному, десертному, мускатному названию (так
    же судит сборка, `rule_sugar`); `None` — сахар карточки. Против направления нельзя ни по крепости («полегче» не крепче
    на 0,5°, «помощнее» не слабее), ни по оси причины на 0,3 и больше (`REASON_GUARDS`), ни по
    сахару («суше» не слаще).
    """
    word = WANT_WORDS[want]
    a_alc, w_alc = alcohol
    a_prof, w_prof = profiles
    shift = _abv_shift(a_alc, w_alc)
    if shift is not None:
        lower, higher = shift
        if (want == "lighter" and higher >= ABV_DIFFERENCE) or (
            want == "fuller" and lower >= ABV_DIFFERENCE
        ):
            return None
    by_grape: dict[str, float] = {}
    if a_prof is not None and w_prof is not None:
        for axis in AXIS_LABELS:
            if a_prof.source(axis) == GRAPE and w_prof.source(axis) == GRAPE:
                by_grape[axis] = round(w_prof.axis(axis) - a_prof.axis(axis), 6)
    for axis, sign in REASON_GUARDS.get(want, ()):
        if axis in by_grape and sign * by_grape[axis] <= -GRAPE_STEP:
            return None
    ours, theirs = _step(anchor_sugar or anchor.sugar), _step(wine.sugar)
    sugar_known = ours is not None and theirs is not None
    if sugar_known and (
        (want == "drier" and theirs > ours) or (want == "sweeter" and theirs < ours)
    ):
        return None

    catalog: str | None = None
    grape: str | None = None
    sugar = strength = colour = False
    axis_used: str | None = None
    for kind in by:
        if kind == "strength" and catalog is None and shift is not None:
            moved = {"lighter": shift[0], "fuller": shift[1]}.get(want, 0.0)
            if moved >= ABV_DIFFERENCE:
                catalog = f"{word}: крепость {degrees_label(w_alc)} против {degrees_label(a_alc)}"
                strength = True
        elif kind == "sugar" and catalog is None and sugar_known:
            if (want == "drier" and theirs < ours) or (want == "sweeter" and theirs > ours):
                other = SUGAR_WORDS[wine.sugar or ""]
                catalog = (
                    f"{word}: {other}, а не {SUGAR_WORDS[anchor.sugar]}"
                    if anchor.sugar
                    else f"{word}: {other} по карточке"
                )
                sugar = True
        elif kind == "colour" and catalog is None and want == "softer":
            if anchor.color in TANNIC_COLOURS and wine.color in LOW_TANNIN_COLOURS:
                catalog = f"{word}: {wine.color.lower()}, а не {anchor.color.lower()}"
                colour = True
        elif grape is None and (want, kind) in REASON_AXES and kind in by_grape:
            sign = REASON_AXES[(want, kind)]
            if sign * by_grape[kind] >= REASON_STEP and set(wine.grapes) != set(anchor.grapes):
                more = "выше" if sign > 0 else "ниже"
                grape = f"{word} по сорту: {AXIS_LABELS[kind]} {more} — {_grape_pair(wine, anchor)}"
                axis_used = kind
    if catalog is None and grape is None:
        return None
    return Move(catalog, grape, sugar, strength, colour, axis_used)


def direction_evaluable(
    want: str, anchor: RecoWine, alcohol: Alcohol, profile: StyleProfile | None
) -> bool:
    """Есть ли у вина хоть один признак, по которому `move_of` может найти сдвиг в сторону `want`.

    Сдвиг считается от фактов самого вина: без них любой кандидат — «никак», и фраза «по
    описаниям разницы нет» была бы неправдой (у AGORA Бастардо без сахара в выгрузке послаще в
    каталоге есть — сравнить просто не с чем). Признаки по направлениям — как в `move_of`:

    * «послаще» — известный сахар;
    * «помягче» — сахар, от которого есть ступень вверх не выше потолка (брют натюр… сухое), или
      ось «по сорту» (танины у красного и оранжевого, кислотность у белого и розового);
    * «посвежее» — сахар, от которого есть ступень вниз, крепость или кислотность «по сорту».
    """
    ours = _step(anchor.sugar)
    axis = _grape_axis(want, anchor)
    by_grape = axis is not None and profile is not None and profile.source(axis[0]) == GRAPE
    if want == "sweeter":
        return ours is not None
    if want == "softer":
        return (ours is not None and ours + 1 <= max(SOFTER_MAX_STEP, ours)) or by_grape
    if want == "fresher":
        lowest = min(SUGAR_STEPS.values())
        return (ours is not None and ours > lowest) or alcohol.value is not None or by_grape
    return True


def not_evaluable_text(want: str, anchor: RecoWine) -> str:
    """Честная фраза, когда направление не оценить: чего именно в каталоге нет.

    «Послаще» без сахара — «Сахар этого вина в каталоге не указан — послаще подобрать не по
    чему»; «помягче» и «посвежее» называют ещё и сорт, а «посвежее» — крепость.
    """
    word = WANT_WORDS[want].lower()
    tail = f"{word} подобрать не по чему"
    if want == "sweeter":
        return f"Сахар этого вина в каталоге не указан — {tail}"
    feel = "мягкость" if want == "softer" else "свежесть"
    grape = f"по сорту {feel} не оценить"
    if want == "fresher":
        missing = "Сахар и крепость этого вина" if anchor.sugar is None else "Крепость этого вина"
        verb = "не указаны" if anchor.sugar is None else "не указана"
        return f"{missing} в каталоге {verb}, а {grape} — {tail}"
    if anchor.sugar is None:
        return f"Сахар этого вина в каталоге не указан, а {grape} — {tail}"
    return f"По сорту {feel} этого вина не оценить — {tail}"


# ------------------------------------------------------------------ ответ
@dataclass(frozen=True, slots=True)
class ShelfPick:
    """Вино выдачи: объяснение, блюда `top` правил сочетаний и чем подтверждено направление."""

    wine: RecoWine
    reasons: tuple[str, ...] = ()
    dishes: tuple[str, ...] = ()
    want_source: str | None = None


@dataclass(frozen=True, slots=True)
class ShelfAnswer:
    """Итог: вина по порядку, откуда пул (`near` | `catalog`) и честные фразы."""

    picks: tuple[ShelfPick, ...]
    pool: str = "near"
    notes: tuple[Note, ...] = ()


def foods_of(pairs: WinePairs | None, dishes: Mapping[str, DishInfo]) -> frozenset[str]:
    """Группы чипа «К чему?», к которым вино подходит: хотя бы одна пара `yes` с блюдом группы.

    Договор, §7: вино подходит чипу `meat`, если у него есть пара `yes` с блюдом группы `meat`
    (`dishes.json`, поле `food`). Нет пар вина — ни одной группы.
    """
    if pairs is None:
        return frozenset()
    return frozenset(
        dish.food
        for dish_id, pair in pairs.dishes.items()
        if pair.verdict == "yes" and (dish := dishes.get(dish_id)) is not None and dish.food
    )


def top_names(
    pairs: WinePairs | None, dishes: Mapping[str, DishInfo], limit: int = TILE_DISHES
) -> tuple[str, ...]:
    """Названия первых блюд `top` вина для плитки; неизвестные блюда пропускаются."""
    if pairs is None:
        return ()
    return tuple(dishes[dish].name for dish in pairs.top if dish in dishes)[:limit]


class Shelf:
    """Подбор «Сомелье у полки» поверх справочника и правил сочетаний.

    Профили стиля всех вин и группы блюд, к которым вино подходит (`foods_of`), считаются один
    раз. `somm=None` — данных сомелье нет: чип блюда даёт честную фразу `no_pairs`.
    """

    def __init__(
        self,
        catalog: RecoCatalog,
        profiles: Mapping[str, StyleProfile] | None = None,
        *,
        somm: SommData | None = None,
    ) -> None:
        self.catalog = catalog
        self.profiles: Mapping[str, StyleProfile] = (
            profiles if profiles is not None else profiles_of(catalog, load_priors())
        )
        self.somm = somm if somm is not None else SommData()
        self._foods: dict[str, frozenset[str]] = {
            wine.slug: foods_of(self.somm.pairs.get(wine.slug), self.somm.dishes)
            for wine in catalog
        }

    @property
    def has_pairs(self) -> bool:
        """Есть ли данные правил сочетаний: без них подбор к блюду недоступен."""
        return bool(self.somm.pairs)

    def goes_with(self, wine: RecoWine, food: str) -> bool:
        """Подходит ли вино чипу: хотя бы одна пара `yes` с блюдом этой группы."""
        return food == "none" or food in self._foods.get(wine.slug, frozenset())

    def dishes_of(self, wine: RecoWine) -> tuple[str, ...]:
        """До трёх блюд `top` правил сочетаний вина; нет данных пар — пусто."""
        return top_names(self.somm.pairs.get(wine.slug), self.somm.dishes, TILE_DISHES)

    # -------------------------------------------------------------- ключи
    def _nearness(self, facts: Facts, wine: RecoWine) -> tuple:
        """Ключ близости по фактам — порядок `facts.py`, но сахар — расстоянием в ступенях."""
        ours, theirs = _step(facts.sugar), _step(wine.sugar)
        sugar_gap = abs(ours - theirs) if ours is not None and theirs is not None else None
        return (
            -jaccard(facts.grapes, wine.grapes) if facts.grapes else 0.0,
            UNKNOWN_SUGAR_GAP if sugar_gap is None else sugar_gap,
            abv_gap(facts.alcohol, self.catalog.alcohol(wine.slug)),
            facts.region is not None and wine.region != facts.region,
            wine.slug,
        )

    def evaluable(self, want: str, anchor: RecoWine) -> bool:
        """Можно ли оценить направление для этого вина (`direction_evaluable`)."""
        return want == "none" or direction_evaluable(
            want, anchor, self.catalog.alcohol(anchor.slug), self.profiles.get(anchor.slug)
        )

    def move(self, want: str, anchor: RecoWine, wine: RecoWine) -> Move | None:
        if want == "none":
            return NO_MOVE
        return move_of(
            want,
            anchor,
            wine,
            alcohol=(self.catalog.alcohol(anchor.slug), self.catalog.alcohol(wine.slug)),
            profiles=(self.profiles.get(anchor.slug), self.profiles.get(wine.slug)),
        )

    def reason_move(
        self,
        want: str,
        by: Sequence[str],
        anchor: RecoWine,
        wine: RecoWine,
        anchor_sugar: str | None = None,
    ) -> Move | None:
        """Сдвиг подборки сомелье от причины «−» (`reason_move`) на крепости и профилях
        справочника."""
        return reason_move(
            want,
            by,
            anchor,
            wine,
            alcohol=(self.catalog.alcohol(anchor.slug), self.catalog.alcohol(wine.slug)),
            profiles=(self.profiles.get(anchor.slug), self.profiles.get(wine.slug)),
            anchor_sugar=anchor_sugar,
        )

    def _candidates(
        self, anchor: RecoWine, facts: Facts, wines: Iterable[RecoWine], food: str, want: str
    ) -> list[tuple[RecoWine, Move]]:
        out = []
        for wine in wines:
            if excluded(wine, facts) or not self.goes_with(wine, food):
                continue
            if want == "none" and not sugar_close(facts.sugar, wine.sugar):
                continue
            move = self.move(want, anchor, wine)
            if move is not None:
                out.append((wine, move))
        return out

    # -------------------------------------------------------------- подбор
    def answer(
        self, anchor: RecoWine, food: str, want: str, limit: int = DEFAULT_LIMIT
    ) -> ShelfAnswer:
        """Три вина под блюдо и направление (`order=reco`) или честная фраза."""
        if food != "none" and not self.has_pairs:
            return ShelfAnswer((), "near", (NO_PAIRS,))
        if not self.evaluable(want, anchor):
            # Сдвиг считать не от чего: ни одного кандидата `move_of` не даст.
            return ShelfAnswer((), "near", (Note(NOT_EVALUABLE, not_evaluable_text(want, anchor)),))
        facts = Facts.of_wine(anchor, self.catalog.alcohol(anchor.slug))
        near = self._candidates(
            anchor, facts, self.catalog.style_pool(anchor.color, anchor.sparkling), food, want
        )
        near = sorted(near, key=lambda item: self._nearness(facts, item[0]))[:NEAR_POOL]
        if len(near) >= limit:
            ranked = sorted(near, key=lambda item: (item[1].tier, self._nearness(facts, item[0])))
        else:
            wide = self._candidates(anchor, facts, self.catalog.pool, food, want)
            ranked = sorted(
                wide,
                key=lambda item: (
                    item[0].color != anchor.color,
                    item[0].sparkling != anchor.sparkling,
                    item[1].tier,
                    self._nearness(facts, item[0]),
                ),
            )
        moves = {wine.slug: move for wine, move in ranked}
        chosen = one_per_winery([wine for wine, _ in ranked], limit)
        picks = tuple(self._pick(facts, wine, moves[wine.slug], food, want) for wine in chosen)
        # «Весь каталог» — только если расширение что-то добавило: вина другого стиля.
        near_slugs = {wine.slug for wine, _ in near}
        pool = "catalog" if any(wine.slug not in near_slugs for wine in chosen) else "near"
        notes: list[Note] = []
        if pool == "catalog":
            notes.append(POOL_EXPANDED)
        if len(picks) < limit:
            notes.append(self._short_note(anchor, facts, len(picks), food, want, limit))
        return ShelfAnswer(picks, pool, tuple(notes))

    def plain(self, anchor: RecoWine, food: str, limit: int = DEFAULT_LIMIT) -> ShelfAnswer:
        """Обычная сортировка: та же категория с блюдом чипа по названию, без объяснений."""
        if food != "none" and not self.has_pairs:
            return ShelfAnswer((), "near", (NO_PAIRS,))
        facts = Facts.of_wine(anchor, self.catalog.alcohol(anchor.slug))
        wines = sorted(
            (
                wine
                for wine in self.catalog.style_pool(anchor.color, anchor.sparkling)
                if not excluded(wine, facts)
                and self.goes_with(wine, food)
                and (anchor.sugar is None or wine.sugar == anchor.sugar)
            ),
            key=name_key,
        )[:limit]
        picks = tuple(ShelfPick(wine, (), self.dishes_of(wine), None) for wine in wines)
        notes = (self._short_note(anchor, facts, len(picks), food, "none", limit),)
        return ShelfAnswer(picks, "near", notes if len(picks) < limit else ())

    def _pick(self, facts: Facts, wine: RecoWine, move: Move, food: str, want: str) -> ShelfPick:
        """Объяснение: блюдо → направление → факты карточки, без повторов и не длиннее четырёх.

        Другой цвет (вино из расширенного пула) называется сразу после направления: это
        главное, чем оно отличается, и при обрезке до четырёх строк он не должен пропасть.
        """
        reasons: list[str] = []
        if food != "none":
            reasons.append(FOOD_REASONS[food])
        reasons.extend(move.reasons)
        shown = facts
        if facts.color and wine.color and wine.color != facts.color:
            reasons.append(f"{wine.color}, а не {facts.color.lower()}")
            shown = replace(shown, color=None)
        if move.sugar:
            shown = replace(shown, sugar=None, sparkling=None)
        if move.grape:
            shown = replace(shown, grapes=())
        alcohol = self.catalog.alcohol(wine.slug)
        reasons.extend(explain(shown, wine, alcohol, strength=not move.strength))
        return ShelfPick(
            wine,
            tuple(reasons[:MAX_REASONS]),
            self.dishes_of(wine),
            move.source if want != "none" else None,
        )

    def _short_note(
        self, anchor: RecoWine, facts: Facts, found: int, food: str, want: str, limit: int
    ) -> Note:
        """Честная фраза, когда вин меньше `limit`: разницы нет вовсе или мало с этим блюдом."""
        if want != "none" and not any(
            not excluded(wine, facts) and self.move(want, anchor, wine) is not None
            for wine in self.catalog.pool
        ):
            if want == "sweeter" and anchor.sugar == "sladkoe":
                text = "Это вино уже сладкое: послаще в каталоге по описаниям не найти"
            else:
                text = f"По описаниям разницы нет — {WANT_WORDS[want].lower()} в каталоге не найти"
            return Note(NO_DIFFERENCE, text)
        if found == 0 and food != "none":
            what = FOOD_TO[food] + (f" {WANT_WORDS[want].lower()}" if want != "none" else "")
            return Note("fewer_than_three", f"{what.capitalize()} в каталоге не нашлось")
        return fewer_note(found, limit)


def shelf_tile(tile: Mapping[str, object], pick: ShelfPick) -> dict[str, object]:
    """Плитка `shelf`: общая плитка рекомендаций плюс блюда и источник направления."""
    return {**tile, "dishes": list(pick.dishes), "want_source": pick.want_source}
