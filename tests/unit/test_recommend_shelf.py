"""«Сомелье у полки» (`app.recommend.shelf`): блюда правил сочетаний, направления, пул и фразы.

Справочник собирается прямо в памяти: якорь — сухой Сира 14° Альфы, у остальных вин пары с
блюдами (`reco_env.somm_data`: вердикт `yes` к блюду группы чипа) и факты подобраны так, чтобы
было видно каждое правило:

    a-syrah-2      Альфа — своя винодельня, в выдачу не идёт
    b-syrah-semi   полусухое Сира: помягче и послаще по каталогу (сахар), не посвежее
    c-merlot       сухое Мерло 14°: помягче по сорту (танины 3,0 против 3,8)
    d-pinot        сухое Пино Нуар 12,5°: посвежее и по каталогу (крепость), и по сорту;
                   помягче по сорту; только к рыбе
    e-saperavi     полусладкое Саперави: послаще; по танинам — не помягче (4,2 против 3,8)
    f-saperavi     сухое Саперави 15°: крепче и таниннее — ни в какую сторону
    g-riesling     белое сухое 12°: только в расширенном пуле, к рыбе
    i-muscat       белое сладкое 16°: послаще всех, в расширенном пуле
    h-no-photo     фото выгрузки нет — не в пуле
"""

from __future__ import annotations

import pytest
from reco_env import row, somm_data

from app.recommend.catalog import SUGAR_STEPS, RecoCatalog, wine_of
from app.recommend.content_filter import check
from app.recommend.facts import Note
from app.recommend.shelf import (
    FOOD_REASONS,
    FOODS,
    NO_PAIRS,
    NOT_EVALUABLE,
    REASON_WANTS,
    SOFTER_MAX_STEP,
    WANTS,
    Shelf,
    not_evaluable_text,
)

#: К чему подходит вино по правилам сочетаний: slug → группы чипа.
PAIRS: dict[str, tuple[str, ...]] = {}


def wine(slug, winery, *, foods, photo=True, **fields):
    PAIRS[slug] = tuple(foods)
    return wine_of(row(slug, winery, photo=photo, **fields))


WINES = [
    wine("a-syrah", "Альфа", title="Альфа Сира", grapes=["syrah"], color="Красное", abv=14.0,
         foods=["meat", "cheese"]),
    wine("a-syrah-2", "Альфа", title="Альфа Сира Второе", grapes=["syrah"], color="Красное",
         sugar="polusuhoe", abv=13.0, foods=["meat", "cheese", "fish"]),
    wine("b-syrah-semi", "Бета", title="Бета Сира", grapes=["syrah"], color="Красное",
         sugar="polusuhoe", abv=14.0, foods=["meat"]),
    wine("c-merlot", "Гамма", title="Гамма Мерло", grapes=["merlot"], color="Красное",
         abv=14.0, foods=["meat"]),
    wine("d-pinot", "Дельта", title="Дельта Пино", grapes=["pinot_noir"], color="Красное",
         abv=12.5, foods=["fish"]),
    wine("e-saperavi", "Эпсилон", title="Эпсилон Саперави", grapes=["saperavi"],
         color="Красное", sugar="polusladkoe", abv=12.0, foods=["cheese"]),
    wine("f-saperavi", "Дзета", title="Дзета Саперави", grapes=["saperavi"], color="Красное",
         abv=15.0, foods=["meat"]),
    wine("g-riesling", "Эта", title="Эта Рислинг", grapes=["riesling"], color="Белое",
         abv=12.0, foods=["fish"]),
    wine("i-muscat", "Йота", title="Йота Мускат", grapes=["muscat"], color="Белое",
         sugar="sladkoe", abv=16.0, foods=["cheese"]),
    wine("h-no-photo", "Тета", title="Тета Мерло", grapes=["merlot"], color="Красное",
         abv=13.0, foods=["meat"], photo=False),
]  # fmt: skip


def make_shelf(wines) -> Shelf:
    return Shelf(RecoCatalog(wines), somm=somm_data(PAIRS))


@pytest.fixture(scope="module")
def shelf() -> Shelf:
    return make_shelf(WINES)


def answer(shelf: Shelf, slug: str, food: str, want: str):
    return shelf.answer(shelf.catalog.get(slug), food, want)


def slugs(result) -> list[str]:
    return [pick.wine.slug for pick in result.picks]


# ------------------------------------------------------------------ блюда
def test_food_chips_follow_pairing_rules(shelf):
    """Вино подходит чипу, если у него есть пара `yes` с блюдом этой группы; оговорка — нет."""
    wines = {w.slug: w for w in WINES}
    assert shelf.goes_with(wines["b-syrah-semi"], "meat")
    assert not shelf.goes_with(wines["d-pinot"], "meat")
    assert shelf.goes_with(wines["d-pinot"], "none")
    # к борщу у всех «оговорка»: чипа у борща нет, и «к рыбе» оговорка не открывает
    assert not shelf.goes_with(wines["c-merlot"], "fish")
    assert shelf.dishes_of(wines["a-syrah"]) == ("Шашлык", "Сырная тарелка", "Борщ")
    assert FOOD_REASONS == {
        "meat": "К мясу — по правилам сочетаний",
        "fish": "К рыбе и морепродуктам — по правилам сочетаний",
        "cheese": "К сырам — по правилам сочетаний",
    }


def test_food_filter_without_direction(shelf):
    result = answer(shelf, "a-syrah", "meat", "none")
    # своя винодельня, вино без фото и вина без мяса — нет; ближе всех тот же сорт
    assert slugs(result) == ["b-syrah-semi", "c-merlot", "f-saperavi"]
    assert result.pool == "near" and result.notes == ()
    first = result.picks[0]
    assert first.reasons[0] == "К мясу — по правилам сочетаний"
    assert first.want_source is None and first.dishes == ("Шашлык", "Борщ")


def test_without_pairing_data_food_chips_say_so():
    """Данных правил сочетаний нет: блюдо выбрано — пусто и честная фраза, без блюда — подбор."""
    local = Shelf(RecoCatalog(WINES))
    anchor = local.catalog.get("a-syrah")
    for result in (local.answer(anchor, "meat", "softer"), local.plain(anchor, "fish")):
        assert result.picks == () and result.notes == (NO_PAIRS,)
    assert NO_PAIRS.text == "Подбор к блюдам недоступен: нет данных правил сочетаний"
    assert check(NO_PAIRS.text).clean
    assert slugs(local.answer(anchor, "none", "softer"))
    assert all(pick.dishes == () for pick in local.answer(anchor, "none", "none").picks)


# ------------------------------------------------------------------ направления
def test_softer_catalog_first_then_by_grape(shelf):
    result = answer(shelf, "a-syrah", "none", "softer")
    assert slugs(result) == ["b-syrah-semi", "c-merlot", "d-pinot"]
    assert [pick.want_source for pick in result.picks] == ["catalog", "grape", "grape"]
    assert result.picks[0].reasons[0] == "Помягче: полусухое, а не сухое"
    assert result.picks[1].reasons[0] == "Помягче по сорту: танины ниже — Мерло, а не Сира"
    assert "e-saperavi" not in slugs(result)  # сахар выше на две ступени и танины выше


def softer_ceiling(anchor) -> int:
    """Потолок «помягче» по правилу, независимо от `move_of`: полусухое или сахар якоря."""
    semidry = SUGAR_STEPS["polusuhoe"]
    return semidry if anchor.sugar is None else max(semidry, SUGAR_STEPS[anchor.sugar])


SEMI_SWEET_MERLOT = wine("k-merlot-semisweet", "Каппа", title="Каппа Мерло", grapes=["merlot"],
                         color="Красное", sugar="polusladkoe", abv=12.0,
                         foods=["meat"])  # fmt: skip
SWEET_MERLOT = wine("m-merlot-sweet", "Мю", title="Мю Мерло", grapes=["merlot"], color="Красное",
                    sugar="sladkoe", abv=12.0, foods=["meat"])  # fmt: skip


def test_softer_ceiling_is_semidry_or_anchor_sugar():
    assert SOFTER_MAX_STEP == SUGAR_STEPS["polusuhoe"]
    wines = {w.slug: w for w in WINES}
    assert softer_ceiling(wines["a-syrah"]) == SUGAR_STEPS["polusuhoe"]  # сухое
    assert softer_ceiling(wines["e-saperavi"]) == SUGAR_STEPS["polusladkoe"]
    assert softer_ceiling(wines["i-muscat"]) == SUGAR_STEPS["sladkoe"]


def test_softer_dry_anchor_never_gets_semisweet_even_by_grape():
    """Сухому якорю «помягче» не бывает полусладким и по сорту: Мерло с танинами ниже — нет.

    Когда-то потолок держал только путь каталога, и к сухому Сира к мясу третьим шло
    полусладкое Мерло «по сорту». Вино с неизвестным сахаром потолок не проверяет — оно остаётся.
    """
    semi_sweet = SEMI_SWEET_MERLOT
    unknown = wine("l-merlot-unknown", "Лямбда", title="Лямбда Мерло", grapes=["merlot"],
                   color="Красное", sugar=None, abv=13.0, foods=["meat"])  # fmt: skip
    local = make_shelf([*WINES, semi_sweet])
    result = local.answer(local.catalog.get("a-syrah"), "meat", "softer")
    assert slugs(result) == ["b-syrah-semi", "c-merlot"]
    assert result.notes == (Note("fewer_than_three", "Нашлось меньше трёх"),)
    # тот же подбор, но у третьего Мерло сахар неизвестен — по сорту оно проходит, как раньше
    local = make_shelf([*WINES, unknown])
    result = local.answer(local.catalog.get("a-syrah"), "meat", "softer")
    assert slugs(result) == ["b-syrah-semi", "c-merlot", "l-merlot-unknown"]
    assert result.picks[2].want_source == "grape"
    # у якоря без сахара потолок — полусухое: полусладкое Мерло и тут не «помягче»
    anchor = wine("x-syrah", "Икс", title="Икс Сира", grapes=["syrah"], color="Красное",
                  sugar=None, abv=14.0, foods=["cheese"])  # fmt: skip
    local = make_shelf([anchor, semi_sweet, unknown])
    assert local.move("softer", anchor, semi_sweet) is None
    assert local.move("softer", anchor, unknown) is not None


def test_softer_semisweet_anchor_gets_same_sugar_by_grape_never_sweet():
    """Полусладкому якорю «помягче» — полусладкое мягче по сорту, но никогда не сладкое.

    Регрессия решения 24.09: потолок — сахар самого якоря, а не полусухое. Раньше полусладкому
    Саперави «помягче» не подбиралось вовсе; сладкое Мерло с теми же танинами — это «послаще».
    """
    local = make_shelf([*WINES, SEMI_SWEET_MERLOT, SWEET_MERLOT])
    saperavi = local.catalog.get("e-saperavi")
    result = local.answer(saperavi, "none", "softer")
    assert slugs(result) == ["k-merlot-semisweet"]
    assert result.picks[0].want_source == "grape"
    assert result.picks[0].reasons[0] == "Помягче по сорту: танины ниже — Мерло, а не Саперави"
    assert result.notes == (Note("fewer_than_three", "Нашлось меньше трёх"),)
    assert local.move("softer", saperavi, SWEET_MERLOT) is None  # ступень вверх — «послаще»
    assert local.move("sweeter", saperavi, SWEET_MERLOT) is not None
    # и сухому Сира ни одно из двух Мерло не «помягче»
    syrah = local.catalog.get("a-syrah")
    assert local.move("softer", syrah, SEMI_SWEET_MERLOT) is None
    assert local.move("softer", syrah, SWEET_MERLOT) is None


def test_softer_semisweet_anchor_without_candidates_says_general_phrase(shelf):
    """Полусладких мягче по сорту нет — фраза общая: сладкий Мускат не идёт, остальные суше."""
    result = answer(shelf, "e-saperavi", "none", "softer")
    assert result.picks == ()
    assert result.notes == (
        Note("no_difference", "По описаниям разницы нет — помягче в каталоге не найти"),
    )


def test_fresher_both_sources_and_widening(shelf):
    result = answer(shelf, "a-syrah", "fish", "fresher")
    # к рыбе из красных — только Пино; расширение добавляет белый Рислинг
    assert slugs(result) == ["d-pinot", "g-riesling"]
    assert result.pool == "catalog"
    pinot = result.picks[0]
    assert pinot.want_source == "catalog"
    assert pinot.reasons[:3] == (
        "К рыбе и морепродуктам — по правилам сочетаний",
        "Посвежее: крепость 12,5° против 14°",
        "Посвежее по сорту: кислотность выше — Пино Нуар, а не Сира",
    )
    assert result.picks[1].reasons[3] == "Белое, а не красное"
    assert [note.code for note in result.notes] == ["pool_expanded", "fewer_than_three"]
    assert result.notes[0].text == (
        "Среди близких по стилю не нашлось — подобрали из всего каталога"
    )


def test_sweeter_widens_to_other_styles(shelf):
    result = answer(shelf, "a-syrah", "none", "sweeter")
    assert slugs(result) == ["b-syrah-semi", "e-saperavi", "i-muscat"]
    assert result.pool == "catalog" and [n.code for n in result.notes] == ["pool_expanded"]
    assert result.picks[1].reasons[0] == "Послаще: полусладкое, а не сухое"
    assert all(pick.want_source == "catalog" for pick in result.picks)


def test_nothing_new_from_widening_keeps_near_pool(shelf):
    """Расширение ничего не добавило — пул «близкий», а честная фраза — «меньше трёх»."""
    result = answer(shelf, "a-syrah", "meat", "sweeter")
    assert slugs(result) == ["b-syrah-semi"]
    assert result.pool == "near"
    assert result.notes == (Note("fewer_than_three", "Нашлось меньше трёх"),)


def test_no_difference_phrases(shelf):
    sweet = answer(shelf, "i-muscat", "none", "sweeter")
    assert sweet.picks == () and sweet.notes == (
        Note("no_difference", "Это вино уже сладкое: послаще в каталоге по описаниям не найти"),
    )
    # «помягче» сладкому — только сладкое мягче по сорту; такого нет, и фраза общая
    softer = answer(shelf, "i-muscat", "none", "softer")
    assert softer.picks == () and softer.notes == (
        Note("no_difference", "По описаниям разницы нет — помягче в каталоге не найти"),
    )
    # сухому Мерло рядом только сухое Саперави с танинами выше — мягче некуда, фраза общая
    merlot = next(w for w in WINES if w.slug == "c-merlot")
    saperavi = next(w for w in WINES if w.slug == "f-saperavi")
    dry = make_shelf([merlot, saperavi]).answer(merlot, "none", "softer")
    assert dry.picks == () and dry.notes == (
        Note("no_difference", "По описаниям разницы нет — помягче в каталоге не найти"),
    )


# ------------------------------------------------------------------ направление не оценить
#: Купаж без сортов в приорах: оси «по сорту» у него нет, и сдвиг считается только от фактов.
def blend(slug="x-blend", **fields):
    base = {"title": "Икс Купаж", "grapes": [], "color": "Красное", "foods": ["meat", "cheese"]}
    return wine(slug, "Икс", **{**base, **fields})


@pytest.mark.parametrize(
    ("fields", "want", "text"),
    [
        ({"sugar": None}, "sweeter",
         "Сахар этого вина в каталоге не указан — послаще подобрать не по чему"),
        ({"sugar": None}, "softer",
         ("Сахар этого вина в каталоге не указан, а по сорту мягкость не оценить — помягче "
          "подобрать не по чему")),
        ({"sugar": "polusladkoe"}, "softer",
         "По сорту мягкость этого вина не оценить — помягче подобрать не по чему"),
        ({"sugar": None, "abv": None}, "fresher",
         ("Сахар и крепость этого вина в каталоге не указаны, а по сорту свежесть не оценить — "
          "посвежее подобрать не по чему")),
        ({"sugar": "brut_nature", "sparkling": True, "abv": None}, "fresher",
         ("Крепость этого вина в каталоге не указана, а по сорту свежесть не оценить — посвежее "
          "подобрать не по чему")),
    ],
)  # fmt: skip
def test_direction_without_facts_is_not_evaluable_not_no_difference(fields, want, text):
    """Сдвиг не от чего считать — фраза `not_evaluable`, а не «по описаниям разницы нет».

    Проверка 24.09: у 376 вин без сахара «послаще» отвечало «разницы нет — послаще в каталоге не
    найти», хотя слаще вина в каталоге есть; сравнить просто не с чем.
    """
    anchor = blend(**fields)
    local = make_shelf([*WINES, anchor])
    assert not local.evaluable(want, anchor)
    for food in FOODS:
        result = local.answer(anchor, food, want)
        assert result.picks == ()
        assert result.notes == (Note(NOT_EVALUABLE, text),), food
        assert check(text).clean and "%" not in text
    # не оценить — значит, ни одно вино каталога правило сдвига и не пропустило бы
    assert all(local.move(want, anchor, other) is None for other in local.catalog)


def test_one_basis_is_enough_to_evaluate(shelf):
    """Сахар, крепость или ось «по сорту» — и направление оценивается как раньше."""
    strong, dry, semi = (
        blend("x-strong", sugar=None, abv=14.0),
        blend("x-dry", sugar="suhoe"),
        blend("x-semi", sugar="polusuhoe"),
    )
    local = make_shelf([*WINES, strong, dry, semi])
    assert local.evaluable("fresher", strong)  # крепость
    assert local.evaluable("softer", dry)  # ступень сахара вверх
    assert not local.evaluable("softer", semi)  # вверх от полусухого — уже «послаще»
    syrah = shelf.catalog.get("a-syrah")
    assert all(shelf.evaluable(want, syrah) for want in (*WANTS, "none"))
    no_sugar = wine("x-syrah", "Икс", title="Икс Сира", grapes=["syrah"], color="Красное",
                    sugar=None, abv=None, foods=["meat"])  # fmt: skip
    local = make_shelf([*WINES, no_sugar])
    assert local.evaluable("softer", no_sugar) and local.evaluable("fresher", no_sugar)
    assert not local.evaluable("sweeter", no_sugar)
    assert not_evaluable_text("sweeter", no_sugar).startswith("Сахар этого вина")


def test_direction_exists_but_not_with_this_food(shelf):
    result = answer(shelf, "a-syrah", "cheese", "fresher")
    assert result.picks == ()
    assert result.notes == (Note("fewer_than_three", "К сыру посвежее в каталоге не нашлось"),)


def test_fresher_never_sweeter_or_stronger(shelf):
    result = answer(shelf, "a-syrah", "meat", "fresher")
    assert "b-syrah-semi" not in slugs(result)  # слаще
    assert "f-saperavi" not in slugs(result)  # крепче на 1°


def test_reason_moves_follow_the_reason(shelf):
    """Подборка сомелье от причины «−» (`reason_move`, третий круг проверки 25.09): сдвиг — только
    по доказательствам причины и только заметный. «Полегче» — крепость ниже на 0,5° или тело ниже
    по сорту на 0,5, «помощнее» — наоборот, «суше» — сахар ниже по карточке, «помягче» — белое
    вместо красного или танины ниже, «посвежее» — кислотность выше; против направления — никак.
    Чипами `/shelf` эти направления не приходят."""
    get = shelf.catalog.get
    syrah, pinot, strong, semi = (
        get("a-syrah"),
        get("d-pinot"),
        get("f-saperavi"),
        get("e-saperavi"),
    )
    merlot, riesling = get("c-merlot"), get("g-riesling")
    both = ("strength", "body")

    lighter = shelf.reason_move("lighter", both, syrah, pinot)
    assert lighter is not None and lighter.strength and lighter.axis == "body"
    assert lighter.catalog == "Полегче: крепость 12,5° против 14°"
    assert lighter.grape == "Полегче по сорту: тело ниже — Пино Нуар, а не Сира"
    assert shelf.reason_move("lighter", both, syrah, strong) is None  # крепче на 1°
    # Мерло легче Сиры по сорту на 0,4 — это шум порога, а не «тело легче».
    assert shelf.reason_move("lighter", both, syrah, merlot) is None
    # Причина — крепость: «полегче» только слабее, тело не в счёт.
    only_abv = shelf.reason_move("lighter", ("strength",), syrah, pinot)
    assert only_abv is not None and only_abv.grape is None and only_abv.strength
    fuller = shelf.reason_move("fuller", both, pinot, syrah)
    assert fuller is not None and fuller.catalog == "Помощнее: крепость 14° против 12,5°"
    assert shelf.reason_move("fuller", both, syrah, pinot) is None

    drier = shelf.reason_move("drier", ("sugar",), semi, syrah)
    assert drier is not None and drier.sugar and drier.catalog == "Суше: сухое, а не полусладкое"
    assert shelf.reason_move("drier", ("sugar",), syrah, semi) is None
    # Сахар вина карточки — по креплёному названию: кандидат суше по своей карточке.
    port = wine("x-port", "Икс", title="Икс Портвейн", grapes=["syrah"], color="Красное",
                sugar=None, abv=None, foods=["cheese"])  # fmt: skip
    by_name = make_shelf([*WINES, port]).reason_move("drier", ("sugar",), port, syrah, "sladkoe")
    assert by_name is not None and by_name.catalog == "Суше: сухое по карточке"

    softer = shelf.reason_move("softer", ("colour", "tannin"), syrah, riesling)
    assert softer is not None and softer.colour and softer.catalog == "Помягче: белое, а не красное"
    assert softer.axis == "tannin" and softer.tier == 0
    tannin = shelf.reason_move("softer", ("colour", "tannin"), syrah, merlot)
    assert tannin is not None and not tannin.colour and tannin.axis == "tannin"
    assert shelf.reason_move("softer", ("colour", "tannin"), merlot, syrah) is None  # терпче

    fresher = shelf.reason_move("fresher", ("acidity",), merlot, pinot)
    assert fresher is not None and fresher.grape.startswith("Посвежее по сорту: кислотность выше")
    assert shelf.reason_move("fresher", ("acidity",), merlot, syrah) is None  # +0,2 — шум
    assert not set(REASON_WANTS) & set(WANTS)


def test_reason_strength_is_a_real_half_degree():
    """«Крепость ниже» — на полградуса с запасом диапазона: «13–14°» против «14°» — не ниже, а
    12,2° против 12,7° — ровно полградуса, хотя во float разница 0,4999…"""
    anchor = wine("x-anchor", "Икс", title="Икс Сира", grapes=["syrah"], color="Красное",
                  abv=12.7, foods=["meat"])  # fmt: skip
    near = wine("y-near", "Игрек", title="Игрек Сира", grapes=["syrah"], color="Красное",
                abv=12.2, foods=["meat"])  # fmt: skip
    ranged = wine("z-range", "Зет", title="Зет Сира", grapes=["syrah"], color="Красное",
                  abv=[12.0, 13.0], foods=["meat"])  # fmt: skip
    local = make_shelf([*WINES, anchor, near, ranged])
    moved = local.reason_move("lighter", ("strength",), anchor, near)
    assert moved is not None and moved.catalog == "Полегче: крепость 12,2° против 12,7°"
    assert local.reason_move("lighter", ("strength",), anchor, ranged) is None
    assert local.reason_move("fuller", ("strength",), near, ranged) is None


def test_grape_axis_needs_known_grapes(shelf):
    """Без сортов в приорах ось «по сорту» не считается: остаются только факты каталога."""
    anchor = wine("x-blend", "Икс", title="Икс Купаж", grapes=[], color="Красное", abv=14.0,
                  foods=["cheese"])  # fmt: skip
    local = make_shelf([*WINES, anchor])
    result = local.answer(anchor, "none", "softer")
    # Мерло и Пино «помягче» только по сорту — у купажа без сортов этой оси нет
    assert set(slugs(result)) == {"a-syrah-2", "b-syrah-semi"}
    assert {pick.want_source for pick in result.picks} == {"catalog"}


def test_grape_reason_names_the_difference_beyond_two_grapes():
    """Первые два сорта общие — объяснение называет все, а не «X и Y, а не X и Y»."""
    anchor = wine("x-cs-merlot", "Икс", title="Икс Купаж", grapes=["cabernet_sauvignon", "merlot"],
                  color="Красное", abv=14.0, foods=["cheese"])  # fmt: skip
    blend = wine("y-cs-merlot-pinot", "Игрек", title="Игрек Купаж",
                 grapes=["cabernet_sauvignon", "merlot", "pinot_noir"], color="Красное",
                 abv=14.0, foods=["cheese"])  # fmt: skip
    result = make_shelf([anchor, blend]).answer(anchor, "none", "softer")
    assert slugs(result) == ["y-cs-merlot-pinot"]
    assert result.picks[0].reasons[0] == (
        "Помягче по сорту: танины ниже — Каберне Совиньон, Мерло и Пино Нуар, "
        "а не Каберне Совиньон и Мерло"
    )


# ------------------------------------------------------------------ обычная сортировка
def test_plain_is_category_with_food_by_name(shelf):
    result = shelf.plain(shelf.catalog.get("a-syrah"), "meat")
    assert slugs(result) == ["c-merlot", "f-saperavi"]  # «Гамма…», «Дзета…»; полусухое — нет
    assert all(pick.reasons == () and pick.want_source is None for pick in result.picks)
    assert result.notes == (Note("fewer_than_three", "Нашлось меньше трёх"),)


# ------------------------------------------------------------------ все сочетания
def test_every_combination_is_three_or_honest_and_clean():
    # с двумя Мерло полусладкому якорю есть что подобрать «помягче», а сладкое — выше потолка
    local = make_shelf([*WINES, SEMI_SWEET_MERLOT, SWEET_MERLOT])
    for anchor in local.catalog.pool:
        for food in FOODS:
            for want in WANTS:
                result = local.answer(anchor, food, want)
                if len(result.picks) < 3:
                    codes = {note.code for note in result.notes}
                    assert codes & {"fewer_than_three", "no_difference", "not_evaluable"}, (
                        anchor.slug,
                        food,
                        want,
                    )
                texts = [n.text for n in result.notes]
                texts += [r for pick in result.picks for r in pick.reasons]
                for text in texts:
                    assert check(text).clean and "%" not in text, text
                for pick in result.picks:
                    assert pick.wine.winery_norm != anchor.winery_norm
                    assert (pick.want_source is None) == (want == "none")
                    if food != "none":
                        assert local.goes_with(pick.wine, food)
                    if want == "softer" and pick.wine.sugar is not None:
                        step = SUGAR_STEPS[pick.wine.sugar]
                        assert step <= softer_ceiling(anchor), (anchor.slug, pick.wine.slug)
                        if anchor.sugar is not None:
                            assert step >= SUGAR_STEPS[anchor.sugar], pick.wine.slug  # не суше
