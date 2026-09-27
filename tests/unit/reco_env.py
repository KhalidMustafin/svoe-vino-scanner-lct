"""Справочник рекомендаций на фейках — для тестов слоя «после поиска».

Только выгрузка организатора (решение 24.09): факты вина — записи разметки (`gt_tokens.jsonl`),
сахар и игристость — по правилу выгрузки из названия и slug, группы — наш словарь
(`wines.jsonl`, `wine_groups.json`). Блюда чипа «К чему?» — правила сочетаний (`data/somm/`).

Шесть позиций `api_env.API_RECORDS` (Альфа Долина, Бета Холмы, Гамма Берег) — без сахара в
названии и slug, как 376 настоящих позиций выгрузки. Плюс вина других виноделен, на которых
видно каждое правило подбора; сахар у них — последним словом slug, как у выгрузки:

    delta-merlot-suhoe             Дельта Сад: вторая позиция винодельни в выгрузке
    delta-merlot                   Дельта Сад: карточка живого портала вне выгрузки — есть в
                                   словаре групп, но не в разметке, и в справочник не входит
    eta-merlot-suhoe, -magnum-…    Эта: одна группа wine_id, магнум неканонический — не в пуле
    theta-merlot-suhoe             Тета: у позиции нет фото выгрузки — не в пуле
    zeta-merlot-polusuhoe          Зета: полусухое — сахар в пределах ступени
    iota-merlot-sladkoe            Йота: сладкое — сахар дальше ступени
    epsilon-blend-suhoe            Эпсилон: Мерло + Саперави, Крым
    kappa-brut, lambda-rose-brut   игристые брют: белое и розовое (брют — в slug)
    gamma-brut                     игристое Гаммы — для «Не тупика» по винодельне
    mu-orange-suhoe                оранжевое тихое

Модуль не собирается pytest (имя не начинается с `test_`).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from catalog import wine_record
from learned_synth import ALFA, BETA, GAMMA

from app.recommend.somm_data import FIXTURE_DIR, DishInfo, Pair, SommData, WinePairs

if TYPE_CHECKING:
    from app.api.service import ScannerService

#: Путь карты фото у правленой карточки: фото не из выгрузки (правка эталонов 23.09).
FOREIGN_PHOTO = "packshots_fixed"


def row(
    slug: str,
    winery: str,
    *,
    title: str,
    grapes: list[str],
    color: str,
    sugar: str | None = "suhoe",
    sparkling: bool = False,
    abv: float | list[float] | None = 13.0,
    region: str = "Кубань",
    wine_id: str | None = None,
    canonical: bool = True,
    in_csv: bool = True,
    photo: bool = True,
) -> dict[str, Any]:
    """Строка фактов вина в форме `organizer_row` — и строка словаря групп `wines.jsonl`."""
    return {
        "slug": slug,
        "wine_id": wine_id or f"W-{slug}",
        "canonical": slug,
        "is_canonical": canonical,
        "in_csv": in_csv,
        "winery": winery,
        "winery_norm": winery.casefold(),
        "title": title,
        "grapes": grapes,
        "grapes_src": "csv",
        "color": color,
        "sugar": sugar,
        "sparkling": sparkling,
        "abv": abv,
        "region": region,
        "photo": f"{slug}.webp" if photo else None,
    }


ROWS: list[dict[str, Any]] = [
    row("alfa-muskat", "Альфа Долина", title="Мускат", grapes=["muscat"], color="Белое",
        sugar=None, abv=12.0, region="Крым"),
    row("alfa-riesling", "Альфа Долина", title="Рислинг", grapes=["riesling"], color="Белое",
        sugar=None, abv=11.5, region="Крым"),
    row("beta-merlot", "Бета Холмы", title="Мерло Терруар", grapes=["merlot"], color="Красное",
        sugar=None, abv=13.5),
    row("beta-shardone", "Бета Холмы", title="Шардоне Резерв", grapes=["chardonnay"],
        color="Белое", sugar=None, abv=12.5),
    row("gamma-saperavi", "Гамма Берег", title="Саперави", grapes=["saperavi"], color="Красное",
        sugar=None, abv=13.0, region="Крым"),
    row("gamma-rkatsiteli", "Гамма Берег", title="Ркацители Янтарь", grapes=["rkatsiteli"],
        color="Белое", sugar=None, abv=12.0, region="Крым"),
    row("gamma-brut", "Гамма Берег", title="Гамма Брют", grapes=["chardonnay"], color="Белое",
        sugar="brut", sparkling=True, abv=12.0, region="Крым"),
    row("delta-merlot", "Дельта Сад", title="Дельта Мерло", grapes=["merlot"], color="Красное",
        abv=None, in_csv=False, wine_id="W-delta"),
    row("delta-merlot-suhoe", "Дельта Сад", title="Дельта Мерло Второе", grapes=["merlot"],
        color="Красное", abv=13.5),
    row("eta-merlot-suhoe", "Эта", title="Эта Мерло", grapes=["merlot"], color="Красное",
        abv=13.5, wine_id="W-eta"),
    row("eta-merlot-magnum-suhoe", "Эта", title="Эта Мерло Магнум", grapes=["merlot"],
        color="Красное", abv=13.5, wine_id="W-eta", canonical=False),
    row("theta-merlot-suhoe", "Тета", title="Тета Мерло", grapes=["merlot"], color="Красное",
        abv=13.5, photo=False),
    row("zeta-merlot-polusuhoe", "Зета", title="Зета Мерло", grapes=["merlot"],
        color="Красное", sugar="polusuhoe", abv=12.0),
    row("iota-merlot-sladkoe", "Йота", title="Йота Мерло", grapes=["merlot"], color="Красное",
        sugar="sladkoe", abv=16.0),
    row("epsilon-blend-suhoe", "Эпсилон", title="Эпсилон Купаж", grapes=["merlot", "saperavi"],
        color="Красное", abv=14.0, region="Крым"),
    row("kappa-brut", "Каппа", title="Каппа Брют", grapes=["chardonnay"], color="Белое",
        sugar="brut", sparkling=True, abv=12.0),
    row("lambda-rose-brut", "Лямбда", title="Лямбда Розе Брют", grapes=["pinot_noir"],
        color="Розовое", sugar="brut", sparkling=True, abv=12.0),
    row("mu-orange-suhoe", "Мю", title="Мю Оранж", grapes=["rkatsiteli"], color="Оранжевое",
        abv=12.5, region="Крым"),
]  # fmt: skip

#: Позиции выгрузки (в разметке); `delta-merlot` — живая карточка портала, её там нет.
ORGANIZER = [item for item in ROWS if item["in_csv"]]

#: Винодельни шести позиций `api_env.API_RECORDS` — те же токены, что у словаря сканера.
_WINERY_FAKES = {"Альфа Долина": ALFA, "Бета Холмы": BETA, "Гамма Берег": GAMMA}

#: Блюда правил сочетаний: по одному на каждую группу чипа и одно без группы.
DISHES: list[dict[str, Any]] = [
    {"id": "shashlyk", "name": "Шашлык", "dative": "к шашлыку", "category": "main_meat",
     "food": "meat", "family": "shashlyk", "aliases": ["шашлык"]},
    {"id": "oysters", "name": "Устрицы", "dative": "к устрицам", "category": "seafood_raw",
     "food": "fish", "family": "oysters", "aliases": ["устрицы"]},
    {"id": "cheese_plate", "name": "Сырная тарелка", "dative": "к сырной тарелке",
     "category": "appetizer", "food": "cheese", "family": "cheese_plate",
     "aliases": ["сырная тарелка"]},
    {"id": "medovik", "name": "Медовик", "dative": "к медовику", "category": "dessert",
     "food": "dessert", "family": "medovik", "aliases": ["медовик"]},
    {"id": "borsch", "name": "Борщ", "dative": "к борщу", "category": "soup", "food": None,
     "family": "borsch", "aliases": ["борщ"]},
]  # fmt: skip
DISH_OF_FOOD = {"meat": "shashlyk", "fish": "oysters", "cheese": "cheese_plate",
                "dessert": "medovik"}  # fmt: skip

#: К чему подходят вина по правилам сочетаний (пары `yes`); у остальных пар — оговорка.
FOODS: dict[str, tuple[str, ...]] = {
    "beta-merlot": ("meat", "cheese"),
    "delta-merlot-suhoe": ("meat",),
    "eta-merlot-suhoe": ("meat", "fish"),
    "zeta-merlot-polusuhoe": ("meat",),
    "iota-merlot-sladkoe": ("meat", "dessert"),
    "epsilon-blend-suhoe": ("cheese",),
    "gamma-saperavi": ("meat",),
    "alfa-riesling": ("fish",),
}


def pairs_json(foods: Mapping[str, Iterable[str]]) -> dict[str, Any]:
    """`pairs.json` договора: у каждого вина `yes` к блюду своей группы, к борщу — оговорка."""
    wines: dict[str, Any] = {}
    for slug, groups in foods.items():
        yes = [DISH_OF_FOOD[food] for food in groups]
        dishes = {dish: ["yes", ["intensity_match"], []] for dish in yes}
        dishes["borsch"] = ["caveat", ["intensity_match"], ["umami_vs_tannin_clash"]]
        wines[slug] = {"top": [*yes, "borsch"], "dishes": dishes}
    return {"version": 1, "built": "фейки тестов", "wines": wines}


def dishes_json() -> dict[str, Any]:
    """`dishes.json` договора: фейковые блюда и настоящие правила заглушки договора."""
    fixture = json.loads((FIXTURE_DIR / "dishes.json").read_text(encoding="utf-8"))
    return {"version": 1, "source": "фейки тестов", "categories": fixture["categories"],
            "dishes": DISHES, "rules": fixture["rules"]}  # fmt: skip


def somm_data(foods: Mapping[str, Iterable[str]]) -> SommData:
    """Данные сомелье в памяти: пары и блюда, остальное пусто."""
    dishes = {
        item["id"]: DishInfo(**{**item, "aliases": tuple(item["aliases"])}) for item in DISHES
    }
    pairs = {
        slug: WinePairs(
            top=tuple(body["top"]),
            dishes={
                dish: Pair(verdict, tuple(plus), tuple(minus))
                for dish, (verdict, plus, minus) in body["dishes"].items()
            },
        )
        for slug, body in pairs_json(foods)["wines"].items()
    }
    return SommData(pairs=pairs, dishes=dishes)


def record_of(item: Mapping[str, Any]) -> dict[str, Any]:
    """Запись разметки (`gt_tokens.jsonl`) для строки фактов: только поля выгрузки."""
    fakes = _WINERY_FAKES.get(item["winery"], {"winery": item["winery"],
                                              "key_tokens": (), "variants": ()})  # fmt: skip
    abv = item["abv"]
    record = wine_record(
        item["slug"],
        **fakes,
        name=item["title"],
        grapes=item["grapes"],
        abv=abv if isinstance(abv, list) else ([abv] if abv is not None else []),
        color=item["color"],
    )
    record["region"] = item["region"]
    record["photo_name"] = item["photo"] or ""
    return record


def write_reco(tmp_path: Path, *, photos: bool = True, somm: bool = True) -> Path:
    """Разметка, словарь групп, данные сомелье, карта фото и лёгкие фото в `tmp_path`.

    Карта фото правленой карточки (`beta-shardone`) ведёт в `packshots_fixed`: такое фото не
    из выгрузки и не отдаётся. Возвращает `tmp_path`.
    """
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir(exist_ok=True)
    with (tmp_path / "gt_tokens.jsonl").open("w", encoding="utf-8") as fh:
        for item in ORGANIZER:
            fh.write(json.dumps(record_of(item), ensure_ascii=False) + "\n")
            if photos and item["photo"]:
                (photo_dir / f"{item['slug']}.webp").write_bytes(
                    b"RIFF-small-" + item["slug"].encode()
                )
    with (tmp_path / "wines.jsonl").open("w", encoding="utf-8") as fh:
        for item in ROWS:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    groups: dict[str, dict[str, Any]] = {}
    for item in ROWS:
        group = groups.setdefault(item["wine_id"], {"wine_id": item["wine_id"], "members": []})
        group["members"].append(item["slug"])
    groups["W-delta"]["members"].insert(0, "delta-merlot-2024")  # член вне выгрузки
    (tmp_path / "wine_groups.json").write_text(
        json.dumps(groups, ensure_ascii=False), encoding="utf-8"
    )
    foreign = tmp_path / FOREIGN_PHOTO
    foreign.mkdir(exist_ok=True)
    (foreign / "beta-shardone.png").write_bytes(b"PNG-foreign")
    (tmp_path / "photo_map.csv").write_text(
        "slug,path,method\n"
        f"beta-merlot,{tmp_path / 'dump' / 'beta-merlot.webp'},sitemap\n"
        f"beta-shardone,{foreign / 'beta-shardone.png'},packshot_fix\n",
        encoding="utf-8",
    )
    if somm:
        directory = tmp_path / "somm"
        directory.mkdir(exist_ok=True)
        for name, body in (("pairs.json", pairs_json(FOODS)), ("dishes.json", dishes_json())):
            (directory / name).write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    return tmp_path


def reco_records(tmp_path: Path) -> list[dict[str, Any]]:
    """Записи разметки, которые положил `write_reco`."""
    text = (tmp_path / "gt_tokens.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def make_reco_service(
    tmp_path: Path, text: str = "Бета Холмы\nМерло", **overrides: Any
) -> ScannerService:
    """Сервис на фейках с загруженным слоем «после поиска» из `write_reco`.

    Карточки — из той же разметки, что справочник (как у настоящего сервиса); словарь,
    признаки и индекс сканера — шесть позиций `api_env.API_RECORDS`. Импорты внутри: тесты
    `app.recommend` берут отсюда только данные и не должны тянуть сервис и FastAPI.
    """
    from api_env import FakeOllama, make_service, settings_for

    from app.api.after_layer import AfterSearch
    from app.api.cards import CatalogCards

    write_reco(tmp_path)
    settings = settings_for(tmp_path, **overrides)
    cards = CatalogCards.build(reco_records(tmp_path), photo_map=settings.photo_map)
    service = make_service(FakeOllama(text), settings=settings, cards=cards)
    service.after = AfterSearch.load(settings, service.cards, service.attrs)
    return service
