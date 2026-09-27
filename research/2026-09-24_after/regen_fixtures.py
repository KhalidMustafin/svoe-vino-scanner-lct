"""Пересобрать заглушки «после поиска» (`tests/fixtures/after/`) живым кодом слоя.

Договор (§5) говорил: если в `facts.py` примут «одна винодельня — одно место» и приоритет
отличия фактом, заглушку `similar.json` нужно перегенерировать. Приняли — вот она, из того же
кода, что у маршрутов.

Д2 (25.09): заглушки `shelf.json` (Массандра, рыба + посвежее) и `shelf_empty.json` (послаще
сладкого) тоже собраны живым `AfterSearch.shelf`.

Д3: `scan_suggest.json` — ответ `/v1/scan` с подсказкой `suggest_not_found` на записанном кадре
поля R086 (Bel Colle Barolo, вина нет в каталоге; сканер отдал чужое вино с p = 0,68, счёт CV
лучшей серии 0,69). Кадр — из `tests/fixtures/after_recorded/ooc_frames.json`. `by_label.json` —
без `exclude`: поле необязательно, и заглушка обязана остаться прежней.

Без портала (решение 24.09): карточки, плитки и подбор — только выгрузка организатора и наши
правила. Поэтому пересобираются и `wine_card.json`, и карточки, кандидаты и `after` в
`scan_found.json`, `scan_check.json`, `scan_not_found.json` — по тем же доказательствам
(`evidence`) и top-5, что записаны в заглушке. `scan_not_found.json` по-прежнему показывает
экран, которого сервер не возвращает (калибровка Д2, `hint.md`): его `state` и `reasons`
записаны, остальное — живое.
Описание выгрузки в git не кладётся — в заглушке текст-заглушка.

Блюда чипа «К чему?» и плиток `shelf*` — `data/somm/` (`SVS_SOMM_DIR`), сборка движков
`scripts/build_somm.py` (24.09, 17:09). Её пересборка меняет и эти заглушки.

    SVS_DATA_DIR=... SVS_DATASET_DIR=... \\
        .venv/Scripts/python.exe research/2026-09-24_after/regen_fixtures.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from app.api.after_layer import AfterSearch, ByLabelBody  # noqa: E402
from app.api.cards import CatalogCards, read_records  # noqa: E402
from app.api.config import ServiceSettings  # noqa: E402
from app.api.service import Confidence, ScanResult, TopItem  # noqa: E402
from app.resolve.attrs import CatalogAttrs  # noqa: E402

FIXTURES = REPO / "tests" / "fixtures" / "after"
RECORDED = REPO / "tests" / "fixtures" / "after_recorded" / "ooc_frames.json"
SUGGEST_FRAME = "R086"
#: Описание выгрузки организатора в git не кладётся (договор, §9).
PLACEHOLDER = "Текст-заглушка вместо описания выгрузки организатора: выгрузка в git не кладётся."
ANCHOR = "shato-pino-shiraz-krasnoe-suhoe-14"
MASSANDRA = "massandra-muskatel-belyy-belye-sorta-vinograda-beloe-sladkoe-16"
ALVEUS = {"winery": "Фанагория", "color": "Оранжевое", "sugar": "brut", "grapes": [], "abv": 12,
          "sparkling": True}  # fmt: skip


def card(after: AfterSearch, slug: str | None) -> dict | None:
    body = after.card(slug)
    if body is not None and body.get("description"):
        body["description"] = PLACEHOLDER
    return body


def rescan(after: AfterSearch, name: str) -> dict:
    """Записанный ответ `/v1/scan` заново: те же доказательства и top-5, живые карточка,
    кандидаты и `after`.

    `scan_not_found` — показ экрана, которого сервер не возвращает: его `state` и `reasons`
    остаются записанными (счёт CV кадра 0,88 выше порога Д2), остальное `after` — живое.
    """
    body = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    result = ScanResult.model_validate({key: body[key] for key in ScanResult.model_fields})
    block = after.after(result)
    if name == "scan_not_found":
        block.update(state=body["after"]["state"], reasons=body["after"]["reasons"])
    return result.scan_body(
        card(after, result.slug), candidates=after.candidates(result), after=block
    )


def scan_suggest(after: AfterSearch) -> dict:
    """`/v1/scan` на записанном кадре R086: чужое вино, счёт CV ниже порога подсказки."""
    frames = json.loads(RECORDED.read_text(encoding="utf-8"))["frames"]
    frame = next(f for f in frames if f["query_id"].startswith(SUGGEST_FRAME))
    top5 = [TopItem(slug=slug, score=p) for slug, p in frame["top5"]]
    result = ScanResult(
        slug=top5[0].slug,
        confidence=Confidence(top1=frame["p_top1"], top5=round(sum(t.score for t in top5), 4)),
        margin=round(top5[0].score - top5[1].score, 4),
        top5=top5,
        outcome=frame["outcome"],
        evidence={
            "cv": {"top5": [{"slug": top5[0].slug, "score": frame["cv_top1"]}]},
            "vlm": frame["vlm"],
            "abstain": {"mode": "off", "fired": False, "best_visual": frame["cv_top1"]},
        },
    )
    return result.scan_body(
        card(after, result.slug), candidates=after.candidates(result), after=after.after(result)
    )


def main() -> int:
    settings = ServiceSettings.from_env(os.environ)
    records = read_records(settings.attrs_path)
    cards = CatalogCards.build(records, csv_path=settings.catalog_csv, photo_map=settings.photo_map)
    after = AfterSearch.load(settings, cards, CatalogAttrs.from_records(records))
    print("справочник:", after.stats())
    bodies = {
        "wine_card": card(after, MASSANDRA),
        "scan_found": rescan(after, "scan_found"),
        "scan_check": rescan(after, "scan_check"),
        "scan_not_found": rescan(after, "scan_not_found"),
        "scan_suggest": scan_suggest(after),
        "similar": after.similar(ANCHOR, limit=3, order="reco"),
        "similar_plain": after.similar(ANCHOR, limit=3, order="plain"),
        "by_label": after.by_label(ByLabelBody(**ALVEUS), limit=3, order="reco"),
        "shelf": after.shelf(MASSANDRA, food="fish", want="fresher", order="reco"),
        "shelf_empty": after.shelf(MASSANDRA, food="none", want="sweeter", order="reco"),
    }
    for name, body in bodies.items():
        path = FIXTURES / f"{name}.json"
        path.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(path.relative_to(REPO))
    for name in ("similar", "shelf"):
        for item in bodies[name]["items"]:
            print(" ", item["winery"], "—", item["name"], ":", " · ".join(item["reasons"]))
    for name in ("scan_found", "scan_check", "scan_not_found", "scan_suggest"):
        after_block = bodies[name]["after"]
        print(f"  {name}: {after_block['state']} {after_block['reasons']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
