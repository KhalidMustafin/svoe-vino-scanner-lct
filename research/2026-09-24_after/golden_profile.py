"""Золотой тест переноса профиля (внутренний план команды вне репозитория, Д2 п. 9).

Оси `derived_*` join CSV «Лозы» посчитаны её `build_profile` из полей Strapi
(`Датасет и подробное задание/analysis/strapi_join.py:345`): сорта, цвет, сахар, тип, регион,
название и год урожая лежат в тех же строках, в колонках `strapi_*`. Здесь — три сверки:

    A  перенос `app.recommend.build.build_profile` на тех же входах `strapi_*` → все 8 осей
       против `derived_*`, допуск 0,05. Приёмка: 2103 из 2103.
    B  профиль сервиса (`profile_of` по справочнику выгрузки) без крепости → сколько вин
       совпадают по осям и почему расходятся остальные: у сервиса свои входы (сорта таксономии
       сканера, сахар и игристость по правилам выгрузки — решение 24.09 без портала).
    C  ось крепости сервиса: по `abv`, а не 3,0 у всех (так и задумано, см. `build.py`).

Join CSV лежит вне репозитория и в него не попадает: в нём `loza_id`, `expert_score` и
`price_rub`. Отсюда в репозиторий идут только числа сверки (`golden_profile.json`) и, с ключом
`--fixture`, фикстура на 30 вин для юнит-теста: slug, входы профиля и ожидаемые оси.

    .venv/Scripts/python.exe research/2026-09-24_after/golden_profile.py [--fixture]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from app.reading.contracts import Color, SugarClass  # noqa: E402
from app.recommend.build import build_profile, load_priors, oak_from_name, profile_of  # noqa: E402
from app.recommend.catalog import RecoCatalog  # noqa: E402
from app.recommend.profile import AXES, RussianPGI, WineKind  # noqa: E402

JOIN = Path(
    r"<корень>\Датасет и подробное задание\analysis"
    r"\strapi_to_loza_join.csv"
)
FIXTURE = REPO / "tests" / "fixtures" / "profile_golden.json"
TOL = 0.05
#: Коды «Лозы» (join CSV) → коды сканера.
COLORS = {"white": Color.WHITE, "red": Color.RED, "rose": Color.ROSE, "orange": Color.ORANGE}
SUGARS = {
    "brut_nature": SugarClass.BRUT_NATURE,
    "extra_brut": SugarClass.EXTRA_BRUT,
    "brut": SugarClass.BRUT,
    "dry": SugarClass.DRY,
    "semi_dry": SugarClass.SEMI_DRY,
    "semi_sweet": SugarClass.SEMI_SWEET,
    "sweet": SugarClass.SWEET,
}
#: Только эти колонки join CSV читаются: входы профиля и ожидаемые оси.
INPUTS = ("strapi_slug", "strapi_name", "strapi_color", "strapi_sugar", "strapi_kind",
          "strapi_vintage", "strapi_grapes", "strapi_region", "derived_basis")  # fmt: skip


def read_join(path: Path) -> list[dict[str, str]]:
    keep = (*INPUTS, *(f"derived_{axis}" for axis in AXES))
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return [{key: row[key] for key in keep} for row in csv.DictReader(fh)]


def inputs_of(row: dict[str, str]) -> dict:
    """Входы `build_profile` из колонок `strapi_*` — в кодах сканера."""
    return {
        "grapes": [code for code in row["strapi_grapes"].split(";") if code],
        "color": COLORS[row["strapi_color"]].value if row["strapi_color"] else None,
        "sugar": SUGARS[row["strapi_sugar"]].value if row["strapi_sugar"] else None,
        "kind": row["strapi_kind"],
        "region": row["strapi_region"],
        "name": row["strapi_name"],
        "vintage": int(row["strapi_vintage"]) if row["strapi_vintage"] else None,
    }


def profile_from_inputs(inputs: dict, priors):
    return build_profile(
        inputs["grapes"],
        Color(inputs["color"]) if inputs["color"] else None,
        SugarClass(inputs["sugar"]) if inputs["sugar"] else None,
        WineKind(inputs["kind"]),
        priors,
        region=RussianPGI(inputs["region"]),
        name=inputs["name"],
        vintage=inputs["vintage"],
    )


def expected_of(row: dict[str, str]) -> dict[str, float]:
    return {axis: float(row[f"derived_{axis}"]) for axis in AXES}


def mismatched_axes(profile, expected: dict[str, float], axes=AXES) -> list[str]:
    return [axis for axis in axes if abs(profile.axis(axis) - expected[axis]) > TOL]


def tags_of(inputs: dict, profile, expected: dict[str, float]) -> set[str]:
    """Признаки строки для фикстуры: хочется покрыть каждую ветку `build_profile`."""
    oak = oak_from_name(inputs["name"])
    age = 2026 - inputs["vintage"] if inputs["vintage"] else None
    tags = {
        f"basis:{profile.basis}",
        f"kind:{inputs['kind']}",
        f"color:{inputs['color']}",
        f"sugar:{inputs['sugar']}",
        f"region:{inputs['region']}",
        "age:none" if age is None else ("age:young" if age < 3 else "age:old"),
        "oak:none" if oak is None else ("oak:fresh" if oak <= 0.5 else "oak:marker"),
    }
    if any(value in (0.0, 5.0) for value in expected.values()):
        tags.add("clamp")
    if inputs["color"] in ("Белое", "Розовое") and profile.basis != "low":
        tags.add("tannin_cap")
    return tags


def pick_fixture(rows: list[dict[str, str]], priors, size: int = 30) -> list[dict]:
    """Жадно — строки, добавляющие больше всего новых признаков; при равенстве — по slug."""
    items = []
    for row in sorted(rows, key=lambda r: r["strapi_slug"]):
        inputs = inputs_of(row)
        expected = expected_of(row)
        profile = profile_from_inputs(inputs, priors)
        items.append((row, inputs, expected, tags_of(inputs, profile, expected)))
    chosen: list[tuple] = []
    seen: set[str] = set()
    while len(chosen) < size:
        best = max(
            (item for item in items if item not in chosen),
            key=lambda item: len(item[3] - seen),
        )
        if not best[3] - seen:
            # Всё покрыто: добираем разными сортами, чтобы в фикстуре были и купажи.
            rest = [i for i in items if i not in chosen and len(i[1]["grapes"]) > 1]
            best = rest[len(chosen) * 37 % len(rest)]
        chosen.append(best)
        seen |= best[3]
    return [
        {
            "slug": row["strapi_slug"],
            **inputs,
            "basis": row["derived_basis"],
            "axes": expected,
        }
        for row, inputs, expected, _ in sorted(chosen, key=lambda item: item[0]["strapi_slug"])
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--join", default=str(JOIN))
    parser.add_argument("--data", default=os.environ.get("SVS_DATA_DIR") or str(REPO / "data"))
    parser.add_argument("--fixture", action="store_true", help="переписать фикстуру на 30 вин")
    parser.add_argument("--out", default=str(HERE / "golden_profile.json"))
    args = parser.parse_args()

    priors = load_priors()
    rows = read_join(Path(args.join))
    print(f"join CSV: {len(rows)} строк, приоров {len(priors)}")

    # A — перенос на входах «Лозы».
    bad_a: dict[str, list[str]] = {}
    basis_a = 0
    worst_a = 0.0
    for row in rows:
        inputs = inputs_of(row)
        profile = profile_from_inputs(inputs, priors)
        expected = expected_of(row)
        worst_a = max(worst_a, *(abs(profile.axis(a) - expected[a]) for a in AXES))
        axes = mismatched_axes(profile, expected)
        if axes:
            bad_a[row["strapi_slug"]] = axes
        basis_a += profile.basis == row["derived_basis"]
    ok_a = len(rows) - len(bad_a)
    print(f"A: все 8 осей в пределах {TOL}: {ok_a}/{len(rows)}; основа совпала {basis_a}/{len(rows)};"
          f" наибольшее отклонение {worst_a:.4f}")

    # B и C — профиль сервиса по справочнику.
    data = Path(args.data)
    catalog = RecoCatalog.load(
        data / "gt" / "gt_tokens.jsonl",
        wines_path=data / "catalog" / "wines.jsonl",
        groups_path=data / "catalog" / "wine_groups.json",
    )
    by_slug = {row["strapi_slug"]: row for row in rows}
    no_alcohol = tuple(axis for axis in AXES if axis != "alcohol")
    ok_b = ok_b7 = 0
    axis_bad_b: Counter[str] = Counter()
    causes: Counter[str] = Counter()
    unexplained: list[str] = []
    alcohol_gap: list[float] = []
    missing = 0
    for slug, row in by_slug.items():
        wine = catalog.get(slug)
        if wine is None:
            missing += 1
            continue
        expected = expected_of(row)
        inputs = inputs_of(row)
        plain_profile = profile_of(wine, priors)  # без крепости — как у «Лозы»
        axes = mismatched_axes(plain_profile, expected)
        axes7 = mismatched_axes(plain_profile, expected, no_alcohol)
        ok_b += not axes
        ok_b7 += not axes7
        axis_bad_b.update(axes)
        if axes7:
            why = []
            if set(wine.grapes) != set(inputs["grapes"]):
                why.append("grapes")
            if wine.sugar != inputs["sugar"]:
                why.append("sugar")
            if wine.sparkling != (inputs["kind"] == "sparkling"):
                why.append("sparkling")
            if wine.color != inputs["color"]:
                why.append("color")
            if wine.year != inputs["vintage"]:
                why.append("year")
            if wine.title != inputs["name"]:
                why.append("name")
            causes["+".join(why) or "unexplained"] += 1
            if not why:
                unexplained.append(slug)
        served = profile_of(wine, priors, alcohol=catalog.alcohol(slug))
        if served.source("alcohol") is not None:
            alcohol_gap.append(served.alcohol - expected["alcohol"])
    compared = len(by_slug) - missing
    print(f"B: сервис без крепости, все 8 осей: {ok_b}/{compared}; 7 осей без крепости: "
          f"{ok_b7}/{compared}; причины расхождений: {dict(causes.most_common())}")
    gaps = sorted(abs(g) for g in alcohol_gap)
    print(f"C: крепость по abv у {len(alcohol_gap)} вин; |Δ| к 3,0 «Лозы»: медиана "
          f"{gaps[len(gaps) // 2]:.2f}, p95 {gaps[int(0.95 * (len(gaps) - 1))]:.2f}")

    result = {
        "rows": len(rows),
        "tolerance": TOL,
        "A_port_on_loza_inputs": {
            "all_axes_ok": ok_a,
            "of": len(rows),
            "basis_ok": basis_a,
            "max_abs_diff": round(worst_a, 4),
            "bad": dict(list(bad_a.items())[:20]),
            "pass": ok_a == len(rows),
        },
        "B_service_inputs_no_abv": {
            "compared": compared,
            "missing_in_catalog": missing,
            "all_axes_ok": ok_b,
            "axes_without_alcohol_ok": ok_b7,
            "bad_by_axis": dict(axis_bad_b.most_common()),
            "causes": dict(causes.most_common()),
            "unexplained": unexplained[:20],
        },
        "C_alcohol_from_abv": {
            "wines": len(alcohol_gap),
            "abs_diff_to_loza_median": round(gaps[len(gaps) // 2], 3),
            "abs_diff_to_loza_p95": round(gaps[int(0.95 * (len(gaps) - 1))], 3),
        },
    }
    Path(args.out).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.out)  # `--out` может быть и вне репозитория
    if args.fixture:
        fixture = pick_fixture(rows, priors)
        FIXTURE.write_text(
            json.dumps(
                {
                    "_comment": "Золотая фикстура профиля: 30 вин join CSV «Лозы» — slug, входы "
                    "build_profile (коды сканера) и оси derived_*. Собрана "
                    "research/2026-09-24_after/golden_profile.py --fixture.",
                    "tolerance": TOL,
                    "wines": fixture,
                },
                ensure_ascii=False,
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
        print(FIXTURE.relative_to(REPO), len(fixture))
    return 0 if ok_a == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
