import csv
import importlib.util
import json
from pathlib import Path

import pytest

from app.reading.contracts import Color, SugarClass
from app.reading.lexicon.build import Lexicon
from app.resolve.attrs import CatalogAttrs

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gt():
    return load_script("build_gt_tokens")


@pytest.fixture(scope="module")
def lexicon_cli():
    return load_script("build_lexicon")


BRUT = "kuban-vino-aristov-xxiv-brut-2023"
EXTRA = "kuban-vino-aristov-xxiv-extra-brut-2022"
MUSCAT = "massandra-muskat-belyy"


@pytest.fixture
def tiny_dataset(tmp_path):
    """Крошечный синтетический каталог во всех форматах входа сборщика."""
    rows = [
        (BRUT, "Аристов XXIV Брют 2023", "Кубань-Вино", "Белое", "Шардоне"),
        (EXTRA, "Аристов XXIV Экстра Брют 2022", "Кубань-Вино", "Белое", "Шардоне, Пино Нуар"),
        (MUSCAT, "Мускат белый", "Массандра", "Белое", "Мускат белый"),
    ]
    csv_path = tmp_path / "catalog.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["Slug", "Название вина", "Винодельня", "Категория", "Сорт винограда"]
            + ["Название фото", "Регион"]
        )
        for slug, name, winery, color, grape in rows:
            writer.writerow([slug, name, winery, color, grape, f"{slug}.webp", "Кубань"])
    clusters = {
        "slug_attrs": {
            BRUT: {"sugar_slug": ["brut"], "year_slug": 2023, "abv_slug": [12.5]},
            EXTRA: {"sugar_slug": ["extra_brut"], "year_slug": 2022},
        },
        "level_B": {
            "clusters": [
                {
                    "cluster_id": 1,
                    "members": [{"slug": BRUT}, {"slug": EXTRA}],
                    "visual_groups": [[BRUT, EXTRA]],
                }
            ]
        },
    }
    live = [{"slug": BRUT, "category": "Белое брют"}, {"slug": MUSCAT, "category": "Белое сладкое"}]
    paths = {
        "csv": csv_path,
        "clusters": tmp_path / "clusters.json",
        "live": tmp_path / "live.json",
        "winery-map": tmp_path / "winery_map.txt",
        "fixes": tmp_path / "gt_fixes.tsv",
        "out-dir": tmp_path / "gt",
    }
    paths["clusters"].write_text(json.dumps(clusters, ensure_ascii=False), encoding="utf-8")
    paths["live"].write_text(json.dumps(live, ensure_ascii=False), encoding="utf-8")
    paths["winery-map"].write_text("Массандра -> ['massandra']\n", encoding="utf-8")
    write_fixes(paths["fixes"], [])  # таблица правок каталога — только для настоящей выгрузки
    return paths


FIX_HEADER = ("slug", "field", "old", "new", "rule", "source_1", "source_2", "cross_check")


def write_fixes(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    """TSV правок: (slug, поле, было, стало) и два выдуманных источника на строку."""
    table = [list(FIX_HEADER)]
    table += [[*row, "тест", "фото: тест", "название: тест", ""] for row in rows]
    path.write_text("".join("\t".join(cells) + "\n" for cells in table), encoding="utf-8")


def args_for(paths) -> list[str]:
    return [item for key, value in paths.items() for item in (f"--{key}", str(value))]


def test_slug_translit_matches_catalog_slugs(gt):
    assert gt.slug_translit("Пино Нуар") == "pino-nuar"
    assert gt.slug_translit("Брют") == "bryut"
    assert gt.lat("Кубань-Вино") == "kuban vino"


def test_sugar_named_in_title(gt):
    assert gt.ru_sugar("Экстра Брют") == ["extra_brut"]
    assert gt.ru_sugar("Demi-Sec") == ["polusladkoe"]
    assert gt.ru_sugar("Полусухое") == ["polusuhoe"]


def test_find_phrases_prefers_longer(gt):
    phrases, spans = gt.find_phrases("гран резерв и резерв", ["резерв", "гран резерв"])
    assert phrases == ["гран резерв", "резерв"]
    assert len(spans) == 2


NO_CLUSTERS = {"slug_attrs": {}, "level_B": {"clusters": []}}


def test_packshot_aliases_only_on_request(gt):
    plain = gt.GtBuilder([], NO_CLUSTERS, [], {}).winery_block("Абрау-Дюрсо")
    assert "abrau estates" not in plain["variants"]
    packshot = gt.GtBuilder([], NO_CLUSTERS, [], {}, packshot_aliases=True)
    block = packshot.winery_block("Абрау-Дюрсо")
    assert "abrau estates" in block["variants"]
    assert block["variant_src"]["abrau estates"].startswith("packshot:")


def test_public_frames_are_not_alias_sources(gt):
    notes = [note for table in (gt.MANUAL_WINERY, gt.PACKSHOT_WINERY) for _, note in table.values()]
    assert not any(q in note for note in notes for q in ("q1", "q2", "q3"))
    builder = gt.GtBuilder([], NO_CLUSTERS, [], {}, packshot_aliases=True)
    assert builder.winery_block("Табия")["variants"] == ["tabiya"]  # транслитерация названия
    assert builder.winery_block("Массандра")["variants"] == ["massandra"]


def test_missing_input_is_reported(gt, tiny_dataset, capsys):
    tiny_dataset["csv"] = tiny_dataset["csv"].with_name("absent.csv")
    assert gt.main(args_for(tiny_dataset)) == 2
    assert "absent.csv" in capsys.readouterr().err


def test_tiny_dataset_to_gt_tokens_and_lexicon(gt, lexicon_cli, tiny_dataset, tmp_path, capsys):
    assert gt.main(args_for(tiny_dataset)) == 0
    out = tiny_dataset["out-dir"]
    lines = (out / "gt_tokens.jsonl").read_text(encoding="utf-8").splitlines()
    recs = {r["slug"]: r for r in map(json.loads, lines)}
    assert sorted(recs) == sorted([BRUT, EXTRA, MUSCAT])

    brut = recs[BRUT]["fields"]
    assert brut["winery"]["key_tokens"] == ["кубань", "вино"]
    assert "aristov" in brut["winery"]["variants"]
    assert (brut["sugar"]["class"], brut["sugar"]["src"]) == ("brut", "live_api")
    assert brut["year"]["value"] == 2023
    assert brut["serial"]["tokens"] == ["XXIV"]
    extra = recs[EXTRA]["fields"]
    assert extra["sugar"]["class"] == "extra_brut"
    assert extra["grape"]["codes"] == ["chardonnay", "pinot_noir"]
    assert "unpublished" in recs[EXTRA]["noise_flags"]
    muscat = recs[MUSCAT]["fields"]
    assert muscat["grape"]["codes"] == ["muscat"]
    assert muscat["winery"]["variants"] == ["massandra"]
    assert muscat["sugar"]["class"] == "sladkoe"

    summary = json.loads((out / "gt_summary.json").read_text(encoding="utf-8"))
    assert summary["n_slugs"] == 3
    assert summary["twin_pairs_level_B"] == 1

    lexicon_path = tmp_path / "index" / "lexicon.json"
    code = lexicon_cli.main(
        ["--gt-tokens", str(out / "gt_tokens.jsonl"), "--out", str(lexicon_path)]
    )
    assert code == 0
    assert '"entries"' in capsys.readouterr().out
    lex = Lexicon.load(lexicon_path)
    assert lex.get("producer", "ARISTOV").canonical == "Кубань-Вино"
    assert lex.get("serial", "XXIV").slugs == {BRUT, EXTRA}
    assert lex.get("sugar", "Extra Brut").slugs == {EXTRA}


# ------------------------------------------------------------------ правки карточек (Э4)
def read_out(paths) -> dict[str, dict]:
    lines = (paths["out-dir"] / "gt_tokens.jsonl").read_text(encoding="utf-8").splitlines()
    return {r["slug"]: r for r in map(json.loads, lines)}


def test_fixes_change_fields_but_keep_the_csv_category(gt, lexicon_cli, tiny_dataset, tmp_path):
    """Правка меняет поле разметки, а «Категория» выгрузки остаётся: её показывают карточки."""
    write_fixes(
        tiny_dataset["fixes"],
        [
            (MUSCAT, "color", "Белое", "null"),
            (BRUT, "sugar", "brut", "extra_brut"),
            (EXTRA, "grape", "Шардоне, Пино Нуар", "Шардоне"),
        ],
    )
    assert gt.main(args_for(tiny_dataset)) == 0
    recs = read_out(tiny_dataset)
    muscat, brut, extra = recs[MUSCAT], recs[BRUT], recs[EXTRA]
    assert muscat["category"] == "Белое" and muscat["fields"]["color"]["class"] is None
    assert not any(t["field"] == "color" for t in muscat["expected_tokens"])
    assert muscat["fixes"] == [{"field": "color", "old": "Белое", "new": None}]
    assert (brut["fields"]["sugar"]["class"], brut["fields"]["sugar"]["src"]) == (
        "extra_brut",
        "gt_fixes",
    )
    assert brut["fields"]["sugar"]["catalog_class"] == "brut"  # что говорит выгрузка — видно
    assert extra["fields"]["grape"]["codes"] == ["chardonnay"]
    summary = json.loads((tiny_dataset["out-dir"] / "gt_summary.json").read_text(encoding="utf-8"))
    assert summary["gt_fixes"]["applied"] == 3 and summary["gt_fixes"]["nulled"] == 1

    # признаки выбора и словарь берут поле, а не «Категорию»
    attrs = CatalogAttrs.load(tiny_dataset["out-dir"] / "gt_tokens.jsonl")
    assert attrs.get(MUSCAT).color is None and attrs.get(BRUT).color is Color.WHITE
    assert attrs.get(BRUT).sugar is SugarClass.EXTRA_BRUT
    lexicon_path = tmp_path / "lexicon.json"
    lexicon_cli.main(
        [
            "--gt-tokens",
            str(tiny_dataset["out-dir"] / "gt_tokens.jsonl"),
            "--out",
            str(lexicon_path),
        ]
    )
    assert Lexicon.load(lexicon_path).get("color", "белое").slugs == {BRUT, EXTRA}


def test_no_fixes_builds_the_catalog_as_is(gt, tiny_dataset):
    write_fixes(tiny_dataset["fixes"], [(MUSCAT, "color", "Белое", "null")])
    assert gt.main([*args_for(tiny_dataset), "--no-fixes"]) == 0
    muscat = read_out(tiny_dataset)[MUSCAT]
    assert muscat["fields"]["color"]["class"] == "Белое" and "fixes" not in muscat


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ((MUSCAT, "color", "Красное", "Розовое"), "в таблице было"),  # выгрузка сменилась
        (("net-takogo-slug", "sugar", "brut", "null"), "нет в выгрузке"),
        ((MUSCAT, "year", "2020", "null"), "не из"),
        ((MUSCAT, "color", "Белое", "Лиловое"), "не из"),
    ],
)
def test_bad_fix_fails_the_build(gt, tiny_dataset, capsys, row, message):
    """Правка не применяется молча: устаревшее «было», чужой slug или поле — отказ сборки."""
    write_fixes(tiny_dataset["fixes"], [row])
    assert gt.main(args_for(tiny_dataset)) == 2
    assert message in capsys.readouterr().err


def test_fix_needs_two_sources(gt, tmp_path):
    path = tmp_path / "fixes.tsv"
    table = [list(FIX_HEADER), [MUSCAT, "color", "Белое", "null", "тест", "фото: тест", "", ""]]
    path.write_text("".join("\t".join(cells) + "\n" for cells in table), encoding="utf-8")
    with pytest.raises(ValueError, match="два источника"):
        gt.load_fixes(path)


def test_repo_fix_table_is_well_formed(gt):
    """Замороженный список Э4: 12 правок, у каждой два источника, значения из словарей сборки."""
    fixes = gt.load_fixes(gt.DEFAULT_FIXES)
    assert len(fixes) == 12
    assert {field for _, field in fixes} == {"color", "sugar", "grape"}
    assert fixes[("oleg", "grape")]["new"] is None
    assert (
        fixes[("fanagoriya-fanagoriya-polusladkoe-rozovoe-sovinon-blan-krasnoe-11-13", "color")][
            "new"
        ]
        == "Розовое"
    )
