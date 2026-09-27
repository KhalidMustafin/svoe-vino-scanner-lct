"""Право в данных сомелье: 38-ФЗ и стоп-лист по каждой строке `data/somm/*.json`.

Строки данных сомелье показываются человеку как есть (темы справочника, фразы правил, подписи
чипов, портреты сортов, фразы подачи) или сверяют текст модели (словарь замка). Поэтому каждая
строка каждого файла проходит `content_filter.check` и стоп-лист сборки: цены, «купить»,
«лучший», «идеальный», превосходные степени, баллы и рейтинги, призывы выпить, `%`. В файлах нет
ни `expert_score`, ни `price`, ни `typical_price`, ни рублей.

Проверяются три источника: сборка на синтетических винах (те же таблицы и справочник «Лозы», что
и в работе, — без выгрузки организатора), заглушки договора и, если есть на машине, собранный
`data/somm` (`SVS_SOMM_DIR` или `<SVS_DATA_DIR>/somm`).
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from app.config import get_settings
from app.recommend.build import load_priors
from app.recommend.content_filter import check
from app.recommend.somm_data import FIXTURE_DIR, SOMM_FILES

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build_somm.py"

#: Запретное в сыром тексте файлов (приёмка дорожки): ключи Роскачества и цены, проценты, рубли.
FORBIDDEN = {
    "expert_score": re.compile(r"expert_score"),
    "price": re.compile(r"price", re.IGNORECASE),
    "typical_price": re.compile(r"typical_price"),
    "%": re.compile(r"%"),
}

WINERIES = (
    "Фанагория",
    "Массандра",
    "Абрау-Дюрсо",
    "Долина Лефкадия",
    "А. Гордиенко & М. Николаев",
    "Шато Пино",
    "Ароматное",
)
REGIONS = ("Кубань", "Крым", "Долина Дона", "Дагестан", "Самара")
STYLES = (
    ("Красное", "suhoe", False),
    ("Белое", "polusuhoe", False),
    ("Белое", "sladkoe", False),
    ("Розовое", "polusladkoe", False),
    ("Оранжевое", None, False),
    ("Белое", "brut", True),
    ("Красное", None, False),
)


@pytest.fixture(scope="module")
def somm() -> Any:
    spec = importlib.util.spec_from_file_location("build_somm", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def synthetic(somm: Any) -> dict[str, Any]:
    """Сборка на винах всех сортов приоров, всех цветов и сахаров — без выгрузки организатора."""
    facts = []
    for index, code in enumerate(sorted(load_priors())):
        color, sugar, sparkling = STYLES[index % len(STYLES)]
        facts.append(
            somm.WineFacts(
                slug=f"wine-{index}",
                name="Портвейн" if index == 4 else f"Вино {index}",
                winery=WINERIES[index % len(WINERIES)],
                region=REGIONS[index % len(REGIONS)],
                color=color,
                grapes=(code,),
                codes=(code,),
                sugar=sugar,
                sparkling=sparkling,
                abv=None if index % 3 else 12.5,
                year=None,
                canonical=index % 5 != 0,
            )
        )
    result = somm.build_from_facts(facts, "проба")
    return {name: json.loads(blob) for name, blob in result.files.items()}


def strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for key, item in value.items() for s in (key, *strings(item))]
    if isinstance(value, list):
        return [s for item in value for s in strings(item)]
    return []


def violations(somm: Any, texts: list[str]) -> list[str]:
    """`content_filter.check` и стоп-лист сборки по каждой различной строке."""
    unique = sorted(set(texts))
    bad = somm.legal_violations(unique)
    # Стоп-лист сборки сам зовёт фильтр; здесь — прямая сверка, что зовёт тот же `check`.
    assert all(check(text).clean for text in unique) or bad
    return bad


def raw_hits(text: str) -> dict[str, int]:
    hits = {name: len(pattern.findall(text)) for name, pattern in FORBIDDEN.items()}
    hits["руб"] = len(re.findall(r"\bрубл(?!ен)|\bруб(?:\.|\b)|₽", text, re.IGNORECASE))
    return hits


# ------------------------------------------------------------------ стоп-лист
@pytest.mark.parametrize(
    "text",
    [
        "Игристый брют — лучший спутник солёных закусок.",
        "Цена — 500 рублей.",
        "Стоимость бутылки невелика.",
        "Попробуйте оба стиля.",
        "В этой подборке оно самое свежее по кислотности.",
        "Получаются великие сладкие вина.",
        "Идеальное сочетание.",
        "Вино недели.",
        "Балл дегустаторов — 92.",
        "Крепость 12 %.",
        "Купите к ужину.",
        "Есть и дешёвая имитация.",
        "expert_score",
    ],
)
def test_stoplist_catches(somm: Any, text: str) -> None:
    assert somm.legal_violations([text]), text


@pytest.mark.parametrize(
    "text",
    [
        "рубленые котлеты",
        "Рубиновый Магарача",
        "Терпкое вино кажется грубым.",
        "Сомелье советует гостю и подаёт.",
        "Надпись обязательна по закону.",
        "Плотные красные подают при 16–18 °C.",
        "Проверяется только самим вином.",
    ],
)
def test_stoplist_spares(somm: Any, text: str) -> None:
    assert somm.legal_violations([text]) == [], text


# ------------------------------------------------------------------ синтетическая сборка
@pytest.mark.parametrize("name", SOMM_FILES)
def test_synthetic_build_is_clean(somm: Any, synthetic: dict[str, Any], name: str) -> None:
    texts = strings(synthetic[name])
    assert texts
    assert violations(somm, texts) == []
    assert raw_hits(json.dumps(synthetic[name], ensure_ascii=False)) == dict.fromkeys(
        (*FORBIDDEN, "руб"), 0
    )


def test_synthetic_texts_cover_all_sources(synthetic: dict[str, Any]) -> None:
    """Проверка выше действительно видит все тексты: правила, подачу, темы, сорта, словарь."""
    assert len(synthetic["dishes.json"]["rules"]) == 39
    assert len(synthetic["dishes.json"]["dishes"]) == 82
    assert len(synthetic["serve.json"]["rules"]) == 9
    assert len(synthetic["knowledge.json"]["topics"]) == 32
    assert len(synthetic["knowledge.json"]["grapes"]) >= 40
    assert len(synthetic["pairs.json"]["wines"]) == len(load_priors())
    assert synthetic["vocab.json"]["descriptors"] and synthetic["vocab.json"]["winery_words"]


def test_no_portal_and_no_loza_brand(synthetic: dict[str, Any]) -> None:
    """Данных портала в продукте нет, и имя «Лозы» человеку не показывается."""
    every = strings(synthetic)
    assert not [text for text in every if "портал" in text.lower() or "portal" in text.lower()]
    topics = synthetic["knowledge.json"]["topics"]
    for topic in topics:
        assert "Лоз" not in topic["answer"] + topic["detail"], topic["id"]
    for grape in synthetic["knowledge.json"]["grapes"].values():
        assert "Лоз" not in grape["text"]
    assert "portal_dish_mapping" not in json.dumps(synthetic, ensure_ascii=False)


# ------------------------------------------------------------------ заглушки договора
@pytest.mark.parametrize("name", SOMM_FILES)
def test_contract_fixtures_pass_content_filter(somm: Any, name: str) -> None:
    """Заглушки — `content_filter` и запретные слова (стоп-лист договора сверяет его тест).

    Стоп-лист сборки строже договорного: заглушка темы `tannins` несёт текст «Лозы» как есть
    («то самое вяжущее»), а сборка переписывает его (`TOPIC_REWRITES`).
    """
    raw = (FIXTURE_DIR / name).read_text(encoding="utf-8")
    texts = sorted(set(strings(json.loads(raw))))
    assert [text for text in texts if not check(text).clean] == []
    assert raw_hits(raw) == dict.fromkeys((*FORBIDDEN, "руб"), 0)


# ------------------------------------------------------------------ собранные данные
def _data_dir() -> Path | None:
    configured = os.environ.get("SVS_SOMM_DIR")
    path = Path(configured) if configured else get_settings().data_dir / "somm"
    return path if all((path / name).is_file() for name in SOMM_FILES) else None


@pytest.mark.parametrize("name", SOMM_FILES)
def test_built_data_is_clean(somm: Any, name: str) -> None:
    directory = _data_dir()
    if directory is None:
        pytest.skip("нет собранного data/somm: scripts/build_somm.py")
    raw = (directory / name).read_text(encoding="utf-8")
    assert raw_hits(raw) == dict.fromkeys((*FORBIDDEN, "руб"), 0)
    assert violations(somm, strings(json.loads(raw))) == []
