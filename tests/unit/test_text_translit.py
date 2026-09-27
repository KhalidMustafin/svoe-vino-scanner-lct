import pytest
from rapidfuzz.distance import Levenshtein

from app.reading.text.translit import (
    romanize,
    skeleton,
    transcribe,
    transcribe_gost,
    transcribe_variants,
)

# Перенесено из «Лозы» (`TestTranscription`).


def test_french_names():
    assert transcribe("chateau") == "шато"
    assert transcribe("tamagne") == "тамань"
    assert transcribe("cabernet") == "каберне"
    assert transcribe("sauvignon") == "совиньон"
    assert transcribe("merlot") == "мерло"
    assert transcribe("champagne") == "шампань"


def test_reading_variants():
    assert "абрау" in transcribe_variants("abrau")  # гостовское чтение
    assert "дюрсо" in transcribe_variants("durso")  # французское чтение
    assert transcribe_variants("тамань") == ("тамань",)  # кириллица как есть


def test_gost_reading_of_russian_names():
    assert transcribe_gost("myskhako") == "мысхако"
    assert transcribe_gost("novy") == "новы"


def test_variants_strip_diacritics():
    assert "сатен" in transcribe_variants("satèn")


# Обратное направление.


@pytest.mark.parametrize(
    ("cyrillic", "latin"),
    [
        ("Сухое", "suhoe"),
        ("Брют", "bryut"),
        ("Пино Нуар", "pino nuar"),
        ("Цимлянский Чёрный", "tsimlyanskiy chernyy"),
        ("Кубань-Вино", "kuban-vino"),
    ],
)
def test_romanize_follows_catalog_slug_rules(cyrillic, latin):
    assert romanize(cyrillic) == latin


# Скелет.


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Блан Де Блан", "Blanc de Blancs"),
        ("Шардоне", "Chardonnay"),
        ("Мускатель", "Muscatel"),
        ("Аристов", "ARISTOV"),
    ],
)
def test_skeleton_matches_label_spellings_exactly(left, right):
    assert skeleton(left) == skeleton(right)


# Пары название/транслит из `gt_tokens.jsonl` (поля variants винодельни, кюве и сорта).
CATALOG_PAIRS = [
    ("шардоне", "shardone"),
    ("пино нуар", "pinot noir"),
    ("пино нуар", "pino nuar"),
    ("рислинг", "riesling"),
    ("фанагория", "fanagoria"),
    ("фанагория", "fanagoriya"),
    ("шато тамань", "chateau tamagne"),
    ("совиньон блан", "sauvignon blanc"),
    ("совиньон блан", "sovinon blan"),
    ("мысхако", "myskhako"),
    ("мерло", "merlot"),
    ("саперави", "saperavi"),
    ("каберне фран", "cabernet franc"),
    ("каберне совиньон", "kaberne sovinon"),
    ("мускат", "muscat"),
    ("сира", "syrah"),
    ("шираз", "shiraz"),
    ("абрау дюрсо", "abrau dyurso"),
    ("абрау дюрсо", "abrau durso"),
    ("ркацители", "rkatsiteli"),
    ("захарьин", "zaharin"),
    ("валерий", "valeriy"),
    ("массандра", "massandra"),
    ("новый свет", "novyy svet"),
    ("новый светъ", "novy svet"),
    ("алиготе", "aligote"),
    ("ведерниковъ", "vedernikov"),
    ("золотая балка", "zolotaya balka"),
    ("бельбек", "belbek"),
    ("собер баш", "sober bash"),
    ("красностоп золотовский", "krasnostop zolotovsky"),
    ("вионье", "viognier"),
    ("кокур белый", "kokur belyy"),
    ("голубицкое", "golubitskoe"),
    ("мальбек", "malbec"),
    ("цимлянский черный", "tsimlyanskiy chernyy"),
    ("темпранильо", "tempranillo"),
    ("семильон", "semillon"),
    ("шенен блан", "chenin blanc"),
    ("литавщук", "litavshchuk"),
    ("литавщук", "litavschuk"),
    ("жемчужная", "zhemchuzhnaya"),
    ("хрусталева", "khrustaleva"),
    ("кинтессенсе", "quintessence"),
    ("эндемы", "endemy"),
    ("цитронный магарача", "tsitronnyy magaracha"),
    ("лефкадия", "lefkadiya"),
    ("солнечная долина", "solnechnaya dolina"),
]


@pytest.mark.parametrize(("cyrillic", "latin"), CATALOG_PAIRS)
def test_skeleton_catalog_pairs_within_one_edit(cyrillic, latin):
    assert Levenshtein.distance(skeleton(cyrillic), skeleton(latin)) <= 1


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Мерло", "Мальбек"),
        ("Пино Нуар", "Пино Гри"),
        ("Брют", "Сухое"),
        ("Рислинг", "Ркацители"),
        ("Шардоне", "Шираз"),
    ],
)
def test_skeleton_keeps_different_words_apart(left, right):
    assert Levenshtein.distance(skeleton(left), skeleton(right)) >= 3


def test_skeleton_folds_homoglyphs_and_soft_signs():
    assert skeleton("MACCAHДPA") == skeleton("Massandra")
    assert skeleton("Кубань") == skeleton("Kuban")
    assert skeleton("Ведерниковъ") == skeleton("Ведерников")


def test_skeleton_merges_i_like_letters():
    assert skeleton("Новый") == skeleton("Novyj") == skeleton("Noviy")


def test_skeleton_keeps_digits_and_word_boundaries():
    assert skeleton("Шато Тамань 2023") == "xato taman 2023"
    assert skeleton("") == ""
