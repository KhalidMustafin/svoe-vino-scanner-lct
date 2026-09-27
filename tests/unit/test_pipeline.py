import json
from pathlib import Path

import numpy as np
import pytest

from app.detect.bottles import TargetSelection
from app.reading.contracts import Box, Reading, SugarClass, TextLine, image_sha1
from app.reading.lexicon.build import build_from_records
from app.reading.lexicon.correct import lookup
from app.reading.pipeline import label_hits, read_label, select_crop
from app.reading.readers.cache import CachedReader
from app.reading.readers.ollama_vlm import OllamaVlmReader
from app.reading.text.tokenize import tokenize

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "ollama"
YEAR_NOW = 2026


def record(slug, winery, *, key=(), wvariants=(), cuvee=(), grapes=(), **extra):
    """Запись в формате `gt_tokens.jsonl`."""
    return {
        "slug": slug,
        "winery": winery,
        "category": extra.get("color"),
        "fields": {
            "winery": {"key_tokens": list(key), "variants": list(wvariants), "brands": []},
            "cuvee": {"tokens": list(cuvee), "variants": []},
            "grape": {
                "values": [value for value, _ in grapes],
                "codes": [code for _, code in grapes],
                "variants": list(extra.get("gvariants", ())),
            },
            "sugar": {"class": extra.get("sugar"), "variants": []},
            "serial": {"tokens": list(extra.get("serial", ())), "keywords": []},
            "color": {"class": extra.get("color"), "variants": []},
        },
    }


@pytest.fixture(scope="module")
def lex():
    return build_from_records(
        [
            record(
                "solnechnaya-dolina-saperavi",
                "Солнечная Долина",
                key=["солнечная", "долина"],
                wvariants=["solnechnaya dolina"],
                grapes=[("Саперави", "saperavi")],
                gvariants=["saperavi"],
                sugar="suhoe",
                color="Красное",
            ),
            record(
                "abrau-donum-4",
                "Абрау-Дюрсо",
                key=["абрау", "дюрсо"],
                cuvee=["donum"],
                serial=["4"],
                sugar="brut",
                color="Белое",
            ),
            record("abrau-brut", "Абрау-Дюрсо", key=["абрау", "дюрсо"], sugar="brut"),
        ]
    )


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class ScriptedReader:
    """Поддельный читатель: заранее записанные строки, время идёт по часам теста."""

    version = "1"
    params_hash = "p"

    def __init__(
        self, id, *lines, status="ok", seconds=0.0, clock=None, box=None, boxes=None, error=None
    ):
        self.id = id
        self.lines = lines
        self.status = status
        self.seconds = seconds
        self.clock = clock
        self.boxes = boxes if boxes is not None else [box] * len(lines)
        self.error = error
        self.calls = []

    def available(self):
        return True

    def read(self, image, *, crop, budget_ms):
        self.calls.append({"shape": image.shape, "crop": crop, "budget_ms": budget_ms})
        if self.clock is not None:
            self.clock.now += self.seconds
        if self.error is not None:
            raise self.error
        return Reading(
            reader=self.id,
            version=self.version,
            params_hash=self.params_hash,
            image_sha1=image_sha1(image),
            crop=crop,
            crop_px=max(image.shape[:2]),
            lines=[
                TextLine(id=i, text=t, box=box)
                for i, (t, box) in enumerate(zip(self.lines, self.boxes, strict=True))
            ],
            status=self.status,
            elapsed_ms=round(self.seconds * 1000),
        )


class FakeTransport:
    def __init__(self, response=None, exc=None):
        self.response = response
        self.exc = exc

    def __call__(self, payload, *, timeout_s):
        if self.exc is not None:
            raise self.exc
        return self.response


def frame(h=2000, w=1200):
    return np.full((h, w, 3), 128, np.uint8)


def recorded_vlm(name="ok", exc=None):
    response = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    return OllamaVlmReader("qwen2.5vl:3b", transport=FakeTransport(response, exc))


def run(readers, lexicon=None, **kwargs):
    kwargs.setdefault("year_now", YEAR_NOW)
    return read_label(frame(), readers=readers, lexicon=lexicon, **kwargs)


def confident(box, confident=True):
    return TargetSelection(target=box, method="detector", score=0.9, confident=confident)


# --- поля из записанного ответа модели ----------------------------------------


def test_fields_from_recorded_vlm_answer(lex):
    result = run([recorded_vlm()], lex)
    fields = result.fields
    assert [r.status for r in result.readings] == ["ok"]
    assert result.degraded == []
    assert "Солнечная Долина" in [e.value for e in fields.producer]
    assert [e.value for e in fields.grapes] == ["saperavi"]
    assert [e.value for e in fields.sugar] == [SugarClass.DRY]
    assert fields.vintage is not None and fields.vintage.value == 2021
    assert fields.abv is not None and fields.abv.value == 13.5
    assert fields.is_wine_label
    key = result.readings[0].key
    assert fields.vintage.sources == [key] and {t.reading for t in result.tokens} == {key}
    assert {
        "crop",
        "read:vlm",
        "read",
        "tokenize",
        "lexicon",
        "fields",
        "unmatched",
        "total",
    } <= set(result.timings_ms)


def test_without_lexicon_rule_fields_stay_and_degraded_is_marked():
    result = run([recorded_vlm()], None)
    fields = result.fields
    assert fields.producer == [] and fields.grapes == [] and fields.unmatched == []
    assert fields.vintage is not None and fields.vintage.value == 2021
    assert [e.value for e in fields.sugar] == [SugarClass.DRY]
    assert result.degraded == ["lexicon_missing"]


def test_unmatched_lists_strong_words_outside_lexicon(lex):
    reader = ScriptedReader("ocr", "Солнечная Долина", "Quinta do Noval", "Сухое вино")
    result = run([reader], lex)
    values = [e.value for e in result.fields.unmatched]
    assert "quinta" in values and "noval" in values
    assert not {"солнечная", "долина", "do", "вино", "сухое"} & set(values)
    assert all(
        e.support == 1 and e.sources == [result.readings[0].key] for e in result.fields.unmatched
    )


def test_unmatched_word_from_two_readings_is_one_evidence(lex):
    a = ScriptedReader("vlm", "Quinta do Noval")
    b = ScriptedReader("ocr", "QUINTA DO NOVAL")
    result = run([a, b], lex)
    noval = next(e for e in result.fields.unmatched if e.value == "noval")
    assert noval.support == 2
    assert noval.sources == [r.key for r in result.readings]


# --- бюджет и деградация -------------------------------------------------------


def test_readers_run_in_order_with_remaining_budget(lex):
    clock = FakeClock()
    a = ScriptedReader("vlm", "Урожай 2021", seconds=1.0, clock=clock)
    b = ScriptedReader("ocr", "Урожай 2021", seconds=0.2, clock=clock)
    result = run([a, b], lex, budget_ms=2500, clock=clock)
    assert a.calls[0]["budget_ms"] == 2500 and b.calls[0]["budget_ms"] == 1500
    assert [r.reader for r in result.readings] == ["vlm", "ocr"]
    assert result.timings_ms["read:vlm"] == 1000 and result.timings_ms["read:ocr"] == 200
    assert result.degraded == []
    assert result.fields.vintage.support == 2


def test_exhausted_budget_skips_next_reader(lex):
    clock = FakeClock()
    a = ScriptedReader("vlm", "Урожай 2021", seconds=3.0, clock=clock)
    b = ScriptedReader("ocr", "Урожай 2019", clock=clock)
    result = run([a, b], lex, budget_ms=2500, clock=clock)
    assert b.calls == []
    skipped = result.readings[1]
    assert (skipped.reader, skipped.status, skipped.elapsed_ms, skipped.lines) == (
        "ocr",
        "timeout",
        0,
        [],
    )
    assert result.degraded == ["ocr_timeout", "budget_exceeded"]
    assert result.fields.vintage.value == 2021  # опоздавшее, но готовое чтение не выбрасывается


def test_vlm_timeout_is_degraded_and_other_reader_fills_fields(lex):
    ocr = ScriptedReader("ocr", "Саперави", "урожай 2019")
    result = run([recorded_vlm(exc=TimeoutError("timed out")), ocr], lex)
    assert [r.status for r in result.readings] == ["timeout", "ok"]
    assert result.degraded == ["vlm_timeout"]
    assert result.fields.vintage.value == 2019
    assert result.fields.vintage.sources == [result.readings[1].key]
    assert [e.value for e in result.fields.grapes] == ["saperavi"]


def test_unavailable_and_crashing_readers_do_not_break_the_scan(lex):
    down = ScriptedReader("vlm", status="unavailable")
    boom = ScriptedReader("easyocr", error=RuntimeError("CUDA out of memory"))
    ok = ScriptedReader("rapidocr", "Урожай 2020")
    result = run([down, boom, ok], lex)
    assert [r.status for r in result.readings] == ["unavailable", "error", "ok"]
    assert "RuntimeError" in result.readings[1].raw
    assert result.degraded == ["vlm_unavailable", "easyocr_error"]
    assert result.fields.vintage.value == 2020


def test_no_readers_is_degraded_empty_result(lex):
    result = run([], lex)
    assert result.readings == [] and result.tokens == []
    assert result.degraded == ["no_readers"]
    assert result.fields.vintage is None and result.fields.unmatched == []


# --- кроп --------------------------------------------------------------------


@pytest.mark.parametrize("target", [None, "unconfident"])
def test_label_crop_without_confident_target_reads_full_frame(lex, target):
    selection = confident(Box(x0=0.3, y0=0.1, x1=0.7, y1=0.9), confident=False) if target else None
    reader = ScriptedReader("vlm", "Урожай 2021")
    result = run([reader], lex, target=selection, crop="label", crop_px=1024)
    assert reader.calls[0]["crop"] == "full" and reader.calls[0]["shape"] == (1024, 614, 3)
    assert result.readings[0].crop == "full"
    assert result.degraded == ["crop_fallback_full"]


def test_confident_label_crop_and_line_boxes_in_frame_coordinates(lex):
    target = confident(Box(x0=0.25, y0=0.1, x1=0.75, y1=0.9))
    reader = ScriptedReader("vlm", "Урожай 2021", box=Box(x0=0.0, y0=0.0, x1=1.0, y1=0.5))
    result = run([reader], lex, target=target, crop="label", crop_px=1024)
    call = reader.calls[0]
    assert call["crop"] == "label" and max(call["shape"][:2]) == 1024
    assert call["shape"][0] > call["shape"][1]  # полоса этикетки выше, чем шире
    assert result.degraded == []
    box = result.readings[0].lines[0].box
    # Этикетка: нижние 70 % рамки бутылки, по ширине — рамка с отступом 8 %.
    assert (box.x0, box.y0, box.x1, box.y1) == pytest.approx((0.21, 0.34, 0.79, 0.62), abs=1e-3)


def test_band_crop_is_the_middle_of_the_label(lex):
    target = confident(Box(x0=0.25, y0=0.1, x1=0.75, y1=0.9))
    label = select_crop(frame(), "label", target=target, crop_px=4000)
    band = select_crop(frame(), "band", target=target, crop_px=4000)
    assert band.crop == "band" and not band.fallback
    assert label.region.x0 < band.region.x0 < band.region.x1 < label.region.x1
    assert (band.region.y0, band.region.y1) == (label.region.y0, label.region.y1)
    assert (band.region.x0 + band.region.x1) / 2 == pytest.approx(0.5, abs=2e-3)
    # Середина увеличена по Ланцошу в 1,5 раза.
    assert band.image.shape[0] == pytest.approx(label.image.shape[0] * 1.5, abs=2)


def test_full_crop_ignores_target_and_keeps_boxes(lex):
    target = confident(Box(x0=0.25, y0=0.1, x1=0.75, y1=0.9))
    reader = ScriptedReader("vlm", "Урожай 2021", box=Box(x0=0.1, y0=0.2, x1=0.3, y1=0.4))
    result = run([reader], lex, target=target, crop="full")
    assert reader.calls[0]["crop"] == "full" and result.degraded == []
    assert result.readings[0].lines[0].box == Box(x0=0.1, y0=0.2, x1=0.3, y1=0.4)


def test_bad_input_is_rejected(lex):
    with pytest.raises(ValueError):
        read_label(np.zeros((10, 10), np.uint8), readers=[], lexicon=lex)
    with pytest.raises(ValueError):
        read_label(frame(), readers=[], lexicon=lex, crop_px=0)


# --- слияние чтений ------------------------------------------------------------


def test_support_counts_reading_with_same_skeleton_outside_lexicon(lex):
    a = ScriptedReader("vlm", "DONUM")
    b = ScriptedReader("ocr", "DONUN")  # одна правка: в словарь не попадает (бюджет 0,5)
    result = run([a, b], lex)
    tokens_b = [t for t in result.tokens if t.reading == result.readings[1].key]
    assert lookup(tokens_b, lex) == []
    donum = next(e for e in result.fields.cuvee if e.value == "donum")
    assert donum.support == 2 and donum.sources == [r.key for r in result.readings]
    assert "donun" not in [e.value for e in result.fields.unmatched]


def test_support_does_not_count_distant_spelling(lex):
    a = ScriptedReader("vlm", "DONUM")
    b = ScriptedReader("ocr", "DOMAN")
    result = run([a, b], lex)
    donum = next(e for e in result.fields.cuvee if e.value == "donum")
    assert donum.support == 1 and donum.sources == [result.readings[0].key]
    assert "doman" in [e.value for e in result.fields.unmatched]


def test_bare_short_number_is_not_a_serial_hit(lex):
    reading = ScriptedReader("ocr", "Абрау-Дюрсо", "сахара не более 4 г/дм3").read(
        frame(), crop="full", budget_ms=1
    )
    tokens = tokenize(reading, lexicon=lex)
    four = [i for i, t in enumerate(tokens) if t.norm == "4"]
    assert any(ids == tuple(four) for ids, _ in lookup(tokens, lex))  # словарь его находит
    assert all(set(ids) != set(four) for ids, _ in label_hits(tokens, lex))
    result = run([ScriptedReader("ocr", "Абрау-Дюрсо", "сахара не более 4 г/дм3")], lex)
    assert result.fields.serial == []
    assert "Абрау-Дюрсо" in [e.value for e in result.fields.producer]


def pino_lex():
    return build_from_records(
        [
            record(
                "shato-pino-pino-nuar",
                "Шато Пино",
                key=["шато", "пино"],
                cuvee=["нуар"],  # остаток названия «Пино Нуар» после винодельни
                grapes=[("Пино Нуар", "pinot_noir")],
                gvariants=["pinot noir", "пино нуар"],
            ),
            record("kuban-tamagne", "Кубань-Вино", key=["кубань"], wvariants=["шато тамань"]),
        ]
    )


def test_phrase_split_into_word_lines_of_one_row_is_one_phrase():
    # Регрессия: EasyOCR кладёт каждое слово ряда в свою строку, и сорт терялся вместе с кюве.
    lex = pino_lex()
    row = [Box(x0=0.20, y0=0.40, x1=0.32, y1=0.45), Box(x0=0.34, y0=0.40, x1=0.46, y1=0.46)]
    fields = run([ScriptedReader("ocr", "Пино", "Нуар", boxes=row)], lex).fields
    assert [e.value for e in fields.grapes] == ["pinot_noir"]
    assert fields.cuvee == []
    plain = run([ScriptedReader("vlm", "Пино", "Нуар")], lex).fields  # без рамок — порядок чтения
    assert [e.value for e in plain.grapes] == ["pinot_noir"]
    apart = [Box(x0=0.05, y0=0.10, x1=0.17, y1=0.15), Box(x0=0.70, y0=0.85, x1=0.82, y1=0.90)]
    far = run([ScriptedReader("ocr", "Пино", "Нуар", boxes=apart)], lex).fields
    assert far.grapes == []


def test_grape_phrase_beats_catalog_cuvee_word_and_lone_phrase_word_is_known():
    lex = pino_lex()
    fields = run([ScriptedReader("ocr", "Пино Нуар", "Шато Тамань")], lex).fields
    assert [e.value for e in fields.grapes] == ["pinot_noir"]
    assert fields.cuvee == []
    assert "Кубань-Вино" in [e.value for e in fields.producer]
    lone = run([ScriptedReader("ocr", "Тамань")], lex).fields
    assert lone.producer == [] and lone.unmatched == []


# --- ловушки таксономии при словаре каталога -------------------------------------


@pytest.fixture(scope="module")
def trap_lex():
    """Словарь, где у сахара, цвета и серий есть все написания, как в собранном каталоге."""
    return build_from_records(
        [
            record("novyj-svet-brut", "Новый Свет", key=["новый", "свет"], sugar="brut"),
            record("yug-sweet", "Юг", key=["юг"], sugar="sladkoe", color="Розовое"),
            record("yug-dry", "Юг", key=["юг"], sugar="suhoe", color="Белое"),
            record("kuban-xxiv", "Кубань-Вино", key=["кубань"], serial=["XXIV", "II"]),
            record("kuban-extra", "Кубань-Вино", key=["кубань"], sugar="extra_brut"),
            record("kuban-nature", "Кубань-Вино", key=["кубань"], sugar="brut_nature"),
        ]
    )


def lexicon_hits(lex, *lines):
    reading = ScriptedReader("vlm", *lines).read(frame(), crop="full", budget_ms=1)
    tokens = tokenize(reading, lexicon=lex)
    return [
        (tuple(tokens[i].norm for i in ids), h.field, h.canonical) for ids, h in lookup(tokens, lex)
    ]


@pytest.mark.parametrize(
    ("lines", "trap", "expected"),
    [
        (("Новый Свет", "Брют"), (("свет",), "sugar", "sladkoe"), [SugarClass.BRUT]),
        (("Prosecco Extra Dry",), (("dry",), "sugar", "suhoe"), []),
        (("Экстра Брют",), (("брют",), "sugar", "brut"), [SugarClass.EXTRA_BRUT]),
        (("Brut Nature",), (("brut",), "sugar", "brut"), [SugarClass.BRUT_NATURE]),
    ],
)
def test_lexicon_sugar_does_not_reopen_taxonomy_traps(trap_lex, lines, trap, expected):
    assert trap in lexicon_hits(trap_lex, *lines)  # словарь ловушку находит
    result = run([ScriptedReader("vlm", *lines)], trap_lex)
    assert [e.value for e in result.fields.sugar] == expected


def test_lexicon_color_and_roman_serial_keep_rule_checks(trap_lex):
    assert (("rose",), "color", "Розовое") in lexicon_hits(trap_lex, "Traminer Rose")
    assert run([ScriptedReader("vlm", "Traminer Rose")], trap_lex).fields.color is None
    assert (("ii",), "serial", "II") in lexicon_hits(trap_lex, "Ю0 II h")
    assert run([ScriptedReader("vlm", "Ю0 II h")], trap_lex).fields.serial == []
    serial = run([ScriptedReader("vlm", "Серия XXIV")], trap_lex).fields.serial
    assert [(e.value, e.matched) for e in serial] == [("XXIV", "XXIV")]


def test_second_reader_does_not_erase_first_readers_values(lex):
    vlm = ScriptedReader("vlm", "Брют", "2023", "12,5% об.")
    ocr = ScriptedReader("easyocr", "Брют", "2021", "12,0% об.")
    alone = run([vlm], lex).fields
    assert (alone.vintage.value, alone.abv.value) == (2023, 12.5)
    both = run([vlm, ocr], lex)
    assert (both.fields.vintage.value, both.fields.abv.value) == (2023, 12.5)
    assert both.fields.vintage.sources == [both.readings[0].key]


def test_thinking_only_vlm_answer_is_degraded_and_not_cached(tmp_path):
    answers = [
        json.loads((FIXTURES / f"{n}.json").read_text("utf-8")) for n in ("thinking_only", "ok")
    ]
    calls = []

    def transport(payload, *, timeout_s):
        calls.append(payload)
        return answers[min(len(calls), 2) - 1]

    cached = CachedReader(OllamaVlmReader("qwen3-vl:8b", transport=transport), tmp_path)
    first = run([cached], None)
    assert first.readings[0].status == "error"
    assert first.degraded == ["vlm_error", "lexicon_missing"]
    second = run([cached], None)  # после исправления Ollama кэш не держит сбой
    assert second.readings[0].status == "ok" and len(calls) == 2
