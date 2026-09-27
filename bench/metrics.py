"""Метрики OCR-модуля на эталонном наборе (синтетика, полевой набор, публичные кадры).

OCR не выбирает вино: текстовый top-1 здесь — только диагностика. Решение принимает
resolve поверх кандидатов CV, поэтому в предсказании оставлено место под `cv_top5`
и `final_top5` — сравнение «CV» против «CV+OCR».

Предсказание — строка JSONL на кадр:

    {"query_id": "...", "raw_text": "...", "latency_ms": 812, "status": "ok",
     "fields": {"winery": "...", "cuvee": "...", "grape": "...", "sugar": "brut",
                "year": 2023, "serial": ["XXIV"]},
     "text_top5": ["slug1", ...],    # «только текст», пустой список — отказ
     "cv_top5": ["slug1", ...],      # кандидаты CV, когда появятся
     "final_top5": ["slug1", ...]}   # CV+OCR после resolve, когда появится

CLI:

    python -m bench.metrics --gt synth_gt.tsv --ann synth_annotations.jsonl --pred p.jsonl
    python -m bench.metrics --gt synth_gt.tsv --ann synth_annotations.jsonl --selftest
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from rapidfuzz.distance import Levenshtein

from app.config import get_settings

NONE_SLUG = "__none__"
CER_HIT = 0.2  # токен «прочитан», если CER не больше
KEY_TOKEN_FIELDS = ("winery", "cuvee", "grape", "year", "serial")
KEY_AGGREGATE = "key_fields"  # сводная строка по KEY_TOKEN_FIELDS

GtRow = tuple[str, str]  # (query_id, slug | __none__)
Prediction = Mapping[str, Any]
Predictions = Mapping[str, Prediction]
GtTokens = Mapping[str, Mapping[str, Any]]
Annotations = Mapping[str, Mapping[str, Any]]
LabelTokens = list[tuple[str, str]]  # (поле, текст)

_PUNCT_RE = re.compile(r"[^\w\s/]+")
_SPACE_RE = re.compile(r"\s+")


# ------------------------------------------------------------------ строки
def _strip_marks(ch: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c))


def norm(value: object) -> str:
    """NFKC, нижний регистр, ё→е; у латиницы снята диакритика; пунктуация, кроме «/», — пробел."""
    s = unicodedata.normalize("NFKC", "" if value is None else str(value)).lower()
    s = s.replace("ё", "е")
    s = "".join(ch if ord(ch) >= 0x0400 else _strip_marks(ch) for ch in s)
    return _SPACE_RE.sub(" ", _PUNCT_RE.sub(" ", s)).strip()


def lev(a: str, b: str) -> int:
    """Расстояние Левенштейна с единичными весами."""
    return Levenshtein.distance(a, b)


def best_cer(token: str, text: str) -> float | None:
    """CER токена против лучшего окна из того же числа слов (±1) в распознанном тексте.

    Порядок строк OCR не важен. `None` — токен пуст после нормализации.
    """
    target = norm(token)
    if not target:
        return None
    words = norm(text).split()
    if not words:
        return 1.0
    n = len(target.split())
    best = 1.0
    for i in range(len(words)):
        for k in (n - 1, n, n + 1):
            if k < 1 or i + k > len(words):
                continue
            best = min(best, lev(target, " ".join(words[i : i + k])) / len(target))
    return best


# ------------------------------------------------------------------ сводки
def rate(num: int, den: int) -> float:
    return round(num / max(1, den), 3)


def percentile(values: Sequence[float], q: float) -> float | None:
    """Ближайший ранг, как в исследовании: без интерполяции."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def latency_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    """p50 и p95 (подходит и для числа токенов промпта)."""
    return {"p50": percentile(values, 0.5), "p95": percentile(values, 0.95), "n": len(values)}


def _cer_row(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "mean": round(sum(values) / len(values), 3),
        "share_le_0.2": round(sum(v <= CER_HIT for v in values) / len(values), 3),
        "n": len(values),
    }


def _top(pred: Prediction | None, key: str) -> list[str]:
    if not pred:
        return []
    return [str(v) for v in pred.get(key) or []]


def _gt_record(slug: str, gtok: GtTokens) -> Mapping[str, Any] | None:
    return None if slug == NONE_SLUG else gtok.get(slug)


# ------------------------------------------------------------------ CER токенов
def expected_label_tokens(
    annotation: Mapping[str, Any] | None, gt_record: Mapping[str, Any] | None
) -> tuple[str, LabelTokens]:
    """Токены для CER: ручная транскрипция кадра, иначе первичные токены каталога.

    Возвращает источник (`front_label_text` | `expected_tokens` | `none`) и пары (поле, текст).
    """
    front = (annotation or {}).get("front_label_text") or []
    tokens = [(str(t.get("field", "")), str(t["text"])) for t in front if t.get("text")]
    if tokens:
        return "front_label_text", tokens
    if gt_record:
        tokens = [
            (str(t["field"]), str(t["text"]))
            for t in gt_record.get("expected_tokens") or []
            if t.get("primary") and t.get("field") in KEY_TOKEN_FIELDS and t.get("text")
        ]
        if tokens:
            return "expected_tokens", tokens
    return "none", []


def token_cer_table(
    samples: Iterable[tuple[Sequence[tuple[str, str]], str]],
) -> dict[str, dict[str, float | int]]:
    """CER по полям для пар (токены кадра, распознанный текст) и сводка по ключевым полям."""
    by_field: dict[str, list[float]] = defaultdict(list)
    for tokens, text in samples:
        for field, token in tokens:
            cer = best_cer(token, text)
            if cer is None:
                continue
            by_field[field].append(cer)
            if field in KEY_TOKEN_FIELDS:
                by_field[KEY_AGGREGATE].append(cer)
    return {field: _cer_row(values) for field, values in by_field.items()}


def key_token_cer(
    gt_rows: Sequence[GtRow], preds: Predictions, gtok: GtTokens
) -> dict[str, dict[str, float | int]]:
    """CER первичных токенов каталога в `raw_text` для кадров из каталога."""
    samples = []
    for qid, slug in gt_rows:
        record = _gt_record(slug, gtok)
        if not record:
            continue
        _, tokens = expected_label_tokens(None, record)
        samples.append((tokens, str((preds.get(qid) or {}).get("raw_text") or "")))
    return token_cer_table(samples)


# ------------------------------------------------------------------ поля
def _blank(value: object) -> bool:
    return value is None or (isinstance(value, str | list | tuple) and len(value) == 0)


def _winery_match(expected: Sequence[str], got: object) -> bool:
    got_n = norm(got)
    costs = [lev(got_n, norm(v)) / len(norm(v)) for v in expected if norm(v)]
    return bool(costs) and min(costs) <= CER_HIT


def _any_token_match(expected: Sequence[str], got: object) -> bool:
    for value in expected:
        cer = best_cer(value, str(got))
        if cer is not None and cer <= CER_HIT:
            return True
    return False


def _sugar_match(expected: str, got: object) -> bool:
    # Список значений верен, только если все они — ожидаемый класс: перебор не награждается.
    values = [got] if isinstance(got, str) else list(got)  # type: ignore[call-overload]
    return bool(values) and {str(v) for v in values} == {expected}


def _year_match(expected: object, got: object) -> bool:
    try:
        return int(expected) == int(got)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return False


def _serial_match(expected: Sequence[str], got: object) -> bool:
    values = [got] if isinstance(got, str) else list(got)  # type: ignore[call-overload]
    return {norm(v) for v in expected} <= {norm(v) for v in values}


def _score(
    counts: dict[str, Counter[str]],
    name: str,
    expected: Any,
    got: Any,
    match: Callable[[Any, Any], bool],
) -> None:
    """Поле: correct / wrong (прочитано не то) / abstain (не прочитано); без эталона — пропуск."""
    if _blank(expected):
        return
    if _blank(got):
        counts[name]["abstain"] += 1
    elif match(expected, got):
        counts[name]["correct"] += 1
    else:
        counts[name]["wrong"] += 1


def field_metrics(
    gt_rows: Sequence[GtRow], preds: Predictions, gtok: GtTokens
) -> dict[str, dict[str, float | int]]:
    """Точность полей против эталона каталога; `wrong_rate` — доля выдумок."""
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for qid, slug in gt_rows:
        record = _gt_record(slug, gtok)
        if not record:
            continue
        got = (preds.get(qid) or {}).get("fields") or {}
        fl = record["fields"]
        winery = [" ".join(fl["winery"]["key_tokens"]), *fl["winery"]["variants"]]
        winery = [v for v in [*winery, *fl["winery"]["brands"]] if v]
        _score(counts, "winery", winery, got.get("winery"), _winery_match)
        cuvee_or_grape = [*fl["cuvee"]["tokens"], *fl["grape"]["variants"]]
        got_cg = " ".join(str(v) for v in (got.get("cuvee"), got.get("grape")) if v)
        _score(counts, "cuvee_or_grape", cuvee_or_grape, got_cg, _any_token_match)
        _score(counts, "sugar", fl["sugar"]["class"], got.get("sugar"), _sugar_match)
        _score(counts, "year", fl["year"]["value"], got.get("year"), _year_match)
        serial = [*fl["serial"]["tokens"], *fl["serial"]["keywords"]]
        _score(counts, "serial", serial, got.get("serial"), _serial_match)
    result = {}
    for name, c in counts.items():
        total = sum(c.values())
        result[name] = {
            "correct": c["correct"],
            "wrong": c["wrong"],
            "abstain": c["abstain"],
            "acc": rate(c["correct"], total),
            "wrong_rate": rate(c["wrong"], total),
        }
    return result


# ------------------------------------------------------------------ ответы (диагностика)
def text_topk(
    gt_rows: Sequence[GtRow], preds: Predictions, *, key: str = "text_top5"
) -> dict[str, float | int]:
    """Top-1 и top-5 по кадрам из каталога."""
    top1 = top5 = n = 0
    for qid, slug in gt_rows:
        if slug == NONE_SLUG:
            continue
        n += 1
        top = _top(preds.get(qid), key)
        top1 += int(bool(top) and top[0] == slug)
        top5 += int(slug in top[:5])
    return {"top1": rate(top1, n), "top5": rate(top5, n), "n": n}


def twin_resolved_top1(
    gt_rows: Sequence[GtRow], preds: Predictions, gtok: GtTokens, *, key: str = "text_top5"
) -> dict[str, float | int]:
    """Кадры, у цели которых есть визуальные двойники: верный ответ и ответ двойником."""
    queries = resolved = by_mate = 0
    for qid, slug in gt_rows:
        record = _gt_record(slug, gtok)
        mates = set(record.get("visual_mates") or []) if record else set()
        if not mates:
            continue
        queries += 1
        top = _top(preds.get(qid), key)
        first = top[0] if top else None
        resolved += int(first == slug)
        by_mate += int(first in mates)
    return {
        "twin_queries": queries,
        "twin_resolved_top1": rate(resolved, queries),
        "twin_answered_by_mate": rate(by_mate, queries),
    }


def answered_by_frame_neighbor(
    gt_rows: Sequence[GtRow], preds: Predictions, ann: Annotations, *, key: str = "text_top5"
) -> dict[str, float | int]:
    """Доля кадров с известными соседями, где первым ответом стал сосед.

    Считаются и кадры вне каталога: ответ «Розовым золотом» на q1 — тоже ответ соседом.
    """
    queries = answered = 0
    for qid, slug in gt_rows:
        neighbors = {
            n["slug"]
            for n in (ann.get(qid) or {}).get("neighbors") or []
            if n.get("slug") and n["slug"] != slug
        }
        if not neighbors:
            continue
        queries += 1
        top = _top(preds.get(qid), key)
        answered += int(bool(top) and top[0] in neighbors)
    return {"neighbor_queries": queries, "answered_by_frame_neighbor": rate(answered, queries)}


def ooc_reject_rate(
    gt_rows: Sequence[GtRow], preds: Predictions, *, key: str = "text_top5"
) -> float | None:
    """Доля отказов на кадрах вне каталога; нет таких кадров — `None`."""
    ooc = [qid for qid, slug in gt_rows if slug == NONE_SLUG]
    if not ooc:
        return None
    return rate(sum(not _top(preds.get(qid), key) for qid in ooc), len(ooc))


def cv_vs_cv_ocr(gt_rows: Sequence[GtRow], preds: Predictions) -> dict[str, Any] | None:
    """«CV» против «CV+OCR»: считается, когда предсказания несут `cv_top5` (и `final_top5`)."""
    rows = [(qid, slug) for qid, slug in gt_rows if "cv_top5" in (preds.get(qid) or {})]
    if not rows:
        return None
    cv = text_topk(rows, preds, key="cv_top5")
    out: dict[str, Any] = {
        "queries": len(rows),
        "cv_top1": cv["top1"],
        "cv_top5": cv["top5"],
        "cv_ooc_reject_rate": ooc_reject_rate(rows, preds, key="cv_top5"),
    }
    if all("final_top5" in preds[qid] for qid, _ in rows):
        final = text_topk(rows, preds, key="final_top5")
        out["cv_ocr_top1"] = final["top1"]
        out["cv_ocr_top5"] = final["top5"]
        out["cv_ocr_ooc_reject_rate"] = ooc_reject_rate(rows, preds, key="final_top5")
        out["delta_top1"] = round(final["top1"] - cv["top1"], 3)
    return out


def status_counts(gt_rows: Sequence[GtRow], preds: Predictions) -> dict[str, int]:
    counts = Counter(
        str(preds[qid].get("status", "unknown")) if qid in preds else "missing"
        for qid, _ in gt_rows
    )
    return dict(sorted(counts.items()))


def evaluate(
    gt_rows: Sequence[GtRow], ann: Annotations, preds: Predictions, gtok: GtTokens
) -> dict[str, Any]:
    in_catalog = sum(slug != NONE_SLUG for _, slug in gt_rows)
    top = text_topk(gt_rows, preds)
    latencies = [
        p["latency_ms"]
        for qid, _ in gt_rows
        if (p := preds.get(qid)) is not None and p.get("latency_ms") is not None
    ]
    return {
        "queries": len(gt_rows),
        "in_catalog": in_catalog,
        "out_of_catalog": len(gt_rows) - in_catalog,
        "field_accuracy": field_metrics(gt_rows, preds, gtok),
        "key_token_cer": key_token_cer(gt_rows, preds, gtok),
        "text_only_top1": top["top1"],
        "text_only_top5": top["top5"],
        **twin_resolved_top1(gt_rows, preds, gtok),
        **answered_by_frame_neighbor(gt_rows, preds, ann),
        "ooc_reject_rate": ooc_reject_rate(gt_rows, preds),
        "cv_vs_cv_ocr": cv_vs_cv_ocr(gt_rows, preds),
        "statuses": status_counts(gt_rows, preds),
        "latency_ms": latency_summary(latencies),
    }


# ------------------------------------------------------------------ самопроверка
def _primary_text(record: Mapping[str, Any]) -> str:
    return " ".join(t["text"] for t in record.get("expected_tokens") or [] if t.get("primary"))


def synth_predictions(
    gt_rows: Sequence[GtRow], ann: Annotations, gtok: GtTokens, *, seed: int = 1
) -> dict[str, dict[str, Any]]:
    """Зашумлённые предсказания из эталона — проверка кода метрик, цифры смысла не имеют."""
    rng = random.Random(seed)
    preds: dict[str, dict[str, Any]] = {}
    for qid, slug in gt_rows:
        neigh = [
            n["slug"] for n in (ann.get(qid) or {}).get("neighbors") or [] if n.get("slug") in gtok
        ]
        mode = rng.random()
        latency = rng.randint(400, 2600)
        record = _gt_record(slug, gtok)
        if record is None:  # вне каталога (или нет эталона): то отказ, то ответ соседом
            top = [neigh[0]] if neigh and mode < 0.3 else []
            preds[qid] = {"query_id": qid, "raw_text": "", "latency_ms": latency}
            preds[qid].update(status="ok", text_top5=top, fields={})
            continue
        raw = _primary_text(record)
        if mode < 0.2 and neigh:  # прочитан сосед
            raw = _primary_text(gtok[neigh[0]])
            top = [neigh[0], slug]
        elif mode < 0.4:  # шум OCR
            raw = "".join(c if rng.random() > 0.15 else rng.choice("оаеil1") for c in raw)
            top = [slug] if rng.random() < 0.5 else []
        else:
            top = [slug]
        f = record["fields"]
        preds[qid] = {
            "query_id": qid,
            "raw_text": raw,
            "latency_ms": latency,
            "status": "ok",
            "text_top5": top,
            "fields": {
                "winery": " ".join(f["winery"]["key_tokens"]),
                "cuvee": " ".join(f["cuvee"]["tokens"]),
                "grape": (f["grape"]["variants"] or [None])[0],
                "sugar": f["sugar"]["class"] if mode > 0.3 else "suhoe",
                "year": f["year"]["value"],
                "serial": f["serial"]["tokens"],
            },
        }
    return preds


# ------------------------------------------------------------------ файлы и CLI
def load_gt_rows(path: Path) -> list[GtRow]:
    """TSV с заголовком: query_id<TAB>slug[<TAB>…]."""
    lines = path.read_text(encoding="utf-8-sig").splitlines()[1:]
    rows = [line.split("\t") for line in lines if line.strip()]
    return [(r[0].strip(), r[1].strip()) for r in rows if len(r) >= 2]


def load_jsonl_by_qid(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                out[rec["query_id"]] = rec
    return out


def load_gt_tokens(path: Path) -> dict[str, dict[str, Any]]:
    """`gt_tokens.jsonl`: эталонные токены этикетки по slug."""
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                out[rec["slug"]] = rec
    return out


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1)


def print_json(obj: Any) -> None:
    """Печать JSON; консоль без UTF-8 получает экранированный вариант."""
    text = dumps(obj)
    try:
        print(text)
    except UnicodeEncodeError:
        print(json.dumps(obj, ensure_ascii=True, indent=1))


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="python -m bench.metrics", description="Метрики OCR-модуля по предсказаниям."
    )
    parser.add_argument(
        "--gt", type=Path, required=True, help="TSV: query_id<TAB>slug; вне каталога __none__"
    )
    parser.add_argument("--ann", type=Path, help="JSONL разметки кадров: соседи, транскрипции")
    parser.add_argument(
        "--gt-tokens", type=Path, default=settings.data_dir / "gt" / "gt_tokens.jsonl"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pred", type=Path, help="JSONL предсказаний, строка на кадр")
    source.add_argument(
        "--selftest", action="store_true", help="зашумлённые предсказания из эталона"
    )
    parser.add_argument("--out", type=Path, help="куда записать JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    gtok = load_gt_tokens(args.gt_tokens) if args.gt_tokens.is_file() else {}
    if not gtok:
        print(f"предупреждение: нет эталонных токенов {args.gt_tokens}", file=sys.stderr)
    gt_rows = load_gt_rows(args.gt)
    ann = load_jsonl_by_qid(args.ann) if args.ann and args.ann.is_file() else {}
    if args.selftest:
        preds: Predictions = synth_predictions(gt_rows, ann, gtok)
    else:
        preds = load_jsonl_by_qid(args.pred)
    result = evaluate(gt_rows, ann, preds, gtok)
    print_json(result)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(dumps(result) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
