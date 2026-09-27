"""Э7, шаг 1 (разбор до PREREG): где правило имени собственного снимает цвет или сахар.

Только поля чтения: ответов кадров этот скрипт не считает, слоя выбора не вызывает.

`_starts_proper_name` (`app/reading/fields.py`) считает прилагательное не среднего рода («БЕЛЫЙ»,
«СУХАЯ») началом имени, если следующее слово того же отрезка не из словаря этикетки. На
WebP-оригинале R014 (= публичный `q-000002`) читатель пишет «МАССАНДАРА» / «БЕЛЫЙ» / «ГОД УРОЖАЯ»:
«год» не из словаря, цвет пуст, ответ — розовый сосед (окно A). Скрипт переписывает все срабатывания
правила по всем записанным чтениям и слово, которое его вызвало, считает слова двух пунктов Э7 и
смену полей цвета и сахара, если слова пункта не начинают имя.

Чтения:
- v2, kr, ooc — строки VLM стенда Э6 (`runs/field25/iters/runs/e6/pkg_{v2,kr,ooc}`): разбор тот же,
  что у `after-search` @1d5b4e9 (Э6 слит без изменений `app/reading`, кроме выброшенного Э6-С);
- студия — dev-чтения модели -goal (`runs/ocr-pairs*-vlm35-m2`, 296 со статусом ok);
- окно A — два живых скана (`gpu_A/0925_1720/public_scan`: `q-000002` WebP и R014 JPEG).

Поля пересобираются кодом этого дерева (`tokenize` → словарь → `extract_fields`) и сверяются с
записанными — проверка способа. Студийные поля записаны до Э6, поэтому для студии сверяются только
сахар, цвет, год, крепость и серия; число расхождений в винодельне, кюве и сорте печатается (это
ровно встречаемость Э6-В, Ш и Г из его PREREG).

    cd <дерево>; PYTHONPATH=. python research/2026-09-26_e7/e7_census.py [--out e7_census.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator
from itertools import pairwise
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

import app.reading.fields as F
from app.reading.contracts import Reading, TextLine
from app.reading.lexicon.build import Lexicon
from app.reading.pipeline import label_search
from app.reading.text.layout import token_segments
from app.reading.text.normalize import norm_token
from app.reading.text.tokenize import tokenize

ROOT = Path(r"<корень>")
SCANNER = ROOT / "svoe-vino-scanner"
E6 = SCANNER / "runs" / "field25" / "iters" / "runs" / "e6"
STUDIO = [SCANNER / "runs" / "ocr-pairs-vlm35-m2", SCANNER / "runs" / "ocr-pairsphone-vlm35-m2"]
PUBLIC = ROOT / "svs-logs" / "somm-2409" / "gpu_A" / "0925_1720" / "public_scan"
LEXICON = SCANNER / "data" / "index" / "lexicon.json"  # c2c7befd = словарь стенда Э6
ALL_FIELDS = ("producer", "cuvee", "grapes", "sugar", "vintage", "serial", "abv", "color")
STUDIO_FIELDS = ("sugar", "vintage", "serial", "abv", "color")


def _words(text: str) -> frozenset[str]:
    return frozenset(norm_token(word) for word in text.split())


#: Пункты Э7 — списки PREREG §2 дословно (в коде — таблицы `app/reading/taxonomy.py`).
UNITS: dict[str, frozenset[str]] = {
    "u": _words(
        "год года году годы лет урожай урожая урожаи выдержка выдержки выдержкой выпуска "
        "vintage millesime millesimato harvest vendemmia annata cosecha jahrgang"
    ),
    "o": _words(
        "объем объема литр литра литров л мл ml cl alc alcohol алк алкоголь крепость спирта "
        "спирт этилового abv vol об"
    ),
}
UNIT_NAMES = {"u": "Э7-У строка урожая и возраста", "o": "Э7-О строка объёма и крепости"}


def parse_key(key: str) -> dict[str, Any]:
    """`Reading.key` → поля `Reading`: `reader@version|params_hash|image_sha1|crop|crop_px`."""
    head, params, image, crop, px = key.rsplit("|", 4)
    reader, _, version = head.partition("@")
    return {
        "reader": reader,
        "version": version,
        "params_hash": params,
        "image_sha1": image,
        "crop": crop,
        "crop_px": int(px),
    }


def first_source(fields: dict[str, Any] | None) -> str | None:
    for name in (*ALL_FIELDS, "unmatched"):
        value = (fields or {}).get(name)
        items = value if isinstance(value, list) else [value] if value else []
        for item in items:
            if item.get("sources"):
                return str(item["sources"][0])
    return None


DUMMY_KEY = "vlm@qwen3.5:4b|f3a017317f04|census|full|1024"


def reads() -> Iterator[tuple[str, str, list[str], str, dict[str, Any] | None]]:
    """(набор, id, строки, ключ чтения, записанные поля) для каждого чтения со статусом ok."""
    for set_key in ("v2", "kr", "ooc"):
        with (E6 / f"pkg_{set_key}" / "predictions.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                vlm = rec.get("vlm") or {}
                if vlm.get("status") != "ok" or not vlm.get("lines"):
                    continue
                fields = (rec.get("text_read") or {}).get("fields")
                key = first_source(fields) or DUMMY_KEY
                yield set_key, rec["query_id"], list(vlm["lines"]), key, fields
    for path in STUDIO:
        with (path / "predictions.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("status") != "ok" or not rec.get("lines"):
                    continue
                texts = [ln["text"] for ln in sorted(rec["lines"], key=lambda ln: ln["id"])]
                qid = f"{path.name}:{rec['query_id']}"
                yield "studio", qid, texts, rec["reader"], rec.get("label_fields")
    for path in sorted(PUBLIC.glob("scan_*_1.json")):
        body = json.loads(path.read_text(encoding="utf-8"))
        yield "public", path.stem, list(body["evidence"]["vlm"]["lines"]), DUMMY_KEY, None


LOG: list[dict[str, Any]] = []
_ORIGINAL = F._starts_proper_name


def _spy(ctx: Any, i: int, form: str, term_ids: set[int], grape_ids: set[int]) -> bool:
    fired = _ORIGINAL(ctx, i, form, term_ids, grape_ids)
    if fired:
        after = F._segment_neighbour(ctx, i, 1)
        before = F._segment_neighbour(ctx, i, -1)
        assert after is not None
        LOG.append(
            {
                "i": i,
                "form": form,
                "word": ctx.tokens[i].norm,
                "after": ctx.tokens[after].norm,
                "before": ctx.tokens[before].norm if before is not None else None,
            }
        )
    return fired


def parse(lines: list[str], key: str, lexicon: Lexicon) -> tuple[Any, list[Any]]:
    reading = Reading(
        **parse_key(key),
        lines=[TextLine(id=i, text=text) for i, text in enumerate(lines)],
        elapsed_ms=0,
    )
    tokens = tokenize(reading, lexicon=lexicon)
    hits = label_search(tokens, lexicon, token_segments(tokens, [reading])).hits
    fields = F.extract_fields(tokens, [(list(ids), hit) for ids, hit in hits], readings=[reading])
    return fields, tokens


def color_sugar(fields: Any) -> tuple[str | None, list[str]]:
    color = str(fields.color.value) if fields.color else None
    return color, sorted(str(e.value) for e in fields.sugar)


def with_vocabulary(words: frozenset[str], lines: list[str], key: str, lexicon: Lexicon) -> Any:
    """Поля, если слова `words` не начинают имя: словарь этикетки правила имени + `words`."""
    saved = F._WINE_VOCABULARY
    F._WINE_VOCABULARY = saved | words  # type: ignore[misc]
    try:
        return parse(lines, key, lexicon)[0]
    finally:
        F._WINE_VOCABULARY = saved  # type: ignore[misc]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "e7_census.json")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    lexicon = Lexicon.load(LEXICON)
    F._starts_proper_name = _spy  # type: ignore[assignment]
    already = F._WINE_VOCABULARY

    n_reads: Counter[str] = Counter()
    mismatch: dict[str, list[str]] = defaultdict(list)
    studio_lexical: Counter[str] = Counter()
    firings: list[dict[str, Any]] = []
    occurrences: dict[str, dict[str, Counter[str]]] = {u: defaultdict(Counter) for u in UNITS}
    adjacent: dict[str, Counter[str]] = defaultdict(Counter)
    changed: dict[str, dict[str, list[dict[str, Any]]]] = {u: defaultdict(list) for u in UNITS}

    for set_key, qid, lines, key, recorded in reads():
        n_reads[set_key] += 1
        LOG.clear()
        fields, tokens = parse(lines, key, lexicon)
        if recorded is not None:
            got = fields.model_dump(mode="json")
            compared = STUDIO_FIELDS if set_key == "studio" else ALL_FIELDS
            if any(got.get(name) != recorded.get(name) for name in compared):
                mismatch[set_key].append(qid)
            if set_key == "studio":
                for name in ("producer", "cuvee", "grapes"):
                    studio_lexical[name] += got.get(name) != recorded.get(name)
        fired = list({item["i"]: item for item in LOG}.values())
        firings += [{"set": set_key, "id": qid, **item, "lines": lines} for item in fired]
        words = [t.norm if t.kind == "word" else None for t in tokens]
        for unit, table in UNITS.items():
            for word in {w for w in words if w} & table:
                occurrences[unit][set_key][word] += 1
        for a, b in pairwise(tokens):
            if (
                a.norm in F.AGREEING_FORMS
                and b.kind == "word"
                and b.norm in UNITS["u"] | UNITS["o"]
            ):
                adjacent[set_key][f"{a.norm} {b.norm}"] += 1
        base = color_sugar(fields)
        for unit, table in UNITS.items():
            if not any(item["after"] in table for item in fired):
                continue
            new = color_sugar(with_vocabulary(table, lines, key, lexicon))
            if new != base:
                changed[unit][set_key].append({"id": qid, "base": base, "new": new, "lines": lines})

    print("чтений:", dict(n_reads))
    print("поля ≠ записанным:", {k: len(v) for k, v in mismatch.items()}, dict(mismatch))
    print("студия, винодельня/кюве/сорт ≠ записанным до Э6 (справочно):", dict(studio_lexical))
    after_total = Counter(f["after"] for f in firings)
    print(f"срабатываний правила имени: {len(firings)}")
    for word, n in after_total.most_common():
        sets = Counter(f["set"] for f in firings if f["after"] == word)
        unit = next((u for u, t in UNITS.items() if word in t), "—")
        print(f"  {word!r:14} {n:2d}  {dict(sets)}  пункт: {unit}")
    for unit, table in UNITS.items():
        print(f"{UNIT_NAMES[unit]}: слов {len(table)}, уже в словаре правила: "
              f"{sorted(table & already)}")  # fmt: skip
        print(
            "  чтений со словом пункта:", {s: sum(c.values()) for s, c in occurrences[unit].items()}
        )
        print("  по словам:", {s: dict(c) for s, c in occurrences[unit].items()})
        print("  смена цвета/сахара:", {s: len(v) for s, v in changed[unit].items()} or "нет")
        for s, rows in changed[unit].items():
            for row in rows:
                print(
                    f"    {s} {row['id']}: {row['base']} → {row['new']}  {' / '.join(row['lines'])}"
                )
    print(
        "прилагательное не ср. рода + слово пункта рядом:",
        {s: dict(c) for s, c in adjacent.items()},
    )
    out = {
        "reads": dict(n_reads),
        "fields_mismatch": dict(mismatch),
        "studio_lexical_vs_pre_e6": dict(studio_lexical),
        "firings": firings,
        "after_total": dict(after_total.most_common()),
        "units": {u: sorted(t) for u, t in UNITS.items()},
        "already_in_rule_vocabulary": {u: sorted(t & already) for u, t in UNITS.items()},
        "occurrences": {u: {s: dict(c) for s, c in v.items()} for u, v in occurrences.items()},
        "adjacent": {s: dict(c) for s, c in adjacent.items()},
        "changed": {u: dict(v) for u, v in changed.items()},
    }
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 1 if any(mismatch.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
