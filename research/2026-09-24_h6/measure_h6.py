"""Замер H6 строго по `PREREG.md`: прочитанное решает выбор внутри винодельни.

Только CPU и записанные прогоны: видеокарта, Ollama и SigLIP не нужны. Правило, источники
атрибутов, база и критерии — из PREREG (коммит ba8ef42), здесь они только исполняются.

Входы:
    runs/field25/iters/runs/pfix_final/iter20   — текущий сервис (после правки эталонов): решение
    runs/…/v2prod (+ v2prod_l29), ooc_v2prod    — только для сведения
    field_dataset/sets/catalog_v2/meta.jsonl    — эталон (slug ∪ acceptable = «то же вино»)
    data/gt/gt_tokens.jsonl                     — атрибуты сервиса (H6-svc)
    field_dataset/catalog/wines_final.jsonl     — справочник вин (H6-dict)

Ворота исправности (PREREG: «пересчёт H5 из дампа совпадает с ответами pfix_final на 353 из 353»):
    A  поля кадра из дампа (`text_read.fields` → `LabelFields`, как у сервиса) + `contradicts()`
       по записанному порядку слоя (`resolve`, 20) дают `whatif["filter"]` — все 5 позиций;
    B  пересчёт кодом ветки (`rank_query_detailed` + `rerank_ambiguous` сервиса) по записанным
       CV и чтению даёт тот же порядок слоя (20 из 20) и тот же ответ H5, что `whatif`;
    C  база воспроизводит 86,0 макро / 84,1 строго микро.
Не прошли — замер не идёт.

Итог 24.09 (`results.json`): ворота A, B и C — 353/353, база 86,0 / 84,1. H6-svc: 10 починок,
2 поломки, строго микро 84,14 → 86,40, но срез R −1 (R076: на этикетке 2024 «полусухое», у
карточки «сухое») — не принят. H6-dict: 11 починок, 4 поломки, R −3 — не принят. Сервис не
меняется.

    .venv/Scripts/python.exe research/2026-09-24_h6/measure_h6.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from app.features.contracts import VisualResult
from app.reading.contracts import LabelFields
from app.resolve.ambiguous import contradicts, rerank_ambiguous
from app.resolve.attrs import (
    CatalogAttrs,
    color_key,
    grape_key,
    norm_key,
    sugar_key,
)
from app.resolve.features import TextRead, readers_of
from app.resolve.learned import LogisticRanker, rank_query_detailed

ROOT = Path(r"<корень>")
DATA = Path(os.environ.get("SVS_DATA_DIR") or ROOT / "svoe-vino-scanner" / "data")
RUNS = ROOT / "svoe-vino-scanner" / "runs" / "field25" / "iters" / "runs"
FD = ROOT / "field_dataset"
MODEL = REPO / "configs" / "resolve" / "s2so400m-vlm35-goal.json"

#: Порог спорного кадра — `service.AMBIGUOUS_P_TOP1` (сервис не импортируем: он тянет модели).
AMBIGUOUS_P_TOP1 = 0.5
#: База PREREG: H5 `pfix_final`, «то же вино» макро и строго микро, в процентах.
BASE_SOFT_MACRO, BASE_STRICT_MICRO = 86.0, 84.1
SLICES = ("R", "M", "L")
BOOT, SEED = 2000, 0


def jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------ кадры прогона
@dataclass(frozen=True)
class Frame:
    """Один кадр записанного прогона: чтение, порядок слоя и ответы."""

    qid: str
    record: dict[str, Any]
    p_top1: float
    slug: str  # ответ прогона: обученный слой без H5
    order: tuple[str, ...]  # `ranking.slugs` сервиса — порядок слоя, 20 кандидатов
    fields: LabelFields | None  # как `read.fields` сервиса: None — чтения нет
    whatif: dict[str, Any] | None

    @property
    def lines(self) -> list[str]:
        return list((self.record.get("vlm") or {}).get("lines") or [])

    @property
    def h5(self) -> str:
        """Ответ H5 так же, как в `eval_v2.py`: при p_top1 < 0,5 — `no_cluster+filter`."""
        if self.p_top1 < AMBIGUOUS_P_TOP1 and self.whatif is not None:
            alt = self.whatif.get("no_cluster+filter") or []
            return alt[0] if alt else self.slug
        return self.slug


def label_fields(record: Mapping[str, Any]) -> LabelFields | None:
    """Поля кадра так, как их отдаёт `_read` сервиса (и `resolve_whatif.py` — модели)."""
    read = record.get("text_read")
    if read and read.get("fields"):
        return LabelFields.model_validate(read["fields"])
    return None


def load_run(
    name: str, *, extra: Sequence[tuple[str, str]] = (), override: str | None = None
) -> tuple[list[Frame], list[str]]:
    """Кадры прогона; `extra` — дополнительные пары (predictions, whatif) того же прогона.

    `override` — прогон, из которого берутся его кадры поверх основного (v2prod_l29 для L29).
    """
    base = RUNS / name / "iter20"
    pairs = [("predictions.jsonl", "whatif.json"), *extra]
    notes: list[str] = []
    records: dict[str, dict[str, Any]] = {}
    whatif: dict[str, dict[str, Any]] = {}
    for pred, wi in pairs:
        for rec in jsonl(base / pred):
            records[rec["query_id"]] = rec
        if (base / wi).is_file():
            for row in json.loads((base / wi).read_text(encoding="utf-8")):
                whatif[row["query_id"]] = row
    if override:
        for rec in jsonl(RUNS / override / "iter20" / "predictions.jsonl"):
            old = records.get(rec["query_id"])
            same = old is not None and _core(old) == _core(rec)
            notes.append(
                f"{rec['query_id']} взят из {override}; в {name} "
                + (
                    "есть и совпадает по slug/p_top1/resolve/чтению"
                    if same
                    else "его нет или он другой"
                )
            )
            records[rec["query_id"]] = rec
    frames = [
        Frame(
            qid=qid,
            record=rec,
            p_top1=float(rec.get("p_top1") or 0.0),
            slug=rec["slug"],
            order=tuple(slug for slug, _ in rec.get("resolve") or []),
            fields=label_fields(rec),
            whatif=whatif.get(qid),
        )
        for qid, rec in records.items()
    ]
    return frames, notes


def _core(rec: Mapping[str, Any]) -> tuple[Any, ...]:
    return (rec["slug"], rec.get("p_top1"), rec.get("resolve"), rec.get("text_read"))


# ------------------------------------------------------------------ источники атрибутов
@dataclass(frozen=True)
class Source:
    """Таблица атрибутов для правила: карточки в ключах сервиса и словарь кодов сортов."""

    name: str
    attrs: CatalogAttrs  # для `contradicts` — ровно та же функция, что у H5
    universe: frozenset[str]  # коды сортов, встречающиеся хотя бы у одной карточки

    def winery(self, slug: str) -> str:
        wine = self.attrs.get(slug)
        return norm_key(wine.winery) if wine else ""

    def grapes(self, slug: str) -> frozenset[str]:
        wine = self.attrs.get(slug)
        return wine.grapes if wine else frozenset()

    def sugar(self, slug: str) -> str | None:
        wine = self.attrs.get(slug)
        return wine.sugar.value if wine and wine.sugar else None

    def color(self, slug: str) -> str | None:
        wine = self.attrs.get(slug)
        return wine.color.value if wine and wine.color else None


def codes(values: Iterable[object]) -> frozenset[str]:
    return frozenset(key for value in values if (key := grape_key(value)))


def service_source() -> Source:
    attrs = CatalogAttrs.load(DATA / "gt" / "gt_tokens.jsonl")
    universe = frozenset().union(*(wine.grapes for wine in attrs))
    return Source("svc", attrs, universe)


def dict_source(svc: Source) -> tuple[Source, dict[str, int]]:
    """Справочник вин поверх сервиса: сахар, цвет, сорта и винодельня — из справочника.

    slug вне справочника — атрибуты сервиса (PREREG). Кодов сортов — всё, что есть у карточек
    справочника (включая 73 карточки живого портала), плюс сервисные у slug вне справочника.
    """
    cards = {rec["slug"]: rec for rec in jsonl(FD / "catalog" / "wines_final.jsonl")}
    wines, diff = [], Counter()
    for wine in svc.attrs:
        card = cards.get(wine.slug)
        if card is None:
            diff["вне справочника"] += 1
            wines.append(wine)
            continue
        new = replace(
            wine,
            winery=str(card.get("winery") or ""),
            sugar=sugar_key(card.get("sugar")),
            color=color_key(card.get("color")),
            grapes=codes(card.get("grapes") or []),
        )
        for field in ("sugar", "color", "grapes"):
            diff[field] += getattr(new, field) != getattr(wine, field)
        diff["winery"] += norm_key(new.winery) != norm_key(wine.winery)
        wines.append(new)
    attrs = CatalogAttrs(wines, meta={"source": "wines_final.jsonl"})
    universe = frozenset().union(
        *(codes(card.get("grapes") or []) for card in cards.values()),
        *(wine.grapes for wine in svc.attrs if wine.slug not in cards),
    )
    return Source("dict", attrs, universe), dict(diff)


# ------------------------------------------------------------------ правило H6
@dataclass(frozen=True)
class Read:
    """Прочитанное кадра в ключах правила: S, c, G."""

    sugar: frozenset[str]
    color: str | None
    grapes: frozenset[str]


def read_of(fields: LabelFields | None, universe: frozenset[str]) -> Read:
    if fields is None:
        return Read(frozenset(), None, frozenset())
    return Read(
        sugar=frozenset(str(item.value) for item in fields.sugar),
        color=str(fields.color.value) if fields.color is not None else None,
        grapes=frozenset(k for item in fields.grapes if (k := grape_key(item.value)) in universe),
    )


def h6(
    frame: Frame, answer: str, src: Source, *, sugar_color: bool = True, grapes: bool = True
) -> str:
    """Ответ H6 для итогового ответа H5 `answer`. Флаги — только для разложения (для сведения)."""
    read = read_of(frame.fields, src.universe)

    def conflict(slug: str) -> bool:
        if sugar_color and contradicts(slug, frame.fields, src.attrs):
            return True
        card = src.grapes(slug)
        return grapes and bool(read.grapes) and bool(card) and not (read.grapes & card)

    def support(slug: str) -> bool:
        if sugar_color and read.sugar and src.sugar(slug) in read.sugar:
            return True
        if sugar_color and read.color is not None and src.color(slug) == read.color:
            return True
        return grapes and bool(read.grapes & src.grapes(slug))

    if not conflict(answer):
        return answer
    winery = src.winery(answer)
    if not winery:
        return answer
    for slug in frame.order:
        if slug == answer or src.winery(slug) != winery:
            continue
        if not conflict(slug) and support(slug):
            return slug
    return answer


# ------------------------------------------------------------------ счёт
def ok_sets(meta: Sequence[Mapping[str, Any]]) -> dict[str, tuple[str, frozenset[str]]]:
    return {
        m["query_id"]: (m["slug"], frozenset({m["slug"], *(m.get("acceptable") or [])}))
        for m in meta
    }


def metrics(meta: Sequence[Mapping[str, Any]], answers: Mapping[str, str]) -> dict[str, Any]:
    """«То же вино» (мягко) и строго: макро по винам (`source`), микро; срезы по букве кадра."""
    by_soft: dict[str, list[bool]] = defaultdict(list)
    by_strict: dict[str, list[bool]] = defaultdict(list)
    sl: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for m in meta:
        a = answers[m["query_id"]]
        soft = a == m["slug"] or a in set(m.get("acceptable") or [])
        strict = a == m["slug"]
        by_soft[m["source"]].append(soft)
        by_strict[m["source"]].append(strict)
        s = sl[m["query_id"][0]]
        s[0] += soft
        s[1] += strict
        s[2] += 1
    n = len(meta)
    return {
        "frames": n,
        "wines": len(by_soft),
        "soft_macro": 100 * float(np.mean([np.mean(v) for v in by_soft.values()])),
        "soft_micro": 100 * sum(map(sum, by_soft.values())) / n,
        "strict_macro": 100 * float(np.mean([np.mean(v) for v in by_strict.values()])),
        "strict_micro": 100 * sum(map(sum, by_strict.values())) / n,
        "slices": {k: {"soft": v[0], "strict": v[1], "n": v[2]} for k, v in sorted(sl.items())},
    }


def per_wine(meta: Sequence[Mapping[str, Any]], answers: Mapping[str, str]) -> dict[str, float]:
    acc: dict[str, list[bool]] = defaultdict(list)
    for m in meta:
        a = answers[m["query_id"]]
        acc[m["source"]].append(a == m["slug"] or a in set(m.get("acceptable") or []))
    return {w: float(np.mean(v)) for w, v in acc.items()}


def bootstrap(
    meta: Sequence[Mapping[str, Any]], base: Mapping[str, str], new: Mapping[str, str]
) -> dict[str, list[float]]:
    """Бутстреп по винам (2000, seed 0) для «то же вино» макро: база, вариант и их разность.

    Выборки — как в `eval_v2.py`: вина в порядке появления в мете, 2000 вызовов `integers`.
    Разность парная: база и вариант на одних и тех же выборках вин.
    """
    b, n = per_wine(meta, base), per_wine(meta, new)
    wines = list(b)
    vb, vn = np.array([b[w] for w in wines]), np.array([n[w] for w in wines])
    rng = np.random.default_rng(SEED)
    idx = np.stack([rng.integers(0, len(wines), len(wines)) for _ in range(BOOT)])
    out = {}
    for name, arr in (("base", vb), ("variant", vn), ("delta", vn - vb)):
        means = 100 * arr[idx].mean(axis=1)
        out[name] = [
            round(float(np.percentile(means, 2.5)), 2),
            round(float(np.percentile(means, 97.5)), 2),
        ]
    return out


def classify(
    qid: str, old: str, new: str, gt: Mapping[str, tuple[str, frozenset[str]]]
) -> tuple[str, str]:
    """Изменение ответа мягко («то же вино») и строго: fix / break / neutral."""
    slug, ok = gt[qid]
    soft = {(False, True): "fix", (True, False): "break"}.get((old in ok, new in ok), "neutral")
    strict = {(False, True): "fix", (True, False): "break"}.get(
        (old == slug, new == slug), "neutral"
    )
    return soft, strict


def decide(
    meta: Sequence[Mapping[str, Any]],
    base: Mapping[str, str],
    new: Mapping[str, str],
    gt: Mapping[str, tuple[str, frozenset[str]]],
) -> dict[str, Any]:
    """Критерии PREREG: починок ≥ 4 × поломок и ≥ 3; строгий микро не ниже базы; срезы ≥ 0."""
    changed = [q for q in new if new[q] != base[q]]
    kinds = [classify(q, base[q], new[q], gt) for q in changed]
    soft = Counter(k[0] for k in kinds)
    strict = Counter(k[1] for k in kinds)
    net = {s: 0 for s in SLICES}
    for q, (k, _) in zip(changed, kinds, strict=True):
        net[q[0]] += {"fix": 1, "break": -1}.get(k, 0)
    mb, mn = metrics(meta, base), metrics(meta, new)
    c1 = soft["fix"] >= 4 * soft["break"] and soft["fix"] >= 3
    c2 = round(mn["strict_micro"], 6) >= round(mb["strict_micro"], 6)
    c3 = all(v >= 0 for v in net.values())
    return {
        "changed": len(changed),
        "soft": {k: soft.get(k, 0) for k in ("fix", "break", "neutral")},
        "strict": {k: strict.get(k, 0) for k in ("fix", "break", "neutral")},
        "slice_net_soft": net,
        "metrics": mn,
        "criteria": {"fix>=4*break&fix>=3": c1, "strict_micro>=base": c2, "slices_net>=0": c3},
        "criterion1_on_strict": strict["fix"] >= 4 * strict["break"] and strict["fix"] >= 3,
        "accepted": c1 and c2 and c3,
    }


# ------------------------------------------------------------------ ворота
def gate_filter(frames: Sequence[Frame], svc: Source) -> dict[str, Any]:
    """A: `contradicts` на полях из дампа по записанному порядку слоя = `whatif["filter"]`."""
    first = full = model5 = service = 0
    bad: list[str] = []
    for f in frames:
        kept = [s for s in f.order if not contradicts(s, f.fields, svc.attrs)]
        ranked = kept or list(f.order)
        wi = f.whatif or {}
        first += bool(wi.get("filter")) and ranked[0] == wi["filter"][0]
        ok = ranked[:5] == list(wi.get("filter") or [])
        full += ok
        model5 += list(f.order[:5]) == list(wi.get("model") or [])
        service += f.slug == (wi.get("service") or [None])[0] == f.order[0]
        if not ok:
            bad.append(f.qid)
    return {
        "frames": len(frames),
        "filter_top1": first,
        "filter_top5": full,
        "order_top5_eq_whatif_model": model5,
        "slug_eq_order0": service,
        "bad": bad[:20],
    }


def gate_recompute(frames: Sequence[Frame], svc: Source, model: LogisticRanker) -> dict[str, Any]:
    """B: код ветки по записанным CV и чтению → тот же порядок слоя и тот же ответ H5."""
    reader = readers_of(model.feature_names)[0]
    order20 = h5_same = h5_top5 = p_close = 0
    bad: list[str] = []
    for f in frames:
        vis = VisualResult.model_validate(f.record["visual"])
        tr = f.record.get("text_read") or {}
        reads = {reader: TextRead(fields=f.fields, raw_text=tr.get("raw_text"))}
        ranking, features = rank_query_detailed(model, vis, reads, svc.attrs)
        same_order = tuple(ranking.slugs) == f.order
        order20 += same_order
        p_close += abs(ranking.p_top1 - f.p_top1) < 6e-4  # в дампе p_top1 округлён до 3 знаков
        ranked = list(ranking.slugs)
        if ranking.p_top1 < AMBIGUOUS_P_TOP1:
            ranked = list(rerank_ambiguous(model, features, f.fields, svc.attrs))
        h5 = ranked[0]
        h5_same += h5 == f.h5
        if f.p_top1 < AMBIGUOUS_P_TOP1 and f.whatif:
            h5_top5 += ranked[:5] == list(f.whatif.get("no_cluster+filter") or [])
        else:
            h5_top5 += 1
        if not (same_order and h5 == f.h5):
            bad.append(f.qid)
    return {
        "frames": len(frames),
        "order20": order20,
        "p_top1_close": p_close,
        "h5_answer": h5_same,
        "h5_top5_on_ambiguous": h5_top5,
        "bad": bad[:20],
    }


# ------------------------------------------------------------------ разбор изменений
def change_rows(
    frames: Mapping[str, Frame],
    base: Mapping[str, str],
    new: Mapping[str, str],
    gt: Mapping[str, tuple[str, frozenset[str]]],
    src: Source,
) -> list[dict[str, Any]]:
    rows = []
    for qid in sorted(q for q in new if new[q] != base[q]):
        f = frames[qid]
        read = read_of(f.fields, src.universe)
        soft, strict = classify(qid, base[qid], new[qid], gt)
        slug, ok = gt[qid]

        def card(s: str) -> dict[str, Any]:
            return {
                "slug": s,
                "sugar": src.sugar(s),
                "color": src.color(s),
                "grapes": sorted(src.grapes(s)),
            }

        rows.append(
            {
                "query_id": qid,
                "slice": qid[0],
                "p_top1": f.p_top1,
                "lines": f.lines,
                "read": {
                    "sugar": sorted(read.sugar),
                    "color": read.color,
                    "grapes": sorted(read.grapes),
                    "grapes_raw": [str(i.value) for i in (f.fields.grapes if f.fields else [])],
                },
                "gt": card(slug),
                "acceptable": sorted(ok - {slug}),
                "old": card(base[qid]),
                "new": card(new[qid]),
                "soft": soft,
                "strict": strict,
            }
        )
    return rows


def show_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    for r in rows:
        rd = r["read"]
        print(
            f"  {r['query_id']} [{r['slice']}] p={r['p_top1']:.3f} {r['soft']}/{r['strict']} | "
            f"прочитано: сахар {rd['sugar']} цвет {rd['color']} сорта {rd['grapes']} "
            f"(сырые {rd['grapes_raw']})"
        )
        print(f"     строки: {' | '.join(r['lines'])}")
        for key in ("gt", "old", "new"):
            c = r[key]
            print(f"     {key:3} {c['slug']} — {c['sugar']}, {c['color']}, {c['grapes']}")


# ------------------------------------------------------------------ прогоны
def answers_of(
    frames: Iterable[Frame], rule: Callable[[Frame, str], str] | None = None
) -> dict[str, str]:
    return {f.qid: rule(f, f.h5) if rule else f.h5 for f in frames}


def variants(svc: Source, dct: Source) -> dict[str, Callable[[Frame, str], str]]:
    return {
        "H6-svc": lambda f, a: h6(f, a, svc),
        "H6-dict": lambda f, a: h6(f, a, dct),
        "svc: сахар+цвет": lambda f, a: h6(f, a, svc, grapes=False),
        "svc: сорта": lambda f, a: h6(f, a, svc, sugar_color=False),
        "dict: сахар+цвет": lambda f, a: h6(f, a, dct, grapes=False),
        "dict: сорта": lambda f, a: h6(f, a, dct, sugar_color=False),
    }


def fmt(m: Mapping[str, Any]) -> str:
    sl = ", ".join(f"{k} {v['soft']}/{v['n']}" for k, v in m["slices"].items())
    return (
        f"мягко макро {m['soft_macro']:.2f}, микро {m['soft_micro']:.2f} | строго макро "
        f"{m['strict_macro']:.2f}, микро {m['strict_micro']:.2f} | срезы мягко: {sl}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "results.json")
    args = ap.parse_args()

    meta = jsonl(FD / "sets" / "catalog_v2" / "meta.jsonl")
    gt = ok_sets(meta)
    svc = service_source()
    dct, dict_diff = dict_source(svc)
    model = LogisticRanker.load(MODEL)
    print(
        f"атрибуты сервиса: {len(svc.attrs)} карточек, кодов сортов {len(svc.universe)}; "
        f"справочник поверх: расхождений {dict_diff}, кодов сортов {len(dct.universe)}"
    )

    # --- ворота на pfix_final
    frames, _ = load_run("pfix_final")
    assert {f.qid for f in frames} == set(gt), "кадры pfix_final ≠ catalog_v2"
    by_qid = {f.qid: f for f in frames}
    ga = gate_filter(frames, svc)
    gb = gate_recompute(frames, svc, model)
    base = answers_of(frames)
    mb = metrics(meta, base)
    gc = {
        "soft_macro": round(mb["soft_macro"], 1),
        "strict_micro": round(mb["strict_micro"], 1),
        "soft_micro": round(mb["soft_micro"], 1),
    }
    gate_ok = (
        ga["filter_top5"] == ga["frames"] == 353
        and gb["order20"] == gb["h5_answer"] == 353
        and gc["soft_macro"] == BASE_SOFT_MACRO
        and gc["strict_micro"] == BASE_STRICT_MICRO
    )
    print(f"\nворота A (contradicts по порядку слоя = whatif filter): {ga}")
    print(f"ворота B (пересчёт кодом ветки): {gb}")
    print(f"ворота C (база): {gc}; база: {fmt(mb)}")
    print(f"ворота пройдены: {gate_ok}")
    result: dict[str, Any] = {
        "prereg": "research/2026-09-24_h6/PREREG.md @ ba8ef42",
        "inputs": {
            "run": "pfix_final/iter20",
            "gt_tokens_sha1": sha1(DATA / "gt" / "gt_tokens.jsonl"),
            "wines_final_sha1": sha1(FD / "catalog" / "wines_final.jsonl"),
            "meta_sha1": sha1(FD / "sets" / "catalog_v2" / "meta.jsonl"),
            "model_sha1": sha1(MODEL),
        },
        "sources": {
            "svc_grape_codes": len(svc.universe),
            "dict_grape_codes": len(dct.universe),
            "dict_vs_svc_diff": dict_diff,
        },
        "gate": {"A_filter": ga, "B_recompute": gb, "C_base": gc, "passed": gate_ok},
        "base": mb,
    }
    if not gate_ok:
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        print("ворота не пройдены — замер не идёт")
        return 1

    # --- решение: H6-svc и H6-dict на pfix_final
    rules = variants(svc, dct)
    result["variants"] = {}
    for name, rule in rules.items():
        new = answers_of(frames, rule)
        d = decide(meta, base, new, gt)
        d["bootstrap_soft_macro"] = bootstrap(meta, base, new)
        src = dct if name.startswith(("H6-dict", "dict")) else svc
        d["changes"] = change_rows(by_qid, base, new, gt, src)
        result["variants"][name] = d
        print(
            f"\n== {name}: изменено {d['changed']}, мягко {d['soft']}, строго {d['strict']}, "
            f"срезы {d['slice_net_soft']}\n   {fmt(d['metrics'])}\n   критерии {d['criteria']} "
            f"→ {'принят' if d['accepted'] else 'не принят'}; бутстреп {d['bootstrap_soft_macro']}"
        )
        if name.startswith("H6-"):
            show_rows(d["changes"])
    passed = [n for n in ("H6-svc", "H6-dict") if result["variants"][n]["accepted"]]
    if passed:
        best = max(
            passed, key=lambda n: (result["variants"][n]["metrics"]["strict_micro"], n == "H6-svc")
        )
        decision = f"принят {best}"
    else:
        decision = "не принят ни один вариант: сервис не меняется"
    result["decision"] = decision
    print(f"\nРЕШЕНИЕ: {decision}")

    # «код сорта хотя бы у одной карточки каталога»: словарь кодов берётся у своего источника.
    # Для сведения — что будет, если взять словарь кодов другого источника.
    result["universe_sensitivity"] = {}
    for name, src, other in (("H6-svc", svc, dct), ("H6-dict", dct, svc)):
        main_ans = answers_of(frames, rules[name])
        swapped = replace(src, universe=other.universe)
        alt = answers_of(frames, lambda f, a, s=swapped: h6(f, a, s))
        d = decide(meta, base, alt, gt)
        result["universe_sensitivity"][name] = {
            "differs_from_main": sorted(q for q in alt if alt[q] != main_ans[q]),
            "soft": d["soft"],
            "strict": d["strict"],
            "slice_net_soft": d["slice_net_soft"],
            "accepted": d["accepted"],
        }
    print(f"словарь кодов сортов другого источника: {result['universe_sensitivity']}")

    # --- для сведения: прогон до правки эталонов
    v2, notes = load_run("v2prod", override="v2prod_l29")
    assert {f.qid for f in v2} == set(gt)
    gv = {
        "A_filter": gate_filter(v2, svc),
        "B_recompute": gate_recompute(v2, svc, model),
        "notes": notes,
    }
    base_v2 = answers_of(v2)
    info_v2 = {"gate": gv, "base": metrics(meta, base_v2), "variants": {}}
    print(
        f"\n== v2prod (+v2prod_l29): {notes}\n   ворота A {gv['A_filter']}\n   ворота B {gv['B_recompute']}"
        f"\n   база: {fmt(info_v2['base'])}"
    )
    for name, rule in rules.items():
        new = answers_of(v2, rule)
        d = decide(meta, base_v2, new, gt)
        d["bootstrap_soft_macro"] = bootstrap(meta, base_v2, new)
        d["changed_qids"] = sorted(q for q in new if new[q] != base_v2[q])
        info_v2["variants"][name] = d
        print(
            f"   {name}: изменено {d['changed']}, мягко {d['soft']}, строго {d['strict']}, срезы "
            f"{d['slice_net_soft']}; {fmt(d['metrics'])}; прошёл бы: {d['accepted']}"
        )
    result["info_v2prod"] = info_v2

    # --- для сведения: 409 кадров вне каталога
    ooc, _ = load_run("ooc_v2prod", extra=[("predictions_vlmfail.jsonl", "whatif_vlmfail.json")])
    ooc_meta = {m["query_id"] for m in jsonl(FD / "sets" / "ooc_v2" / "meta.jsonl")}
    assert {f.qid for f in ooc} == ooc_meta, "кадры ooc_v2prod ≠ ooc_v2"
    go = {"A_filter": gate_filter(ooc, svc), "B_recompute": gate_recompute(ooc, svc, model)}
    base_ooc = answers_of(ooc)
    info_ooc: dict[str, Any] = {"frames": len(ooc), "gate": go, "variants": {}}
    print(
        f"\n== ooc_v2prod ({len(ooc)}): ворота A {go['A_filter']}\n   ворота B {go['B_recompute']}"
    )
    for name, rule in rules.items():
        new = answers_of(ooc, rule)
        changed = sorted(q for q in new if new[q] != base_ooc[q])
        amb = sum(next(f for f in ooc if f.qid == q).p_top1 < AMBIGUOUS_P_TOP1 for q in changed)
        info_ooc["variants"][name] = {
            "changed": len(changed),
            "changed_at_p_lt_0_5": amb,
            "qids": changed,
        }
        print(f"   {name}: ответ меняется у {len(changed)} (из них при p_top1 < 0,5 — {amb})")
    result["info_ooc_v2prod"] = info_ooc

    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nзаписано {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
