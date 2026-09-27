"""Э8 по `PREREG.md`: отбор 8 карточек, подготовка фото krasnostop, CPU-векторы, кандидаты индекса.

Общий `svoe-vino-scanner/data` только читается; всё пишется в `--out` (по умолчанию
`svs-logs/somm-2409/acc_plan/e8/`), запись внутрь `data` — отказ.

Файлы в `--out`:
- `packshots/<карточка>.png` — фото krasnostop после подготовки замера 2 (`rsk_nowm`);
- `visual-s2so400m.e8.npz` — кандидат CSV-комплекта: строки 8 карточек удалены, 24 новые в конце
  (механика `apply_packshot_fix.py`), прочие строки — байт в байт из базы;
- `visual-s2so400m-live71.e8.npz` — живой комплект поверх кандидата (как `build_live_set.py`);
- `visual-s2so400m.reorder.npz` — ворота М5: та же механика, но в конец перенесены прежние строки;
- `build_info.json` — отбор, ворота М1–М3, sha1 всех входов и выходов, ревизия весов.

    PYTHONPATH=. python research/2026-09-26_e8/e8_build.py            # CPU, ~1–2 мин
    PYTHONPATH=. python research/2026-09-26_e8/e8_build.py --select-only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = Path(r"<корень>")
FD = ROOT / "field_dataset"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(FD / "tools"))

from app.features.index import IndexMeta, VisualIndex  # noqa: E402
from build_index import PACKSHOT_VIEWS, iter_entries  # noqa: E402
from pseudo import white_to_alpha  # noqa: E402

DATA = Path(os.environ.get("SVS_DATA_DIR") or ROOT / "svoe-vino-scanner" / "data")
BASE_INDEX = DATA / "index" / "visual-s2so400m.npz"
LIVE_INDEX = DATA / "index" / "visual-s2so400m-live71.npz"
BASE_SHA1 = "ccd3a01f16379a3f0a8a1e7e8dc0623c7622bda5"
LIVE_SHA1 = "d0a53644"
PHOTO_MAP = DATA / "catalog" / "slug_photo_map.csv"
REPL = REPO / "research" / "2026-09-23_packshot-fix" / "replacements.tsv"
KR_META = FD / "sets" / "krasnostop_v1" / "meta.jsonl"
GROUPS = FD / "catalog" / "wine_groups_final.json"
MATCHES = ROOT / "krastostop" / "match" / "matches.jsonl"
M2 = ROOT / "krastostop" / "measure" / "candidate"
OUT = ROOT / "svs-logs" / "somm-2409" / "acc_plan" / "e8"
#: зона водяного знака «КРАСНОСТОП» — как `build_candidate_index.py` замера 2
WM_X0, WM_Y0 = 0.5, 0.85
#: список PREREG §2 — отбор обязан дать ровно его
EXPECTED = {
    "abrau-dyurso-abrau-kupazh-tyomnyy-suhoe-kaberne-sovinon-krasnoe-13": "K0007-orig-0",
    "derbent-vino-endemy-saperavi-krasnoe-suhoe-13": "K0570-orig-0",
    "derbent-vino-endemy-shardone-beloe-suhoe-13": "K0592-orig-0",
    "legato-legato-sovinon-blan-beloe-suhoe-125": "K0342-orig-0",
    "skalistyy-bereg-shyopot-tsvetov-risling-beloe-suhoe-109": "K0693-orig-0",
    "sober-bash-krasnostop-krasnostop-zolotovskiy-krasnoe-suhoe-14": "K0320-orig-0",
    "vibes-vermentino-viognier-barrel-fermented-2022": "K0780-orig-0",
    "vinodelnya-vedernikov-gubernatorskoe-golubok-krasnoe-suhoe-11": "K0853-orig-0",
}


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


def vol_key(r: dict[str, Any]) -> tuple:
    """Как `assemble.vol_key` сборки набора: 0,75 л, затем проверенные глазами, затем kr_id."""
    v = r.get("volume_l") or 0
    return (abs(v - 0.75) > 1e-6, r.get("match_method") != "verified", r["kr_id"])


# ------------------------------------------------------------------ отбор (PREREG §2)
def select() -> tuple[list[tuple[str, dict[str, Any]]], dict[str, Any]]:
    repl = list(csv.DictReader(open(REPL, encoding="utf-8"), delimiter="\t"))
    wrong = {r["slug"] for r in repl if r["issue"] == "wrong_photo"}
    none = {r["slug"] for r in repl if r["ext_source"] == "none"}
    with open(PHOTO_MAP, encoding="utf-8-sig", newline="") as fh:
        fixed = {r["slug"] for r in csv.DictReader(fh) if r.get("method") == "packshot_fix"}
    pool = (wrong | none) - fixed
    groups = json.loads(GROUPS.read_text(encoding="utf-8"))
    wine = {s: w for w, g in groups.items() for s in g.get("members") or []}
    meta = jsonl(KR_META)
    matches = {r["kr_id"]: r for r in jsonl(MATCHES)}
    chosen, why = [], Counter()
    for card in sorted(pool):
        w = wine.get(card, card)
        same_wine = [r for r in meta if wine.get(r["slug"], r["slug"]) == w]
        ok = [
            r
            for r in same_wine
            if not r.get("same_packshot")
            and not r.get("packaging")
            and (matches.get(r["kr_id"]) or {}).get("confidence") == "high"
        ]
        if not same_wine:
            why["нет кадров krasnostop"] += 1
        elif not ok:
            why["только same_packshot / packaging / не high"] += 1
        else:
            chosen.append((card, sorted(ok, key=vol_key)[0]))
    stats = {
        "wrong_photo": len(wrong),
        "ext_source_none": len(none),
        "union": len(wrong | none),
        "already_packshot_fix": sorted((wrong | none) & fixed),
        "pool": len(pool),
        "chosen": len(chosen),
        "not_chosen": dict(why),
    }
    got = {c: r["query_id"] for c, r in chosen}
    if got != EXPECTED:
        raise SystemExit(f"отбор разошёлся со списком PREREG: {got}")
    return chosen, stats


# ------------------------------------------------------------------ подготовка (как замер 2)
def alpha_box(alpha: np.ndarray, amin: int = 8) -> tuple[int, int, int, int]:
    m = alpha >= amin
    ys, xs = np.flatnonzero(m.any(axis=1)), np.flatnonzero(m.any(axis=0))
    if not len(xs):
        return 0, 0, alpha.shape[1], alpha.shape[0]
    return int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1


def drop_watermark(rgba: np.ndarray) -> tuple[np.ndarray, int]:
    """Копия `build_candidate_index.drop_watermark`: мелкие компоненты альфы в зоне знака."""
    import cv2

    alpha = rgba[..., 3]
    h, w = alpha.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats((alpha >= 8).astype(np.uint8), 8)
    if n <= 2:
        return rgba, 0
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    drop = [
        c
        for c in range(1, n)
        if c != big
        and st[c, cv2.CC_STAT_LEFT] >= WM_X0 * w
        and st[c, cv2.CC_STAT_TOP] >= WM_Y0 * h
    ]
    if not drop:
        return rgba, 0
    out = rgba.copy()
    out[..., 3][np.isin(lab, drop)] = 0
    return out, len(drop)


def prepare(src: Path, dst: Path) -> dict[str, Any]:
    rgba = white_to_alpha(np.array(Image.open(src).convert("RGB")))
    box0 = alpha_box(rgba[..., 3])
    rgba, dropped = drop_watermark(rgba)
    box1 = alpha_box(rgba[..., 3])
    Image.fromarray(rgba, "RGBA").save(dst)
    return {"box_rsk": list(box0), "box": list(box1), "wm_components_dropped": dropped}


# ------------------------------------------------------------------ индекс: сырые массивы npz
def load_raw(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def save_raw(path: Path, slugs: list[str], views: list[str], vectors: np.ndarray, meta: str,
             base_rows: np.ndarray | None = None) -> None:
    arrays = {
        "slugs": np.array(slugs, dtype=np.str_),
        "views": np.array(views, dtype=np.str_),
        "vectors": vectors.astype(np.float16),
        "meta": np.array(meta, dtype=np.str_),
    }
    if base_rows is not None:
        arrays["base_rows"] = base_rows
    with path.open("wb") as fh:
        np.savez(fh, **arrays)


def swap(base: dict[str, Any], cards: set[str], new_slugs: list[str], new_views: list[str],
         new_vec: np.ndarray) -> tuple[list[str], list[str], np.ndarray, str]:
    """Механика `apply_packshot_fix.py`: строки `cards` вон, новые — в конец; прочие как есть."""
    slugs = [str(s) for s in base["slugs"].tolist()]
    views = [str(v) for v in base["views"].tolist()]
    keep = [i for i, s in enumerate(slugs) if s not in cards]
    out_slugs = [slugs[i] for i in keep] + list(new_slugs)
    out_views = [views[i] for i in keep] + list(new_views)
    vec = np.vstack([base["vectors"][keep], new_vec.astype(np.float16)])
    meta = IndexMeta.model_validate_json(str(base["meta"].item()))
    meta = meta.model_copy(update={"n_slugs": len(set(out_slugs)), "n_vectors": len(out_slugs)})
    return out_slugs, out_views, vec, meta.model_dump_json()


def hf_revision(model: str) -> dict[str, Any]:
    base = Path.home() / ".cache" / "huggingface" / "hub" / ("models--" + model.replace("/", "--"))
    ref = base / "refs" / "main"
    snaps = sorted(p.name for p in (base / "snapshots").glob("*")) if (base / "snapshots").exists() else []
    return {"refs_main": ref.read_text().strip() if ref.exists() else None, "snapshots": snaps}


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def ref_check(base_idx: VisualIndex, embedder: Any, k: int = 8) -> dict[str, Any]:
    """Как `ref_check` замера 2: 8 эталонов каталога тем же кодом = их векторы в базе."""
    with open(PHOTO_MAP, encoding="utf-8-sig", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("method") != "packshot_fix"]
    counts = Counter(base_idx.slugs)
    pick = [
        (r["slug"], Path(r["path"]))
        for r in sorted(rows, key=lambda r: r["slug"])
        if counts[r["slug"]] == 3 and Path(r["path"]).exists()
    ][:: max(1, len(rows) // k)][:k]
    st: dict[str, Any] = {"skipped": [], "read_s": 0.0, "read": 0}
    fresh = VisualIndex.build(iter_entries(pick, PACKSHOT_VIEWS, st), embedder)
    pos = {(s, v): i for i, (s, v) in enumerate(zip(base_idx.slugs, base_idx.views, strict=True))}
    cs = [
        cos(x, base_idx.vectors[pos[(s, v)]])
        for s, v, x in zip(fresh.slugs, fresh.views, fresh.vectors, strict=True)
    ]
    return {"slugs": [s for s, _ in pick], "n": len(cs), "min_cos": min(cs), "mean_cos": float(np.mean(cs)),
            "passed": len(cs) == 3 * k and min(cs) >= 0.999}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--select-only", action="store_true")
    args = ap.parse_args()
    out = args.out.resolve()
    if DATA.resolve() in (out, *out.parents):
        print(f"отказ: {out} внутри общих данных {DATA}")
        return 2
    t0 = time.perf_counter()
    for path, want in ((BASE_INDEX, BASE_SHA1), (LIVE_INDEX, LIVE_SHA1)):
        if not sha1(path).startswith(want):
            print(f"{path.name}: sha1 {sha1(path)} ≠ {want} — общие данные сменились, стоп")
            return 2
    chosen, sel = select()
    print(f"отбор: {json.dumps(sel, ensure_ascii=False)}")
    for card, r in chosen:
        print(f"  {card} ← {r['query_id']} {r['kr_id']} ({r['volume_l']} л, {r['match_method']})")
    if args.select_only:
        return 0
    out.mkdir(parents=True, exist_ok=True)
    (out / "packshots").mkdir(exist_ok=True)
    info: dict[str, Any] = {"prereg": "research/2026-09-26_e8/PREREG.md", "selection": sel, "cards": []}

    # --- М1: подготовка = PNG замера 2
    m2_png = {r["query_id"]: Path(r["png"]) for r in csv.DictReader(open(M2 / "added.tsv", encoding="utf-8"), delimiter="\t")}
    photos, m1 = [], {}
    for card, r in chosen:
        dst = out / "packshots" / f"{card}.png"
        prep = prepare(Path(r["image"]), dst)
        ref = m2_png[r["query_id"]]
        same = np.array_equal(np.array(Image.open(dst)), np.array(Image.open(ref)))
        m1[card] = same
        photos.append((card, dst))
        info["cards"].append({"card": card, "query_id": r["query_id"], "kr_id": r["kr_id"], "frame_slug": r["slug"],
                              "volume_l": r.get("volume_l"), "match_method": r.get("match_method"),
                              "image": r["image"], "image_sha1": sha1(Path(r["image"])), "png": str(dst),
                              "png_sha1": sha1(dst), "m2_png": str(ref), "m2_pixels_equal": same, **prep})
    info["M1_prep_equals_m2"] = all(m1.values())
    print(f"М1 подготовка = замер 2: {sum(m1.values())}/{len(m1)}")
    if not info["M1_prep_equals_m2"]:
        (out / "build_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
        return 1

    # --- векторы на CPU
    from app.features.embedder import SiglipEmbedder

    base_idx = VisualIndex.load(BASE_INDEX)
    model = base_idx.meta.model
    embedder = SiglipEmbedder(model, "cpu", "float32")
    info["model"], info["hf_revision"], info["device"] = model, hf_revision(model), "cpu"
    st: dict[str, Any] = {"read": 0, "read_s": 0.0, "skipped": []}
    fresh = VisualIndex.build(iter_entries(photos, PACKSHOT_VIEWS, st), embedder)
    if st["skipped"] or len(fresh) != 3 * len(photos):
        print(f"виды не посчитались: {st['skipped']}, векторов {len(fresh)}")
        return 1
    assert list(fresh.views) == list(PACKSHOT_VIEWS) * len(photos)

    # --- М2: CPU против GPU-векторов замера 2 и пересчёт эталонов каталога
    m2 = load_raw(M2 / "visual-s2so400m-kr.npz")
    m2_slugs = [str(s) for s in m2["slugs"].tolist()]
    m2_views = [str(v) for v in m2["views"].tolist()]
    n_base = len(base_idx)
    frame_of = {c: r["slug"] for c, r in chosen}
    m2_cos = {}
    for s, v, x in zip(fresh.slugs, fresh.views, fresh.vectors, strict=True):
        rows = [i for i in range(n_base, len(m2_slugs)) if m2_slugs[i] == frame_of[s] and m2_views[i] == v]
        assert len(rows) == 1, (s, v, rows)
        m2_cos[f"{s}|{v}"] = cos(x, m2["vectors"][rows[0]])
    rc = ref_check(base_idx, embedder)
    info["M2_cpu_vs_m2_gpu"] = {"min_cos": min(m2_cos.values()), "per_view": m2_cos,
                                "passed": min(m2_cos.values()) >= 0.999}
    info["M2_ref_check"] = rc
    print(f"М2 CPU против GPU замера 2: min cos {min(m2_cos.values()):.5f}; эталоны каталога: min cos "
          f"{rc['min_cos']:.5f} ({rc['n']} векторов) → {info['M2_cpu_vs_m2_gpu']['passed'] and rc['passed']}")

    # --- кандидаты
    base = load_raw(BASE_INDEX)
    live = load_raw(LIVE_INDEX)
    cards = {c for c, _ in chosen}
    c_slugs, c_views, c_vec, c_meta = swap(base, cards, list(fresh.slugs), list(fresh.views),
                                           np.asarray(fresh.vectors, np.float32))
    cand = out / "visual-s2so400m.e8.npz"
    save_raw(cand, c_slugs, c_views, c_vec, c_meta)
    # М5: прежние строки 8 карточек — в конец, в том же порядке, что в базе
    b_slugs = [str(s) for s in base["slugs"].tolist()]
    old = [i for i, s in enumerate(b_slugs) if s in cards]
    r_slugs, r_views, r_vec, r_meta = swap(base, cards, [b_slugs[i] for i in old],
                                           [str(base["views"][i]) for i in old], base["vectors"][old])
    reorder = out / "visual-s2so400m.reorder.npz"
    save_raw(reorder, r_slugs, r_views, r_vec, r_meta)
    # живой комплект: кандидат CSV + строки живых карточек, маска — первые 6 312
    n_csv = len(b_slugs)
    assert live["base_rows"][:n_csv].all() and not live["base_rows"][n_csv:].any()
    l_slugs = c_slugs + [str(s) for s in live["slugs"][n_csv:].tolist()]
    l_views = c_views + [str(v) for v in live["views"][n_csv:].tolist()]
    l_vec = np.vstack([c_vec, live["vectors"][n_csv:]])
    l_meta = IndexMeta.model_validate_json(str(live["meta"].item()))
    l_meta_s = l_meta.model_copy(update={"n_slugs": len(set(l_slugs)), "n_vectors": len(l_slugs)}).model_dump_json()
    live_cand = out / "visual-s2so400m-live71.e8.npz"
    save_raw(live_cand, l_slugs, l_views, l_vec, l_meta_s, base_rows=live["base_rows"].copy())

    # --- М3: сборка
    c = load_raw(cand)
    lc = load_raw(live_cand)
    keep = [i for i, s in enumerate(b_slugs) if s not in cards]
    n_keep = len(keep)
    cs = [str(s) for s in c["slugs"].tolist()]
    cv = [str(v) for v in c["views"].tolist()]
    m3 = {
        "rows": len(cs) == n_csv,
        "other_rows_bytes_equal": bool(np.array_equal(c["vectors"][:n_keep].view(np.uint16),
                                                      base["vectors"][keep].view(np.uint16))),
        "other_slugs_views_order": cs[:n_keep] == [b_slugs[i] for i in keep]
        and cv[:n_keep] == [str(base["views"][i]) for i in keep],
        "cards_3_views": all(Counter(cs[n_keep:])[k] == 3 for k in cards) and len(cs[n_keep:]) == 24
        and cv[n_keep:] == list(PACKSHOT_VIEWS) * len(cards),
        "same_meta_model": IndexMeta.model_validate_json(str(c["meta"].item())).model_dump(exclude={"built_at"})
        == IndexMeta.model_validate_json(str(base["meta"].item())).model_dump(exclude={"built_at"}),
        "n_slugs": len(set(cs)) == len(set(b_slugs)),
        "live_head_equals_candidate": bool(np.array_equal(lc["vectors"][:n_csv].view(np.uint16), c["vectors"].view(np.uint16)))
        and [str(s) for s in lc["slugs"][:n_csv].tolist()] == cs,
        "live_tail_equals_live71": bool(np.array_equal(lc["vectors"][n_csv:].view(np.uint16), live["vectors"][n_csv:].view(np.uint16)))
        and lc["slugs"][n_csv:].tolist() == live["slugs"][n_csv:].tolist(),
        "live_mask_same": bool(np.array_equal(lc["base_rows"], live["base_rows"])),
        "reorder_is_permutation": sorted(zip(r_slugs, r_views, map(bytes, r_vec.view(np.uint16)), strict=True))
        == sorted(zip(b_slugs, [str(v) for v in base["views"].tolist()], map(bytes, base["vectors"].view(np.uint16)), strict=True)),
    }
    for p in (cand, reorder, live_cand):
        VisualIndex.load(p, model=model)  # паспорт и маска читаются сервисом
    info["M3_build"] = m3
    info["M3_passed"] = all(m3.values())
    print(f"М3 сборка: {m3}")
    info["files"] = {
        "base_index": {"path": str(BASE_INDEX), "sha1": sha1(BASE_INDEX)},
        "live_index": {"path": str(LIVE_INDEX), "sha1": sha1(LIVE_INDEX)},
        "candidate": {"path": str(cand), "sha1": sha1(cand)},
        "candidate_live71": {"path": str(live_cand), "sha1": sha1(live_cand)},
        "reorder": {"path": str(reorder), "sha1": sha1(reorder)},
        "m2_candidate": {"path": str(M2 / "visual-s2so400m-kr.npz"), "sha1": sha1(M2 / "visual-s2so400m-kr.npz")},
    }
    info["wall_s"] = round(time.perf_counter() - t0, 1)
    (out / "build_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    ok = info["M1_prep_equals_m2"] and info["M2_cpu_vs_m2_gpu"]["passed"] and rc["passed"] and info["M3_passed"]
    print(f"кандидат {cand.name} sha1 {info['files']['candidate']['sha1'][:10]}, живой {info['files']['candidate_live71']['sha1'][:10]}, "
          f"перестановка {info['files']['reorder']['sha1'][:10]}; ворота М1–М3: {ok}; {info['wall_s']} с")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
