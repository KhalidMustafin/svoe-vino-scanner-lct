"""Выборка 20 кадров для чистого замера 4b против 8b (задача B).

5 portal_real (оригиналы webp организатора), 3 оригинала HEIC из my_dataset (по photos.jsonl),
12 — photos/*.jpg (M и L). Все вина разные (gt_slug ∪ acceptable не пересекаются).
Кадры прошлых GPU-прогонов (selection.json, mini, vlm_direct) исключены.
Пишет imgs/, queries.tsv, selection.json.
"""
from __future__ import annotations

import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"<корень>")
FD = ROOT / "field_dataset"
SP = Path(r"<временная папка>")
OUT = SP / "clean_latency"
IMGS = OUT / "imgs"
SEED = 20260923_2  # задача B

meta = [json.loads(x) for x in open(FD / "sets/catalog_orig/meta.jsonl", encoding="utf-8")]
labels = {d["id"]: d for d in (json.loads(x) for x in open(FD / "labels.jsonl", encoding="utf-8"))}
photos = {d["id"]: d for d in (json.loads(x) for x in open(FD / "photos.jsonl", encoding="utf-8"))}
frames8 = {f["qid"]: f for f in json.load(open(SP / "harness/frames.json", encoding="utf-8"))}

# кадры, уже прогнанные в прошлых GPU-замерах
used: set[str] = {s["q"].split("-")[0] for s in json.load(open(SP / "gpu_service/selection.json", encoding="utf-8"))}
used |= {"M159", "R060", "L04"}
for fn in ("vlm_direct_svc_up.json", "vlm_direct_ollama_alone.json"):
    used |= {r["q"].split("-")[0] for r in json.load(open(SP / "gpu_service" / fn, encoding="utf-8"))}
print("исключено кадров прошлых GPU-прогонов:", len(used))

def sha1(p: Path) -> str:
    return hashlib.sha1(p.read_bytes()).hexdigest()

rows = []
for m in meta:
    pid = m["photo"]
    lab = labels[pid]
    assert lab["gt_slug"] == m["slug"], (pid, lab["gt_slug"], m["slug"])
    assert sorted(lab["acceptable"]) == sorted(m["acceptable"]), pid
    if pid in used:
        continue
    rows.append(m)
print("кандидатов:", len(rows))

rng = random.Random(SEED)
rng.shuffle(rows)
taken: set[str] = set()
sel: list[dict] = []

def free(m: dict) -> bool:
    wines = {m["slug"], *m["acceptable"]}
    return not (wines & taken)

def take(m: dict, kind: str, src: Path, ext: str) -> None:
    taken.update({m["slug"], *m["acceptable"]})
    sel.append(dict(q=m["query_id"], photo=m["photo"], kind=kind, src=str(src), ext=ext,
                    gt=m["slug"], acc=m["acceptable"], dataset_source=m["dataset_source"]))

need = {"portal_webp": 5, "heic": 3, "jpg": 12}
for m in rows:
    if not free(m):
        continue
    ph = photos[m["photo"]]
    srcf = Path(ph["src_file"])
    if m["dataset_source"] == "portal_real" and need["portal_webp"] > 0 and ph["src_format"] == "webp" and srcf.exists():
        take(m, "portal_webp", srcf, ".webp"); need["portal_webp"] -= 1
    elif m["photo"].startswith("M") and ph["src_format"] == "heic" and need["heic"] > 0 and srcf.exists():
        take(m, "heic", srcf, ".heic"); need["heic"] -= 1
    elif m["dataset_source"] != "portal_real" and need["jpg"] > 0 and (FD / m["image"]).exists():
        take(m, "jpg", FD / m["image"], ".jpg"); need["jpg"] -= 1
    if not any(need.values()):
        break
assert not any(need.values()), need

IMGS.mkdir(parents=True, exist_ok=True)
lines = ["query_id\timage_path"]
for s in sel:
    dst = IMGS / f"{s['photo']}{s['ext']}"
    shutil.copy2(s["src"], dst)
    s["file"] = dst.name
    s["bytes"] = dst.stat().st_size
    s["sha1"] = sha1(dst)
    ph = photos[s["photo"]]
    s["sha1_photos_jsonl"] = ph["sha1"]
    s["orig_sha1_ok"] = (s["sha1"] == ph["sha1"]) if s["kind"] != "jpg" else None
    s["src_format"] = ph["src_format"]
    s["wh"] = [ph["width"], ph["height"]]
    f8 = frames8.get(s["q"], {})
    s["dump8_h5_soft"] = f8.get("h5_soft")
    s["dump8_vlm_ms"] = f8.get("vlm_elapsed")
    lines.append(f"{s['q']}\t{s['file']}")
(OUT / "queries.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
json.dump(sel, open(OUT / "selection.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for s in sel:
    print(s["q"], s["kind"], s["src_format"], s["bytes"], s["orig_sha1_ok"], s["dump8_h5_soft"], s["dump8_vlm_ms"], s["gt"][:60])
print("вин разных:", len({s["gt"] for s in sel}), "кадров:", len(sel))
print("по видам:", {k: sum(s["kind"] == k for s in sel) for k in ("portal_webp", "heic", "jpg")})
print("источники:", {k: sum(s["dataset_source"] == k for s in sel) for k in ("set_l", "phone_m", "portal_real")})
