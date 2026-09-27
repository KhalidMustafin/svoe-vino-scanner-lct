"""Задача B, п.3: обманывает ли прогрев кэш промптов Ollama. Код сервиса, без самого сервиса.

python cache_probe.py 4b|8b

Повторяет путь прогрева `ScanService.warm()`: `reader.warm(synthetic_label(WARM_FRAME_PX))`, затем
прогревочный «скан» — `_jpeg(frame)` -> `decode_on_backgrounds` -> `read_label(... crop=VLM_CROP)`
(как `_scan` -> `_read`), и повтор того же скана (WARM_SCAN_ATTEMPTS=2). Потом два реальных кадра:
новый и сразу повтор. Транспорт обёрнут: из ответа Ollama пишутся prompt_eval_count/duration,
eval_count/duration, load_duration и sha1 картинки в запросе. Модель выгружается до и после.
Таймаут чтения 60 с — меряем, а не обрываем.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(r"<корень>\svoe-vino-scanner")
sys.path.insert(0, str(REPO))
from app.api.service import (  # noqa: E402
    BACKGROUND, MAX_PIXELS, VLM_CROP, VLM_CROP_PX, WARM_FRAME_PX, WHITE, _jpeg, decode_on_backgrounds, read_label,
)
from app.reading.readers.ollama_vlm import OllamaVlmReader, UrllibTransport  # noqa: E402
from app.reading.warmup import DEFAULT_WARMUP_BUDGET_MS, synthetic_label  # noqa: E402

CL = Path(__file__).resolve().parent
MODELS = {"4b": "qwen3.5:4b", "8b": "qwen3-vl:8b-instruct"}
cfg = sys.argv[1]
model = MODELS[cfg]
O = "http://127.0.0.1:11434"


def post(path: str, data: dict) -> dict:
    req = urllib.request.Request(O + path, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def unload() -> list:
    for m in MODELS.values():
        post("/api/generate", {"model": m, "keep_alive": 0})
    for _ in range(40):
        ps = json.loads(urllib.request.urlopen(O + "/api/ps").read().decode())["models"]
        if not ps:
            break
        time.sleep(0.5)
    return ps


rows: list[dict] = []
tag = {"v": ""}


class Rec:
    def __init__(self) -> None:
        self.inner = UrllibTransport(O + "/api/chat")

    def __call__(self, payload: dict, *, timeout_s: float) -> dict:
        img = payload["messages"][0]["images"][0]
        t = time.perf_counter()
        d = self.inner(payload, timeout_s=timeout_s)
        ms = lambda k: round((d.get(k) or 0) / 1e6)  # noqa: E731
        row = dict(step=tag["v"], img_sha1=hashlib.sha1(img.encode()).hexdigest()[:12], img_b64_len=len(img),
                   wall_ms=round((time.perf_counter() - t) * 1000), load_ms=ms("load_duration"),
                   prompt_eval_count=d.get("prompt_eval_count"), prompt_eval_ms=ms("prompt_eval_duration"),
                   eval_count=d.get("eval_count"), eval_ms=ms("eval_duration"), total_ms=ms("total_duration"))
        rows.append(row)
        print(cfg, json.dumps(row, ensure_ascii=False), flush=True)
        return d


print("выгрузка до:", unload())
reader = OllamaVlmReader(model, timeout_s=60, keep_alive="5m", transport=Rec())
frame = synthetic_label(WARM_FRAME_PX)
tag["v"] = "warm_readers (reader.warm, синтетика)"
reader.warm(frame, budget_ms=DEFAULT_WARMUP_BUDGET_MS)
_, ocr = decode_on_backgrounds(_jpeg(frame), (BACKGROUND, WHITE), max_pixels=MAX_PIXELS)
for i in (1, 2):
    tag["v"] = f"warm_scan попытка {i} (синтетика через _jpeg/decode/crop)"
    read_label(ocr, readers=[reader], lexicon=None, target=None, crop=VLM_CROP, crop_px=VLM_CROP_PX, budget_ms=60000)
sel = json.load(open(CL / "selection.json", encoding="utf-8"))
for s in (sel[1], sel[12]):
    data = (CL / "imgs" / s["file"]).read_bytes()
    _, ocr = decode_on_backgrounds(data, (BACKGROUND, WHITE), max_pixels=MAX_PIXELS)
    for k in ("новый кадр", "тот же кадр сразу"):
        tag["v"] = f"{s['q']} {k}"
        read_label(ocr, readers=[reader], lexicon=None, target=None, crop=VLM_CROP, crop_px=VLM_CROP_PX, budget_ms=60000)
print("выгрузка после:", unload())
json.dump(rows, open(CL / f"cache_probe_{cfg}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
