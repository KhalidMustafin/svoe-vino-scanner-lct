"""Задача B: откуда разброс скорости VLM ×1,7–1,9 между одинаковыми вызовами (prefill ~330 или ~600 мс у 4b).

python clocks_probe.py 4b|8b

Модель выгружается, грузится на синтетике, затем 20 кадров выборки читаются тем же путём, что в сервисе
(без сервиса). Параллельно nvidia-smi пишет частоты SM/памяти, P-state и мощность каждые 100 мс.
На каждый вызов — prefill/decode из ответа Ollama и медианные частоты в окне вызова.
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(r"<корень>\svoe-vino-scanner")
sys.path.insert(0, str(REPO))
from app.api.service import BACKGROUND, MAX_PIXELS, VLM_CROP, VLM_CROP_PX, WARM_FRAME_PX, WHITE, decode_on_backgrounds, read_label  # noqa: E402
from app.reading.readers.ollama_vlm import OllamaVlmReader, UrllibTransport  # noqa: E402
from app.reading.warmup import synthetic_label  # noqa: E402

CL = Path(__file__).resolve().parent
MODELS = {"4b": "qwen3.5:4b", "8b": "qwen3-vl:8b-instruct"}
cfg = sys.argv[1]
O = "http://127.0.0.1:11434"


def post(path, data):
    req = urllib.request.Request(O + path, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read().decode())


def unload():
    for m in MODELS.values():
        post("/api/generate", {"model": m, "keep_alive": 0})
    for _ in range(40):
        if not json.loads(urllib.request.urlopen(O + "/api/ps").read().decode())["models"]:
            return []
        time.sleep(0.5)
    return "не выгрузилась"


rows, cur = [], {"q": ""}


class Rec:
    def __init__(self):
        self.inner = UrllibTransport(O + "/api/chat")

    def __call__(self, payload, *, timeout_s):
        t0 = datetime.now()
        d = self.inner(payload, timeout_s=timeout_s)
        ms = lambda k: round((d.get(k) or 0) / 1e6)  # noqa: E731
        rows.append(dict(q=cur["q"], t0=t0, t1=datetime.now(), prompt_eval_count=d.get("prompt_eval_count"),
                         prompt_eval_ms=ms("prompt_eval_duration"), eval_count=d.get("eval_count"), eval_ms=ms("eval_duration"),
                         total_ms=ms("total_duration")))
        return d


print("выгрузка:", unload())
smi_f = CL / f"clocks_{cfg}.csv"
smi = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,clocks.sm,clocks.mem,pstate,power.draw,utilization.gpu",
                        "--format=csv,noheader,nounits", "-lms", "100", "-f", str(smi_f)])
try:
    reader = OllamaVlmReader(MODELS[cfg], timeout_s=60, keep_alive="5m", transport=Rec())
    cur["q"] = "warm"
    reader.warm(synthetic_label(WARM_FRAME_PX), budget_ms=60000)
    for s in json.load(open(CL / "selection.json", encoding="utf-8")):
        _, ocr = decode_on_backgrounds((CL / "imgs" / s["file"]).read_bytes(), (BACKGROUND, WHITE), max_pixels=MAX_PIXELS)
        cur["q"] = s["q"]
        read_label(ocr, readers=[reader], lexicon=None, target=None, crop=VLM_CROP, crop_px=VLM_CROP_PX, budget_ms=60000)
finally:
    time.sleep(0.5)
    subprocess.run(["taskkill", "/PID", str(smi.pid), "/F"], capture_output=True)
    print("выгрузка после:", unload())
samples = []
for line in smi_f.read_text().splitlines():
    p = [x.strip() for x in line.split(",")]
    try:
        samples.append((datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f"), int(p[1]), int(p[2]), p[3], float(p[4]), int(p[5])))
    except (ValueError, IndexError):
        pass
out = []
for r in rows:
    w = [x for x in samples if r["t0"] <= x[0] <= r["t1"]]
    sm = statistics.median([x[1] for x in w]) if w else None
    mem = statistics.median([x[2] for x in w]) if w else None
    pst = sorted({x[3] for x in w})
    pw = round(statistics.median([x[4] for x in w])) if w else None
    rate = round(r["prompt_eval_count"] / r["prompt_eval_ms"] * 1000) if r["prompt_eval_ms"] else None
    tps = round(r["eval_count"] / r["eval_ms"] * 1000) if r["eval_ms"] else None
    o = dict(q=r["q"], prefill_ms=r["prompt_eval_ms"], prefill_tok_s=rate, decode_tok_s=tps, total_ms=r["total_ms"],
             sm_mhz=sm, mem_mhz=mem, pstate=pst, power_w=pw, n=len(w))
    out.append(o)
    print(cfg, json.dumps(o, ensure_ascii=False), flush=True)
json.dump(out, open(CL / f"clocks_probe_{cfg}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
