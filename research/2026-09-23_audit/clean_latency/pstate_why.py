"""Задача B: почему под нагрузкой VLM карта уходит в P3 (SM ~800 МГц, память 5001 МГц). Только чтение состояния.

Грузит 4b, 20 с подряд читает кадры выборки; на 2-й и 12-й секунде снимает `nvidia-smi -q -d PERFORMANCE,CLOCK`
и строку частот/причин. Потом выгружает модель. Итог — pstate_why.out рядом.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(r"<корень>\svoe-vino-scanner")
sys.path.insert(0, str(REPO))
from app.api.service import BACKGROUND, MAX_PIXELS, VLM_CROP, VLM_CROP_PX, WARM_FRAME_PX, WHITE, decode_on_backgrounds, read_label  # noqa: E402
from app.reading.readers.ollama_vlm import OllamaVlmReader  # noqa: E402
from app.reading.warmup import synthetic_label  # noqa: E402

CL = Path(__file__).resolve().parent
O = "http://127.0.0.1:11434"


def post(path, data):
    req = urllib.request.Request(O + path, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read().decode())


def unload():
    for m in ("qwen3.5:4b", "qwen3-vl:8b-instruct"):
        post("/api/generate", {"model": m, "keep_alive": 0})
    time.sleep(1)
    return json.loads(urllib.request.urlopen(O + "/api/ps").read().decode())["models"]


print("выгрузка:", unload())
reader = OllamaVlmReader("qwen3.5:4b", timeout_s=60, keep_alive="5m")
reader.warm(synthetic_label(WARM_FRAME_PX), budget_ms=60000)
ocrs = [decode_on_backgrounds((CL / "imgs" / s["file"]).read_bytes(), (BACKGROUND, WHITE), max_pixels=MAX_PIXELS)[1]
        for s in json.load(open(CL / "selection.json", encoding="utf-8"))]
stop = threading.Event()


def load():
    i = 0
    while not stop.is_set():
        read_label(ocrs[i % len(ocrs)], readers=[reader], lexicon=None, target=None, crop=VLM_CROP, crop_px=VLM_CROP_PX, budget_ms=60000)
        i += 1


th = threading.Thread(target=load, daemon=True)
t0 = time.time()
th.start()
for at in (2, 12):
    time.sleep(max(0, t0 + at - time.time()))
    q = subprocess.run(["nvidia-smi", "--query-gpu=pstate,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,power.draw,power.limit,"
                        "temperature.gpu,utilization.gpu,clocks_event_reasons.active", "--format=csv"], capture_output=True, text=True).stdout
    print(f"--- t={at} с\n{q}", flush=True)
    full = subprocess.run(["nvidia-smi", "-q", "-d", "PERFORMANCE,CLOCK"], capture_output=True, text=True).stdout
    print("\n".join(l for l in full.splitlines() if l.strip() and not l.startswith("=")), flush=True)
stop.set()
th.join(timeout=30)
print("выгрузка после:", unload())
