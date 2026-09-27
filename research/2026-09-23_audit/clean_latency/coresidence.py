"""Задача B: замедляет ли соседство с сервисом (SigLIP в той же видеокарте) чтение VLM — без помехи CPU.

python coresidence.py 4b|8b

Одни и те же 5 кадров читаются тем же путём, что в сервисе (decode_on_backgrounds -> read_label, crop full),
прямым вызовом Ollama в двух условиях:
  A) сервис остановлен (в видеопамяти только llama-server);
  B) сервис поднят и простаивает (SigLIP fp32 держит свою память), модель перезагружена.
Перед каждым условием модель выгружается (keep_alive:0) — кэш промптов пуст; каждый кадр читается один раз.
Из ответа Ollama пишутся prompt_eval / eval / total. Итог — clean_latency/coresidence_<cfg>.json
"""
from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(r"<корень>\svoe-vino-scanner")
sys.path.insert(0, str(REPO))
from app.api.service import BACKGROUND, MAX_PIXELS, VLM_CROP, VLM_CROP_PX, WARM_FRAME_PX, WHITE, decode_on_backgrounds, read_label  # noqa: E402
from app.reading.readers.ollama_vlm import OllamaVlmReader, UrllibTransport  # noqa: E402
from app.reading.warmup import synthetic_label  # noqa: E402

CL = Path(__file__).resolve().parent
PY = REPO / ".venv/Scripts/python.exe"
DATASET = "<корень>/Датасет и подробное задание/unpacked/Датасет"
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


def smi() -> str:
    return subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()


rows: list[dict] = []
cur = {"cond": "", "q": ""}


class Rec:
    def __init__(self) -> None:
        self.inner = UrllibTransport(O + "/api/chat")

    def __call__(self, payload: dict, *, timeout_s: float) -> dict:
        t = time.perf_counter()
        d = self.inner(payload, timeout_s=timeout_s)
        ms = lambda k: round((d.get(k) or 0) / 1e6)  # noqa: E731
        row = dict(cond=cur["cond"], q=cur["q"], wall_ms=round((time.perf_counter() - t) * 1000), load_ms=ms("load_duration"),
                   prompt_eval_count=d.get("prompt_eval_count"), prompt_eval_ms=ms("prompt_eval_duration"),
                   eval_count=d.get("eval_count"), eval_ms=ms("eval_duration"), total_ms=ms("total_duration"))
        rows.append(row)
        print(cfg, json.dumps(row, ensure_ascii=False), flush=True)
        return d


sel = json.load(open(CL / "selection.json", encoding="utf-8"))
ext = json.load(open(CL / "extras.json", encoding="utf-8"))
frames = [(e["q"], Path(e["file"])) for e in ext] + [(s["q"], CL / "imgs" / s["file"]) for s in (sel[1], sel[12])]
ocrs = [(q, decode_on_backgrounds(p.read_bytes(), (BACKGROUND, WHITE), max_pixels=MAX_PIXELS)[1]) for q, p in frames]


def run(cond: str) -> None:
    print(cond, "выгрузка:", unload(), "nvidia-smi:", smi(), flush=True)
    reader = OllamaVlmReader(model, timeout_s=60, keep_alive="5m", transport=Rec())
    cur.update(cond=cond, q="warm_synthetic")
    reader.warm(synthetic_label(WARM_FRAME_PX), budget_ms=60000)
    print(cond, "после загрузки nvidia-smi:", smi(), flush=True)
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(CL / "vram_snap.ps1"), "-Tag",
                        f"coresidence_{cfg}_{cond}"], capture_output=True, text=True, encoding="cp866", errors="replace")
    with open(CL / f"coresidence_{cfg}_vram.txt", "a", encoding="utf-8") as f:
        f.write(r.stdout + "\n")
    print(cond, "VRAM:", " | ".join(l.strip() for l in r.stdout.splitlines() if re.search(r"llama|python|phys_0 (dedicated|shared)", l)), flush=True)
    for q, ocr in ocrs:
        cur.update(q=q)
        read_label(ocr, readers=[reader], lexicon=None, target=None, crop=VLM_CROP, crop_px=VLM_CROP_PX, budget_ms=60000)


# A: сервис остановлен
assert socket.socket().connect_ex(("127.0.0.1", 8080)) != 0, "порт 8080 занят"
run("A_alone")
unload()
# B: сервис поднят и простаивает
env = {k: v for k, v in os.environ.items() if not k.startswith("SVS_") and k != "CUDA_VISIBLE_DEVICES"}
env.update(PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", SVS_VLM_MODEL=model, SVS_BUDGET_MS="8000", SVS_VLM_TIMEOUT_MS="5000",
           SVS_DATASET_DIR=DATASET)
log = open(CL / f"coresidence_{cfg}_service.log", "a", encoding="utf-8")
proc = subprocess.Popen([str(PY), "-m", "app.api"], cwd=str(REPO), env=env, stdout=log, stderr=subprocess.STDOUT)
t0 = time.time()
h: dict = {}
while time.time() - t0 < 300:
    try:
        h = json.loads(urllib.request.urlopen("http://127.0.0.1:8080/v1/health", timeout=5).read().decode())
        if h.get("status") != "starting":
            break
    except Exception:  # noqa: BLE001
        time.sleep(0.5)
print("сервис:", h.get("status"), f"{time.time() - t0:.1f} с", flush=True)
try:
    run("B_service_idle")
finally:
    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    time.sleep(1)
    print("после остановки: выгрузка", unload(), "порт 8080 открыт:", socket.socket().connect_ex(("127.0.0.1", 8080)) == 0,
          "nvidia-smi:", smi(), flush=True)
json.dump(rows, open(CL / f"coresidence_{cfg}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for q, _ in frames:
    a = next(r for r in rows if r["cond"] == "A_alone" and r["q"] == q)
    b = next(r for r in rows if r["cond"] == "B_service_idle" and r["q"] == q)
    print(f"{cfg} {q:12s} prompt_eval A {a['prompt_eval_ms']:5d} / B {b['prompt_eval_ms']:5d} мс; eval A {a['eval_ms']:5d} ({a['eval_count']} ток) / "
          f"B {b['eval_ms']:5d} ({b['eval_count']} ток); total A {a['total_ms']:5d} / B {b['total_ms']:5d}")
