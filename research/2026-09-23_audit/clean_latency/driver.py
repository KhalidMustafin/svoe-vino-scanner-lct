"""Замер задержки: чистый замер живого сервиса для одной конфигурации читателя.

python driver.py 4b|8b

Порядок: выгрузить обе модели в Ollama (/api/ps пуст) -> снимок VRAM -> логгеры nvidia-smi и CPU ->
поднять `python -m app.api` (лог в файл) -> ждать ready (холодный старт) -> снимок VRAM и /api/ps ->
20 кадров штатным participant_test.sh -> /v1/health (scans.degraded) -> снимок VRAM ->
/v1/scan: 3 новых кадра, повтор последнего, 1 кадр из двадцати -> остановить сервис -> выгрузить модель ->
остановить логгеры -> итоговый снимок VRAM. Всё пишется в clean_latency/<cfg>/.
Код репозитория не меняется; сервис пишет только в свой лог (полевые маршруты не вызываются).
"""
from __future__ import annotations

import json
import os
import random
import re
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"<корень>")
REPO = ROOT / "svoe-vino-scanner"
PY = REPO / ".venv/Scripts/python.exe"
FD = ROOT / "field_dataset"
DATASET = "<корень>/Датасет и подробное задание/unpacked/Датасет"
PT = DATASET + "/eval/participant_test.sh"
BASH = r"C:\Program Files\Git\bin\bash.exe"
SPD = Path(r"<временная папка>")
CL = SPD / "clean_latency"
MODELS = {"4b": "qwen3.5:4b", "8b": "qwen3-vl:8b-instruct"}
OLLAMA = "http://127.0.0.1:11434"
SVC = "http://127.0.0.1:8080"

cfg = sys.argv[1]
model = MODELS[cfg.split("_")[0]]  # 4b, 8b, 4b_r2, 8b_r2 ...
OUT = CL / cfg
OUT.mkdir(parents=True, exist_ok=True)
assert not (OUT / "predictions.jsonl").exists(), "прогон этой конфигурации уже есть"
LOGF = open(OUT / "driver.log", "a", encoding="utf-8")


def log(*a) -> None:
    s = f"{datetime.now():%H:%M:%S.%f}"[:-3] + " " + " ".join(str(x) for x in a)
    print(s, flush=True)
    LOGF.write(s + "\n"); LOGF.flush()


def http(url: str, data: dict | None = None, timeout: float = 30) -> dict:
    body = None if data is None else json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"} if body else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def vram(tag: str) -> None:
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(CL / "vram_snap.ps1"), "-Tag", tag],
                       capture_output=True, text=True, encoding="cp866", errors="replace")
    with open(OUT / "vram.txt", "a", encoding="utf-8") as f:
        f.write(r.stdout + r.stderr + "\n")
    log(f"VRAM[{tag}]:", " | ".join(l.strip() for l in r.stdout.splitlines() if l.strip() and not l.startswith("---")))


def ps(tag: str) -> dict:
    d = http(OLLAMA + "/api/ps")
    log(f"/api/ps[{tag}]:", json.dumps([{k: m.get(k) for k in ("name", "size", "size_vram", "context_length", "expires_at")} for m in d.get("models", [])]))
    return d


def unload_all(tag: str) -> None:
    for m in MODELS.values():
        try:
            d = http(OLLAMA + "/api/generate", {"model": m, "keep_alive": 0}, timeout=60)
            log(f"unload {m}: done_reason={d.get('done_reason')}")
        except Exception as e:  # noqa: BLE001
            log(f"unload {m}: {type(e).__name__}: {e}")
    for _ in range(60):
        if not http(OLLAMA + "/api/ps").get("models"):
            break
        time.sleep(0.5)
    ps(tag)


def health(name: str) -> dict:
    d = http(SVC + "/v1/health", timeout=10)
    (OUT / f"{name}.json").write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    return d


# ---------------------------------------------------------------- 0. исходное состояние
assert not port_open(8080), "порт 8080 занят"
log(f"=== конфигурация {cfg}: {model}")
unload_all("до старта")
vram("pre")

smi = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,memory.used,utilization.gpu,pstate", "--format=csv,noheader",
                        "-lms", "500", "-f", str(OUT / "smi.csv")])
cpu = subprocess.Popen(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(CL / "cpu_log.ps1"),
                        "-Out", str(OUT / "cpu.tsv"), "-Seconds", "1200"])
marks: dict[str, str] = {}


def mark(k: str) -> None:
    marks[k] = f"{datetime.now():%H:%M:%S.%f}"[:-3]
    log("mark", k)


# ---------------------------------------------------------------- 1. сервис
env = {k: v for k, v in os.environ.items() if not k.startswith("SVS_") and k != "CUDA_VISIBLE_DEVICES"}
env.update(PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", SVS_VLM_MODEL=model, SVS_BUDGET_MS="8000",
           SVS_VLM_TIMEOUT_MS="5000", SVS_DATASET_DIR=DATASET)
svc_log = open(OUT / "service.log", "a", encoding="utf-8")
svc_log.write(f"=== start {datetime.now():%Y-%m-%d %H:%M:%S.%f} cfg={cfg}\n")
for k in sorted(k for k in env if k.startswith("SVS_")):
    svc_log.write(f"{k}={env[k]}\n")
svc_log.flush()
mark("svc_start")
t0 = time.time()
proc = subprocess.Popen([str(PY), "-m", "app.api"], cwd=str(REPO), env=env, stdout=svc_log, stderr=subprocess.STDOUT)
log("service pid", proc.pid)
first_conn = None
ready = None
while time.time() - t0 < 600:
    if proc.poll() is not None:
        log("сервис упал, код", proc.returncode); break
    try:
        h = http(SVC + "/v1/health", timeout=5)
        first_conn = first_conn or time.time()
        if h.get("status") != "starting":
            ready = time.time(); break
    except Exception:  # noqa: BLE001
        pass
    time.sleep(0.25)
assert ready is not None, "сервис не поднялся"
mark("svc_ready")
hr = health("health_ready")
log(f"ready: {ready - t0:.1f} с от запуска (порт открыт через {first_conn - t0:.1f} с); status={hr['status']} "
    f"reasons={hr.get('degraded_reasons')} warnings={len(hr.get('warnings') or [])} warm={json.dumps(hr.get('warm'), ensure_ascii=False)}")
time.sleep(3)
vram("ready")
ps_ready = ps("ready")

# ---------------------------------------------------------------- 2. штатный скрипт организатора
envb = dict(os.environ)
envb["PATH"] = r"<домашняя папка>\bin;" + envb.get("PATH", "")
mark("pt_start")
tp = time.time()
r = subprocess.run([BASH, PT, "--images-dir", str(CL / "imgs").replace("\\", "/"), "--manifest", str(CL / "queries.tsv").replace("\\", "/"),
                    "--output", str(OUT / "predictions.jsonl").replace("\\", "/")], env=envb, capture_output=True, text=True,
                   encoding="utf-8", errors="replace")
pt_s = time.time() - tp
mark("pt_end")
log(f"participant_test.sh: код {r.returncode}, {pt_s:.1f} с; stdout={r.stdout[-500:]!r} stderr={r.stderr[-800:]!r}")
ha = health("health_after_pt")
log("health после прогона:", json.dumps(ha.get("scans"), ensure_ascii=False), "status=", ha.get("status"))
vram("after_pt")
ps("after_pt")

# ---------------------------------------------------------------- 3. разбор по стадиям (после основного прогона)
sel = json.load(open(CL / "selection.json", encoding="utf-8"))
extras_p = CL / "extras.json"
if not extras_p.exists():
    meta = [json.loads(x) for x in open(FD / "sets/catalog_orig/meta.jsonl", encoding="utf-8")]
    taken = {w for s in sel for w in (s["gt"], *s["acc"])}
    usedq = {s["photo"] for s in sel} | {"M159", "R060", "L04"}
    usedq |= {s["q"].split("-")[0] for s in json.load(open(SPD / "gpu_service/selection.json", encoding="utf-8"))}
    for fn in ("vlm_direct_svc_up.json", "vlm_direct_ollama_alone.json"):
        usedq |= {x["q"].split("-")[0] for x in json.load(open(SPD / "gpu_service" / fn, encoding="utf-8"))}
    pool = [m for m in meta if m["photo"].startswith("M") and m["photo"] not in usedq]
    random.Random(777).shuffle(pool)
    ex = []
    for m in pool:
        if {m["slug"], *m["acceptable"]} & taken:
            continue
        taken |= {m["slug"], *m["acceptable"]}
        ex.append(dict(q=m["query_id"], photo=m["photo"], file=str(FD / m["image"]), gt=m["slug"], acc=m["acceptable"]))
        if len(ex) == 3:
            break
    extras_p.write_text(json.dumps(ex, ensure_ascii=False, indent=1), encoding="utf-8")
extras = json.load(open(extras_p, encoding="utf-8"))
plan = [(e["q"], e["file"], "new") for e in extras]
plan.append((extras[-1]["q"], extras[-1]["file"], "repeat_immediate"))
plan.append((sel[0]["q"], str(CL / "imgs" / sel[0]["file"]), "repeat_after_20"))
scans = []
mark("scan_start")
for q, f, kind in plan:
    t = time.time()
    rr = subprocess.run([r"C:\Windows\System32\curl.exe", "-s", "-F", f"image=@{f}", SVC + "/v1/scan"], capture_output=True)
    wall = (time.time() - t) * 1000
    body = json.loads(rr.stdout.decode("utf-8"))
    ev = body.get("evidence") or {}
    row = dict(q=q, kind=kind, wall_ms=round(wall), slug=body.get("slug"), degraded=body.get("degraded"),
               timings_ms=body.get("timings_ms"), p=(body.get("confidence") or {}).get("top1"))
    scans.append(dict(row, evidence_keys=sorted(ev.keys()) if isinstance(ev, dict) else None, body=body))
    log("scan", json.dumps(row, ensure_ascii=False))
mark("scan_end")
json.dump(scans, open(OUT / "scans.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
health("health_end")

# ---------------------------------------------------------------- 4. остановка
subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
for _ in range(40):
    if not port_open(8080):
        break
    time.sleep(0.25)
log("порт 8080 открыт после остановки:", port_open(8080))
mark("svc_stopped")
unload_all("после остановки")
time.sleep(2)
subprocess.run(["taskkill", "/PID", str(smi.pid), "/T", "/F"], capture_output=True)
subprocess.run(["taskkill", "/PID", str(cpu.pid), "/T", "/F"], capture_output=True)
vram("final")
json.dump(dict(cfg=cfg, model=model, cold_start_s=round(ready - t0, 1), port_open_s=round(first_conn - t0, 1),
               pt_seconds=round(pt_s, 1), pt_code=r.returncode, marks=marks,
               ps_ready=ps_ready), open(OUT / "run_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
log("готово")
