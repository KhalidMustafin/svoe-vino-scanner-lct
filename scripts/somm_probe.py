"""Зонд живого голоса сомелье для окна видеокарты: скорость, доля прошедших, без перезагрузки.

    python scripts/somm_probe.py --out runs/somm_probe.json
    python scripts/somm_probe.py --runs 20 --preempt 20 --facts facts.jsonl
    python scripts/somm_probe.py --service http://127.0.0.1:8080 --slugs <slug> <slug>

**Прямой режим** (по умолчанию) — клиент голоса и проверки без сервиса, на той Ollama и с той
моделью, что у сервиса (`SVS_OLLAMA_URL`, `SVS_VLM_MODEL`, `SVS_VLM_KEEP_ALIVE`):

1. `options` — клиент голоса собирается `OllamaText.mirroring` из читателя этикетки, собранного
   так же, как в `ScannerService.load`, и сверяется с ним: модель, `keep_alive`, `num_ctx`,
   ключи тела и `options`, нет параметров загрузки. Расхождение — Ollama перезагрузит модель;
2. `ps_before` / `ps_after` — `/api/ps` до и после: та же модель в памяти, `expires_at` не
   укорочен. `reloads` — вызовы с `load_duration` дольше секунды;
3. `calls` — `--runs` вызовов на каждый пакет фактов: пакеты с `voice: true` из заглушек
   договора (`tests/fixtures/somm/ask_*.ndjson`) и строки `--facts` (события `facts` потока
   `POST /v1/sommelier/ask`, по одному JSON на строку) — только те, что голос сервиса сейчас
   пересказывает (`answers.voice_fits`: у `dish_check` — вердикт «да» без вин подборки; старые
   записи с `voice: true` при оговорке не мерятся). Для каждого — статус, время до первого
   токена, полное время, токены и код первой непройденной проверки. `pass_rate` — доля
   прошедших среди ответивших;
4. `preempt` — `--preempt` обрывов: вызов идёт `--preempt-after-ms`, затем обрыв, как от скана.
   Мерится, когда вернулся клиент (`cancel_return_ms`) и через сколько короткий запрос с теми же
   опциями получил первый токен (`next_ttft_ms`) — столько ждал бы скан, пока Ollama отдаёт
   слот; `baseline_ttft_ms` — тот же запрос без обрыва.

**Режим сервиса** (`--service`) — насквозь через `GET /v1/wines/{slug}/sommelier` и
`POST /v1/sommelier/ask` по всем чипам карточки: `generated`, `reason`, `guard`, время событий
`facts` и `text`. Нужен сервис с маршрутами сомелье и `SVS_SOMM_LIVE=1`.

Тексты модели пишутся только в отчёт `--out` (без `--no-texts`), не в журнал и не на экран.
Сеть — только до указанных Ollama и сервиса; прокси окружения не используются.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app.sommelier import guards
from app.sommelier.answers import voice_fits
from app.sommelier.entity_lock import EntityLock
from app.sommelier.ollama_text import ChatResult, OllamaText
from app.sommelier.voice import build_messages

FIXTURES = REPO / "tests" / "fixtures" / "somm"
#: Параметры загрузки модели: их в теле голоса быть не должно (перезагрузка).
LOAD_OPTIONS = ("num_gpu", "num_thread", "num_batch", "use_mmap", "use_mlock", "main_gpu")
#: `load_duration` дольше этого — модель грузилась заново.
RELOAD_MS = 1000
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")

VERDICT_WORDS = {
    "yes": "подходит",
    "caveat": "подходит с оговоркой",
    "no": "скорее нет",
    "neutral": "правила сочетаний молчат",
}
INTENT_LABELS = {
    "what_to_eat": "к чему подать это вино",
    "dish_check": "подходит ли вино к блюду «{dish}»",
    "softer": "вина помягче",
    "fresher": "вина посвежее",
    "replace": "чем заменить вино",
    "guided": "подбор к блюду",
}


# ---------------------------------------------------------------------- пакет фактов
@dataclass(frozen=True)
class ProbeFacts:
    """Пакет фактов зонда — те же поля, что у `FactsPackage` (`guards.FactsLike`).

    Собирается из тела события `facts` и карточки якоря по §6.4 договора, пока пакет
    маршрутизатора (`app/sommelier/answers.py`) не отдаёт свои строки для модели наружу.
    """

    public: Mapping[str, Any]
    verdict_template: str
    voiced: bool
    intent_label: str
    voice_input: tuple[str, ...]
    allowed_entities: frozenset[str]
    allowed_numbers: frozenset[str]
    allowed_text: str


def _wine_line(wine: Mapping[str, Any]) -> str:
    parts = [
        str(wine.get("name") or ""),
        str(wine.get("winery") or ""),
        str(wine.get("region") or ""),
        str(wine.get("style_label") or "").lower(),
        ", ".join(wine.get("grapes") or []) or "сорт не указан",
    ]
    line = " · ".join(part for part in parts if part)
    return f"{line} — {wine['pill']}" if wine.get("pill") else line


def facts_from_event(
    public: Mapping[str, Any], anchor: Mapping[str, Any], *, template: str | None = None
) -> ProbeFacts:
    """Пакет по телу события `facts` (без `type`, `t_ms`) и карточке якоря с `serve`."""
    template = str(public.get("verdict_template") or "") if template is None else template
    grapes = [str(grape) for grape in anchor.get("grapes") or []]
    lines = [
        f"Вино: {anchor['name']}",
        f"Винодельня: {anchor['winery']}",
        f"Регион: {anchor.get('region') or 'не указан'}",
        f"Стиль: {str(anchor.get('style_label') or '').lower()}",
        f"Сорта: {', '.join(grapes)}" if grapes else "Сорт не указан",
    ]
    serve = anchor.get("serve") or {}
    if serve.get("temperature_c"):
        low, high = serve["temperature_c"]
        lines.append(f"Подача: {low}–{high} °C")
    for dish in public.get("dishes") or []:
        line = f"Блюдо: {dish['name']} — {VERDICT_WORDS.get(dish['verdict'], dish['verdict'])}"
        pluses = ", ".join(chip["text"] for chip in dish.get("plus") or [])
        minuses = ", ".join(chip["text"] for chip in dish.get("minus") or [])
        if pluses:
            line += f"; за: {pluses}"
        if minuses:
            line += f"; против: {minuses}"
        lines.append(line)
    if public.get("wines_title"):
        lines.append(f"Подборка: {public['wines_title']}")
    for wine in public.get("wines") or []:
        lines.append(f"Вино подборки: {_wine_line(wine)}")

    entities = {str(anchor["name"]), str(anchor["winery"]), str(anchor.get("region") or "")}
    entities.update(grapes)
    for dish in public.get("dishes") or []:
        entities.add(str(dish["name"]))
    for wine in public.get("wines") or []:
        entities.update(str(wine.get(key) or "") for key in ("name", "winery", "region"))
        entities.update(str(grape) for grape in wine.get("grapes") or [])
    text = "\n".join([*lines, template])
    dishes = [str(dish["name"]) for dish in public.get("dishes") or []]
    intent = str(public.get("intent") or "")
    label = INTENT_LABELS.get(intent, intent).format(dish=dishes[0] if dishes else "")
    return ProbeFacts(
        public=dict(public),
        verdict_template=template,
        voiced=bool(public.get("voice")),
        intent_label=label,
        voice_input=tuple(lines),
        allowed_entities=frozenset(entity for entity in entities if entity),
        allowed_numbers=frozenset(_NUMBER_RE.findall(text)),
        allowed_text=text,
    )


def fixture_anchor() -> dict[str, Any]:
    """Якорь заглушек — Cru Lermont Saperavi: карточка без портала и подача из `sommelier_red`."""
    card = json.loads((FIXTURES / "wine_card_noportal.json").read_text(encoding="utf-8"))
    red = json.loads((FIXTURES / "sommelier_red.json").read_text(encoding="utf-8"))
    return {**card, "serve": red["serve"]}


def _facts_events(lines: Iterable[str]) -> list[dict[str, Any]]:
    out = []
    for line in lines:
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("type", "facts") == "facts":
            out.append({k: v for k, v in event.items() if k not in {"type", "t_ms"}})
    return out


def _fits(public: Mapping[str, Any]) -> bool:
    """Пересказывает ли голос сервиса такой пакет сейчас (договор, §6.5)."""
    return voice_fits(
        str(public.get("intent") or ""), public.get("dishes") or [], public.get("wines") or []
    )


def fixture_packages() -> list[tuple[str, ProbeFacts]]:
    """Пакеты с голосом из потоков заглушек, по одному на намерение."""
    anchor = fixture_anchor()
    seen: set[str] = set()
    packages = []
    for path in sorted(FIXTURES.glob("ask_*.ndjson")):
        for public in _facts_events(path.read_text(encoding="utf-8").splitlines()):
            if public.get("voice") and _fits(public) and public["intent"] not in seen:
                seen.add(public["intent"])
                packages.append((path.stem, facts_from_event(public, anchor)))
    return packages


def file_packages(path: Path, anchor: Mapping[str, Any]) -> list[tuple[str, ProbeFacts]]:
    """Пакеты из JSONL: событие `facts` или `{"facts": …, "anchor": …}` на строку."""
    packages = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        public = row.get("facts", row)
        public = {k: v for k, v in public.items() if k not in {"type", "t_ms"}}
        facts = facts_from_event(public, row.get("anchor") or anchor)
        if facts.voiced and _fits(public):
            packages.append((f"{path.name}:{number}", facts))
    return packages


def load_lock(directory: Path | None) -> tuple[EntityLock, str]:
    """Замок по `vocab.json` из `SVS_SOMM_DIR` (или `--somm-dir`), иначе из заглушек."""
    for candidate in ([directory / "vocab.json"] if directory else []) + [FIXTURES / "vocab.json"]:
        if candidate.is_file():
            data = json.loads(candidate.read_text(encoding="utf-8"))
            return EntityLock.from_mapping(data), str(candidate)
    raise FileNotFoundError("нет vocab.json ни в каталоге сомелье, ни в заглушках")


# ---------------------------------------------------------------------- Ollama
def mirror_reader(ollama: str | None, model: str | None) -> tuple[Any, OllamaText]:
    """Читатель этикетки, собранный как в `ScannerService.load`, и голос, зеркалящий его."""
    from app.api.config import ServiceSettings
    from app.reading.readers.ollama_vlm import OllamaVlmReader

    settings = ServiceSettings.from_env()
    reader = OllamaVlmReader(
        model or settings.vlm_model,
        ollama or settings.ollama_url,
        timeout_s=settings.vlm_timeout_ms / 1000,
        keep_alive=settings.vlm_keep_alive,
    )
    return reader, OllamaText.mirroring(reader)


def options_report(reader: Any, text: OllamaText) -> dict[str, Any]:
    ours = text.payload([{"role": "user", "content": "проверка"}])
    theirs = reader.payload("")
    return {
        "model": ours["model"],
        "keep_alive": ours["keep_alive"],
        "num_ctx": ours["options"]["num_ctx"],
        "num_predict": ours["options"]["num_predict"],
        "model_equal": ours["model"] == theirs["model"],
        "keep_alive_equal": ours["keep_alive"] == theirs["keep_alive"],
        "num_ctx_equal": ours["options"]["num_ctx"] == theirs["options"]["num_ctx"],
        "top_keys_equal": set(ours) == set(theirs),
        "option_keys_equal": set(ours["options"]) == set(theirs["options"]),
        "no_load_options": not set(ours["options"]) & set(LOAD_OPTIONS),
        "think_false": ours["think"] is False and theirs["think"] is False,
    }


def get_json(url: str, *, timeout: float = 5.0) -> Any:
    with _OPENER.open(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def ps(base: str) -> dict[str, Any]:
    """Модели в памяти Ollama: имя, `expires_at`, видеопамять."""
    try:
        data = get_json(f"{base.rstrip('/')}/api/ps")
    except (OSError, ValueError, urllib.error.URLError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "models": [
            {key: model.get(key) for key in ("name", "expires_at", "size_vram")}
            for model in data.get("models") or []
        ]
    }


def ps_compare(before: Mapping[str, Any], after: Mapping[str, Any], model: str) -> dict[str, Any]:
    def entry(state: Mapping[str, Any]) -> Mapping[str, Any] | None:
        return next((m for m in state.get("models") or [] if m.get("name") == model), None)

    first, last = entry(before), entry(after)
    return {
        "loaded_before": first is not None,
        "loaded_after": last is not None,
        "same_models": [m.get("name") for m in before.get("models") or []]
        == [m.get("name") for m in after.get("models") or []],
        "expires_not_shortened": bool(
            first
            and last
            and str(last.get("expires_at") or "") >= str(first.get("expires_at") or "")
        ),
    }


# ---------------------------------------------------------------------- замеры
def quantiles(values: Sequence[float]) -> dict[str, float | None]:
    data = sorted(float(v) for v in values if v is not None)
    if not data:
        return {"n": 0, "p50": None, "p95": None, "max": None}
    p95 = data[min(len(data) - 1, max(0, round(0.95 * len(data)) - 1))]
    return {"n": len(data), "p50": statistics.median(data), "p95": p95, "max": data[-1]}


def run_calls(
    client: OllamaText,
    packages: Sequence[tuple[str, ProbeFacts]],
    lock: EntityLock,
    *,
    runs: int,
    timeout_s: float,
    texts: bool,
) -> list[dict[str, Any]]:
    records = []
    for name, facts in packages:
        messages = build_messages(facts)
        for attempt in range(runs):
            chat = client.chat(messages, timeout_s=timeout_s, cancel=threading.Event())
            guard = None
            if chat.status in {"ok", "thinking_only"}:
                guard = guards.check(
                    chat.content,
                    facts,
                    lock,
                    done_reason=chat.done_reason,
                    thinking_only=chat.status == "thinking_only",
                )
            record = {
                "package": name,
                "intent": facts.public.get("intent"),
                "attempt": attempt,
                "status": chat.status,
                "guard": guard,
                "passed": chat.status == "ok" and guard is None,
                **_timing(chat),
                "input_chars": sum(len(m["content"]) for m in messages),
            }
            if texts:
                record["text"] = chat.content
            records.append(record)
    return records


def _timing(chat: ChatResult) -> dict[str, Any]:
    return {
        "elapsed_ms": chat.elapsed_ms,
        "first_token_ms": chat.first_token_ms,
        "eval_count": chat.eval_count,
        "prompt_eval_count": chat.prompt_eval_count,
        "load_ms": chat.load_ms,
        "done_reason": chat.done_reason,
        "error": chat.error,
    }


def _short_ttft(client: OllamaText, timeout_s: float) -> int | None:
    short = OllamaText(
        client.url, client.model, keep_alive=client.keep_alive, num_ctx=client.num_ctx,
        num_predict=1,
    )  # fmt: skip
    chat = short.chat(
        [{"role": "user", "content": "Ответь одним словом: да."}],
        timeout_s=timeout_s,
        cancel=threading.Event(),
    )
    return chat.first_token_ms if chat.status in {"ok", "thinking_only"} else None


def run_preempt(
    client: OllamaText,
    facts: ProbeFacts,
    *,
    count: int,
    after_ms: int,
    timeout_s: float,
) -> dict[str, Any]:
    """Обрыв посреди ответа и сколько после него ждёт следующий запрос к той же модели."""
    baseline = [_short_ttft(client, timeout_s) for _ in range(max(1, min(count, 5)))]
    messages = build_messages(facts)
    rows = []
    for _ in range(count):
        aborts: list[Any] = []
        out: dict[str, Any] = {}
        cancel = threading.Event()

        def call(cancel: threading.Event = cancel, aborts: list[Any] = aborts, out=out) -> None:
            out["chat"] = client.chat(
                messages, timeout_s=timeout_s, cancel=cancel, on_abort=aborts.append
            )
            out["returned"] = time.perf_counter()

        thread = threading.Thread(target=call)
        thread.start()
        time.sleep(after_ms / 1000)
        started = time.perf_counter()
        cancel.set()
        for abort in aborts:
            abort()
        thread.join()
        chat: ChatResult = out["chat"]
        rows.append(
            {
                "status": chat.status,
                "cancel_return_ms": round((out["returned"] - started) * 1000, 1),
                "next_ttft_ms": _short_ttft(client, timeout_s),
            }
        )
    return {
        "after_ms": after_ms,
        "cancelled": sum(row["status"] == "cancelled" for row in rows),
        "finished_before_cancel": sum(row["status"] != "cancelled" for row in rows),
        "cancel_return_ms": quantiles([row["cancel_return_ms"] for row in rows]),
        "next_ttft_ms": quantiles([row["next_ttft_ms"] for row in rows]),
        "baseline_ttft_ms": quantiles(baseline),
        "rows": rows,
    }


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    answered = [r for r in records if r["status"] in {"ok", "thinking_only"}]
    passed = [r for r in records if r["passed"]]
    return {
        "calls": len(records),
        "answered": len(answered),
        "passed": len(passed),
        "pass_rate": round(len(passed) / len(answered), 3) if answered else None,
        "statuses": dict(Counter(r["status"] for r in records)),
        "guards": dict(Counter(r["guard"] for r in records if r["guard"])),
        "reloads": sum((r.get("load_ms") or 0) > RELOAD_MS for r in records),
        "elapsed_ms": quantiles([r["elapsed_ms"] for r in answered]),
        "first_token_ms": quantiles([r["first_token_ms"] for r in answered]),
        "eval_count": quantiles([r["eval_count"] for r in answered]),
        "prompt_eval_count": quantiles([r["prompt_eval_count"] for r in answered]),
        "by_intent": {
            intent: {
                "calls": sum(r["intent"] == intent for r in records),
                "passed": sum(r["intent"] == intent and r["passed"] for r in records),
            }
            for intent in sorted({str(r["intent"]) for r in records})
        },
    }


# ---------------------------------------------------------------------- режим сервиса
def _post_ndjson(url: str, body: Mapping[str, Any], timeout: float) -> list[dict[str, Any]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with _OPENER.open(request, timeout=timeout) as response:
        return [json.loads(line) for line in response.read().decode("utf-8").splitlines() if line]


def run_service(base: str, slugs: Sequence[str], *, timeout_s: float) -> list[dict[str, Any]]:
    """Все чипы карточки каждого slug насквозь через маршруты сомелье."""
    base = base.rstrip("/")
    records = []
    for slug in slugs:
        quoted = urllib.parse.quote(slug)
        try:
            card = get_json(f"{base}/v1/wines/{quoted}/sommelier", timeout=timeout_s)
        except (OSError, ValueError, urllib.error.URLError) as exc:
            records.append({"slug": slug, "error": f"{type(exc).__name__}: {exc}"})
            continue
        for chip in card.get("chips") or []:
            body = {"slug": slug, "chip": chip["id"], "args": chip.get("args") or {}}
            started = time.perf_counter()
            try:
                events = _post_ndjson(f"{base}/v1/sommelier/ask", body, timeout_s + 5)
            except (OSError, ValueError, urllib.error.URLError) as exc:
                records.append({"slug": slug, "chip": chip["id"], "error": type(exc).__name__})
                continue
            facts = next((e for e in events if e["type"] == "facts"), {})
            text = next((e for e in events if e["type"] == "text"), {})
            records.append(
                {
                    "slug": slug,
                    "chip": chip["id"],
                    "intent": facts.get("intent"),
                    "voice": facts.get("voice"),
                    "facts_ms": facts.get("t_ms"),
                    "text_ms": text.get("t_ms"),
                    "wall_ms": round((time.perf_counter() - started) * 1000),
                    "generated": text.get("generated"),
                    "reason": text.get("reason"),
                    "guard": text.get("guard"),
                    "stages": [e["id"] for e in events if e["type"] == "stage"],
                }
            )
    return records


def summarize_service(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    voiced = [r for r in records if r.get("voice")]
    generated = [r for r in voiced if r.get("generated")]
    return {
        "asks": len(records),
        "errors": sum("error" in r for r in records),
        "voiced": len(voiced),
        "generated": len(generated),
        "pass_rate": round(len(generated) / len(voiced), 3) if voiced else None,
        "reasons": dict(Counter(r.get("reason") for r in voiced if r.get("reason"))),
        "guards": dict(Counter(r.get("guard") for r in voiced if r.get("guard"))),
        "facts_ms": quantiles([r["facts_ms"] for r in records if r.get("facts_ms") is not None]),
        "text_ms": quantiles([r["text_ms"] for r in generated if r.get("text_ms") is not None]),
    }


# ---------------------------------------------------------------------- запуск
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/somm_probe.py", description="Зонд живого голоса сомелье."
    )
    parser.add_argument("--ollama", default=None, help="адрес Ollama; по умолчанию SVS_OLLAMA_URL")
    parser.add_argument("--model", default=None, help="тег модели; по умолчанию SVS_VLM_MODEL")
    parser.add_argument("--runs", type=int, default=5, help="вызовов на пакет фактов")
    parser.add_argument("--timeout-ms", type=int, default=4000, help="как SVS_SOMM_TIMEOUT_MS")
    parser.add_argument("--facts", type=Path, default=None, help="JSONL событий facts")
    parser.add_argument("--no-fixtures", action="store_true", help="без пакетов заглушек")
    parser.add_argument("--somm-dir", type=Path, default=None, help="каталог с vocab.json")
    parser.add_argument("--preempt", type=int, default=0, help="сколько обрывов замерить")
    parser.add_argument("--preempt-after-ms", type=int, default=300)
    parser.add_argument("--service", default=None, help="адрес сервиса для режима насквозь")
    parser.add_argument("--slugs", nargs="*", default=None, help="slug для режима сервиса")
    parser.add_argument("--out", type=Path, default=None, help="отчёт JSON")
    parser.add_argument("--no-texts", action="store_true", help="не класть тексты в отчёт")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    timeout_s = args.timeout_ms / 1000
    report: dict[str, Any] = {"started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    reader, client = mirror_reader(args.ollama, args.model)
    report["options"] = options_report(reader, client)
    report["ps_before"] = ps(client.url)

    if args.service:
        slugs = args.slugs or [fixture_anchor()["slug"]]
        records = run_service(args.service, slugs, timeout_s=timeout_s)
        report["service"] = {"summary": summarize_service(records), "asks": records}
    else:
        packages = [] if args.no_fixtures else fixture_packages()
        if args.facts:
            packages += file_packages(args.facts, fixture_anchor())
        if not packages:
            print("нет пакетов фактов с голосом", file=sys.stderr)
            return 2
        somm_dir = args.somm_dir or _somm_dir_from_env()
        lock, vocab = load_lock(somm_dir)
        report["vocab"] = vocab
        records = run_calls(
            client, packages, lock, runs=args.runs, timeout_s=timeout_s, texts=not args.no_texts
        )
        report["summary"] = summarize(records)
        report["calls"] = records
        if args.preempt:
            report["preempt"] = run_preempt(
                client,
                packages[0][1],
                count=args.preempt,
                after_ms=args.preempt_after_ms,
                timeout_s=timeout_s,
            )

    report["ps_after"] = ps(client.url)
    report["ps"] = ps_compare(report["ps_before"], report["ps_after"], client.model)
    _print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"отчёт: {args.out}")
    return 0


def _somm_dir_from_env() -> Path | None:
    import os

    raw = (os.environ.get("SVS_SOMM_DIR") or "").strip()
    if raw:
        return Path(raw)
    data = (os.environ.get("SVS_DATA_DIR") or "").strip()
    return Path(data) / "somm" if data else REPO / "data" / "somm"


def _print(report: Mapping[str, Any]) -> None:
    options = report["options"]
    same = all(v for k, v in options.items() if k.endswith(("_equal", "_options", "_false")))
    print(
        f"модель {options['model']} keep_alive={options['keep_alive']} "
        f"num_ctx={options['num_ctx']} num_predict={options['num_predict']}; "
        f"опции как у читателя: {same}"
    )
    print(f"/api/ps: {json.dumps(report['ps'], ensure_ascii=False)}")
    if "summary" in report:
        summary = report["summary"]
        print(
            f"вызовов {summary['calls']}, ответили {summary['answered']}, прошли проверки "
            f"{summary['passed']} (доля {summary['pass_rate']}), перезагрузок {summary['reloads']}"
        )
        print(f"первый токен {summary['first_token_ms']}")
        print(f"полное время {summary['elapsed_ms']}")
        print(f"статусы {summary['statuses']}; проверки {summary['guards']}")
    if "preempt" in report:
        preempt = report["preempt"]
        print(
            f"обрыв через {preempt['after_ms']} мс: возврат {preempt['cancel_return_ms']}, "
            f"следующий запрос {preempt['next_ttft_ms']}, без обрыва {preempt['baseline_ttft_ms']}"
        )
    if "service" in report:
        print(f"сервис: {json.dumps(report['service']['summary'], ensure_ascii=False)}")


if __name__ == "__main__":
    raise SystemExit(main())
