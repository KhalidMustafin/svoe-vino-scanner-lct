"""Лист «Сомелье» `app/api/static/somm.js` и `somm.css`: договор, право и поведение.

Договор — `docs/api-sommelier.md`, §8: `window.SommUI` с методами §8.1, классы `.somm-` и токены
§8.2, «роза ветров» и шкалы по правилам §8.1, поток NDJSON по §3. Здесь проверяются исходники,
как у страницы продукта (`test_product_page.py`): без внешних адресов, без разметки строкой, без
хранилища браузера, строки проходят `content_filter`, знака процента нет. Поведение — на Node,
если он есть: `somm.js` грузится в маленький поддельный DOM, рисует заглушки
`tests/fixtures/somm/` и читает потоки `ask_*.ndjson`, порезанные посреди букв.

Заглушка стенда `scripts/somm_stub.py` отвечает по договору: те же маршруты, 404 и 422, поток
NDJSON той же грамматики.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from app.recommend.content_filter import check

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "app" / "api" / "static"
JS = STATIC / "somm.js"
CSS = STATIC / "somm.css"
CONTRACT = ROOT / "docs" / "api-sommelier.md"
FIXTURES = ROOT / "tests" / "fixtures" / "somm"
STUB = ROOT / "scripts" / "somm_stub.py"
#: Предел листа — 1000 строк: 900 не хватило на легенду наложения и силуэт без фото.
MAX_LINES = 1000
LABEL_AI = "Текст — ИИ, подбор — алгоритм"
LABEL_ALGO = "Текст и подбор — алгоритм"
#: Адрес наружу в любом виде — как в `test_product_page.py` для файлов оформления.
URL_RE = re.compile(r"(?:https?:)?//[A-Za-z0-9.-]+\.[a-z]{2,}")
#: Стоп-слова 38-ФЗ и проверок голоса (договор §1 и §6.5): в строках листа их нет.
STOP = re.compile(
    r"купи|покупк|цен[аыуе]\b|стоимост|₽|руб|лучш|идеальн|вино недели|превосходн|шедевр"
    r"|безупречн|уникальн|рейтинг|балл|звёзд|скидк|акци[яи]\b|попробуйте|выпейте|наслаждайтесь",
    re.IGNORECASE,
)
TYPE_SCALE = {36, 32, 24, 18, 16, 14, 12}
EVENTS = ("stage", "facts", "text", "error", "done")
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


@pytest.fixture(scope="module")
def js() -> str:
    return JS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def css() -> str:
    return CSS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def code(js: str) -> str:
    """Скрипт без комментариев: только то, что исполняется."""
    text = re.sub(r"/\*.*?\*/", "", js, flags=re.DOTALL)
    return re.sub(r"(?m)^\s*//.*$|\s//\s.*$", "", text)


def literals(code: str) -> list[str]:
    return re.findall(r'"((?:[^"\\\n]|\\.)*)"', code)


def contract() -> str:
    return CONTRACT.read_text(encoding="utf-8")


def tokens() -> dict[str, Any]:
    return json.loads((FIXTURES / "tokens.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- исходники


def test_files_are_small_and_clean(js: str, css: str) -> None:
    assert len(js.splitlines()) <= MAX_LINES
    for text in (js, css):
        assert [hex(ord(c)) for c in text if ord(c) < 32 and c not in "\n\r\t"] == []
    assert js.lstrip().startswith("/*") and "(function () {" in js and '"use strict";' in js


def test_no_external_resources(js: str, css: str) -> None:
    for text in (js, css):
        assert not URL_RE.search(text), URL_RE.findall(text)
        assert "@import" not in text and "googleapis" not in text and "gstatic" not in text
    assert "url(" not in css


def test_api_text_never_becomes_markup(code: str) -> None:
    """Всё из API — только textContent: разметки строкой, eval и таймеров-строк нет вовсе."""
    for bad in (
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
        "eval(",
        "Function(",
    ):
        assert bad not in code, bad
    assert not re.search(r"set(?:Timeout|Interval)\(\s*[\"']", code)


def test_no_storage_and_no_logging(js: str) -> None:
    """Разговор живёт в памяти вкладки, вопрос не пишется никуда (договор §1 «Журнал», §8.1)."""
    for bad in ("localStorage", "sessionStorage", "indexedDB", "document.cookie", "console."):
        assert bad not in js, bad


def test_no_percent_sign(js: str, css: str) -> None:
    """Процентов нет нигде: точку на шкале ставит переменная `--at`, «%» — только единица CSS."""
    assert "%" not in js
    assert re.findall(r"(?<![0-9])%", css) == []
    assert not re.search(r"content:\s*\"[^\"]*%", css)


def test_every_string_passes_content_filter(code: str) -> None:
    strings = {s for s in literals(code) if re.search("[А-Яа-яЁё]", s)}
    assert len(strings) > 40
    bad = {s: check(s).violations for s in strings if not check(s).clean}
    assert bad == {}
    assert [s for s in strings if STOP.search(s)] == []


def test_no_comforting_fake_stages(js: str) -> None:
    """Этапы — только из потока (§3.3): «почти готово» и своих строк этапов в файле нет."""
    assert "почти готово" not in js.lower()
    for stage in ("Проверяю правила", "Сомелье формулирует", "Сверяю имена", "Ищу в каталоге"):
        assert stage not in js, stage


def test_labels_are_exactly_contract(code: str) -> None:
    text = contract()
    assert f'`"{LABEL_AI}"`' in text and f'`"{LABEL_ALGO}"`' in text
    assert f'var LABEL_AI = "{LABEL_AI}";' in code
    assert f'var LABEL_ALGO = "{LABEL_ALGO}";' in code
    # Метка ИИ — только при generated: true и точной метке сервера; иначе — метка алгоритма.
    assert "t.generated === true && t.label === LABEL_AI" in code
    assert code.count(f'"{LABEL_AI}"') == 1 and code.count(f'"{LABEL_ALGO}"') == 1


def test_sommui_interface_matches_contract(code: str) -> None:
    block = re.search(r"```js\nwindow\.SommUI = \{(.*?)\n\};\n```", contract(), re.DOTALL)
    assert block
    names = re.findall(r"^\s*([a-zA-Z]+)[(:]", block.group(1), re.MULTILINE)
    ours = re.search(r"window\.SommUI = \{(.*?)\n  \};", code, re.DOTALL)
    assert ours
    keys = re.findall(r"^\s*([a-zA-Z]+):", ours.group(1), re.MULTILINE)
    assert keys == names
    assert "version: 1," in ours.group(1)
    host = re.search(r"```js\n\{\n(.*?)\n\}\n```", contract(), re.DOTALL)
    assert host
    callbacks = re.findall(r"^\s*([a-zA-Z]+):", host.group(1), re.MULTILINE)
    ours_host = re.search(r"var host = \{(.*?)\n  \};", code, re.DOTALL)
    assert ours_host
    assert re.findall(r"^    ([a-zA-Z]+):", ours_host.group(1), re.MULTILINE) == callbacks


def test_api_paths_follow_contract(code: str) -> None:
    text = contract()
    paths = {s for s in literals(code) if s.startswith("/v1/")}
    assert paths == {"/v1/sommelier/ask?order=", "/v1/wines/"}
    assert '"/sommelier?order="' in code
    assert "## 3. `POST /v1/sommelier/ask?order=reco|plain`" in text
    assert "## 2. `GET /v1/wines/{slug}/sommelier?order=reco|plain`" in text
    # Тело вопроса — ровно поля §3.1: slug и одно из question / chip (+ args), context как есть.
    for field in ("body.question", "body.chip", "body.args", "body.context"):
        assert field in code, field
    assert '"Content-Type": "application/json"' in code and 'Accept: "application/x-ndjson"' in code


def test_stream_is_read_line_by_line_and_can_be_cut(code: str) -> None:
    for part in (
        "getReader()",
        'new TextDecoder("utf-8")',
        "{ stream: true }",
        'split("\\n")',
        "new AbortController()",
        "signal: ctrl.signal",
        "ctrl.abort()",
        '"pagehide"',
    ):
        assert part in code, part
    for event in EVENTS:
        assert f'event.type === "{event}"' in code, event
        assert f"`{event}`" in contract()


def test_question_limit_matches_contract(code: str) -> None:
    assert "до 200 символов" in contract()
    assert "var MAX_QUESTION = 200;" in code
    assert "maxlength: MAX_QUESTION" in code and ".slice(0, MAX_QUESTION)" in code


def test_caret_and_radar_numbers_follow_contract(code: str) -> None:
    text = contract()
    assert f"var CARET_MS = {tokens()['caret_ms_per_char']};" in code
    assert "18 мс на символ" in text
    assert "var RW = 260, RH = 220, RCX = 130, RCY = 110, RR = 66, RMIN = 0.14;" in code
    assert "`viewBox 0 0 260 220`, центр (130, 110), радиус 66" in text
    assert "R * max(value, 0.14)" in text
    order = re.search(r"var AXES = \[(.*?)\];", code)
    assert order
    axes = re.findall(r'"([a-z_]+)"', order.group(1))
    table = re.findall(r"^\| `([a-z_]+)` \| [А-Я]", text, re.MULTILINE)
    assert axes == table[:8]


def test_reduced_motion_is_respected(code: str, css: str) -> None:
    assert 'matchMedia("(prefers-reduced-motion: reduce)")' in code
    block = css[css.index("@media (prefers-reduced-motion: reduce)") :]
    for cls in (".somm-sheet", ".somm-caret", ".somm-stage--now", ".somm-stage__dot"):
        assert cls in block, cls
    assert "animation: none" in block and "scroll-behavior: auto" in block


def test_css_only_contract_tokens(css: str) -> None:
    """Цвета, шрифты, радиусы и темп — только токены §8.2 (и своя `--at` для точки на шкале)."""
    table = tokens()
    known = {
        name
        for group in ("colors", "gradients", "fonts", "layout", "motion")
        for name in table[group]
    }
    used = set(re.findall(r"var\((--[a-z0-9-]+)", css))
    assert used - known == {"--at"}
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", css)
    assert "rgb" not in css and "hsl" not in css


def test_original_point_rings_the_suggested_one(css: str) -> None:
    """Исходное вино на шкале — серое кольцо шире точки предложенного, без заливки.

    Значения совпали («Помягче» по другой оси) — видно оба: точка в кольце, как серая полоса
    под контуром на «розе ветров».
    """

    def rule(selector: str) -> str:
        found = re.search(re.escape(selector) + r" \{([^}]*)\}", css)
        assert found, selector
        return found.group(1)

    def width(selector: str) -> int:
        return int(re.search(r"width: (\d+)px", rule(selector)).group(1))

    assert "background: transparent" in rule(".somm-point--orig")
    assert width(".somm-point--orig") > width(".somm-point")
    mini = ".somm-track--mini .somm-point"
    assert width(mini + "--orig") > width(mini)
    assert css.index(".somm-track--mini .somm-point {") < css.index(mini + "--orig {")


def test_style_marks_match_grape_and_thumb_is_visible(css: str) -> None:
    """«По стилю» рисуется как «по сорту» (договор §8.1): полая точка, то же кольцо легенды и
    пунктир 4 3 у «розы» — у одного вина бывает только одна из двух оценок, различает подпись.

    Силуэт в шапке листа (28×42 на светлом фоне) заметнее, чем в плитке: рамка стекла темнее,
    заливка прозрачнее.
    """
    for part in ("point", "legend__mark", "radar__edge", "radar__dot"):
        assert re.search(rf"\.somm-{part}--grape, \.somm-{part}--style \{{", css), part
    assert ".somm-radar__edge--grape, .somm-radar__edge--style { stroke-dasharray: 4 3; }" in css
    thumb = re.search(r"\.somm-thumb \.somm-bottle__glass \{([^}]*)\}", css)
    assert thumb and "stroke-opacity: .8" in thumb.group(1) and "fill-opacity: .4" in thumb.group(1)


def test_sheet_switch_is_a_finger_target(css: str) -> None:
    """Переключатель плашки листа — путь назад к рекомендациям: цель касания не ниже 32px, как у
    переключателя плашки карточки (index.html, `.switch`)."""
    switch = re.search(r"\.somm-switch \{([^}]*)\}", css)
    assert switch and "min-height: 32px" in switch.group(1)
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    card = re.search(r"\.switch \{([^}]*)\}", page)
    assert card and "min-height: 32px" in card.group(1)


def test_css_font_sizes_follow_type_scale(css: str) -> None:
    sizes = {int(n) for n in re.findall(r"font(?:-size)?:[^;]*?\b(\d+)px", css)}
    assert sizes and sizes <= TYPE_SCALE, sizes - TYPE_SCALE


def test_everything_is_prefixed(code: str, css: str) -> None:
    """Классы листа — только `.somm-`: классы страницы (`.sheet`, `.chip`…) не трогаются (§8.2)."""
    selectors = set(
        re.findall(r"\.([a-zA-Z][\w-]*)", re.sub(r"/\*.*?\*/|\d*\.\d+", "", css, flags=re.DOTALL))
    )
    assert selectors and all(name.startswith("somm-") for name in selectors), selectors
    classes: list[str] = []
    classes += re.findall(r'\bh\("[a-z]+", "([^"]+)"', code)
    classes += re.findall(r'button\("([^"]+)"', code)
    classes += re.findall(r'"class": "([^"]+)"', code)
    classes += re.findall(r'className = "([^"]+)"', code)
    assert len(classes) > 40
    for value in classes:
        assert all(part.startswith("somm-") for part in value.split()), value


def test_notice_149_and_harm_footer(code: str) -> None:
    """Одна плашка 149-ФЗ на лист с «Обычной сортировкой», дальше — значок (i); подвал 18+."""
    assert '"Обычная сортировка"' in code and "sheet.notice = true;" in code
    assert "infoMark(note149)" in code and 'role: "switch"' in code
    assert '"Чрезмерное употребление алкоголя вредит вашему здоровью. 18+"' in code


# ---------------------------------------------------------------- поведение на Node

HARNESS = r"""
"use strict";
const fs = require("fs");
const path = require("path");
const [, , SRC, FIX, MODE] = process.argv;
const out = {};

class Text { constructor(v) { this.nodeValue = String(v); this.parentNode = null; this.childNodes = []; }
  get textContent() { return this.nodeValue; } }
class El {
  constructor(tag, ns) {
    this.localName = tag; this.ns = ns || "html"; this.childNodes = []; this.parentNode = null;
    this.attributes = {}; this.listeners = {}; this.className = ""; this.hidden = false; this.value = "";
    this.checked = false; this.scrollTop = 0; this.offsetTop = 0;
    const props = {}; this.style = { props, setProperty(k, v) { props[k] = v; } };
  }
  get firstChild() { return this.childNodes[0] || null; }
  get lastChild() { return this.childNodes[this.childNodes.length - 1] || null; }
  appendChild(c) { if (c.parentNode) c.parentNode.removeChild(c); c.parentNode = this; this.childNodes.push(c); return c; }
  insertBefore(c, ref) {
    if (!ref) return this.appendChild(c);
    if (c.parentNode) c.parentNode.removeChild(c);
    c.parentNode = this; this.childNodes.splice(this.childNodes.indexOf(ref), 0, c); return c;
  }
  removeChild(c) { const i = this.childNodes.indexOf(c); if (i < 0) throw new Error("not a child"); this.childNodes.splice(i, 1); c.parentNode = null; return c; }
  replaceChild(n, o) { if (n.parentNode) n.parentNode.removeChild(n); const i = this.childNodes.indexOf(o); this.childNodes[i] = n; n.parentNode = this; o.parentNode = null; return o; }
  get textContent() { return this.childNodes.map((c) => c.textContent).join(""); }
  set textContent(v) { this.childNodes.forEach((c) => { c.parentNode = null; }); this.childNodes = []; if (String(v)) this.appendChild(new Text(v)); }
  setAttribute(k, v) { if (k === "class") this.className = String(v); else this.attributes[k] = String(v); }
  getAttribute(k) { return k === "class" ? this.className : (k in this.attributes ? this.attributes[k] : null); }
  removeAttribute(k) { delete this.attributes[k]; }
  get classList() {
    const el = this; const set = () => new Set(el.className.split(/\s+/).filter(Boolean));
    return {
      add(c) { const s = set(); s.add(c); el.className = [...s].join(" "); },
      remove(c) { const s = set(); s.delete(c); el.className = [...s].join(" "); },
      toggle(c, on) { const s = set(); if (on === undefined ? !s.has(c) : on) s.add(c); else s.delete(c); el.className = [...s].join(" "); },
      contains(c) { return set().has(c); }
    };
  }
  addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }
  removeEventListener(t, f) { this.listeners[t] = (this.listeners[t] || []).filter((g) => g !== f); }
  fire(t, extra) { (this.listeners[t] || []).slice().forEach((f) => f(Object.assign({ preventDefault() {} }, extra || {}))); }
  focus() {}
  all(cls) {
    const found = [];
    const walk = (n) => n.childNodes.forEach((c) => { if (c instanceof El) { if (c.classList.contains(cls)) found.push(c); walk(c); } });
    walk(this); return found;
  }
}
const docListeners = {};
global.window = global;
if (MODE !== "bare") {
  global.document = {
    body: new El("body"), activeElement: null,
    createElement: (t) => new El(t), createElementNS: (ns, t) => new El(t, ns), createTextNode: (v) => new Text(v),
    addEventListener(t, f) { (docListeners[t] = docListeners[t] || []).push(f); },
    removeEventListener(t, f) { docListeners[t] = (docListeners[t] || []).filter((g) => g !== f); }
  };
  global.DOMParser = class { parseFromString() { return { body: { firstChild: { namespaceURI: "svg-ns" } } }; } };
}
let reduced = true;
global.matchMedia = () => ({ matches: reduced });
const winListeners = {};
global.addEventListener = (t, f) => { (winListeners[t] = winListeners[t] || []).push(f); };
eval(fs.readFileSync(SRC, "utf8"));
out.keys = Object.keys(window.SommUI);
out.version = window.SommUI.version;
if (MODE === "bare") { console.log(JSON.stringify(out)); process.exit(0); }

const U = window.SommUI;
const load = (name) => JSON.parse(fs.readFileSync(path.join(FIX, name), "utf8"));
// Мускатель — как до оценки по стилю: оси `style` пустые (заглушка бывает в обоих видах, а
// «по стилю» проверяется отдельно, на профилях ниже).
const legacy = (d) => Object.assign({}, d, { profile: d.profile.map((a) => (a.source === "style"
  ? Object.assign({}, a, { value: null, source: null }) : a)) });
const card = { red: load("sommelier_red.json"), sweet: legacy(load("sommelier_sweet.json")),
  sparkling: load("sommelier_sparkling.json"), nosugar: load("sommelier_nosugar.json") };
const texts = (el) => el.all("somm-radar__label").map((t) => t.textContent);

// «Роза ветров» и шкалы.
out.radar = {};
for (const [key, data] of Object.entries(card)) {
  const el = new El("div");
  const ok = U.radar(el, data.profile);
  const svg = el.firstChild;
  out.radar[key] = { ok, dots: el.all("somm-radar__dot").length, grape: el.all("somm-radar__dot--grape").length,
    edges: el.all("somm-radar__edge").length, dashed: el.all("somm-radar__edge--grape").length,
    labels: texts(el), aria: svg ? svg.getAttribute("aria-label") : null, role: svg ? svg.getAttribute("role") : null };
}
const two = card.red.profile.map((a, i) => (i < 2 ? a : Object.assign({}, a, { value: null, source: null })));
const twoEl = new El("div");
out.radarTwo = { ok: U.radar(twoEl, two), kids: twoEl.childNodes.length };
const half = card.red.profile.map((a) => (a.axis === "tannin" ? Object.assign({}, a, { value: null, source: null }) : a));
const cmpEl = new El("div");
U.radar(cmpEl, card.nosugar.profile, { profile: half, name: "якорь" });
out.radarCompare = { labels: texts(cmpEl), orig: cmpEl.all("somm-radar__edge--orig").length,
  aria: cmpEl.firstChild.getAttribute("aria-label") };
out.scales = {};
for (const [key, data] of Object.entries(card)) {
  const el = new El("div");
  const ok = U.scales(el, data.profile);
  out.scales[key] = { ok, rows: el.all("somm-scale").map((r) => r.all("somm-scale__name")[0].textContent),
    at: el.all("somm-point").map((p) => Number(p.style.props["--at"])), text: el.textContent,
    hollow: el.all("somm-point--grape").length };
}
// Карточка рисует одну общую легенду сама: у шкал её нет.
const bare = new El("div");
U.scales(bare, card.red.profile, { legend: false });
out.scalesBare = { rows: bare.all("somm-scale").length, legend: bare.all("somm-legend").length };
// Заметка и вход.
const noteEl = new El("div");
out.note = { ok: U.note(noteEl, card.red), text: noteEl.textContent,
  verdict: noteEl.all("somm-verdict").map((v) => v.textContent), label: noteEl.all("somm-label").map((l) => l.textContent),
  rows: noteEl.all("somm-note__chips").length, chips: noteEl.all("somm-chip").map((c) => c.textContent) };
const noNotes = new El("div");
U.note(noNotes, Object.assign({}, card.red, { chips: card.red.chips.filter((c) => c.id !== "softer" && c.id !== "fresher") }));
out.noteNoWays = noNotes.all("somm-note__chips").length;
// «По стилю»: сорта нет в приорах — оси сорта оценены по цвету и сахару (источник style).
const styled = card.nosugar.profile.map((a) => (a.source === "grape" ? Object.assign({}, a, { source: "style" }) : a));
const mixed = card.red.profile.map((a) => (a.axis === "oak" || a.axis === "aroma_intensity" ? Object.assign({}, a, { source: "style" }) : a));
const marksOf = (el) => el.all("somm-legend__item").map((i) => ({ text: i.textContent,
  marks: i.all("somm-legend__mark").map((m) => m.className.split("--")[1]) }));
const radarKinds = (el) => ({ style: el.all("somm-radar__dot--style").length, grape: el.all("somm-radar__dot--grape").length,
  dotted: el.all("somm-radar__edge--style").length, dashed: el.all("somm-radar__edge--grape").length,
  edges: el.all("somm-radar__edge").length, aria: el.firstChild ? el.firstChild.getAttribute("aria-label") : null });
const styleRadar = new El("div"), mixedRadar = new El("div"), styleScales = new El("div"), mixedScales = new El("div");
out.style = { ok: U.radar(styleRadar, styled), radar: radarKinds(styleRadar) };
U.radar(mixedRadar, mixed);
out.style.mixed = radarKinds(mixedRadar);
U.scales(styleScales, styled);
U.scales(mixedScales, mixed, { axes: ["sweetness", "tannin", "oak"] });
out.style.scales = { src: styleScales.all("somm-scale__src").map((s) => s.textContent),
  hollow: styleScales.all("somm-point--style").length, legend: marksOf(styleScales) };
out.style.mixedScales = { src: mixedScales.all("somm-scale__src").map((s) => s.textContent), legend: marksOf(mixedScales) };
const badNote = new El("div");
out.noteEmpty = { ok: U.note(badNote, {}), hidden: badNote.hidden };
const entryEl = new El("div");
U.entry(entryEl, card.red);
const entryOff = new El("div");
U.entry(entryOff, Object.assign({}, card.red, { input: false }));
out.entry = { chips: entryEl.all("somm-chip").map((c) => c.textContent), input: entryEl.all("somm-ask__input").length,
  inputOff: entryOff.all("somm-ask__input").length };
// Карточка: заметка с рядом направлений, ниже «Спросить сомелье» того же вина — без повтора ряда;
// у другого вина и после заметки без ряда — все чипы.
const notedEl = new El("div"), entryNoted = new El("div"), entryOther = new El("div"), entryPlain = new El("div");
U.note(notedEl, card.red);
U.entry(entryNoted, card.red);
U.entry(entryOther, Object.assign({}, card.red, { slug: "other-wine" }));
U.note(new El("div"), Object.assign({}, card.red, { chips: card.red.chips.filter((c) => c.id !== "softer" && c.id !== "fresher") }));
U.entry(entryPlain, card.red);
out.entryNoted = { noted: entryNoted.all("somm-chip").map((c) => c.textContent),
  other: entryOther.all("somm-chip").length, plain: entryPlain.all("somm-chip").length };

// Поток: заглушки NDJSON режутся по 7 байт — посреди русских букв.
const calls = [];
const queue = [];
let hang = null;
global.fetch = (url, init) => {
  init = init || {};
  calls.push({ url, method: init.method || "GET", body: init.body ? JSON.parse(init.body) : null });
  if (!init.method) return Promise.resolve(new Response(JSON.stringify(card.red), { status: 200 }));
  const next = queue.shift();
  if (next && next.status) return Promise.resolve(new Response(JSON.stringify(next.body), { status: next.status }));
  const bytes = new TextEncoder().encode(next.text);
  const stream = new ReadableStream({
    start(ctrl) {
      if (init.signal) init.signal.addEventListener("abort", () => { try { ctrl.error(new Error("AbortError")); } catch (e) {} });
      const upto = next.hangAfter ? next.text.indexOf("\n", next.text.indexOf('"facts"')) + 1 : bytes.length;
      const cut = next.hangAfter ? new TextEncoder().encode(next.text.slice(0, upto)).length : bytes.length;
      for (let i = 0; i < cut; i += 7) ctrl.enqueue(bytes.slice(i, Math.min(i + 7, cut)));
      if (!next.hangAfter) ctrl.close(); else hang = init.signal;
    }
  });
  return Promise.resolve(new Response(stream, { status: 200, headers: { "Content-Type": "application/x-ndjson" } }));
};
const nd = (name) => fs.readFileSync(path.join(FIX, name), "utf8");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const sheetState = [];
let order = "reco";
U.init({ order: () => order, setOrder: (o) => { order = o; }, onSheet: (open) => sheetState.push(open),
  openWine: (slug) => sheetState.push("wine:" + slug), scan: () => sheetState.push("scan") });
const wrap = () => document.body.childNodes.find((n) => n.className === "somm-wrap");
const turns = () => wrap().all("somm-turn");
const snap = (turn) => ({ text: turn.textContent, labels: turn.all("somm-label").map((l) => l.textContent),
  now: turn.all("somm-stage--now").length, done: turn.all("somm-stage--done").map((s) => s.textContent),
  chips: turn.all("somm-turn__tail")[0].all("somm-chip").map((c) => c.textContent),
  notice: turn.all("somm-notice").length, info: turn.all("somm-info").length, tiles: turn.all("somm-tile").length,
  minis: turn.all("somm-mini").length, rules: turn.all("somm-rule").map((r) => r.textContent),
  heads: turn.all("somm-dish__verdict").map((v) => v.textContent), srcs: turn.all("somm-rule__src").length,
  caption: turn.all("somm-rules__src").map((c) => c.textContent),
  verdict: (turn.all("somm-verdict")[0] || { textContent: null }).textContent });

// Фокус в модальном листе: Tab и Shift+Tab с краёв листа не уходят на страницу под затемнением.
if (MODE === "focus") {
  El.prototype.focus = function () { document.activeElement = this; };
  El.prototype.getClientRects = function () { return this.hidden ? [] : [{}]; };
  El.prototype.contains = function (n) { for (; n; n = n.parentNode) if (n === this) return true; return false; };
  El.prototype.querySelectorAll = function (sel) {
    const tags = sel.split(",").map((s) => s.trim()); const found = [];
    const walk = (n) => n.childNodes.forEach((c) => { if (c instanceof El) { if (tags.includes(c.localName)) found.push(c); walk(c); } });
    walk(this); return found;
  };
  const tab = (shiftKey) => {
    let stop = false;
    (docListeners.keydown || []).forEach((f) => f({ key: "Tab", shiftKey, preventDefault() { stop = true; } }));
    return { stop, at: document.activeElement.className };
  };
  U.open(card.red.slug, { name: card.red.name, data: card.red });
  const items = wrap().querySelectorAll("button, input");
  const page = document.body.appendChild(new El("button"));
  out.focus = { first: items[0].className, last: items[items.length - 1].className, fromPanel: tab(true) };
  out.focus.fromLast = tab(false);
  out.focus.fromFirst = tab(true);
  items[1].focus();
  out.focus.middle = tab(false);
  page.focus();
  out.focus.fromPage = tab(false);
  U.close();
  out.focus.afterClose = (docListeners.keydown || []).length;
  console.log(JSON.stringify(out));
  process.exit(0);
}

(async () => {
  queue.push({ text: nd("ask_dish_check_live.ndjson") });
  U.open(card.red.slug, { name: card.red.name, data: card.red, question: "  а к борщу\n подойдёт?  " });
  out.openState = { isOpen: U.isOpen(), sheet: sheetState.slice(), inputs: wrap().all("somm-ask__input").length,
    bottle: wrap().all("somm-thumb")[0].all("somm-bottle")[0].childNodes.map((n) => n.className) };
  await sleep(60);
  out.live = snap(turns()[0]);
  out.liveLegend = turns()[0].all("somm-wines__head")[0].all("somm-legend__item").map((i) => ({ text: i.textContent,
    marks: i.all("somm-legend__mark").map((m) => m.className.split("--")[1]) }));
  out.liveCall = calls[calls.length - 1];
  // Уточнение чипом: контекст прошлого ответа уходит как есть, чипы остаются только под последним.
  queue.push({ text: nd("ask_what_to_eat_busy.ndjson") });
  turns()[0].all("somm-chip").find((c) => c.textContent === "А к шашлыку?").fire("click");
  await sleep(60);
  out.chipCall = calls[calls.length - 1];
  out.afterChip = { first: snap(turns()[0]), second: snap(turns()[1]) };
  // Забракованный текст, ошибка сборки, отказ, шаг «Помогите выбрать».
  for (const name of ["ask_dish_check_guard.ndjson", "ask_error.ndjson", "ask_refusal.ndjson", "ask_guided_step.ndjson"]) {
    queue.push({ text: nd(name) });
    const form = wrap().all("somm-ask")[0];
    form.all("somm-ask__input")[0].value = "вопрос " + name;
    form.fire("submit");
    await sleep(60);
    out[name] = snap(turns()[turns().length - 1]);
  }
  out.notices = wrap().all("somm-notice").length;
  out.followBlocks = wrap().all("somm-turn__tail").filter((t) => t.all("somm-chip").length).length;
  // Каретка без reduced motion: короткий живой текст допечатывается и каретка уходит.
  reduced = false;
  const short = nd("ask_dish_check_live.ndjson").replace(/"text": "К борщу — да, с оговоркой: [^"]*"/, '"text": "К борщу — да."');
  queue.push({ text: short });
  wrap().all("somm-chip")[0].fire("click");
  await sleep(40);
  out.typing = { caret: wrap().all("somm-caret").length };
  await sleep(400);
  const last = turns()[turns().length - 1];
  out.typed = { verdict: last.all("somm-verdict")[0].textContent, caret: wrap().all("somm-caret").length,
    ghost: wrap().all("somm-ghost").length, labels: last.all("somm-label").map((l) => l.textContent) };
  reduced = true;
  // Обрыв: поток встал после facts, лист закрыли — запрос оборван, этап без галочки снят.
  queue.push({ text: nd("ask_dish_check_live.ndjson"), hangAfter: true });
  wrap().all("somm-chip")[0].fire("click");
  await sleep(40);
  const hung = turns()[turns().length - 1];
  out.hung = { now: hung.all("somm-stage--now").length };
  U.close();
  await sleep(20);
  out.closed = { aborted: Boolean(hang && hang.aborted), isOpen: U.isOpen(), sheet: sheetState.slice(),
    inBody: Boolean(wrap()), now: hung.all("somm-stage--now").length, verdict: snap(hung).verdict,
    labels: snap(hung).labels };
  // Снова открыли: уточнения под последним ходом вернулись; 422 «вопрос текстом выключен» прячет поле.
  U.open(card.red.slug, { name: card.red.name });
  out.reopen = { chips: snap(turns()[turns().length - 1]).chips.length, calls: calls.length };
  queue.push({ status: 422, body: { detail: "вопрос текстом выключен" } });
  const form = wrap().all("somm-ask")[0];
  form.all("somm-ask__input")[0].value = "ещё вопрос";
  form.fire("submit");
  await sleep(40);
  out.inputOff = { inputs: wrap().all("somm-ask__input").length, camera: wrap().all("somm-camera").length,
    text: snap(turns()[turns().length - 1]).text };
  // Плитка открывает карточку, камера — новый скан; оба сначала прячут лист без записи истории.
  wrap().all("somm-camera")[0].fire("click");
  out.scan = { isOpen: U.isOpen(), sheet: sheetState.slice() };
  order = "plain";
  queue.push({ text: nd("ask_softer_plain.ndjson") });
  U.open(card.red.slug, { name: card.red.name, chip: "softer" });
  await sleep(60);
  out.plainCall = calls[calls.length - 1];
  const plain = turns()[turns().length - 1];
  out.plain = snap(plain);
  out.plainSwitch = wrap().all("somm-switch__input").map((s) => s.checked);
  out.plainNotices = wrap().all("somm-notice").map((n) => n.textContent);
  plain.all("somm-tile")[0].fire("click");
  out.tile = { isOpen: U.isOpen(), sheet: sheetState.slice() };
  // Уход со страницы посреди ответа: поток оборван, незаконченный этап снят, а уточнения
  // остались — страница может вернуться из кэша «назад/вперёд» с открытым листом.
  order = "reco";
  queue.push({ text: nd("ask_dish_check_live.ndjson"), hangAfter: true });
  U.open(card.red.slug, { name: card.red.name, chip: "what_to_eat" });
  await sleep(40);
  (winListeners.pagehide || []).forEach((f) => f({}));
  await sleep(20);
  const left = turns()[turns().length - 1];
  out.pagehide = { aborted: Boolean(hang && hang.aborted), now: left.all("somm-stage--now").length,
    chips: snap(left).chips.length, labels: snap(left).labels };
  // «Чем заменить»: мини-шкал нет — «роза ветров» первого вина поверх исходного и легенда
  // только нарисованных меток. Ответ без поля compare.overlay — первое вино подборки.
  const replace = nd("ask_dish_check_live.ndjson").split("\n").filter(Boolean).map((line) => {
    const e = JSON.parse(line);
    if (e.type === "facts") {
      Object.assign(e, { intent: "replace", wines_title: "Чем заменить" });
      e.compare.axes = [];
      delete e.compare.overlay;
    }
    return JSON.stringify(e);
  }).join("\n") + "\n";
  queue.push({ text: replace });
  U.open(card.red.slug, { name: card.red.name, chip: "replace" });
  await sleep(60);
  const swap = turns()[turns().length - 1];
  const fig = swap.all("somm-overlay")[0];
  out.replace = { minis: swap.all("somm-mini").length, overlay: Boolean(fig),
    caption: fig ? fig.all("somm-eyebrow")[0].textContent : null,
    aria: fig ? fig.all("somm-radar__svg")[0].getAttribute("aria-label") : null,
    orig: fig ? fig.all("somm-radar__edge--orig").length : 0,
    legend: fig ? fig.all("somm-legend__item").map((i) => ({ text: i.textContent,
      marks: i.all("somm-legend__mark").map((m) => m.className.split("--")[1]) })) : [] };
  // compare.overlay (договор §1): поверх — вино, которое выбрал сервис; null — все совпали с
  // исходным, наложения нет.
  out.overlayPick = [];
  for (const pick of ["belbek-pino-nuar-krasnoe-suhoe-133", null]) {
    queue.push({ text: replace.split("\n").filter(Boolean).map((line) => {
      const e = JSON.parse(line);
      if (e.type === "facts") e.compare.overlay = pick;
      return JSON.stringify(e);
    }).join("\n") + "\n" });
    U.open(card.red.slug, { name: card.red.name, chip: "replace" });
    await sleep(60);
    out.overlayPick.push(turns()[turns().length - 1].all("somm-overlay").map((f) => f.all("somm-legend__item")[1].textContent));
  }
  // Одно блюдо, у всех чипов один источник: пометка — раз, строкой под чипами.
  const shared = nd("ask_dish_check_guard.ndjson").split("\n").filter(Boolean).map((line) => {
    const e = JSON.parse(line);
    if (e.type === "facts") e.dishes.forEach((d) => d.plus.concat(d.minus).forEach((r) => { r.source = "grape"; }));
    return JSON.stringify(e);
  }).join("\n") + "\n";
  queue.push({ text: shared });
  U.open(card.red.slug, { name: card.red.name, chip: "dish_check", args: { dish: "borsch" } });
  await sleep(60);
  out.shared = snap(turns()[turns().length - 1]);
  // Ряд направлений под заметкой: «Помягче» открывает лист этого вина с чипом. Лист новый, а
  // сортировка обычная: плашка одна — «Обычная сортировка: по названию» с включённым
  // переключателем; выключили — та же подборка уже с рекомендациями, плашка не удваивается.
  order = "plain";
  const noteSp = new El("div");
  U.note(noteSp, card.sparkling);
  queue.push({ text: nd("ask_softer_plain.ndjson") });
  noteSp.all("somm-chip").find((c) => c.textContent === "Помягче").fire("click");
  await sleep(60);
  const plainFirst = turns()[turns().length - 1];
  const sw = plainFirst.all("somm-switch__input")[0];
  out.notePlain = { call: calls[calls.length - 1], slug: wrap().all("somm-who__wine")[0].textContent,
    notices: wrap().all("somm-notice").map((n) => n.textContent), checked: sw ? sw.checked : null, turns: turns().length };
  queue.push({ text: nd("ask_dish_check_live.ndjson") });
  sw.checked = false;
  sw.fire("change");
  await sleep(60);
  out.backToReco = { order, call: calls[calls.length - 1], notices: wrap().all("somm-notice").map((n) => n.textContent),
    checked: sw.checked, turns: turns().length, info: snap(turns()[turns().length - 1]).info };
  console.log(JSON.stringify(out));
})().catch((e) => { console.error(e && e.stack || e); process.exit(1); });
"""


@pytest.fixture(scope="module")
def node() -> str:
    exe = shutil.which("node")
    if not exe:
        pytest.skip("Node нет на этой машине")
    return exe


def run_harness(node: str, tmp: Path, mode: str) -> dict[str, Any]:
    harness = tmp / "harness.js"
    harness.write_text(HARNESS, encoding="utf-8")
    done = subprocess.run(
        [node, str(harness), str(JS), str(FIXTURES), mode],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def run(node: str, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return run_harness(node, tmp_path_factory.mktemp("somm"), "full")


def test_load_only_declares_sommui(node: str, tmp_path: Path) -> None:
    """При загрузке файл в DOM не лезет: без `document` он только объявляет window.SommUI."""
    out = run_harness(node, tmp_path, "bare")
    assert out["keys"] == [
        "version",
        "init",
        "note",
        "radar",
        "scales",
        "entry",
        "open",
        "close",
        "isOpen",
        "say",
    ]
    assert out["version"] == 1


def test_radar_draws_only_known_axes(run: dict[str, Any]) -> None:
    radar = run["radar"]
    red = radar["red"]
    assert red["ok"] and red["role"] == "img" and red["dots"] == 8 and red["grape"] == 5
    assert red["labels"] == [
        "Сладость",
        "Кислотность",
        "Танины",
        "Тело",
        "Крепость",
        "Дуб",
        "Аромат",
        "Пузырьки",
    ]
    # Отрезок пунктирный, если касается вершины «по сорту»: у Cru Lermont сплошной один —
    # «Пузырьки — Сладость», обе из карточки.
    assert red["edges"] == 8 and red["dashed"] == 7
    assert red["aria"].startswith("Профиль вкуса: ") and not re.search(r"\d", red["aria"])
    # Мускатель: сорта нет в приорах — пять осей не рисуются вовсе, остаются три из карточки.
    sweet = radar["sweet"]
    assert sweet["ok"] and sweet["dots"] == 3 and sweet["grape"] == 0 and sweet["dashed"] == 0
    assert sweet["labels"] == ["Сладость", "Крепость", "Пузырьки"]
    assert radar["nosugar"]["dots"] == 6 and "Сладость" not in radar["nosugar"]["labels"]
    assert run["radarTwo"] == {"ok": False, "kids": 0}
    # Сравнение: ось, где у одного из вин null, не рисуется у обоих; якорь — контур --orig.
    compare = run["radarCompare"]
    assert "Танины" not in compare["labels"] and "Сладость" not in compare["labels"]
    assert compare["orig"] == len(compare["labels"])
    # Подпись для скринридера называет и якорь: «розу» сравнивают с ним.
    assert compare["aria"].endswith("; серым контуром для сравнения — якорь")
    assert not re.search(r"\d", compare["aria"])


def test_scales_default_axes_and_sources(run: dict[str, Any]) -> None:
    scales = run["scales"]
    assert scales["red"]["rows"] == ["Сладость", "Крепость", "Кислотность", "Танины", "Тело"]
    assert scales["red"]["hollow"] == 3
    assert scales["sparkling"]["rows"][-1] == "Пузырьки" and len(scales["sparkling"]["rows"]) == 6
    assert scales["nosugar"]["rows"] == ["Кислотность", "Танины", "Тело"]
    assert scales["sweet"]["rows"] == ["Сладость", "Крепость"]
    # Легенда — только нарисованные виды точек: у Мускателя «по сорту» нет.
    assert "по сорту — типично" not in scales["sweet"]["text"]
    assert "по сорту — типично" in scales["red"]["text"]
    for data in scales.values():
        assert all(0 <= at <= 1 for at in data["at"]) and "%" not in data["text"]
    # `legend: false` — шкалы без легенды: карточка рисует одну общую на «розу» и шкалы.
    assert run["scalesBare"] == {"rows": 5, "legend": 0}


def test_style_source_is_hollow_and_named(run: dict[str, Any]) -> None:
    """Источник оси `style` — оценка по винам того же стиля, когда сорта нет в приорах.

    Рисуется как «по сорту» (договор §8.1): полая точка и пунктир отрезков к ней; класс отрезка
    — по более слабой вершине (каталог < сорт < стиль). Подпись шкалы и легенда — «по стилю», в
    `aria-label` — тоже словами.
    """
    style = run["style"]
    radar = style["radar"]
    assert style["ok"] and radar["style"] == 5 and radar["grape"] == 0
    assert radar["edges"] == 6 and radar["dotted"] == 6 and radar["dashed"] == 0
    assert ", по стилю" in radar["aria"] and "по сорту" not in radar["aria"]
    # Красное с дубом и ароматом «по стилю»: пунктир — только между вершинами сорта и каталога.
    mixed = style["mixed"]
    assert mixed["style"] == 2 and mixed["grape"] == 3
    assert mixed["dotted"] == 3 and mixed["dashed"] == 4 and mixed["edges"] == 8
    assert style["scales"]["src"] == ["по стилю"] * 3 and style["scales"]["hollow"] == 3
    legend_style = "по стилю — типично для стиля, не замер этого вина"
    assert style["scales"]["legend"] == [{"text": legend_style, "marks": ["style"]}]
    assert style["mixedScales"]["src"] == ["из карточки", "по сорту", "по стилю"]
    assert [item["marks"] for item in style["mixedScales"]["legend"]] == [
        ["catalog"],
        ["grape"],
        ["style"],
    ]


def test_dish_source_is_written_once(run: dict[str, Any]) -> None:
    """У всех чипов блюда один источник — он пишется раз, а не «· по сорту» у каждого чипа.

    Список блюд — в строке блюда рядом с вердиктом; одно блюдо ответа — строкой под чипами.
    Источники разные (у «Борща» есть чип каталога) — пометка у каждого чипа, как раньше.
    """
    what = run["afterChip"]["second"]
    assert what["heads"] == [
        "подходит · по сорту",
        "подходит с оговоркой · по сорту",
        "подходит с оговоркой · по сорту",
    ]
    assert what["srcs"] == 0 and not [r for r in what["rules"] if "по сорту" in r]
    shared = run["shared"]
    assert shared["caption"] == ["Правила сочетаний — по сорту"] and shared["srcs"] == 0
    live = run["live"]
    assert live["caption"] == [] and live["srcs"] == 3
    assert live["rules"][0] == "+ Равны по силе вкуса · по сорту"


def test_note_row_opens_sheet_and_plain_switch_returns_reco(run: dict[str, Any]) -> None:
    """Чип ряда под заметкой открывает лист с этим чипом; при обычной сортировке в листе своя
    плашка с включённым переключателем — выключили, и та же подборка идёт с рекомендациями.

    Плашка одна на лист: после переключения новой не появляется, у второй подборки — значок (i).
    """
    plain = run["notePlain"]
    assert plain["call"]["url"] == "/v1/sommelier/ask?order=plain"
    assert plain["call"]["body"] == {
        "slug": "fanagoriya-alveus-ultra-cuvee-brut-shardone-beloe-bryut-12",
        "chip": "softer",
    }
    assert plain["slug"].startswith("о Alveus") and plain["turns"] == 1
    assert plain["notices"] == [
        "Обычная сортировка: по названию, без подбора" + "Обычная сортировка"
    ]
    assert plain["checked"] is True
    back = run["backToReco"]
    assert back["order"] == "reco" and back["checked"] is False
    assert back["call"]["url"] == "/v1/sommelier/ask?order=reco"
    assert back["call"]["body"]["chip"] == "softer" and back["turns"] == 2
    # Текст плашки — о подборке под ней (она по названию), переключатель — о сортировке листа.
    assert back["notices"] == [
        "Обычная сортировка: по названию, без подбора" + "Обычная сортировка"
    ]
    assert back["info"] == 1


def test_note_and_entry(run: dict[str, Any]) -> None:
    red = json.loads((FIXTURES / "sommelier_red.json").read_text(encoding="utf-8"))
    note = run["note"]
    # «16–18 °C» не рвётся в узкой ленте, тире не начинает строку: пробел перед градусами и перед
    # тире неразрывный, текст тот же (у реплик листа — так же, ниже «К борщу — да»).
    tight = re.sub(r"(\d)–(\d)", "\\1\u2060–\u2060\\2", red["note"]["text"])
    assert note["ok"] and note["verdict"] == [
        tight.replace(" °C", "\u00a0°C").replace(" —", "\u00a0—")
    ]
    assert "16\u2060–\u206018\u00a0°C" in note["verdict"][0]
    assert note["label"] == [LABEL_ALGO] and LABEL_AI not in note["text"]
    # Под заметкой — короткий ряд направлений из чипов ответа: «помягче» одним касанием.
    assert note["rows"] == 1 and note["chips"] == ["Помягче", "Посвежее"]
    assert note["text"].index(LABEL_ALGO) < note["text"].index("Помягче")
    # Направлений в чипах нет (голос без них) — нет и ряда.
    assert run["noteNoWays"] == 0
    assert run["noteEmpty"] == {"ok": False, "hidden": True}
    assert run["entry"]["chips"] == [c["text"] for c in red["chips"]]
    assert run["entry"]["input"] == 1 and run["entry"]["inputOff"] == 0
    # Ряд под заметкой уже дал «Помягче · Посвежее»: во «Спросить сомелье» того же вина их нет.
    noted = run["entryNoted"]
    ways = {"softer", "fresher", "sweeter"}
    assert noted["noted"] == [c["text"] for c in red["chips"] if c["id"] not in ways]
    assert len(noted["noted"]) < len(red["chips"])
    assert noted["other"] == noted["plain"] == len(red["chips"])


def test_live_answer_follows_answer_order(run: dict[str, Any]) -> None:
    assert run["openState"] == {
        "isOpen": True,
        "sheet": [True],
        "inputs": 1,
        # Фото в шапке нет — силуэт: стекло, колпачок, этикетка и блик, а не битая картинка.
        "bottle": [
            "somm-bottle__glass",
            "somm-bottle__cap",
            "somm-bottle__label",
            "somm-bottle__shine",
        ],
    }
    call = run["liveCall"]
    assert call["method"] == "POST" and call["url"] == "/v1/sommelier/ask?order=reco"
    # Вопрос уходит очищенным, контекста у первого хода нет.
    assert call["body"] == {
        "slug": "fanagoriya-cru-lermont-saperavi-saperavi-krasnoe-suhoe-135",
        "question": "а к борщу подойдёт?",
    }
    live = run["live"]
    assert live["now"] == 0
    assert live["done"] == [
        "Открыл карточку",
        "Проверил правила сочетаний",
        "Посмотрел каталог",
        "Сомелье сформулировал",
        "Сверил имена и числа",
    ]
    assert live["verdict"].startswith("К борщу\u00a0— да, с оговоркой: вино и борщ")
    assert live["labels"] == [LABEL_AI]
    assert live["notice"] == 1 and live["tiles"] == 3 and live["minis"] == 3
    assert live["rules"][0].startswith("+ Равны по силе вкуса") and live["rules"][-1].startswith(
        "− Бульон"
    )
    assert live["chips"] == ["А к шашлыку?", "Как подать", "Посвежее"]
    # Легенда мини-шкал: исходное вино с винодельней (у предложенных то же «Пино Нуар» и
    # «Саперави» бывает у многих) и полая точка предложенного — танины здесь «по сорту».
    assert run["liveLegend"] == [
        {"text": "Cru Lermont Saperavi · Фанагория — исходное", "marks": ["orig"]},
        {"text": "предложенное", "marks": ["grape"]},
    ]
    text = live["text"]
    # Порядок Answer.tsx: вопрос → этапы → «Коротко» → подборка → уточнения → метка.
    marks = [
        "а к борщу подойдёт?",
        "Открыл карточку",
        "Коротко",
        "Помягче к борщу",
        "А к шашлыку?",
        LABEL_AI,
    ]
    assert [text.index(m) for m in marks] == sorted(text.index(m) for m in marks)


def test_follow_up_chip_sends_context_and_moves_chips(run: dict[str, Any]) -> None:
    body = run["chipCall"]["body"]
    assert body["chip"] == "dish_check" and body["args"] == {"dish": "shashlyk_baranina"}
    assert body["context"]["dish"] == "borsch" and len(body["context"]["shown"]) == 3
    first, second = run["afterChip"]["first"], run["afterChip"]["second"]
    assert first["chips"] == [] and second["chips"] == [
        "А к шашлыку?",
        "Как подать",
        "Чем заменить",
    ]
    assert second["labels"] == [LABEL_ALGO] and second["notice"] == 0
    assert "Бородинский хлеб с салом" in second["text"] and "подходит с оговоркой" in second["text"]


def test_guard_error_refusal_and_guided(run: dict[str, Any]) -> None:
    guard = run["ask_dish_check_guard.ndjson"]
    assert guard["labels"] == [LABEL_ALGO] and guard["verdict"].startswith(
        "К борщу\u00a0— да, с оговоркой."
    )
    # Плашка 149-ФЗ одна на лист: у второй подборки — значок (i).
    assert guard["notice"] == 0 and guard["info"] == 1 and run["notices"] == 1
    error = run["ask_error.ndjson"]
    assert "Не получилось собрать ответ — повторите вопрос" in error["text"]
    assert error["done"] == [] and error["now"] == 0 and error["chips"]
    refusal = run["ask_refusal.ndjson"]
    assert refusal["verdict"].startswith("О покупке и стоимости не рассказываю")
    assert refusal["chips"] == ["Как подать", "К чему подать"] and refusal["tiles"] == 0
    guided = run["ask_guided_step.ndjson"]
    assert guided["verdict"] == "К чему подбираете вино?"
    assert guided["chips"] == ["К мясу", "К рыбе", "К сыру", "К десерту", "Без блюда"]
    assert run["followBlocks"] == 1


def test_caret_types_verified_text_without_layout_jump(run: dict[str, Any]) -> None:
    assert run["typing"]["caret"] == 1
    assert run["typed"] == {
        "verdict": "К борщу\u00a0— да.",
        "caret": 0,
        "ghost": 0,
        "labels": [LABEL_AI],
    }


def test_close_aborts_stream_and_keeps_template(run: dict[str, Any]) -> None:
    assert run["hung"]["now"] == 1
    closed = run["closed"]
    assert closed["aborted"] and not closed["isOpen"] and not closed["inBody"]
    assert closed["sheet"][-1] is False and closed["now"] == 0
    assert closed["verdict"].startswith("К борщу\u00a0— да, с оговоркой. Насыщенный")
    assert closed["labels"] == [LABEL_ALGO]
    # Повторное открытие того же вина: разговор на месте, уточнения под последним ходом.
    assert run["reopen"]["chips"] > 0


def test_sheet_keeps_focus_inside(node: str, tmp_path: Path) -> None:
    """Лист модальный (`aria-modal`): Tab с последней кнопки — на первую, Shift+Tab — наоборот."""
    focus = run_harness(node, tmp_path, "focus")["focus"]
    close, send = "somm-round somm-round--ghost", "somm-round somm-round--accent"
    assert focus["first"] == close and focus["last"] == send
    assert focus["fromPanel"] == {"stop": True, "at": send}
    assert focus["fromLast"] == {"stop": True, "at": close}
    assert focus["fromFirst"] == {"stop": True, "at": send}
    # Посреди листа Tab не трогается — ход делает браузер; со страницы фокус возвращается в лист.
    assert focus["middle"] == {"stop": False, "at": "somm-chip"}
    assert focus["fromPage"] == {"stop": True, "at": close}
    assert focus["afterClose"] == 0


def test_input_off_hides_field_and_camera_stays(run: dict[str, Any]) -> None:
    off = run["inputOff"]
    assert off["inputs"] == 0 and off["camera"] == 1
    assert "Вопрос текстом выключен — выберите вопрос ниже" in off["text"]
    # Камера: лист прячется без onSheet(false) — запись истории заменяет сама страница.
    assert run["scan"]["isOpen"] is False and run["scan"]["sheet"][-1] == "scan"


def test_plain_order_and_tile_opens_wine(run: dict[str, Any]) -> None:
    assert run["plainCall"]["url"] == "/v1/sommelier/ask?order=plain"
    assert run["plainCall"]["body"]["chip"] == "softer"
    plain = run["plain"]
    assert plain["notice"] == 0 and plain["info"] == 0 and plain["minis"] == 0
    assert plain["labels"] == [LABEL_ALGO] and plain["tiles"] == 3
    # Сортировку сменили вне листа (на карточке): переключатель плашки при открытии листа — в тон,
    # а текст — нет: над плашкой по-прежнему рекомендательная подборка, и 149-ФЗ у неё остаётся.
    assert run["plainSwitch"] == [True]
    assert run["plainNotices"] == ["Применяются рекомендательные технологии" + "Обычная сортировка"]
    assert run["tile"]["isOpen"] is False
    assert run["tile"]["sheet"][-1] == "wine:a-gordienko-m-nikolaev-pino-nuar-krasnoe-suhoe-135"


def test_pagehide_cuts_stream_but_keeps_follow_up_chips(run: dict[str, Any]) -> None:
    """Уход со страницы обрывает поток; уточнения остаются для возврата из кэша «назад/вперёд»."""
    assert run["pagehide"]["aborted"] and run["pagehide"]["now"] == 0
    assert run["pagehide"]["chips"] > 0 and run["pagehide"]["labels"] == [LABEL_ALGO]


def test_replace_overlay_legend_names_only_drawn_marks(run: dict[str, Any]) -> None:
    """«Чем заменить»: «роза» первого вина поверх исходного и легенда всего, что на ней есть.

    Серая полоса — исходное вино (с винодельней: названия часто совпадают), контур акцента —
    предложенное; точка «из карточки» со сплошным отрезком и полая точка с пунктиром «по сорту» —
    только те, что нарисованы.
    """
    replace = run["replace"]
    assert replace["overlay"] and replace["minis"] == 0 and replace["orig"] == 8
    assert replace["caption"] == "Профиль рядом с исходным"
    assert replace["aria"].endswith(
        "; серым контуром для сравнения — Cru Lermont Saperavi · Фанагория"
    )
    assert replace["legend"] == [
        {"text": "Cru Lermont Saperavi · Фанагория — исходное", "marks": ["orig-line"]},
        {"text": "Пино Нуар · А. Гордиенко & М. Николаев", "marks": ["own-line"]},
        {"text": "из карточки каталога", "marks": ["catalog", "solid"]},
        {"text": "по сорту — типично для сорта, не замер этого вина", "marks": ["grape", "dash"]},
    ]


def test_replace_overlay_follows_compare_overlay(run: dict[str, Any]) -> None:
    """Сервис выбирает вино для наложения сам (`compare.overlay`, сильнее всех отличное от
    исходного): поверх — оно, а не первое в подборке; `null` — профили совпали, наложения нет.
    Ответ без поля — первое вино подборки (выше)."""
    assert run["overlayPick"] == [["Пино Нуар · Бельбек"], []]


# ---------------------------------------------------------------- заглушка стенда


@pytest.fixture(scope="module")
def stub() -> Any:
    spec = importlib.util.spec_from_file_location("somm_stub", STUB)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def serve(stub: Any, **kwargs: Any) -> Iterator[str]:
    server = stub.make_server(port=0, speed=0.0, **kwargs)
    stub.serve_in_thread(server)
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture(scope="module")
def base(stub: Any) -> Iterator[str]:
    yield from serve(stub)


@pytest.fixture(scope="module")
def base_off(stub: Any) -> Iterator[str]:
    yield from serve(stub, ask_input=False, live=False)


def call(url: str, body: Any = None) -> tuple[int, str, bytes]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"} if data else {}
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with OPENER.open(request, timeout=10) as response:
            return response.status, response.headers.get("Content-Type", ""), response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers.get("Content-Type", ""), error.read()


def events(raw: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in raw.decode("utf-8").split("\n") if line]


def grammar(stream: list[dict[str, Any]]) -> str:
    """`stage* ( facts stage* text | error ) done` — договор §3.2."""
    kinds = "".join(
        {"stage": "s", "facts": "f", "text": "t", "error": "e", "done": "d"}[e["type"]]
        for e in stream
    )
    assert re.fullmatch(r"s*(?:fs*t|e)d", kinds), kinds
    return kinds


RED = "fanagoriya-cru-lermont-saperavi-saperavi-krasnoe-suhoe-135"
ASKS = [
    {"chip": "what_to_eat"},
    {"chip": "serve"},
    {"chip": "softer"},
    {"chip": "fresher"},
    {"chip": "replace"},
    {"chip": "grape", "args": {"grape": "saperavi"}},
    {"chip": "term", "args": {"topic": "brut_scale"}},
    {"chip": "guided"},
    {"chip": "guided", "args": {"food": "meat"}},
    {"chip": "guided", "args": {"food": "meat", "want": "softer"}},
    {"chip": "dish_check", "args": {"dish": "shashlyk_baranina"}},
    {"question": "а к борщу подойдёт?"},
    {"question": "где купить подешевле?"},
    {"question": "привет"},
    {"question": "что-то непонятное"},
    {"question": "ошибка"},
]


def test_stub_serves_stand_and_assets(base: str) -> None:
    status, ctype, body = call(base + "/")
    assert status == 200 and ctype.startswith("text/html")
    assert b'<script src="/static/somm.js"></script>' in body and b"/static/somm.css" in body
    for name, kind in (("somm.js", "text/javascript"), ("somm.css", "text/css")):
        status, ctype, data = call(f"{base}/static/{name}")
        assert status == 200 and ctype.startswith(kind) and data == (STATIC / name).read_bytes()
    assert call(base + "/static/index.html")[0] == 404
    # Стенд ведёт историю листа, как страница: поздний popstate своего «назад» не закрывает
    # лист, открытый заново (гонка, найденная при проверке листа).
    assert b"backing = true;" in body and b"if (SommUI.isOpen() && !entry) push();" in body


def test_stub_card_route(base: str, base_off: str) -> None:
    status, _, body = call(f"{base}/v1/wines/{RED}/sommelier?order=plain")
    card = json.loads(body)
    assert status == 200 and card["order"] == "plain" and card["notice_149"] is None
    assert card["live"] is True and card["input"] is True
    off = json.loads(call(f"{base_off}/v1/wines/{RED}/sommelier")[2])
    assert off["live"] is False and off["input"] is False and off["notice_149"]
    assert call(f"{base}/v1/wines/nope/sommelier")[0] == 404
    assert call(f"{base}/v1/wines/{RED}/sommelier?order=best")[0] == 422


@pytest.mark.parametrize("ask", ASKS, ids=lambda a: a.get("chip") or a["question"])
def test_stub_streams_follow_grammar(base: str, ask: dict[str, Any]) -> None:
    status, ctype, raw = call(base + "/v1/sommelier/ask?order=reco", dict(ask, slug=RED))
    assert status == 200 and ctype.startswith("application/x-ndjson")
    stream = events(raw)
    kinds = grammar(stream)
    if "f" not in kinds:
        return
    facts = next(e for e in stream if e["type"] == "facts")
    text = next(e for e in stream if e["type"] == "text")
    assert (text["generated"] is True) == (text["label"] == LABEL_AI) == (text["reason"] is None)
    if facts["intent"] in ("softer", "fresher", "replace", "guided"):
        assert facts["voice"] is False and text["reason"] == "not_voiced"  # подборки — шаблон
    assert bool(facts["notice_149"]) == bool(facts["wines"])
    for line in [facts["verdict_template"], text["text"], *(c["text"] for c in facts["chips"])]:
        assert check(line).clean and "%" not in line.replace("% об.", ""), line


def test_stub_validates_like_contract(base: str, base_off: str) -> None:
    ask = base + "/v1/sommelier/ask"
    assert call(ask, {"slug": RED})[0] == 422
    assert call(ask, {"slug": RED, "chip": "serve", "question": "как подать"})[0] == 422
    assert call(ask, {"slug": RED, "chip": "price"})[0] == 422
    assert call(ask, {"slug": RED, "question": "я" * 201})[0] == 422
    assert call(ask, {"slug": "nope", "chip": "serve"})[0] == 404
    status, _, body = call(base_off + "/v1/sommelier/ask", {"slug": RED, "question": "как подать"})
    assert status == 422 and json.loads(body) == {"detail": "вопрос текстом выключен"}
    # «off» — у намерений с голосом; подборки голоса не имеют вовсе (`not_voiced`)
    for chip, reason in (("what_to_eat", "off"), ("softer", "not_voiced")):
        stream = events(call(base_off + "/v1/sommelier/ask", {"slug": RED, "chip": chip})[2])
        text = next(e for e in stream if e["type"] == "text")
        assert text["reason"] == reason and text["label"] == LABEL_ALGO
    plain = events(call(ask + "?order=plain", {"slug": RED, "chip": "softer"})[2])
    facts = next(e for e in plain if e["type"] == "facts")
    assert facts["notice_149"] is None and facts["compare"] is None
    assert all(w["pill"] is None and w["reasons"] == [] for w in facts["wines"])
