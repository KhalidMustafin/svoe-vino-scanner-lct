"""Страница продукта `app/api/static/index.html`: право, работа без сети и договоры API.

План вне репо (§3, §8), договор сомелье §8: страница не тянет внешних ресурсов (шрифт и иконки
локальные, внешняя ссылка только на карточку vino-svoe.ru), в статике нет стоп-слов рекламы
алкоголя, знак процента только в «% об.», лист 18+ стоит перед контентом, у подбора одна плашка
149-ФЗ с «Обычной сортировкой», данных портала нет, а «роза ветров» рисуется по правилам
договора. Здесь проверяется исходник страницы; то же на живом DOM после прохода сценария
проверяет `scripts/ui_screens.py`.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from app.recommend.content_filter import check

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "app" / "api" / "static"
PAGE = STATIC / "index.html"
FONTS = STATIC / "fonts"
SOMM = ROOT / "tests" / "fixtures" / "somm"
#: Подмножества Playfair Display 500 и их unicode-range — как в CSS Google Fonts (v40),
#: откуда взяты файлы. Вьетнамское подмножество не нужно: его букв на странице нет.
FONT_FACES = {
    "PlayfairDisplay-Medium-cyrillic.woff2": "U+0301, U+0400-045F, U+0490-0491, U+04B0-04B1, U+2116",
    "PlayfairDisplay-Medium-latin-ext.woff2": (
        "U+0100-02BA, U+02BD-02C5, U+02C7-02CC, U+02CE-02D7, U+02DD-02FF, U+0304, U+0308, U+0329,"
        " U+1D00-1DBF, U+1E00-1E9F, U+1EF2-1EFF, U+2020, U+20A0-20AB, U+20AD-20C0, U+2113,"
        " U+2C60-2C7F, U+A720-A7FF"
    ),
    "PlayfairDisplay-Medium-latin.woff2": (
        "U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, U+0304, U+0308,"
        " U+0329, U+2000-206F, U+20AC, U+2122, U+2191, U+2193, U+2212, U+2215, U+FEFF, U+FFFD"
    ),
}
LOCAL_FACES = 'local("Playfair Display Medium"), local("PlayfairDisplay-Medium")'
SUGGEST_TITLE = "Похоже, этой бутылки может не быть в каталоге"
#: Пустой значок: браузер не просит /favicon.ico, которого у сервиса нет.
FAVICON = '<link rel="icon" href="data:,">'
#: Лист сомелье — файлы дорожки листа (договор сомелье, §8); страница их только подключает.
SOMM_CSS = '<link rel="stylesheet" href="/static/somm.css">'
SOMM_JS = "/static/somm.js"
SHEET_FILES = {"somm.js", "somm.css"}
#: Что отдаёт `/static` (`app.api.page.ASSET_SUFFIXES`) — повторено, чтобы тест не тянул FastAPI.
ASSET_SUFFIXES = {
    ".js",
    ".css",
    ".woff2",
    ".woff",
    ".svg",
    ".png",
    ".webp",
    ".jpg",
    ".jpeg",
    ".ico",
}
TEXT_ASSETS = {".svg", ".css", ".js"}
CONTRACT = ROOT / "docs" / "api-after-search.md"
SOMM_CONTRACT = ROOT / "docs" / "api-sommelier.md"
PORTAL = "https://vino-svoe.ru/wines/"
SVG_NS = "http://www.w3.org/2000/svg"
NOTICE = "Применяются рекомендательные технологии"
ALGO = "Текст и подбор — алгоритм"
HARM = "Чрезмерное употребление алкоголя вредит вашему здоровью."
STOP = re.compile(
    r"купи|покупк|цен[аыуе]\b|₽|руб\.|рубл|лучш|идеальн|вино недели|рейтинг|publicrating|скидк"
    r"|акци[яи]\b|\bprice",
    re.IGNORECASE,
)
# Токены портала (снимок 23.09) и цвет «Лозы», которого в продукте быть не должно.
PORTAL_TOKENS = ("#fefdfa", "#fdf9ed", "#2c2a28", "#8f3d42", "#723135", "#d7d4d2")
LOZA_ACCENT = "#a5182b"
CARD_REGION = re.compile(r"// @card-begin.*?// @card-end", re.DOTALL)
RADAR_REGION = re.compile(r"// @radar-begin.*?// @radar-end", re.DOTALL)
#: Шкала кеглей договора сомелье (§8.2) и предел на страницу.
TYPE_SCALE = {36, 32, 24, 18, 16, 14, 12}
MAX_SIZES = 6
#: Поля карточки из снимка и статуса живого портала — в продукте их нет (решение 24.09).
PORTAL_FIELDS = ("dishes", "temperature", "category_gradient", "live_category", "published")
#: Хозяин листа и методы `window.SommUI` — договор сомелье, §8.1.
HOST_KEYS = {"order", "setOrder", "openWine", "scan", "onSheet", "localUrl", "portalUrl"}
SOMMUI = {"version", "init", "note", "radar", "scales", "entry", "open", "close", "isOpen", "say"}


@pytest.fixture(scope="module")
def html() -> str:
    return PAGE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def script(html: str) -> str:
    """Код страницы — только встроенный скрипт, без подключённого листа сомелье."""
    return "\n".join(re.findall(r"<script>(.*?)</script>", html, re.DOTALL))


@pytest.fixture(scope="module")
def style(html: str) -> str:
    return "\n".join(re.findall(r"<style\b[^>]*>(.*?)</style>", html, re.DOTALL))


@pytest.fixture(scope="module")
def markup(html: str) -> str:
    """Разметка без стилей, скриптов и комментариев — то, что видит человек."""
    text = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", "", html, flags=re.DOTALL)
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)


def load(name: str) -> Any:
    return json.loads((SOMM / name).read_text(encoding="utf-8"))


def only_known(profile: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Профиль так, как его отдавал сервис до оценки по стилю: оси `style` — `null`.

    Заглушка Мускателя (сорта нет в приорах) бывает в обоих видах; «по стилю» проверяется
    отдельно — на профилях, где оси без источника заполнены оценкой по стилю.
    """
    return [dict(a, value=None, source=None) if a["source"] == "style" else a for a in profile]


def block(script: str, start: str, end: str = "\n  }\n") -> str:
    """Кусок скрипта от `start` до конца функции верхнего уровня."""
    tail = script[script.index(start) :]
    return tail[: tail.index(end)]


def test_page_is_one_file_plus_sheet(html: str) -> None:
    assert html.startswith("<!DOCTYPE html>")
    assert '<html lang="ru">' in html
    control = [hex(ord(c)) for c in html if ord(c) < 32 and c not in "\n\r\t"]
    assert control == []
    # Внешний скрипт один — лист сомелье, и он стоит до кода страницы: window.SommUI к её
    # запуску уже объявлен (или его нет вовсе).
    assert re.findall(r'<script\b[^>]*\bsrc="([^"]+)"', html) == [SOMM_JS]
    assert html.index(f'<script src="{SOMM_JS}"></script>') < html.index("<script>\n")
    # <link> — пустой значок `data:,` (без него браузер просит /favicon.ico) и стили листа.
    assert re.findall(r"<link\b[^>]*>", html) == [FAVICON, SOMM_CSS]
    assert html.index(SOMM_CSS) < html.index("<style>") < html.index("</head>")


def test_no_external_resources(html: str, style: str) -> None:
    urls = re.findall(r"https?://[^\s\"'<>)]+", html)
    # Кроме карточки портала — только пространство имён SVG для «розы ветров» (не запрос).
    assert urls and all(url.startswith(PORTAL) or url == SVG_NS for url in urls), urls
    assert not re.search(r"""(?:src|href|action)\s*=\s*["']?(?:https?:)?//""", html)
    assert "@import" not in html
    assert "googleapis" not in html and "gstatic" not in html
    css_urls = re.findall(r"url\(\s*[\"']?([^\"')]+)", style)
    assert all(u.startswith("/static/") for u in css_urls), css_urls


def test_no_font_service_anywhere_in_static() -> None:
    """Шрифт скачан один раз и лежит рядом: ни один файл статики не ссылается на Google Fonts."""
    for path in STATIC.rglob("*"):
        if path.is_file():
            data = path.read_bytes()
            assert b"googleapis" not in data and b"gstatic" not in data, path


def test_static_references_exist(html: str) -> None:
    """Каждый файл `/static/…`, который просит страница, лежит рядом и отдаётся `/static`.

    Иначе браузер пишет 404 в консоль. Шрифт — три woff2 в `static/fonts/`. Файлы листа сомелье
    приносит дорожка листа: до слияния их может не быть, и страница работает без них.
    """
    quoted = re.findall(r"""(?:["'(]|url\(\s*)/static/([A-Za-z0-9_./-]+)""", html)
    assert {f"fonts/{name}" for name in FONT_FACES} <= set(quoted)
    assert SHEET_FILES <= set(quoted)
    for ref in set(quoted):
        path = STATIC / ref
        assert ".." not in ref, ref
        assert path.suffix.lower() in ASSET_SUFFIXES, f"/static не отдаёт {path.suffix}"
        assert ref in SHEET_FILES or path.is_file(), f"страница просит /static/{ref}, а файла нет"


def test_static_assets_have_no_external_urls() -> None:
    """Файлы оформления и листа (svg, css, js) тоже не ходят наружу."""
    for path in STATIC.rglob("*"):
        suffix = path.suffix.lower()
        if suffix not in ASSET_SUFFIXES or suffix not in TEXT_ASSETS:
            continue
        text = path.read_text(encoding="utf-8", errors="replace").replace(SVG_NS, "")
        assert not re.search(r"(?:https?:)?//[A-Za-z0-9.-]+\.[a-z]{2,}", text), path


def unicode_ranges(value: str) -> list[tuple[int, int]]:
    """`U+0400-045F, U+2116` → [(0x400, 0x45F), (0x2116, 0x2116)]."""
    out = []
    for part in value.split(","):
        low, _, high = part.strip().removeprefix("U+").partition("-")
        out.append((int(low, 16), int(high or low, 16)))
    return out


def test_font_playfair_local_first_then_own_files(style: str) -> None:
    """Playfair Display 500: сначала из системы (local), потом свой woff2 из `/static/fonts/`.

    Три подмножества с unicode-range Google Fonts: браузер качает только те, чьи буквы есть на
    экране. `font-display: swap` — текст виден запасным шрифтом, пока woff2 грузится.
    """
    faces = re.findall(r"@font-face\s*\{(.*?)\}", style, re.DOTALL)
    assert len(faces) == len(FONT_FACES)
    seen = {}
    for body in faces:
        rules = {
            key.strip(): value.strip()
            for key, value in re.findall(r"([a-z-]+):\s*([^;]+);", body, re.DOTALL)
        }
        assert rules["font-family"] == '"Playfair Display"'
        assert rules["font-style"] == "normal" and rules["font-weight"] == "500"
        assert rules["font-display"] == "swap"
        src = " ".join(rules["src"].split())
        assert src.startswith(LOCAL_FACES + ", "), src
        refs = re.findall(r'url\("/static/fonts/([^"]+)"\) format\("woff2"\)', src)
        assert len(refs) == 1 and src.endswith('format("woff2")'), src
        seen[refs[0]] = " ".join(rules["unicode-range"].split())
    assert seen == FONT_FACES
    for name in FONT_FACES:
        assert (FONTS / name).read_bytes()[:4] == b"wOF2", name
    stack = re.search(r"--display:\s*([^;]+);", style)
    assert stack and stack.group(1).startswith('"Playfair Display"')
    assert stack.group(1).rstrip().endswith("serif") and "Georgia" in stack.group(1)
    assert "Noto Serif" in stack.group(1)


def test_font_licence_lies_next_to_files() -> None:
    """SIL OFL 1.1 требует класть лицензию рядом со шрифтом; отдавать её `/static` не нужно."""
    licence = (FONTS / "OFL.txt").read_text(encoding="utf-8")
    assert "SIL Open Font License, Version 1.1" in licence
    assert "Playfair Display" in licence


def test_font_covers_every_letter_of_the_page(markup: str, script: str) -> None:
    """Каждая буква текста страницы попадает в unicode-range одного из трёх файлов.

    Иначе этот символ в заголовке рисовался бы запасным шрифтом посреди Playfair. Поэтому
    галочки и стрелки на странице — SVG, а не символы.
    """
    ranges = [r for value in FONT_FACES.values() for r in unicode_ranges(value)]
    text = re.sub(r"<[^>]+>", " ", markup) + " ".join(re.findall(r'"([^"]*)"', script))
    missing = sorted(
        {
            ch
            for ch in text
            if not ch.isspace() and not any(lo <= ord(ch) <= hi for lo, hi in ranges)
        }
    )
    assert missing == [], [f"U+{ord(ch):04X}" for ch in missing]


def test_portal_tokens_not_loza(html: str) -> None:
    low = html.lower()
    assert LOZA_ACCENT not in low
    for token in PORTAL_TOKENS:
        assert token in low, token


def test_tokens_follow_sommelier_contract(style: str) -> None:
    """Все токены договора сомелье (§8.2, заглушка `tokens.json`) объявлены на :root как есть.

    Лист сомелье берёт цвета, шрифты и радиусы только из них.
    """
    root = re.search(r":root\s*\{(.*?)\}", style, re.DOTALL)
    assert root
    declared = {
        key: " ".join(value.split())
        for key, value in re.findall(r"(--[a-z0-9-]+):\s*([^;]+);", root.group(1))
    }
    tokens = load("tokens.json")
    for group in ("colors", "gradients", "fonts", "layout", "motion"):
        for key, value in tokens[group].items():
            assert declared.get(key) == value, (key, declared.get(key), value)


def css_rules(style: str) -> list[tuple[str, str]]:
    """(селектор, тело) каждого простого правила стилей."""
    return [(sel.strip(), body) for sel, body in re.findall(r"([^{}]+)\{([^{}]*)\}", style)]


def sizes_of(body: str) -> list[int]:
    return [int(v) for v in re.findall(r"font(?:-size)?:[^;]*?\b(\d+)px", body)]


def test_type_scale(style: str) -> None:
    """Кегли — только из шкалы договора, и их не больше шести на всю страницу."""
    sizes = [(rule, size) for rule, body in css_rules(style) for size in sizes_of(body)]
    assert sizes, "проверка устарела: кеглей в стилях не нашлось"
    assert [(rule, size) for rule, size in sizes if size not in TYPE_SCALE] == []
    assert len({size for _, size in sizes}) <= MAX_SIZES, sorted({size for _, size in sizes})


def test_small_text_is_dark_enough(style: str) -> None:
    """`--text-2` (3,9:1 на фоне) — только для текста от 18px; мелкий текст — `--text-3`."""
    for selector, body in css_rules(style):
        if re.search(r"(?<![-\w])color:\s*var\(--text-2\)", body):
            assert sizes_of(body) and min(sizes_of(body)) >= 18, selector


def test_no_stop_words(html: str) -> None:
    assert STOP.findall(html) == []


def test_no_percent_outside_card(markup: str, script: str) -> None:
    """Знак процента печатается только крепостью «% об.» — в одном помеченном месте карточки."""
    assert "%" not in markup
    regions = CARD_REGION.findall(script)
    assert len(regions) == 1
    assert "%" in regions[0]
    assert "%" not in CARD_REGION.sub("", script)
    assert re.findall(r'" % об\."', regions[0])


def test_age_gate_stands_before_content(html: str, markup: str) -> None:
    app = re.search(r'<div class="app" id="app"([^>]*)>', markup)
    assert app and "inert" in app.group(1) and 'aria-hidden="true"' in app.group(1)
    age = re.search(r'<div class="([^"]*)" id="age"[^>]*>(.*?)<script', html, re.DOTALL)
    assert age, "нет листа 18+"
    assert "hidden" not in age.group(1).split()
    assert "18+" in age.group(2)
    assert 'id="ageYes"' in age.group(2)
    assert html.index('id="age"') > html.index('id="app"')  # лист поверх страницы, не внутри


def test_harm_footer_on_every_screen(markup: str) -> None:
    """Подвал «вредит здоровью. 18+» — вне экранов, поэтому стоит на каждом."""
    footer = re.search(r'<footer class="foot">(.*?)</footer>', markup, re.DOTALL)
    assert footer
    text = " ".join(re.sub(r"<[^>]+>", " ", footer.group(1)).split())
    assert HARM in text and "18+" in text
    assert markup.index("</main>") < markup.index('<footer class="foot">')
    assert check(HARM).clean


def test_storage_is_guarded(script: str) -> None:
    lines = [line for line in script.splitlines() if "localStorage" in line]
    assert lines, "хранилище не используется — проверка устарела"
    for line in lines:
        assert "try {" in line and "catch" in line, line
    assert "sessionStorage" not in script


def test_one_notice_per_screen(markup: str, script: str) -> None:
    """Плашка 149-ФЗ одна: элемент один, скрипт ставит его над первым готовым блоком подбора.

    У остальных блоков подбора — значок (i). Переключатель — «Обычная сортировка», общий
    `svs.order` на все подборки.
    """
    assert len(re.findall(r'class="notice[ "]', markup)) == 1
    assert 'id="notice"' in markup
    reco = re.findall(r'<section class="[^"]*" id="(\w+)" data-reco', markup)
    assert set(reco) == {"dishBlock", "similar", "nfSame", "nfSim"}
    place = block(script, "function placeNotice")
    assert "blocks[0].parentNode.insertBefore(notice, blocks[0]);" in place
    assert "head.appendChild(infoButton());" in place
    assert 'block.getAttribute("data-state") === "ready"' in place
    assert f'var NOTICE = "{NOTICE}";' in script
    assert '"Обычная сортировка"' in script and 'store.set("svs.order", order);' in script


def test_reco_blocks_follow_order(script: str) -> None:
    """Все подборки идут с общей сортировкой: блюда сомелье, похожие и «нет в каталоге»."""
    assert '"/sommelier?order=" + order' in script
    assert '"/similar?limit=3&order=" + order' in script
    assert '"/v1/similar/by-label?limit=3&order=" + order' in script
    assert "Object.keys(reloads).forEach" in script


def test_api_paths_follow_contract(script: str) -> None:
    contract = CONTRACT.read_text(encoding="utf-8")
    somm = SOMM_CONTRACT.read_text(encoding="utf-8")
    literals = set(re.findall(r'"(/v1/[^"]*)"', script))
    assert literals
    allowed = ("/v1/scan", "/v1/wines/", "/v1/similar/by-label")
    assert all(lit.startswith(allowed) for lit in literals), literals
    for route in ("POST /v1/scan", "GET /v1/wines/{slug}", "/similar", "POST /v1/similar/by-label"):
        assert route in contract, route
    assert "GET /v1/wines/{slug}/sommelier" in somm
    # Продукт ходит только в скан и справочные маршруты: полевой контур пишет кадры на диск.
    assert "/v1/field" not in script and "/v1/eval" not in script
    # Вопросы сомелье задаёт лист; «Подобрать к ужину» заменили его чипы (договор, §8.3).
    assert "/v1/sommelier/ask" not in script and "/shelf" not in script


def page_strings(markup: str, script: str) -> set[str]:
    """Русские строки: текст разметки, подписи (aria-label, title, alt) и литералы скрипта."""
    cyr = "А-Яа-яЁё"
    text = {t.strip() for t in re.findall(rf">([^<]*[{cyr}][^<]*)<", markup)}
    text |= set(re.findall(rf'(?:aria-label|title|alt)="([^"]*[{cyr}][^"]*)"', markup))
    # Литерал — в пределах строки и с учётом `\"`: так кавычка в комментарии не сдвигает пары.
    literals = re.findall(r'"((?:[^"\\\n]|\\.)*)"', script)
    text |= {lit for lit in literals if re.search(f"[{cyr}]", lit)}
    return text


def test_no_portal_data_on_page(script: str, markup: str) -> None:
    """Решение 24.09: блюд, температуры, иконок, градиента и статуса портала на странице нет."""
    for field in PORTAL_FIELDS:
        assert f"card.{field}" not in script, field
    assert "/v1/icons" not in script and "icon_url" not in script
    strings = page_strings(markup, script)
    assert [s for s in strings if "портал" in s.lower()] == [], "подписи «из портала»"
    # Фон шапки — наш токен по цвету выгрузки, а не поле ответа.
    assert '$("hero").className = "hero hero--" + tone(card.color_label);' in script


def test_card_blocks_follow_sommelier_map(markup: str, script: str) -> None:
    """Карточка собирается по карте договора сомелье (§8.3): блоки и источники их данных."""
    order = [
        'id="hero"',
        'id="wineTtl"',
        'id="read"',
        'id="note"',
        'id="taste"',
        'id="askBox"',
        'id="kpis"',
        'id="dishBlock"',
        'id="descBlock"',
        'id="similar"',
    ]
    at = [markup.index(anchor) for anchor in order]
    assert at == sorted(at), "порядок блоков карточки"
    assert "ИИ прочитал на этикетке" in markup
    assert "Описание из каталога «Своё Вино»" in markup
    assert "Открыть на vino-svoe.ru" in markup
    assert f'var ALGO = "{ALGO}";' in script
    assert 'h("p", { class: "small note__label", text: ALGO })' in script
    for read in (
        "somm.profile",
        "somm.serve",
        "somm.dishes",
        "somm.notice_149",
        "serve.temperature_c",
    ):
        assert read in script, read


def test_sommelier_sheet_is_optional(script: str) -> None:
    """Страница живёт и без листа сомелье: `window.SommUI` договора §8.1 — если он есть.

    Хозяин листа — ровно поля договора; заметку, «розу ветров» и шкалы страница рисует сама,
    если метода нет или он упал, а вход в лист без листа не рисуется.
    """
    assert "window.SommUI && window.SommUI.version === 1 ? window.SommUI : null" in script
    host = block(script, "var host = {", "\n  };\n")
    assert set(re.findall(r"^    (\w+):", host, re.MULTILINE)) == HOST_KEYS
    used = set(re.findall(r"SOMM\.(\w+)", script)) | set(re.findall(r'viaSomm\("(\w+)"', script))
    assert used <= SOMMUI, used - SOMMUI
    for name in ("note", "radar", "scales", "say"):
        assert f'viaSomm("{name}", function' in script, name
    assert "SOMM.init(host);" in script
    assert 'if (SOMM && typeof SOMM.entry === "function")' in script
    # Лист подключается только после листа 18+: init — в unlock(), а не при загрузке.
    assert script.count("SOMM.init(host);") == 1
    assert "SOMM.init(host);" in block(script, "  function unlock()")
    # Шкалы листа — без своей легенды: одна общая легенда блока «Какое это вино».
    assert "drawScalesUI(scales, profile, { legend: false })" in script
    assert "tasteLegend(profile, drawn, ruled);" in script


def test_sheet_history(script: str) -> None:
    """Лист открыт — наверху истории его запись: «назад» закрывает лист, а не карточку.

    Страница, уводя с листа сама (тап по вину, камера), запись листа заменяет, а не копит.
    Закрыл лист человек — запись снимает своё `history.back()`; оно асинхронное, и его поздний
    popstate не закрывает лист, открытый заново до его прихода (гонка, найденная при проверке листа): такой
    popstate ставит запись нового листа, а не закрывает его.
    """
    push = block(script, "  function pushSheet()")
    assert 'history.pushState({ at: at, sheet: "somm" }, ""); sheetEntry = true;' in push
    sheet = block(script, "    onSheet: function (isOpen)", "\n    },\n")
    assert "if (!sheetEntry && !backing) pushSheet();" in sheet
    assert "history.back()" in sheet and "if (!sheetEntry || quiet) return;" in sheet
    assert sheet.index("backing = true;") < sheet.index("history.back()")
    assert 'open({ kind: "wine", slug: slug }, leaveSheet());' in script
    pop = block(script, 'window.addEventListener("popstate"', "\n  });\n")
    # Сначала — своё «назад» листа: экран тот же, открытый заново лист получает свою запись.
    own = pop[pop.index("if (backing) {") : pop.index('if (sheetEntry && state.sheet !== "somm")')]
    assert "backing = false;" in own and 'state.at === at && state.sheet !== "somm"' in own
    assert "if (sheetOpen() && !sheetEntry) pushSheet();" in own and "return;" in own
    assert 'if (sheetEntry && state.sheet !== "somm")' in pop
    assert "SOMM.close({ fromHistory: true })" in pop


def test_own_photo_stays_on_the_phone(script: str) -> None:
    """Фото человека уходит на сервер один раз (/v1/scan) и живёт только адресом blob: вкладки."""
    assert script.count('fetch("/v1/scan"') == 1
    assert script.count("URL.createObjectURL(file)") == 1
    assert "URL.revokeObjectURL(shotUrl)" in script
    assert "FileReader" not in script and "toDataURL" not in script
    # Своё фото показывается только тем экранам, что пришли из этого же скана.
    assert "view && view.shot && view.shot === shotUrl ? shotUrl : null" in script


def test_own_photo_hides_when_browser_cannot_draw(script: str) -> None:
    """HEIC сервис разбирает, а браузер не рисует: у каждой картинки с кадром человека — onerror.

    Иначе в шапке, на экране check и «нет в каталоге» стоит значок битой картинки.
    """
    uses = re.findall(r"(\w+)\.src = (?:shotUrl|shot)\b", script)
    assert len(set(uses)) >= 4, "страница не показывает кадр — проверка устарела"
    for name in set(uses):
        assert f"{name}.onerror" in script, f"{name}.src без обработчика ошибки"


def test_own_photo_crop_keeps_the_label(style: str) -> None:
    """Кадр человека в рамке (шапка карточки, «нет в каталоге») обрезается по центру этикетки.

    `object-fit: cover` по центру оставлял от высокой бутылки плечики, а у снимка бутылки на
    белом фоне — тёмное пятно: этикетка типичного кадра — чуть ниже середины.
    """
    rule = re.search(r"\.uphoto img \{([^}]*)\}", style)
    assert rule and "object-fit: cover" in rule.group(1)
    assert "object-position: 50% 62%" in rule.group(1)


def test_every_error_code_has_text(script: str) -> None:
    table = re.search(r"var FAIL = \{(.*?)\};", script, re.DOTALL)
    assert table
    keys = set(re.findall(r"^\s*([a-z_]+):", table.group(1), re.MULTILINE))
    codes = {"no_image", "bad_request", "too_large", "decode", "cv", "not_ready", "internal"}
    assert codes <= keys, codes - keys


def test_portal_link_is_checked_before_use(script: str) -> None:
    assert f'var PORTAL = "{PORTAL}";' in script
    assert "portalUrl(card.portal_url)" in script
    assert 'rel="noopener noreferrer"' in PAGE.read_text(encoding="utf-8")


def examples_of(script: str) -> list[dict[str, str]]:
    """«Посмотреть на примере» со страницы: slug, название, винодельня и цвет."""
    table = re.search(r"var EXAMPLES = \[(.*?)\];", script, re.DOTALL)
    assert table
    rows = re.findall(
        r'\{ slug: "([a-z0-9-]+)", name: "([^"]+)", winery: "([^"]+)", color: "([^"]+)" \}',
        table.group(1),
    )
    return [dict(zip(("slug", "name", "winery", "color"), row, strict=True)) for row in rows]


def test_examples_are_catalog_wines(script: str, markup: str) -> None:
    """«Посмотреть на примере» — три вина выгрузки, без камеры и скана.

    Три разные винодельни и три цвета: пример показывает разные «розы ветров». Что у каждого
    8 известных осей из 8 и фото выгрузки, сверяет `test_api_page.py` на настоящих данных.
    """
    examples = examples_of(script)
    assert len(examples) == 3 and len({e["slug"] for e in examples}) == 3
    assert len({e["winery"] for e in examples}) == 3
    assert len({e["color"] for e in examples}) == 3
    assert {e["color"] for e in examples} <= {"Красное", "Белое", "Розовое", "Оранжевое"}
    assert 'open({ kind: "wine", slug: example.slug, example: true })' in script
    assert "Пример из каталога" in markup and "Посмотреть на примере" in markup
    # «Попробовать» над тремя бутылками читается как призыв к вину (38-ФЗ): на странице его нет
    # ни в каком виде — и у подборок, и в ошибках скана «повторите», а не «попробуйте».
    assert "опроб" not in markup.lower() and "опроб" not in script.lower()
    # Карточка примера помечена прямо над шапкой, а не только на главной.
    assert markup.index('id="exampleNote"') < markup.index('id="hero"')
    assert 'show($("exampleNote"), o.state === "example");' in script


def test_silhouette_instead_of_missing_photo(script: str, style: str) -> None:
    """`photo_url: null` или битая картинка — силуэт бутылки, а не значок битой картинки.

    Везде, где страница рисует фото каталога: шапка карточки (с честной подписью), плитки,
    лента вариантов, «Может, это …?» и примеры — все через `framePhoto` или `heroPhoto`.
    """
    bottle = re.search(r"var BOTTLE = (.*?);\n", script, re.DOTALL)
    assert bottle
    parts = re.findall(r'class="(bottle[^"]*)"', bottle.group(1))
    assert parts == ["bottle", "bottle__glass", "bottle__cap", "bottle__label", "bottle__shine"]
    for part in parts:
        assert re.search(rf"\.{part}\b[^{{]*\{{", style), part
    frame = block(script, "  function framePhoto")
    assert "if (!src) { fallback(); return box; }" in frame
    assert 'img.addEventListener("error", fallback);' in frame
    hero = block(script, "  function heroPhoto")
    assert 'none("фото в каталоге нет")' in hero and 'none("фото не загрузилось")' in hero
    # Все фото вина на странице — через эти две функции: своего <img> фото каталога нет.
    assert script.count('h("img"') == 2


def test_every_page_string_passes_content_filter(markup: str, script: str) -> None:
    """Все русские строки страницы проходят фильтр рекламы алкоголя (`content_filter`)."""
    strings = page_strings(markup, script)
    assert SUGGEST_TITLE in strings and ALGO in strings and len(strings) > 80
    bad = {text: check(text).violations for text in strings if not check(text).clean}
    assert bad == {}


def test_suggest_not_found_banner_and_primary_button(markup: str, script: str) -> None:
    """`after.suggest_not_found` при `check`: плашка и «Моего вина здесь нет» главной кнопкой.

    Поле читается строго: нет его (сервис старше договора) или не `true` — прежний экран.
    Карточка и варианты остаются: это подсказка, а не отказ.
    """
    assert "suggest: after.suggest_not_found === true" in script
    assert 'var suggest = o.state === "check" && o.suggest === true;' in script
    assert "suggest_not_found: false" in script  # ответ старого сервиса — без подсказки
    texts = dict(re.findall(r'var (SUGGEST_[A-Z_]+) = "([^"]+)";', script))
    assert texts["SUGGEST_TITLE"] == SUGGEST_TITLE
    assert set(texts) == {"SUGGEST_TITLE", "SUGGEST_TEXT", "SUGGEST_TEXT_EMPTY"}
    for text in texts.values():
        assert check(text).clean and not STOP.search(text) and "%" not in text, text
    top = re.search(r'<button class="([^"]*)" id="notHereTop"[^>]*>([^<]+)</button>', markup)
    assert top, "нет главной кнопки «Моего вина здесь нет»"
    assert {"btn--primary", "hidden"} <= set(top.group(1).split())
    assert top.group(2) == "Моего вина здесь нет"
    # Под плашкой и над вариантами; кнопка под вариантами при подсказке прячется.
    assert markup.index('id="banner"') < markup.index('id="notHereTop"') < markup.index('id="alts"')
    assert "show(topButton, suggest);" in script and 'show($("notHere"), !suggest);' in script
    # Варианты свёрнуты только при карточке: без неё список остаётся раскрытым выбором.
    assert 'var pick = o.state === "check" && !(suggest && card);' in script
    assert "setAlts(pick);" in script
    assert 'show($("wine"), Boolean(card));' in script  # карточка не прячется


def test_check_screen_asks_one_question(markup: str, script: str) -> None:
    """Экран check: один вопрос сомелье (`after.question`) и лента миниатюр вариантов.

    Вариант ответа открывает карточку своего вина; в ленте открытое вино отмечено, а слова,
    которыми названия различаются, выделены жирным.
    """
    assert markup.index('id="checkq"') < markup.index('id="alts"') < markup.index('id="wine"')
    assert "question: after.question || null" in script
    assert "var q = o.question;" in script and "(q && q.options) || []" in script
    assert '"aria-current": here ? "true" : null' in script
    assert 'here ? h("span", { class: "alt__now", text: "открыто" }) : null' in script
    assert 'out.push(plain ? part : h("b", { text: part }));' in script


def test_not_here_sends_exclude(script: str) -> None:
    """«Моего вина здесь нет» шлёт `exclude`: самое похожее, показанные варианты и их карточки.

    Не больше 20 slug, как в договоре; сервис старше договора лишнее поле пропускает.
    """
    assert "var MAX_EXCLUDE = 20;" in script
    assert "if (card) markSeen(o.seen, card.slug);" in script  # самое похожее и открытая карточка
    # Варианты — только раскрытые: главная кнопка над свёрнутым списком не убирает из подборки
    # то, чего человек не видел.
    shown = "if (alts.open) alts.list.forEach(function (cand) { markSeen(alts.seen, cand.slug); });"
    assert shown in block(script, "function setAlts")
    assert "markSeen" not in block(script, "function fillAlts")
    scan = 'open({ kind: "scan", body: body, after: afterOf(body), seen: [], shot: shot });'
    assert scan in script
    assert "seen: o.seen" in script and "seen: view.seen" in script  # один список на скан
    assert "exclude: (o.seen || []).slice(0, MAX_EXCLUDE)" in script
    assert len(re.findall(r"onclick = function \(\) \{ notHere\(o\); \};", script)) == 2
    payload = block(script, "function byLabelPayload")
    assert "payload.exclude = view.exclude.slice(0, MAX_EXCLUDE);" in payload
    assert "JSON.stringify(byLabelPayload(view))" in script


def test_not_found_same_winery_without_maybe_and_three_shown(markup: str, script: str) -> None:
    """«Нет в каталоге»: вина из «Может, это …?» нет среди вин винодельни — оно стоит выше.

    Вин винодельни до 6 (договор «после поиска», §6): видны три, остальные — по кнопке «Ещё»,
    чтобы подбор по стилю не уходил на три экрана вниз.
    """
    body = block(script, "  function renderNotFound(view)")
    assert 'var maybeSlug = view.mode === "auto" && card ? card.slug : null;' in body
    assert "return item && item.slug !== maybeSlug;" in body and "sameTiles(sameItems);" in body
    assert "var SAME_SHOWN = 3;" in script
    more = block(script, "  function sameTiles(items)")
    assert 'tiles($("nfSameList"), items.slice(0, SAME_SHOWN));' in more
    assert 'tiles($("nfSameList"), items); hide(more);' in more and "show(more, rest > 0);" in more
    assert (
        markup.index('id="nfSameList"')
        < markup.index('id="nfSameMore"')
        < markup.index('id="nfSim"')
    )
    # Загрузка, пустой ответ и сбой прячут кнопку прошлой подборки.
    assert body.count('hide($("nfSameMore"));') == 3


def test_dash_never_starts_a_persona_line(script: str, style: str) -> None:
    """Тире в репликах и честных фразах страницы не начинает строку: пробел перед ним неразрывный."""
    assert (
        'return String(text || "").replace(/ —/g, "\\u00a0—")'
        '.replace(/(\\d)–(\\d)/g, "$1\\u2060–\\u2060$2").replace(/ °C/g, "\\u00a0°C");'
    ) in block(script, "  function dash(text)")
    assert "function say(el, text) { return sayUI(el, dash(text)); }" in script
    assert "el.textContent = dash(text || fallback);" in script
    # Заметка карточки — копией ответа с тем же пробелом; ответ, который получает лист, цел.
    assert 'drawNote($("note"), noteOf(entryData(somm, card)))' in script
    assert "note.text = dash(note.text);" in block(script, "  function noteOf(somm)")
    # Одна плитка «Крепость» (подача не нашлась) — во всю ширину: auto-fit схлопывает пустую колонку.
    assert "grid-template-columns: repeat(auto-fit, minmax(128px, 1fr));" in style


def test_scan_stages_are_honest(script: str) -> None:
    """Этапы скана — человеческие, последний держится до ответа, «почти готово» не пишется."""
    stages = re.search(r"var STAGES = \[(.*?)\];", script, re.DOTALL)
    assert stages
    texts = [json.loads(f'"{t}"') for t in re.findall(r'text: "([^"]+)"', stages.group(1))]
    # Название каталога — через неразрывный пробел: на 360 px «Своё / Вино» не рвётся.
    assert texts == [
        "Отправляю фото",
        "Смотрю на бутылку",
        "Читаю этикетку",
        "Сверяю с каталогом «Своё Вино»",
    ]
    assert re.findall(r'done: "([^"]*)"', stages.group(1))[-1] == ""
    literals = re.findall(r'"([^"\n]*)"', script)
    assert not [lit for lit in literals if "почти готово" in lit]


# ---------------------------------------------------------------- «роза ветров» в Node

NODE_HARNESS = r"""
const fs = require("fs");
const html = fs.readFileSync(process.argv[2], "utf8");
const code = html.match(/\/\/ @radar-begin[\s\S]*?\/\/ @radar-end/)[0];
class Node {
  constructor(tag) { this.tag = tag; this.attrs = {}; this.children = []; this.text = ""; }
  setAttribute(key, value) { this.attrs[key] = String(value); }
  appendChild(child) { this.children.push(child); return child; }
  set textContent(value) { this.children = []; this.text = String(value); }
  get textContent() { return this.text; }
  get childNodes() { return this.children; }
}
global.document = { createElementNS: (ns, tag) => new Node(tag) };
eval(code);
const dump = (n) => ({ tag: n.tag, attrs: n.attrs, text: n.text, children: n.children.map(dump) });
const profiles = JSON.parse(fs.readFileSync(0, "utf8"));
console.log(JSON.stringify(profiles.map((profile) => {
  const el = new Node("div");
  const drawn = drawRadar(el, profile);
  return { drawn, svg: el.children.length ? dump(el.children[0]) : null };
})));
"""


def run_radar(profiles: list[list[dict[str, Any]]], tmp_path: Path) -> list[dict[str, Any]]:
    """Код «розы ветров» со страницы как есть — в Node с игрушечным DOM."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("нет Node.js: «розу ветров» проверяет только исходник")
    harness = tmp_path / "radar.js"
    harness.write_text(NODE_HARNESS, encoding="utf-8")
    done = subprocess.run(
        [node, str(harness), str(PAGE)],
        input=json.dumps(profiles, ensure_ascii=False),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=True,
    )
    return json.loads(done.stdout)


def kids(svg: dict[str, Any], tag: str, cls: str) -> list[dict[str, Any]]:
    return [
        k for k in svg["children"] if k["tag"] == tag and cls in k["attrs"].get("class", "").split()
    ]


def test_radar_code_follows_contract(script: str) -> None:
    region = RADAR_REGION.findall(script)
    assert len(region) == 1
    code = region[0]
    assert 'viewBox: "0 0 260 220"' in code
    assert "cx: 130, cy: 110, r: 66, floor: 0.14" in code
    assert 'attrs["stroke-dasharray"] = "4 3"' in code
    assert 'role: "img"' in code and '"Профиль вкуса: "' in code
    assert "if (known.length < 3) return false;" in code
    assert "%" not in code


def test_radar_draws_known_axes_only(tmp_path: Path) -> None:
    """«Роза ветров» (P0): неизвестная ось не рисуется вовсе, «по сорту» — пунктир и полая точка.

    Прогон кода страницы в Node на профилях заглушек договора: красное (8 осей), игристое,
    без сахара и крепости (6 осей), сладкое без сорта в приорах (3 оси) и профиль с двумя
    осями — его не рисуют.
    """
    red = load("sommelier_red.json")["profile"]
    sparkling = load("sommelier_sparkling.json")["profile"]
    nosugar = load("sommelier_nosugar.json")["profile"]
    sweet = only_known(load("sommelier_sweet.json")["profile"])
    two = [dict(axis, value=None, source=None) for axis in red]
    two[0], two[4] = red[0], red[4]
    out = run_radar([red, sparkling, nosugar, sweet, two], tmp_path)

    assert [o["drawn"] for o in out] == [True, True, True, True, False]
    assert out[4]["svg"] is None
    for profile, result in zip((red, sparkling, nosugar, sweet), out[:4], strict=True):
        svg = result["svg"]
        # Сладкое — без осей «по стилю» (`only_known`): их рисунок — в тесте ниже.
        known = [(i, axis) for i, axis in enumerate(profile) if axis["source"] is not None]
        assert svg["attrs"]["viewBox"] == "0 0 260 220" and svg["attrs"]["role"] == "img"
        label = svg["attrs"]["aria-label"]
        assert label.startswith("Профиль вкуса: ") and not re.search(r"\d|%", label)
        assert len(kids(svg, "circle", "radar__ring")) == 3
        # Лучи, подписи и вершины — только у известных осей, в порядке договора.
        assert len(kids(svg, "line", "radar__ray")) == len(known)
        labels = [t["text"] for t in kids(svg, "text", "radar__label")]
        assert labels == [axis["label"] for _, axis in known]
        dots = kids(svg, "circle", "radar__dot")
        assert len(dots) == len(known)
        for dot, (i, axis) in zip(dots, known, strict=True):
            assert ("radar__dot--grape" in dot["attrs"]["class"]) == (axis["source"] == "grape")
            # Вершина — на радиусе R·max(value, 0.14), луч i — от 12 часов по часовой стрелке.
            x, y = float(dot["attrs"]["cx"]), float(dot["attrs"]["cy"])
            radius = 66 * max(axis["value"], 0.14)
            angle = i / 8 * 2 * math.pi
            assert abs(x - (130 + radius * math.sin(angle))) < 0.2
            assert abs(y - (110 - radius * math.cos(angle))) < 0.2
        # Ребро к вершине «по сорту» — пунктир 4 3, между двумя вершинами каталога — сплошное.
        edges = kids(svg, "line", "radar__edge")
        assert len(edges) == len(known)
        for k, edge in enumerate(edges):
            ends = (known[k][1]["source"], known[(k + 1) % len(known)][1]["source"])
            assert (edge["attrs"].get("stroke-dasharray") == "4 3") == ("grape" in ends)
        polygon = kids(svg, "polygon", "radar__area")
        assert len(polygon) == 1 and len(polygon[0]["attrs"]["points"].split()) == len(known)
        assert not [t for t in labels if re.search(r"\d|%", t)]
    # Сладкое без сорта в приорах — только оси каталога: ни одного пунктира.
    sweet_edges = kids(out[3]["svg"], "line", "radar__edge")
    assert not [e for e in sweet_edges if "stroke-dasharray" in e["attrs"]]
    # Без сахара и крепости нет ни подписей, ни лучей этих осей.
    labels = [t["text"] for t in kids(out[2]["svg"], "text", "radar__label")]
    assert "Сладость" not in labels and "Крепость" not in labels


def test_radar_style_source_is_dotted(tmp_path: Path) -> None:
    """Источник `style` (сорта нет в приорах — оценка по винам того же стиля) рисуется как «по
    сорту» (договор §8.1): вершина полая, рёбра к ней — пунктир 4 3, подпись — «по стилю».

    Так у 31 вина, где осей из карточки меньше трёх, «роза» всё равно рисуется.
    """
    red = load("sommelier_red.json")["profile"]
    sweet = only_known(load("sommelier_sweet.json")["profile"])
    styled = [dict(a, value=0.4, source="style") if a["source"] is None else a for a in sweet]
    mixed = [dict(a, source="style") if a["axis"] in ("oak", "aroma_intensity") else a for a in red]
    out = run_radar([styled, mixed], tmp_path)
    assert [o["drawn"] for o in out] == [True, True]
    for profile, result in zip((styled, mixed), out, strict=True):
        svg = result["svg"]
        dots = kids(svg, "circle", "radar__dot")
        for dot, axis in zip(dots, profile, strict=True):
            assert ("radar__dot--style" in dot["attrs"]["class"]) == (axis["source"] == "style")
            assert ("radar__dot--grape" in dot["attrs"]["class"]) == (axis["source"] == "grape")
        for k, edge in enumerate(kids(svg, "line", "radar__edge")):
            ends = {profile[k]["source"], profile[(k + 1) % 8]["source"]}
            want = "4 3" if ends & {"grape", "style"} else None
            assert edge["attrs"].get("stroke-dasharray") == want, (k, ends)
        assert "(по стилю)" in svg["attrs"]["aria-label"]


LEGEND_HARNESS = r"""
const fs = require("fs");
const html = fs.readFileSync(process.argv[2], "utf8");
const radar = html.match(/\/\/ @radar-begin[\s\S]*?\/\/ @radar-end/)[0];
const legend = html.match(/\/\/ @legend-begin[\s\S]*?\/\/ @legend-end/)[0];
class Node {
  constructor(tag) { this.tag = tag; this.attrs = {}; this.children = []; this.text = ""; this.hidden = false; }
  setAttribute(key, value) { this.attrs[key] = String(value); }
  appendChild(child) { this.children.push(child); return child; }
  set textContent(value) { this.children = []; this.text = String(value); }
  get textContent() { return this.text + this.children.map((c) => c.textContent).join(""); }
  get childNodes() { return this.children; }
}
global.document = { createElementNS: (ns, tag) => new Node(tag) };
const box = new Node("p");
global.$ = () => box;
global.show = (el, on) => { el.hidden = on === false; };
global.h = (tag, props, kids) => {
  const node = new Node(tag);
  if (props && props.class) node.attrs.class = props.class;
  (kids || []).forEach((kid) => {
    if (kid === null || kid === undefined) return;
    if (typeof kid === "string") { const t = new Node("#text"); t.text = kid; node.appendChild(t); } else node.appendChild(kid);
  });
  return node;
};
eval(radar + legend);
const cases = JSON.parse(fs.readFileSync(0, "utf8"));
console.log(JSON.stringify(cases.map(([profile, radarDrawn, scalesDrawn]) => {
  tasteLegend(profile, radarDrawn, scalesDrawn);
  return { hidden: box.hidden, items: box.children.map((item) => ({
    marks: item.children.filter((k) => k.tag === "i").map((k) => k.attrs.class.replace("lg lg--", "")),
    text: item.children.filter((k) => k.tag === "#text").map((k) => k.text).join("") })) };
})));
"""


def test_one_taste_legend_names_only_drawn_marks(tmp_path: Path) -> None:
    """Одна легенда «Какое это вино» на «розу» и шкалы — и только то, что нарисовано.

    Сплошной отрезок — только если две соседние вершины «розы» из карточки; пунктир — только
    при вершине «по сорту» на нарисованной «розе». У вина без сахара и крепости единственная
    вершина из карточки (пузырьки) — между двумя «по сорту», и сплошного отрезка нет.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("нет Node.js: легенду проверяет только исходник")
    harness = tmp_path / "legend.js"
    harness.write_text(LEGEND_HARNESS, encoding="utf-8")
    red = load("sommelier_red.json")["profile"]
    sweet = only_known(load("sommelier_sweet.json")["profile"])
    nosugar = load("sommelier_nosugar.json")["profile"]
    cases = [[red, True, True], [sweet, True, True], [nosugar, True, True], [red, False, True]]
    cases.append([red, False, False])
    styled = [dict(a, value=0.4, source="style") if a["source"] is None else a for a in sweet]
    cases += [[styled, True, True], [styled, False, True]]
    done = subprocess.run(
        [node, str(harness), str(PAGE)],
        input=json.dumps(cases, ensure_ascii=False),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=True,
    )
    out = json.loads(done.stdout)
    catalog, grape = "из карточки каталога", "по сорту — типично для сорта, не замер этого вина"
    assert out[0] == {
        "hidden": False,
        "items": [
            {"marks": ["dot", "solid"], "text": catalog},
            {"marks": ["ring", "dash"], "text": grape},
        ],
    }
    assert out[1]["items"] == [{"marks": ["dot", "solid"], "text": catalog}]
    assert out[2]["items"] == [
        {"marks": ["dot"], "text": catalog},
        {"marks": ["ring", "dash"], "text": grape},
    ]
    # «Розы» нет — нет и её отрезков в легенде; нет ни «розы», ни шкал — легенды нет.
    assert out[3]["items"] == [
        {"marks": ["dot"], "text": catalog},
        {"marks": ["ring"], "text": grape},
    ]
    assert out[4] == {"hidden": True, "items": []}
    # «По стилю» — те же кольцо и пунктир, подпись своя; без «розы» — одно кольцо.
    style = "по стилю — типично для стиля, не замер этого вина"
    assert out[5]["items"] == [
        {"marks": ["dot", "solid"], "text": catalog},
        {"marks": ["ring", "dash"], "text": style},
    ]
    assert out[6]["items"] == [
        {"marks": ["dot"], "text": catalog},
        {"marks": ["ring"], "text": style},
    ]


DISHES_HARNESS = r"""
const fs = require("fs");
const html = fs.readFileSync(process.argv[2], "utf8");
const code = html.match(/\/\/ @dishes-begin[\s\S]*?\/\/ @dishes-end/)[0];
class Node {
  constructor(tag) { this.tag = tag; this.attrs = {}; this.children = []; this.text = ""; }
  appendChild(child) { this.children.push(child); return child; }
  set textContent(value) { this.children = []; this.text = String(value); }
  get textContent() { return this.text + this.children.map((c) => c.textContent).join(""); }
}
global.h = (tag, props, kids) => {
  const node = new Node(tag);
  if (props && props.class) node.attrs.class = props.class;
  if (props && props.text) node.text = props.text;
  (kids || []).forEach((kid) => {
    if (kid === null || kid === undefined) return;
    if (typeof kid === "string") { const t = new Node("#text"); t.text = kid; node.appendChild(t); } else node.appendChild(kid);
  });
  return node;
};
global.lookup = (map, key) => (Object.prototype.hasOwnProperty.call(map, key) ? map[key] : undefined);
global.glyph = () => new Node("span");
global.ICONS = { dish: "" };
global.DISH_ICONS = {};
global.VERDICTS = { yes: "подходит", caveat: "подходит с оговоркой" };
global.SOURCES = { grape: "по сорту", type: "типично для стиля" };
eval(code);
const list = new Node("ul");
drawDishes(list, JSON.parse(fs.readFileSync(0, "utf8")));
const rows = (li) => li.children[1].children.filter((k) => k.attrs.class === "rules");
console.log(JSON.stringify(list.children.map((li) => ({
  chips: rows(li).flatMap((row) => row.children.map((chip) => chip.children[0].text)),
  small: rows(li).flatMap((row) => row.children.flatMap((chip) => chip.children.filter((k) => k.tag === "small").map((k) => k.text))),
  verdict: li.children[1].children.filter((k) => k.attrs.class === "dish__verdict").map((k) => k.text) }))));
"""


def test_card_dish_chips_do_not_repeat(tmp_path: Path) -> None:
    """«К чему подать» на карточке: у каждого блюда сильнейший «+» и «−», но не тот, что выше.

    У Саперави все три блюда начинались с «Кислотность освежает жирное» — три одинаковых чипа
    подряд. Теперь у блюда ниже — следующее по силе правило этого же блюда; другого нет — то же.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("нет Node.js")
    harness = tmp_path / "dishes.js"
    harness.write_text(DISHES_HARNESS, encoding="utf-8")
    dishes = load("sommelier_red.json")["dishes"]
    out = run_dishes(node, harness, dishes)
    assert [d["chips"] for d in out] == [
        ["+ Кислотность освежает жирное"],
        ["+ Равны по силе вкуса", "− Бульон подчёркивает терпкость"],
        ["+ Свежесть вровень с блюдом", "− Бульон подчёркивает терпкость"],
    ]


def run_dishes(node: str, harness: Path, dishes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    done = subprocess.run(
        [node, str(harness), str(PAGE)],
        input=json.dumps(dishes, ensure_ascii=False),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=True,
    )
    return json.loads(done.stdout)


def test_card_dish_source_is_written_once(tmp_path: Path) -> None:
    """Источник, общий для всех чипов блюда, пишется раз — в строке вердикта, а не у чипа.

    У Cru Lermont все правила сверялись с осями «по сорту»: «подходит · по сорту» и чипы без
    пометки. У Рислинга без сахара источники разные (сорт и типичное для стиля) — пометка у
    каждого чипа, строка вердикта — без неё.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("нет Node.js")
    harness = tmp_path / "dishes.js"
    harness.write_text(DISHES_HARNESS, encoding="utf-8")
    red = run_dishes(node, harness, load("sommelier_red.json")["dishes"])
    assert [d["verdict"] for d in red] == [
        ["подходит · по сорту"],
        ["подходит с оговоркой · по сорту"],
        ["подходит с оговоркой · по сорту"],
    ]
    assert [d["small"] for d in red] == [[], [], []]
    nosugar = run_dishes(node, harness, load("sommelier_nosugar.json")["dishes"])
    assert [d["verdict"] for d in nosugar] == [["подходит"]] * 3
    assert all(d["small"] for d in nosugar)
