"""Скриншоты страницы продукта 390×844 и проверка того, что на них показано.

Страницу отдаёт заглушка `scripts/ui_stub_server.py` (поднимается здесь же, на свободном порту),
сценарий за человека проходит `scripts/ui_stub_driver.js`: 18+, выбор фото, кнопки и чипы.

Снимки листа сомелье (`somm_…`) идут на стенде `scripts/somm_stub.py` (тоже здесь же, на
свободном порту, поток без пауз): тот же драйвер жмёт чипы «Помогите выбрать» → «К мясу» →
«Помягче» и «К чему подать» → «А к шашлыку?» и ждёт конца каждого хода.

Безголовый Chrome не делает окно уже 500 px, поэтому снимок идёт с обёртки `/__stub/frame`:
страница во фрейме ровно 390×844, а картинка окна обрезается до фрейма.

Кроме снимков, для каждого состояния снимается DOM той же страницы (`--dump-dom`) и проверяется:
- драйвер дошёл до конца (`data-shot="ready"`), а не упал на полпути;
- в блоках подбора (блюда, похожие, «нет в каталоге») нет знака процента, а во всём видимом
  тексте он стоит только в «% об.»;
- в видимом тексте страницы нет стоп-слов (цена, «купить», «лучший», «вино недели»…);
- текст до 14px контрастен не ниже 4,5:1 (замер драйвера, фон — худший цвет градиента);
- кеглей не больше шести, и все из шкалы 36/32/24/18/16/14/12;
- плашка 149-ФЗ на экране одна или её нет.

Снимки пишутся в `--out` (обязательно); в `docs/screens/` — снимки на заглушках, без данных
портала (страница — 26.09, лист сомелье и экран ожидания — 27.09).

Запуск из корня рабочего дерева (видеокарта не нужна, сервис не нужен):

    python scripts/ui_screens.py --out <папка> --data-dir data --csv <выгрузка>
    python scripts/ui_screens.py --out <папка> --only found check --no-screens   # только DOM
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
WIDTH, HEIGHT = 390, 844
SHOTS = (
    "age",
    "home",
    "busy",
    "found",
    "taste",
    "dishes",
    "check",
    "suggest",
    "not_found",
    "error",
    "similar",
    "example",
    "example_taste",
    "example_dishes",
)
#: Снимки листа сомелье на стенде и вино стенда (`?wine=`): подборка «три вина других виноделен»
#: у заглушки есть только у красного, вопрос о блюде — у вина снимков страницы (Мускатель).
SOMM_SHOTS = {
    "somm_guided": "red",
    "somm_want": "red",
    "somm_three": "red",
    "somm_dish": "sweet",
}
# Блоки подбора: в них не бывает знака процента (право, договор docs/api-after-search.md, §1).
RECO_IDS = {"similar", "dishBlock", "notice", "nfSame", "nfSim", "nfNote"}
STOP = re.compile(
    r"купи|покупк|цен[аыуе]\b|₽|руб\.|рубл|лучш|идеальн|вино недели|рейтинг|скидк|акци[яи]\b",
    re.IGNORECASE,
)
#: Шкала кеглей страницы (договор сомелье, §8.2) и их предел на экране.
TYPE_SCALE = frozenset({36, 32, 24, 18, 16, 14, 12})
MAX_SIZES = 6


def load_stub(name: str = "ui_stub_server") -> Any:
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def shot_query(shot: str) -> str:
    """Параметры адреса снимка: у листа сомелье — ещё и вино стенда."""
    wine = SOMM_SHOTS.get(shot)
    return f"wine={wine}&shot={shot}" if wine else f"shot={shot}"


class Texts(HTMLParser):
    """Видимый текст страницы и отдельно — текст внутри блоков рекомендаций."""

    VOID = frozenset(
        [
            "area",
            "base",
            "br",
            "col",
            "embed",
            "hr",
            "img",
            "input",
            "link",
            "meta",
            "source",
            "track",
            "wbr",
        ]
    )

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[tuple[str, bool, bool]] = []  # (тег, в рекомендациях, скрыт)
        self.visible: list[str] = []
        self.reco: list[str] = []
        self.status = ""
        self.marks: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "html":
            self.status = values.get("data-shot") or ""
            self.marks = {k: v or "" for k, v in values.items() if k.startswith("data-")}
        if tag in self.VOID:
            return
        parent_reco = self.stack[-1][1] if self.stack else False
        parent_hidden = self.stack[-1][2] if self.stack else False
        classes = (values.get("class") or "").split()
        hidden = parent_hidden or "hidden" in classes or tag in {"script", "style", "template"}
        self.stack.append((tag, parent_reco or values.get("id") in RECO_IDS, hidden))

    def handle_endtag(self, tag: str) -> None:
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                return

    def handle_data(self, data: str) -> None:
        if not self.stack or not data.strip():
            return
        _, reco, hidden = self.stack[-1]
        if hidden:
            return
        self.visible.append(data.strip())
        if reco:
            self.reco.append(data.strip())


def chrome(args: list[str], profile: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    cmd = [
        str(CHROME),
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-extensions",
        f"--user-data-dir={profile}",
        "--force-device-scale-factor=1",
        "--virtual-time-budget=20000",
        *args,
    ]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,  # код возврата Chrome ничего не говорит: судим по DOM и снимку
    )


def check_dom(base: str, shot: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as profile:
        done = chrome(
            ["--window-size=500,1000", "--dump-dom", f"{base}/?{shot_query(shot)}"], profile
        )
    parser = Texts()
    parser.feed(done.stdout)
    visible, reco = " ".join(parser.visible), " ".join(parser.reco)
    marks = parser.marks
    sizes = sorted({int(x) for x in marks.get("data-sizes", "").split(",") if x}, reverse=True)
    return {
        "status": parser.status or "нет",
        "reco_chars": len(reco),
        "reco_percent": reco.count("%"),
        "percent": int(marks.get("data-percent") or 0),
        "stop": sorted({m.group(0).lower() for m in STOP.finditer(visible)}),
        "low": [x for x in marks.get("data-low", "").split(" | ") if x],
        "sizes": sizes,
        "notices": int(marks.get("data-notices") or 0),
    }


def problems(dom: dict[str, Any]) -> list[str]:
    """Что не так на снимке; пусто — всё по правилам."""
    out = []
    if dom["status"] != "ready":
        out.append(f"драйвер: {dom['status']}")
    if dom["reco_percent"] or dom["percent"]:
        out.append(f"знак % вне «% об.»: {dom['percent']}, в подборе: {dom['reco_percent']}")
    if dom["stop"]:
        out.append("стоп-слова: " + ", ".join(dom["stop"]))
    if dom["low"]:
        out.append("контраст ниже 4,5:1: " + "; ".join(dom["low"][:4]))
    odd = [size for size in dom["sizes"] if size not in TYPE_SCALE]
    if odd or len(dom["sizes"]) > MAX_SIZES:
        out.append(f"кегли {dom['sizes']}: вне шкалы {odd}, предел {MAX_SIZES}")
    if dom["notices"] > 1:
        out.append(f"плашек 149-ФЗ на экране: {dom['notices']}")
    return out


def screenshot(base: str, shot: str, out: Path, height: int = HEIGHT) -> int:
    from PIL import Image

    with tempfile.TemporaryDirectory() as profile:
        raw = Path(profile) / "raw.png"
        chrome(
            [
                f"--window-size=500,{height + 160}",
                f"--screenshot={raw}",
                f"{base}/__stub/frame?{shot_query(shot)}&w={WIDTH}&h={height}",
            ],
            profile,
        )
        with Image.open(raw) as image:
            image.crop((0, 0, WIDTH, height)).convert("RGB").save(out, optimize=True)
    return out.stat().st_size


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="папка для снимков")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ["SVS_DATA_DIR"]) if os.environ.get("SVS_DATA_DIR") else None,
    )
    parser.add_argument("--csv", type=Path, default=None, help="выгрузка организатора")
    every = [*SHOTS, *SOMM_SHOTS]
    parser.add_argument("--only", nargs="*", choices=every, default=every)
    parser.add_argument("--no-screens", action="store_true", help="только проверка DOM")
    parser.add_argument(
        "--height", type=int, default=HEIGHT, help="высота фрейма: больше 844 — вся страница"
    )
    args = parser.parse_args(argv)
    if not CHROME.is_file():
        print(f"нет Chrome: {CHROME}", file=sys.stderr)
        return 2

    stub = load_stub()
    csv_path = args.csv or stub.default_csv()
    server = stub.make_server(port=0, data_dir=args.data_dir, delay=0.3, csv_path=csv_path)
    stub.serve_in_thread(server)
    # Стенд листа: поток без пауз (speed=0) — ход кончается сразу, каретку ждёт драйвер.
    somm = load_stub("somm_stub")
    somm_server = somm.make_server(port=0, data_dir=args.data_dir, speed=0.0)
    somm.serve_in_thread(somm_server)
    page_base = f"http://127.0.0.1:{server.server_address[1]}"
    somm_base = f"http://127.0.0.1:{somm_server.server_address[1]}"
    args.out.mkdir(parents=True, exist_ok=True)
    failed = 0
    try:
        print(f"{'снимок':<14} {'png, байт':>10} {'кегли':<22} {'149':>3}  замечания")
        for shot in args.only:
            base = somm_base if shot in SOMM_SHOTS else page_base
            dom = check_dom(base, shot)
            name = shot if args.height == HEIGHT else f"{shot}_{args.height}"
            size = (
                0
                if args.no_screens
                else screenshot(base, shot, args.out / f"{name}.png", args.height)
            )
            bad = problems(dom)
            failed += bool(bad)
            sizes = "/".join(str(x) for x in dom["sizes"])
            print(f"{shot:<14} {size:>10} {sizes:<22} {dom['notices']:>3}  {'; '.join(bad) or '—'}")
    finally:
        for running in (server, somm_server):
            running.shutdown()
            running.server_close()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
