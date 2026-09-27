"""Честный протокол фундаментального трека (26.09): пути снимка данных, разбиение, стражи, счёт.

Каждый скрипт обучения и разработки импортирует отсюда `assert_no_test` и вызывает его на своих
кадрах и картинках до любой работы. kr-test выдаёт только `test_frames(цель)`: база
(`baseline`, EVAL_PROTOCOL.md §4) и итоговая оценка кандидата (`final:<имя>`, §5).

Разбиение — `split.json` рядом (коммит до любой модели), копия — `svs-logs/.../fund/protocol/`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(r"<корень>")
FUND = ROOT / "svs-logs" / "somm-2409" / "fund"
FROZEN = FUND / "frozen_data"  # снимок общего data 25.09 19:51 (SHA1SUMS.txt внутри)
FROZEN_FD = FROZEN / "field_dataset"  # meta.jsonl наборов и словарь вин на тот же момент
FROZEN_DATASET = FROZEN / "dataset"  # strapi_output0709.csv
PROTOCOL = FUND / "protocol"
SPLIT = HERE / "split.json"
SPLIT_SEED = 20260926

ITERS = ROOT / "svoe-vino-scanner" / "runs" / "field25" / "iters"
READS = ITERS / "cache" / "reads"  # кэш чтений стенда, ключ — sha1 кадра, дописывается только
QEMB = ROOT / "svs-logs" / "somm-2409" / "acc_plan" / "retrieval"
FD = ROOT / "field_dataset"  # картинки полевых кадров (пути в meta относительные)


def jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def sha1_file(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def norm_path(p: str | Path) -> str:
    return os.path.normcase(os.path.abspath(str(p)))


# ------------------------------------------------------------------ разбиение и стражи
@cache
def load_split() -> dict[str, Any]:
    return json.loads(SPLIT.read_text(encoding="utf-8"))


@cache
def forbidden_ids() -> frozenset[str]:
    """query_id kr-test: основные и same_packshot вин kr-test."""
    s = load_split()
    return frozenset(s["kr_test"]) | frozenset(s["kr_test_same_packshot"])


@cache
def forbidden_images() -> frozenset[str]:
    """Картинки krasnostop, которые нельзя брать ни в обучение, ни в разработку."""
    return frozenset(norm_path(p) for p in load_split()["forbidden_kr_images"])


@cache
def allowed_kr_images() -> frozenset[str]:
    """Единственные картинки krastostop/images, которые можно брать в обучение: кадры kr-dev."""
    return frozenset(norm_path(p) for p in load_split()["allowed_kr_images"])


class TestLeak(AssertionError):
    """Кадр или картинка kr-test попали туда, где их быть не должно."""


def assert_no_test(qids: Iterable[str] = (), images: Iterable[str | Path] = ()) -> int:
    """Падает, если среди кадров есть kr-test, а среди картинок — картинка krasnostop вне kr-dev.

    Возвращает число проверенных элементов: скрипт пишет его в свой лог.
    """
    bad_ids = forbidden_ids()
    n = 0
    for q in qids:
        n += 1
        if q in bad_ids:
            raise TestLeak(f"kr-test в обучении или разработке: {q}")
    kr_root = norm_path(ROOT / "krastostop")
    ok_kr = allowed_kr_images()
    bad_img = forbidden_images()
    for p in images:
        n += 1
        np_ = norm_path(p)
        if np_ in bad_img:
            raise TestLeak(f"картинка kr-test: {p}")
        if np_.startswith(kr_root) and np_ not in ok_kr:
            raise TestLeak(f"картинка krasnostop вне kr-dev: {p}")
    return n


def dev_frames() -> list[str]:
    """kr-dev: основные кадры (без same_packshot) — пул разработки."""
    return list(load_split()["kr_dev"])


LEDGER = PROTOCOL / "test_ledger.md"


def test_frames(purpose: str) -> list[str]:
    """kr-test. Только база (`baseline`, §4 протокола) и итоговая оценка кандидата (`final:<имя>`, §5).

    Каждое открытие дописывает строку «открыт» в журнал `test_ledger.md` до того, как скрипт
    увидит хоть один кадр: просмотр без записи невозможен.
    """
    if purpose != "baseline" and not purpose.startswith("final:"):
        raise TestLeak("kr-test открывают только база и итоговая оценка (EVAL_PROTOCOL.md §3, §5)")
    from datetime import datetime

    stamp = datetime.now().strftime("%d.%m %H:%M:%S")
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(f"| {stamp} | открыт | {purpose} | {Path(sys.argv[0]).name} | — |\n")
    return list(load_split()["kr_test"])


# ------------------------------------------------------------------ вина и группы
@cache
def wine_of() -> dict[str, str]:
    """slug -> wine_id словаря `wine_groups_final.json` снимка."""
    groups = json.loads(
        (FROZEN_FD / "catalog" / "wine_groups_final.json").read_text(encoding="utf-8")
    )
    return {slug: wid for wid, g in groups.items() for slug in g.get("members") or []}


@cache
def wine_attrs() -> dict[str, dict[str, Any]]:
    return json.loads(
        (FROZEN_FD / "catalog" / "wine_groups_final.json").read_text(encoding="utf-8")
    )


class DSU:
    def __init__(self) -> None:
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def label_nodes(m: Mapping[str, Any]) -> list[str]:
    """Узлы «того же вина» кадра: slug, acceptable и их wine_id."""
    wines = wine_of()
    slugs = [m["slug"], *(m.get("acceptable") or [])]
    return [f"s:{s}" for s in slugs] + [f"w:{wines[s]}" for s in slugs if s in wines]


def group_key(dsu: DSU, m: Mapping[str, Any]) -> str:
    nodes = label_nodes(m)
    for a in nodes[1:]:
        dsu.union(nodes[0], a)
    return dsu.find(nodes[0])


# ------------------------------------------------------------------ счёт
def correct(answer: str | None, m: Mapping[str, Any]) -> bool:
    """«То же вино»: ответ в {slug, acceptable} метки."""
    return answer is not None and answer in {m["slug"], *(m.get("acceptable") or [])}


def strict(answer: str | None, m: Mapping[str, Any]) -> bool:
    return answer is not None and answer == m["slug"]


def boot_ci_by_group(
    ok: Sequence[bool], groups: Sequence[str], *, seed: int = 0, n_boot: int = 10000
) -> list[float]:
    """95 % интервал микро-точности бутстрэпом по группам вин (кадры группы — вместе)."""
    keys = sorted(set(groups))
    idx = {k: i for i, k in enumerate(keys)}
    hit = np.zeros(len(keys))
    cnt = np.zeros(len(keys))
    for o, g in zip(ok, groups, strict=True):
        hit[idx[g]] += bool(o)
        cnt[idx[g]] += 1
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    micro = hit[draws].sum(axis=1) / cnt[draws].sum(axis=1)
    return [
        round(100 * float(np.percentile(micro, 2.5)), 2),
        round(100 * float(np.percentile(micro, 97.5)), 2),
    ]


def sign_test(fixes: int, breaks: int) -> float:
    """Точный двусторонний биномиальный тест знаков по кадрам, где верность сменилась."""
    n = fixes + breaks
    if n == 0:
        return 1.0
    k = max(fixes, breaks)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2**n
    return min(1.0, 2 * tail)
