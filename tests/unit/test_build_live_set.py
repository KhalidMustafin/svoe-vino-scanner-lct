"""`scripts/build_live_set.py`: комплект «CSV + живые карточки» на игрушечном каталоге.

Проверяется то, на чём держатся ворота Э3: строки базового индекса и gt идут в комплект как
есть (байт в байт), маска `base_rows` отмечает ровно их, карточки-дубли вин CSV отсекаются по
группам вин, словарь собирается по живому gt.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from catalog import TOY_RECORDS, wine_record

from app.config import REPO_ROOT
from app.features.contracts import IndexMeta
from app.features.index import VisualIndex
from app.reading.lexicon.build import Lexicon

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import build_live_set

VIEWS = ("bottle", "label", "band")


def write_index(path: Path, slugs: list[str], seed: int) -> VisualIndex:
    rng = np.random.default_rng(seed)
    rows = [s for s in slugs for _ in VIEWS]
    views = [v for _ in slugs for v in VIEWS]
    vectors = rng.normal(size=(len(rows), 4)).astype(np.float32)
    meta = IndexMeta(
        model="fake/pixel", dim=4, views=list(VIEWS), n_slugs=len(slugs), n_vectors=len(rows)
    )
    index = VisualIndex(rows, views, vectors, meta)  # type: ignore[arg-type]
    index.save(path)
    return index


@pytest.fixture
def live_inputs(tmp_path: Path) -> dict[str, Path]:
    data = tmp_path / "data"
    (data / "index").mkdir(parents=True)
    (data / "gt").mkdir(parents=True)
    base = [r["slug"] for r in TOY_RECORDS[:3]]
    write_index(data / "index" / "visual-s2so400m.npz", base, seed=0)
    with (data / "gt" / "gt_tokens.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for record in TOY_RECORDS[:3]:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    cards = [
        wine_record("portal-merlo", winery="Портальная", key_tokens=("портальная",), name="Мерло"),
        wine_record("portal-dup", winery="Тестовая Долина", name="Алиготе Баррель"),
        wine_record("portal-rose", winery="Портальная", key_tokens=("портальная",), name="Розе"),
    ]
    new_cards = tmp_path / "new_cards.jsonl"
    new_cards.write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\r\n" for c in cards), encoding="utf-8"
    )
    # Индекс стенда: другие векторы старых slug (их брать нельзя) и виды двух новых карточек.
    write_index(tmp_path / "extra.npz", [*base, "portal-merlo", "portal-rose"], seed=1)
    groups = {
        "W1": {"members": [base[0], "portal-dup"]},
        "W2": {"members": ["portal-merlo"]},
        "W3": {"members": ["portal-rose"]},
    }
    (tmp_path / "groups.json").write_text(json.dumps(groups), encoding="utf-8")
    return {
        "data": data,
        "extra": tmp_path / "extra.npz",
        "groups": tmp_path / "groups.json",
        "cards": new_cards,
    }


def test_live_set_keeps_the_base_as_is_and_adds_only_new_wines(live_inputs):
    data = live_inputs["data"]
    rc = build_live_set.main(
        [
            "--data-dir", str(data),
            "--extra-index", str(live_inputs["extra"]),
            "--wine-groups", str(live_inputs["groups"]),
            "--new-cards", str(live_inputs["cards"]),
        ]
    )  # fmt: skip
    assert rc == 0
    with (
        np.load(data / "index" / "visual-s2so400m.npz") as base,
        np.load(data / "index" / "visual-s2so400m-live71.npz") as live,
    ):
        n = len(base["slugs"])
        assert np.array_equal(live["vectors"][:n], base["vectors"])  # те же float16
        assert live["slugs"][:n].tolist() == base["slugs"].tolist()
        assert live["slugs"][n:].tolist() == ["portal-merlo"] * 3 + ["portal-rose"] * 3
        assert live["base_rows"].tolist() == [True] * n + [False] * 6
    index = VisualIndex.load(data / "index" / "visual-s2so400m-live71.npz")
    assert index.n_base == 9 and len(index) == 15 and index.meta.n_slugs == 5

    base_gt = (data / "gt" / "gt_tokens.jsonl").read_bytes()
    live_gt = (data / "gt" / "gt_tokens-live71.jsonl").read_bytes()
    assert live_gt.startswith(base_gt) and b"\r" not in live_gt
    added = [json.loads(x)["slug"] for x in live_gt[len(base_gt) :].decode("utf-8").splitlines()]
    assert added == ["portal-merlo", "portal-rose"]  # дубль вина CSV отсечён

    lexicon = Lexicon.load(data / "index" / "lexicon-live71.json")
    assert lexicon.meta["source_sha1"] == build_live_set.sha1(
        data / "gt" / "gt_tokens-live71.jsonl"
    )
    assert lexicon.n_slugs == 5
    assert any("portal-merlo" in entry.slugs for entry in lexicon.entries)


def test_live_set_refuses_an_extra_index_without_the_cards(live_inputs, tmp_path):
    write_index(tmp_path / "short.npz", ["portal-merlo"], seed=2)
    with pytest.raises(SystemExit, match="не те"):
        build_live_set.main(
            [
                "--data-dir", str(live_inputs["data"]),
                "--extra-index", str(tmp_path / "short.npz"),
                "--wine-groups", str(live_inputs["groups"]),
                "--new-cards", str(live_inputs["cards"]),
            ]
        )  # fmt: skip


def test_csv_duplicates_are_members_of_a_group_with_a_csv_slug():
    groups = {
        "W1": {"members": ["a", "x"]},
        "W2": {"members": ["y"]},
        "W3": {"members": ["z", "b"]},
    }
    assert build_live_set.csv_duplicates(["x", "y", "z"], {"a", "b"}, groups) == {"x", "z"}
