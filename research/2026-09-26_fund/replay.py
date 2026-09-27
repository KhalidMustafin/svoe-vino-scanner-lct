"""CPU-реплей сервиса (код этого дерева) на снимке данных fund — без видеокарты и без Ollama.

Как стенд `e3_measure.py` (Э3), но на `frozen_data` и с точками подмены для кандидатов:
- поиск по картинке — `index._rank` по CPU-векторам запросов (`acc_plan/retrieval/qemb_*.npz`,
  окна bottle, label, band, full × 1152, float16) вместо SigLIP; `cv_fn(Q)` подменяет выдачу
  (например, счёт другой модели по всем slug → `candidates_from_scores`);
- чтение этикетки — из кэша стенда (`runs/field25/iters/cache/reads`, читатель
  `vlm|qwen3.5:4b|f3a017317f04`); промах — сбой кадра (адрес Ollama — порт 9, туда не дойдёт);
- разбор чтения (`read_label`: словарь, Э6), признаки, ранкер `-goal`, H5, P1 (Э1), Э2 и данные
  Э4 — код и файлы сервиса как есть. Другая модель выбора — `model_path` (формат
  `app/resolve/learned.py`).

`scan()` гоняет кадр целиком и записывает всё, что нужно для повторов без картинки:
top-20 CV, отрыв, чтение (поля + сырой текст + статус), признаки кандидатов, счёт ранкера и
ответ. `resolve()` повторяет слой выбора по записанным CV и чтению за миллисекунды —
равенство с `scan()` проверяет `build_trainpool.py`.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import protocol as P

import app.api.service as service_mod
from app.api.config import CV_PER_SLUG, ServiceSettings
from app.api.service import ScannerService, _Run
from app.features.contracts import Candidate, VisualResult
from app.features.embedder import unit_rows
from app.reading.contracts import LabelFields, Reading
from app.resolve.features import TextRead

READER_IDENT = "vlm|qwen3.5:4b|f3a017317f04"
QFILES = {
    "catalog_v2": "qemb_catalog_v2.npz",
    "krasnostop_v1": "qemb_krasnostop_v1.npz",
    "ooc_v2": "qemb_ooc_v2.npz",
}
CvFn = Callable[[np.ndarray], tuple[list[Candidate], float]]


def load_qemb(set_name: str) -> tuple[list[str], np.ndarray, list[str]]:
    """qids, векторы (N, 4, 1152) float16 — окна запроса в порядке names, и names."""
    z = np.load(P.QEMB / QFILES[set_name])
    return [str(q) for q in z["qids"]], z["vectors"], [str(n) for n in z["names"]]


def image_path(m: Mapping[str, Any]) -> Path:
    p = Path(m["image"])
    return p if p.is_absolute() else P.FD / p


def sha1_bytes(*parts: bytes) -> str:
    import hashlib

    h = hashlib.sha1()
    for part in parts:
        h.update(part)
    return h.hexdigest()


def cand_json(visual: VisualResult) -> list[list[Any]]:
    return [[c.slug, c.score, c.view, c.rank] for c in visual.candidates]


def visual_of(cands: list[list[Any]], margin: float, model: str = "") -> VisualResult:
    return VisualResult(
        candidates=[Candidate(slug=s, score=sc, view=v, rank=r) for s, sc, v, r in cands],
        margin=margin,
        timings_ms={},
        model=model,
    )


def read_json(read: TextRead) -> dict[str, Any]:
    return {
        "fields": read.fields.model_dump(mode="json") if read.fields is not None else None,
        "raw_text": read.raw_text,
    }


def read_of(d: Mapping[str, Any]) -> TextRead:
    f = d.get("fields")
    return TextRead(
        fields=LabelFields.model_validate(f) if f is not None else None, raw_text=d.get("raw_text")
    )


class FundReplay:
    """Сервис на снимке данных с CV по векторам запросов и чтением из кэша стенда."""

    def __init__(
        self,
        *,
        model_path: Path | None = None,
        live: bool = False,
        extra_env: Mapping[str, str] | None = None,
    ) -> None:
        env = {
            "SVS_DATA_DIR": str(P.FROZEN),
            "SVS_DATASET_DIR": str(P.FROZEN_DATASET),
            "SVS_CACHE_DIR": str(P.FROZEN / "cache"),
            "SVS_LIVE_CARDS": "1" if live else "0",
            "SVS_OLLAMA_URL": "http://127.0.0.1:9",  # промах кэша не дойдёт до Ollama
            "SVS_DEVICE": "cpu",
            "SVS_BUDGET_MS": "30000",
            "SVS_VLM_TIMEOUT_MS": "15000",
        }
        if model_path is not None:
            env["SVS_RESOLVE_MODEL"] = str(model_path)
        # Кандидат PREREG_final за флагом сервиса: SVS_CANDIDATE, SVS_CV_ADAPTER (путь к карте).
        env.update(extra_env or {})
        self.settings = ServiceSettings.from_env(env)
        self.svc = svc = ScannerService.load(self.settings)
        self.cv_fn: CvFn | None = None
        self._visual: VisualResult | None = None
        self.cap: dict[str, Any] = {}
        self.hits = self.misses = 0
        #: Чтение, которое вернёт читатель на следующем кадре вместо кэша стенда (`None` — кэш).
        self.forced_reading: Reading | None = None
        locked = svc.vlm
        assert locked is not None
        ident = f"{locked.id}|{locked.version}|{locked.params_hash}"
        assert ident == READER_IDENT, ident

        def cached_read(image: np.ndarray, *, crop: str, budget_ms: int) -> Reading:
            if self.forced_reading is not None:  # записанное чтение не из кэша стенда (студия)
                self.hits += 1
                return self.forced_reading
            key = sha1_bytes(
                image.tobytes(), str(image.shape).encode(), crop.encode(), ident.encode()
            )
            path = P.READS / f"{key}.json"
            if not path.exists():
                self.misses += 1
                raise RuntimeError("промах кэша чтений")
            self.hits += 1
            return Reading.model_validate_json(path.read_text(encoding="utf-8"))

        locked.read = cached_read  # type: ignore[method-assign]

        def search(image: np.ndarray, run: _Run) -> VisualResult:
            assert self._visual is not None
            return self._visual

        svc._search = search  # type: ignore[method-assign]
        orig_read = svc._read

        def spy_read(image: np.ndarray, run: _Run) -> Any:
            reads, fields = orig_read(image, run)
            self.cap["reads"], self.cap["fields"] = reads, fields
            self.cap["vlm_evidence"] = run.evidence.get("vlm")
            return reads, fields

        svc._read = spy_read  # type: ignore[method-assign]
        orig_rank = service_mod.rank_query_detailed

        def spy_rank(*a: Any, **kw: Any) -> Any:
            ranking, features = orig_rank(*a, **kw)
            self.cap["ranking"], self.cap["features"] = ranking, features
            return ranking, features

        service_mod.rank_query_detailed = spy_rank  # на весь процесс: сервис зовёт по имени модуля

    # -------------------------------------------------------------- CV
    def default_cv(self, Q: np.ndarray) -> tuple[list[Candidate], float]:
        return self.svc.index._rank(Q, self.settings.top_k, CV_PER_SLUG)

    def cv_scores(self, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Счёт zmax по всем slug (порядок `index.slug_order`) и номер строки индекса, давшей его."""
        idx = self.svc.index
        from app.features.index import align_pairs

        if idx.adapter is not None:  # индекс уже в пространстве адаптера — запрос туда же
            Q = idx.adapter.apply(Q)
        sims = align_pairs(idx.vectors @ Q.T, idx.view_groups, idx.base_rows)
        best = sims.max(axis=1)
        scores = np.full(idx.n_slugs, -np.inf, dtype=np.float32)
        np.maximum.at(scores, idx.slug_ids, best)
        order = np.argsort(-best, kind="stable")
        _, first = np.unique(idx.slug_ids[order], return_index=True)
        return scores, order[first]

    def candidates_from_scores(
        self, scores: np.ndarray, best_row: np.ndarray | None = None, top_k: int | None = None
    ) -> tuple[list[Candidate], float]:
        """Выдача CV из счёта по всем slug — как `_rank` (обрезка снизу нулём, отрыв по каталогу)."""
        idx = self.svc.index
        k = top_k or self.settings.top_k
        order = np.argsort(-scores, kind="stable")
        views = idx.views
        cands = [
            Candidate(
                slug=idx.slug_order[int(s)],
                score=float(max(0.0, scores[s])),
                view=views[int(best_row[s])] if best_row is not None else "bottle",
                rank=r,
            )
            for r, s in enumerate(order[:k], start=1)
        ]
        margin = float(scores[order[0]] - scores[order[1]]) if len(order) > 1 else 0.0
        return cands, max(0.0, margin)

    # -------------------------------------------------------------- кадр целиком
    def scan(self, data: bytes, qvec: np.ndarray) -> dict[str, Any]:
        """Кадр целиком; запись хранит всё, что нужно `resolve_row` для повтора без картинки."""
        Q = unit_rows(np.asarray(qvec, dtype=np.float32))
        cands, margin = (self.cv_fn or self.default_cv)(Q)
        self._visual = VisualResult(
            candidates=cands, margin=margin, timings_ms={}, model=self.svc.index.meta.model
        )
        self.cap = {}
        before = self.misses
        result = self.svc.scan(data)
        ok = self.misses == before and result.outcome != "error" and "reads" in self.cap
        row: dict[str, Any] = {"ok": ok, "answer": result.slug, "outcome": result.outcome}
        row["cv"] = cand_json(self._visual)
        row["margin"] = margin
        if not ok:
            row["error"] = result.error
            return row
        ev = result.evidence
        read = self.cap["reads"][self.svc.reader_key]
        vlm = self.cap.get("vlm_evidence") or {}
        row.update(
            p_answer=result.confidence.top1,
            read=read_json(read),
            vlm_status=vlm.get("status"),
            vlm_lines=len(vlm.get("lines") or []),
            resolve=ev.get("resolve"),
            degraded=list(result.degraded),
        )
        ranking = self.cap.get("ranking")
        features = self.cap.get("features")
        if ranking is not None and features is not None:
            names = list(self.svc.model.feature_names)
            row["rank_slugs"] = list(ranking.slugs)
            row["rank_scores"] = [round(float(s), 6) for s in ranking.scores]
            row["p_top1"] = ranking.p_top1
            row["feat_slugs"] = list(features.slugs)
            row["features"] = features.matrix(names).round(6).tolist() if features.rows else []
        return row

    # -------------------------------------------------------------- слой выбора по записи
    def resolve(
        self,
        visual: VisualResult,
        read: TextRead,
        *,
        vlm_status: str | None = "ok",
        vlm_lines: int = 1,
    ) -> dict[str, Any]:
        """Ответ сервиса на готовых CV и чтении: ранкер, H5, P1, Э2 — без картинки."""
        t = time.perf_counter()
        run = _Run(time.perf_counter, t, t + 30)
        run.evidence["vlm"] = {"status": vlm_status, "lines": ["x"] * vlm_lines}
        key = self.svc.reader_key
        ans = self.svc._resolve(visual, {key: read}, read.fields, run)
        return {"answer": ans["slug"], "resolve": run.evidence.get("resolve")}

    def resolve_row(self, row: Mapping[str, Any]) -> str:
        """Ответ по записи `trainpool` (поля cv, margin, read, vlm_status, vlm_lines)."""
        v = visual_of(row["cv"], row["margin"], self.svc.index.meta.model)
        return self.resolve(
            v,
            read_of(row["read"]),
            vlm_status=row.get("vlm_status"),
            vlm_lines=row.get("vlm_lines", 0),
        )["answer"]
