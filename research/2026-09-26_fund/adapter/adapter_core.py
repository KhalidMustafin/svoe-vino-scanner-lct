"""Обучаемый адаптер поиска поверх замороженных векторов SigLIP2 so400m (фундаментальный трек, CPU).

Идея: небольшая остаточная проекция `x -> norm(x + Δ(x))` сдвигает векторы так, чтобы окна
реального кадра (bottle, label, band, full) ложились ближе к видам эталона своей карточки, чем
к чужим. Учится InfoNCE по всему каталогу (2 103 карточки × виды) через тот же счёт, что у
сервиса (`zmax`: стандартизация пары «окно × вид» по каталогу, максимум по окнам и видам) —
поэтому после обучения адаптер встаёт в сервис без переделки поиска: запрос — матрица поверх
вектора SigLIP, индекс — та же (или своя) матрица, применённая к сохранённым векторам.

Стороны (`side`):
    q       только запрос, индекс как есть;
    shared  одна проекция и для запроса, и для индекса (метрика);
    asym    своя проекция у запроса и у индекса.

Форма (`kind`): `lowrank` (Δ = x U Vᵀ, ранг r), `full` (Δ = x A), `mlp` (1 скрытый слой).
Начало — Δ = 0, то есть ровно нынешний поиск.

Отрицательные — все карточки каталога; те, что в одной `visual_group` с верной (та же этикетка,
другой объём или год), из знаменателя убраны: визуально их не различить, и учить это — шум.
Карточки той же винодельни — трудные отрицательные, их вес в знаменателе `beta` (1 — без
выделения).

Ничего из kr-test здесь нет и быть не может: пул берётся из `trainpool.*`, и каждый вызов
`load_pool` прогоняет `protocol.assert_no_test` по всем id и картинкам.
"""

from __future__ import annotations

import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
FUND_DIR = HERE.parent
sys.path.insert(0, str(FUND_DIR))

import protocol as P  # noqa: E402

OUT = P.FUND / "adapter"
WINDOWS = ("bottle", "label", "band", "full")


# ------------------------------------------------------------------ пул
@dataclass
class Pool:
    rows: list[dict[str, Any]]
    Q: np.ndarray  # (N, 4, d) float32, единичная норма
    pos: list[np.ndarray]  # номера slug (порядок index.slug_order) «того же вина»; пусто — вне каталога
    groups: np.ndarray  # (N,) str
    sets: np.ndarray  # (N,) str
    feats: np.ndarray  # (N, 20, 34) признаки продукта (для проверки)

    def __len__(self) -> int:
        return len(self.rows)


def load_pool(slug_order: Sequence[str], log: Any = print) -> Pool:
    rows = P.jsonl(P.PROTOCOL / "trainpool.jsonl")
    z = np.load(P.PROTOCOL / "trainpool.npz")
    n = P.assert_no_test([r["id"] for r in rows], [r["image"] for r in rows])
    log(f"assert_no_test: проверено {n} id и картинок пула, kr-test нет")
    assert [str(i) for i in z["ids"]] == [r["id"] for r in rows]
    assert [str(x) for x in z["qemb_names"]] == list(WINDOWS)
    pos_of = {s: i for i, s in enumerate(slug_order)}
    Q = z["qemb"].astype(np.float32)
    Q /= np.linalg.norm(Q, axis=2, keepdims=True)
    pos = []
    for r in rows:
        if not r["in_catalog"]:
            pos.append(np.zeros(0, dtype=np.int64))
            continue
        slugs = [r["slug"], *(r.get("acceptable") or [])]
        # 5 меток v2 — карточки вне индекса CSV (vostok-krasnoe, merlo-2 …): у них верного нет
        ids = sorted({pos_of[s] for s in slugs if s in pos_of})
        pos.append(np.asarray(ids, dtype=np.int64))
    return Pool(
        rows=rows,
        Q=Q,
        pos=pos,
        groups=np.array([r["group"] for r in rows]),
        sets=np.array([r["set"] for r in rows]),
        feats=z["features"],
    )


# ------------------------------------------------------------------ индекс
@dataclass
class IndexData:
    """Индекс сервиса в порядке «вид за видом» — так стандартизация zmax — это reshape."""

    vectors: np.ndarray  # (N, d) float32 в исходном порядке строк сервиса
    views: list[str]
    slug_ids: np.ndarray  # (N,) номер slug строки
    slug_order: list[str]
    view_rows: list[np.ndarray]  # строки каждого вида (как index.view_groups)
    perm: np.ndarray  # порядок строк «вид за видом»
    n_per_view: int
    winery: np.ndarray  # (S,) номер винодельни slug
    vgroup: np.ndarray  # (S,) номер visual_group slug или -1

    @property
    def n_slugs(self) -> int:
        return len(self.slug_order)


def index_data(svc: Any) -> IndexData:
    idx = svc.index
    groups = [np.asarray(g) for g in idx.view_groups]
    sizes = {len(g) for g in groups}
    assert len(sizes) == 1, sizes  # у каждого вида одинаковое число строк
    assert idx.base_rows is None
    perm = np.concatenate(groups)
    attrs = svc.attrs
    wmap: dict[str, int] = {}
    gmap: dict[str, int] = {}
    winery = np.zeros(idx.n_slugs, dtype=np.int64)
    vgroup = np.full(idx.n_slugs, -1, dtype=np.int64)
    for i, s in enumerate(idx.slug_order):
        w = attrs.get(s)
        key = (w.winery or "").strip().lower() if w else f"?{s}"
        winery[i] = wmap.setdefault(key, len(wmap))
        if w is not None and w.visual_group:
            vgroup[i] = gmap.setdefault(w.visual_group, len(gmap))
    return IndexData(
        vectors=np.asarray(idx.vectors, dtype=np.float32),
        views=list(idx.views),
        slug_ids=np.asarray(idx.slug_ids),
        slug_order=list(idx.slug_order),
        view_rows=groups,
        perm=perm,
        n_per_view=sizes.pop(),
        winery=winery,
        vgroup=vgroup,
    )


# ------------------------------------------------------------------ адаптер
class Residual(torch.nn.Module):
    """x -> norm(x + Δ(x)); Δ = 0 в начале."""

    def __init__(self, dim: int, kind: str, rank: int = 128, hidden: int = 512) -> None:
        super().__init__()
        self.kind = kind
        if kind == "lowrank":
            self.U = torch.nn.Parameter(torch.randn(dim, rank) / math.sqrt(dim))
            self.V = torch.nn.Parameter(torch.zeros(rank, dim))
        elif kind == "full":
            self.A = torch.nn.Parameter(torch.zeros(dim, dim))
        elif kind == "mlp":
            self.l1 = torch.nn.Linear(dim, hidden)
            self.l2 = torch.nn.Linear(hidden, dim)
            torch.nn.init.zeros_(self.l2.weight)
            torch.nn.init.zeros_(self.l2.bias)
        else:
            raise ValueError(kind)

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "lowrank":
            return (x @ self.U) @ self.V
        if self.kind == "full":
            return x @ self.A
        return self.l2(F.gelu(self.l1(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x + self.delta(x), dim=-1)

    def reg(self) -> torch.Tensor:
        """L2 к тождеству: штраф на то, что сдвигает векторы."""
        if self.kind == "lowrank":
            return (self.U @ self.V).pow(2).sum()
        if self.kind == "full":
            return self.A.pow(2).sum()
        return self.l2.weight.pow(2).sum() + self.l1.weight.pow(2).sum() * 1e-2


@dataclass
class Config:
    side: str = "q"  # q | shared | asym
    kind: str = "lowrank"
    rank: int = 128
    hidden: int = 512
    lr: float = 2e-3
    wd: float = 1e-3  # вес reg()
    epochs: int = 30
    batch: int = 64
    beta: float = 1.0  # вес отрицательных той же винодельни в знаменателе
    tau0: float = 0.02  # начальная температура (учится)
    mask_vgroup: bool = True
    val_frac: float = 0.15  # доля групп обучения под раннюю остановку
    sets: tuple[str, ...] = ("pairs", "pairs_phone", "catalog_v2", "kr_dev", "kr_dev_sp")
    seed: int = 0
    threads: int = 8

    def name(self) -> str:
        form = {"lowrank": f"r{self.rank}", "full": "full", "mlp": f"mlp{self.hidden}"}[self.kind]
        return f"{self.side}-{form}-wd{self.wd:g}-b{self.beta:g}-e{self.epochs}"


class Adapter(torch.nn.Module):
    def __init__(self, cfg: Config, dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        mk = lambda: Residual(dim, cfg.kind, cfg.rank, cfg.hidden)  # noqa: E731
        self.q = mk()
        self.i = None if cfg.side == "q" else (self.q if cfg.side == "shared" else mk())
        self.log_tau = torch.nn.Parameter(torch.tensor(math.log(cfg.tau0)))

    def fq(self, x: torch.Tensor) -> torch.Tensor:
        return self.q(x)

    def fi(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.i is None else self.i(x)

    def reg(self) -> torch.Tensor:
        r = self.q.reg()
        if self.i is not None and self.i is not self.q:
            r = r + self.i.reg()
        return r

    # numpy-обёртки для инференса
    @torch.no_grad()
    def np_q(self, Q: np.ndarray) -> np.ndarray:
        return self.fq(torch.from_numpy(np.ascontiguousarray(Q, dtype=np.float32))).numpy()

    @torch.no_grad()
    def np_i(self, V: np.ndarray) -> np.ndarray:
        return self.fi(torch.from_numpy(np.ascontiguousarray(V, dtype=np.float32))).numpy()


def zmax_torch(
    Qa: torch.Tensor, Ivm: torch.Tensor, n_per_view: int, slug_vm: torch.Tensor, n_slugs: int
) -> torch.Tensor:
    """Счёт сервиса (`align_pairs` + max по окнам + max по slug), дифференцируемый.

    Qa (B, W, d); Ivm (N, d) — строки индекса «вид за видом»; slug_vm (N,) — slug строки.
    """
    B = Qa.shape[0]
    S = torch.einsum("nd,bwd->bnw", Ivm, Qa)  # (B, N, W)
    M = S.mean(dim=(1, 2), keepdim=True)
    s = S.std(dim=(1, 2), keepdim=True, correction=0)
    V = S.reshape(B, -1, n_per_view, S.shape[2])  # (B, views, rows, W)
    mu = V.mean(dim=2, keepdim=True)
    sd = V.std(dim=2, keepdim=True, correction=0)
    A = M.unsqueeze(1) + s.unsqueeze(1) * (V - mu) / sd
    row = A.amax(dim=3).reshape(B, -1)  # (B, N)
    out = torch.full((B, n_slugs), -1e4, dtype=row.dtype)
    return out.scatter_reduce(1, slug_vm.unsqueeze(0).expand(B, -1), row, "amax", include_self=True)


# ------------------------------------------------------------------ счёт numpy (как сервис)
def cv_scores_np(Qa: np.ndarray, Ia: np.ndarray, ix: IndexData) -> tuple[np.ndarray, np.ndarray]:
    """Ровно `FundReplay.cv_scores`, но с адаптированными запросом и индексом."""
    from app.features.index import align_pairs

    sims = align_pairs(Ia @ Qa.T, ix.view_rows, None)
    best = sims.max(axis=1)
    scores = np.full(ix.n_slugs, -np.inf, dtype=np.float32)
    np.maximum.at(scores, ix.slug_ids, best)
    order = np.argsort(-best, kind="stable")
    _, first = np.unique(ix.slug_ids[order], return_index=True)
    return scores, order[first]


def all_scores(model: Adapter | None, pool_Q: np.ndarray, ix: IndexData) -> np.ndarray:
    """Счёт zmax по всем slug для набора кадров (N, S), быстро (батчами на torch)."""
    V = torch.from_numpy(ix.vectors[ix.perm])
    slug_vm = torch.from_numpy(ix.slug_ids[ix.perm])
    out = np.zeros((len(pool_Q), ix.n_slugs), dtype=np.float32)
    with torch.no_grad():
        Ia = V if model is None else model.fi(V)
        for a in range(0, len(pool_Q), 128):
            q = torch.from_numpy(pool_Q[a : a + 128])
            qa = q if model is None else model.fq(q)
            out[a : a + 128] = zmax_torch(qa, Ia, ix.n_per_view, slug_vm, ix.n_slugs).numpy()
    return out


# ------------------------------------------------------------------ обучение
def split_groups(groups: Sequence[str], frac: float, seed: int) -> set[str]:
    keys = sorted(set(groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    return set(keys[: int(round(frac * len(keys)))])


@dataclass
class TrainLog:
    cfg: dict[str, Any]
    n_train: int = 0
    n_val: int = 0
    best_epoch: int = -1
    history: list[dict[str, float]] = field(default_factory=list)
    seconds: float = 0.0


def _loss(
    model: Adapter,
    Qb: torch.Tensor,
    posb: list[np.ndarray],
    Ivm: torch.Tensor,
    ix: IndexData,
    slug_vm: torch.Tensor,
    cfg: Config,
    winery_t: torch.Tensor,
    vgroup_t: torch.Tensor,
) -> torch.Tensor:
    scores = zmax_torch(model.fq(Qb), Ivm, ix.n_per_view, slug_vm, ix.n_slugs)
    logits = scores / model.log_tau.exp()
    B = len(posb)
    posm = torch.zeros(B, ix.n_slugs, dtype=torch.bool)
    for b, p in enumerate(posb):
        posm[b, torch.from_numpy(p)] = True
    allowed = torch.ones_like(posm)
    if cfg.mask_vgroup:
        for b, p in enumerate(posb):
            g = vgroup_t[torch.from_numpy(p)]
            g = g[g >= 0]
            if len(g):
                allowed[b] &= ~torch.isin(vgroup_t, g)
        allowed |= posm
    den = logits.masked_fill(~allowed, -1e9)
    if cfg.beta != 1.0:
        same = torch.zeros_like(posm)
        for b, p in enumerate(posb):
            same[b] = torch.isin(winery_t, winery_t[torch.from_numpy(p)])
        same &= ~posm
        den = den + same.float() * math.log(cfg.beta)
    num = logits.masked_fill(~posm, -1e9)
    return (torch.logsumexp(den, 1) - torch.logsumexp(num, 1)).mean()


def train_adapter(
    pool: Pool, train_idx: np.ndarray, ix: IndexData, cfg: Config, log: Any = None
) -> tuple[Adapter, TrainLog]:
    """Адаптер на кадрах `train_idx` (только в каталоге и из `cfg.sets`); ранняя остановка по
    доле групп обучения (`val_frac`) — ни одного кадра вне `train_idx`."""
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.threads)
    t0 = time.perf_counter()
    keep = np.array(
        [i for i in train_idx if len(pool.pos[i]) and pool.sets[i] in cfg.sets], dtype=np.int64
    )
    val_groups = split_groups(pool.groups[keep], cfg.val_frac, cfg.seed + 7) if cfg.val_frac else set()
    is_val = np.array([pool.groups[i] in val_groups for i in keep])
    tr, va = keep[~is_val], keep[is_val]
    model = Adapter(cfg, pool.Q.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    V = torch.from_numpy(ix.vectors[ix.perm])
    slug_vm = torch.from_numpy(ix.slug_ids[ix.perm])
    winery_t = torch.from_numpy(ix.winery)
    vgroup_t = torch.from_numpy(ix.vgroup)
    Qt = torch.from_numpy(pool.Q)
    rng = np.random.default_rng(cfg.seed)
    tl = TrainLog(cfg=asdict(cfg), n_train=len(tr), n_val=len(va))
    best: tuple[float, float] | None = None
    best_state = None

    def evaluate() -> tuple[float, float]:
        if not len(va):
            return 0.0, 0.0
        model.eval()
        with torch.no_grad():
            Ia = model.fi(V)
            sc = zmax_torch(model.fq(Qt[va]), Ia, ix.n_per_view, slug_vm, ix.n_slugs).numpy()
            loss = float(
                _loss(model, Qt[va], [pool.pos[i] for i in va], Ia, ix, slug_vm, cfg, winery_t, vgroup_t)
            )
        model.train()
        top = sc.argmax(1)
        acc = float(np.mean([top[k] in pool.pos[i] for k, i in enumerate(va)]))
        return acc, loss

    acc0, loss0 = evaluate()
    tl.history.append({"epoch": 0, "val_top1": acc0, "val_loss": loss0})
    best, best_state, tl.best_epoch = (acc0, -loss0), {k: v.clone() for k, v in model.state_dict().items()}, 0
    for ep in range(1, cfg.epochs + 1):
        order = rng.permutation(tr)
        tot = 0.0
        for a in range(0, len(order), cfg.batch):
            b = order[a : a + cfg.batch]
            Ia = model.fi(V)
            loss = _loss(model, Qt[b], [pool.pos[i] for i in b], Ia, ix, slug_vm, cfg, winery_t, vgroup_t)
            obj = loss + cfg.wd * model.reg()
            opt.zero_grad()
            obj.backward()
            opt.step()
            tot += loss.item() * len(b)
        acc, vloss = evaluate()
        tl.history.append(
            {"epoch": ep, "train_loss": tot / max(1, len(tr)), "val_top1": acc, "val_loss": vloss,
             "tau": float(model.log_tau.exp())}
        )
        if log:
            log(f"  ep {ep:2d} train {tot / max(1, len(tr)):.4f} val top1 {acc:.4f} loss {vloss:.4f}")
        key = (acc, -vloss)
        if len(va) == 0 or key > best:
            best, tl.best_epoch = key, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    tl.seconds = time.perf_counter() - t0
    return model, tl


# ------------------------------------------------------------------ разрез по группам
def group_kfold(groups: Sequence[str], sets: Sequence[str], k: int, seed: int) -> np.ndarray:
    """Номер фолда кадра. Группы — целиком; крупные первыми, каждая в фолд с меньшим числом кадров
    (с учётом набора: штраф |Δ кадров| + |Δ кадров того же набора|)."""
    groups = np.asarray(groups)
    sets = np.asarray(sets)
    keys = sorted(set(groups.tolist()))
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    members = {g: np.flatnonzero(groups == g) for g in keys}
    keys.sort(key=lambda g: -len(members[g]))
    set_names = sorted(set(sets.tolist()))
    cnt = np.zeros(k)
    per = np.zeros((k, len(set_names)))
    fold = np.full(len(groups), -1, dtype=np.int64)
    for g in keys:
        m = members[g]
        vec = np.array([np.sum(sets[m] == s) for s in set_names])
        cost = cnt + (per * (vec > 0)).sum(1)
        f = int(np.argmin(cost))
        fold[m] = f
        cnt[f] += len(m)
        per[f] += vec
    return fold


def cfg_from(d: Mapping[str, Any]) -> Config:
    d = dict(d)
    if "sets" in d:
        d["sets"] = tuple(d["sets"])
    return Config(**d)
