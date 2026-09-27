"""Замкнутые линейные адаптеры поиска: PCA-выбеливание каталога и обученное выбеливание по парам.

Классика поиска экземпляра (Jégou & Chum 2012; Radenović et al. 2016, «learned whitening»):

- `pcaw` — без меток: центр и главные оси индекса эталонов, веса осей `λ^(-α/2)` с усадкой;
  общие для всех бутылок направления («любая бутылка вина похожа на любую») гасятся;
- `lw` — по парам «окно кадра — вид эталона своей карточки»: `C_S` — ковариация разностей
  совпадающих пар, `P₁ = (C_S + εI)^(-1/2)` гасит то, чем кадр отличается от своего эталона
  (свет, фон, ракурс, телефон), затем поворот на главные оси индекса в новом пространстве.

Обе проекции — одна матрица `d × D` и вектор центра: к запросу — после SigLIP, к индексу — один раз
при сборке. Параметров обучения нет, кроме усадки и числа осей.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LinearMap:
    mean: np.ndarray  # (d,)
    W: np.ndarray  # (d, D)
    name: str = ""

    def apply(self, X: np.ndarray) -> np.ndarray:
        Y = (np.asarray(X, dtype=np.float32) - self.mean) @ self.W
        n = np.linalg.norm(Y, axis=-1, keepdims=True)
        return (Y / np.maximum(n, 1e-12)).astype(np.float32)

    np_q = apply
    np_i = apply


def _eigh_desc(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lam, U = np.linalg.eigh(C.astype(np.float64))
    order = np.argsort(-lam)
    return np.maximum(lam[order], 0.0), U[:, order]


def pca_whitening(index_vecs: np.ndarray, *, alpha: float = 0.5, shrink: float = 0.1,
                  dims: int | None = None) -> LinearMap:
    """Выбеливание по индексу: веса осей (λ + shrink·mean λ)^(-α/2)."""
    X = np.asarray(index_vecs, dtype=np.float64)
    m = X.mean(0)
    lam, U = _eigh_desc(np.cov((X - m).T, bias=True))
    D = dims or X.shape[1]
    reg = lam[:D] + shrink * lam.mean()
    W = U[:, :D] * (reg ** (-alpha / 2))
    return LinearMap(m.astype(np.float32), W.astype(np.float32), f"pcaw-a{alpha}-s{shrink}-d{D}")


def learned_whitening(q: np.ndarray, p: np.ndarray, index_vecs: np.ndarray, *, shrink: float = 0.1,
                      dims: int | None = None, rotate: bool = True) -> LinearMap:
    """Radenović LW: q[i] и p[i] — совпадающая пара (окно кадра, вид эталона верной карточки)."""
    q = np.asarray(q, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    X = np.asarray(index_vecs, dtype=np.float64)
    m = X.mean(0)
    diff = q - p
    Cs = diff.T @ diff / len(diff)
    lam, U = _eigh_desc(Cs)
    reg = lam + shrink * lam.mean()
    P1 = U * (reg ** -0.5)  # (d, d): x -> x @ P1 = Cs^(-1/2) (в базисе U)
    if not rotate:
        D = dims or X.shape[1]
        return LinearMap(m.astype(np.float32), P1[:, :D].astype(np.float32), f"lw-norot-s{shrink}")
    Xw = (X - m) @ P1
    lam2, U2 = _eigh_desc(np.cov(Xw.T, bias=True))
    D = dims or X.shape[1]
    W = P1 @ U2[:, :D]
    return LinearMap(m.astype(np.float32), W.astype(np.float32), f"lw-s{shrink}-d{D}")
