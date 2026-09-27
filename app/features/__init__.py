"""Визуальный канал: виды кадра, их векторы и индекс эталонов каталога."""

from app.features.contracts import (
    VIEWS,
    Aggregation,
    Candidate,
    Embedder,
    IndexMeta,
    ViewName,
    VisualResult,
)
from app.features.embedder import (
    DEFAULT_MODEL,
    LETTERBOX_SIDE,
    ModelNotAvailable,
    SiglipEmbedder,
    letterbox,
    unit_rows,
)
from app.features.index import IndexMismatch, VisualIndex
from app.features.views import (
    BACKGROUND,
    MAX_SIDE,
    QUERY_WINDOWS,
    WINDOWS_NOTE,
    alpha_bounds,
    flatten_alpha,
    from_bottle,
    from_packshot,
    from_query,
    order_views,
)

__all__ = [
    "BACKGROUND",
    "DEFAULT_MODEL",
    "LETTERBOX_SIDE",
    "MAX_SIDE",
    "QUERY_WINDOWS",
    "VIEWS",
    "WINDOWS_NOTE",
    "Aggregation",
    "Candidate",
    "Embedder",
    "IndexMeta",
    "IndexMismatch",
    "ModelNotAvailable",
    "SiglipEmbedder",
    "ViewName",
    "VisualIndex",
    "VisualResult",
    "alpha_bounds",
    "flatten_alpha",
    "from_bottle",
    "from_packshot",
    "from_query",
    "letterbox",
    "order_views",
    "unit_rows",
]
