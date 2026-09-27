"""RapidOCR с явно закреплённым PP-OCRv5 на ONNX Runtime CPU.

Сверено с исходниками установленного rapidocr 3.9.2 (движок не запускался):
- в config.yaml по умолчанию Det/Rec — PP-OCRv6 small, поэтому версия задаётся явно
  и сверяется с `engine.cfg` после создания;
- `ParseParams.update_batch` принимает engine_type/model_type/ocr_version только как Enum,
  строка даёт TypeError — параметры переводятся в Enum перед созданием;
- в default_models.yaml для v5 есть eslav только mobile: `rec_model_type="server"` падает
  с `RapidOcrConfigError`, а не подменяется молча и не выдаётся за «недоступен»;
- ndarray с тремя каналами движок считает BGR.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import time
from typing import Any, Literal

import numpy as np

from app.reading.contracts import CropName, Reading, TextLine, params_hash
from app.reading.readers.base import (
    box_from_quad,
    elapsed_ms_since,
    ensure_rgb,
    make_reading,
    quad_angle,
    sort_reading_order,
)

POSTPROCESS_VERSION = "1"
OCR_VERSION = "PP-OCRv5"

# Поле параметра → имя Enum в rapidocr.utils.typings.
_ENUM_FIELDS = {
    "engine_type": "EngineType",
    "model_type": "ModelType",
    "ocr_version": "OCRVersion",
}
_LANG_ENUMS = {"Det": "LangDet", "Rec": "LangRec"}
# Сообщения rapidocr о неверной конфигурации — это ошибка настройки, а не недоступность.
_CONFIG_ERROR_MARKERS = (
    "Unsupported",  # «Unsupported configuration», «Unsupported Rec.lang_type=...»
    "Invalid OCR configuration",
    "must be Enum",
    "is not a valid key",
)


class RapidOcrConfigError(RuntimeError):
    """Движок собрался не с той версией или моделью, что запрошена."""


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


class RapidOcrReader:
    id = "rapidocr"

    def __init__(
        self,
        *,
        rec_model_type: Literal["mobile", "server"] = "mobile",
        # mobile на CPU в 3–5 раз быстрее server при том же тексте (замер 15.09.2026).
        det_model_type: Literal["mobile", "server"] = "mobile",
        rec_lang: str = "eslav",
        use_cls: bool = False,
        text_score: float = 0.5,
    ) -> None:
        self.rec_model_type = rec_model_type
        self.det_model_type = det_model_type
        self.rec_lang = rec_lang
        self.use_cls = use_cls
        self.text_score = text_score
        self.version = f"{_version('rapidocr')}+{OCR_VERSION}"
        self.params_hash = params_hash(
            {
                "ocr_version": OCR_VERSION,
                "det_model_type": det_model_type,
                "rec_model_type": rec_model_type,
                "rec_lang": rec_lang,
                "use_cls": use_cls,
                "text_score": text_score,
                "postprocess": POSTPROCESS_VERSION,
            }
        )
        self._engine: Any = None

    def engine_params(self) -> dict[str, Any]:
        """Параметры RapidOCR строками — читаемо и без импорта пакета. В Enum — `to_enums`."""
        return {
            "Global.use_cls": self.use_cls,
            "Global.text_score": self.text_score,
            "EngineConfig.onnxruntime.use_cuda": False,
            "Det.engine_type": "onnxruntime",
            "Det.lang_type": "ch",
            "Det.model_type": self.det_model_type,
            "Det.ocr_version": OCR_VERSION,
            "Rec.engine_type": "onnxruntime",
            "Rec.lang_type": self.rec_lang,
            "Rec.model_type": self.rec_model_type,
            "Rec.ocr_version": OCR_VERSION,
        }

    @staticmethod
    def to_enums(params: dict[str, Any], typings: Any) -> dict[str, Any]:
        """Строковые значения Det/Rec → Enum из `rapidocr.utils.typings`."""
        out: dict[str, Any] = {}
        for dotted, value in params.items():
            section, _, field = dotted.partition(".")
            enum_name = _ENUM_FIELDS.get(field)
            if field == "lang_type":
                enum_name = _LANG_ENUMS.get(section)
            if section in ("Det", "Rec") and enum_name and isinstance(value, str):
                value = getattr(typings, enum_name)(value)
            out[dotted] = value
        return out

    def available(self) -> bool:
        return (
            importlib.util.find_spec("rapidocr") is not None
            and importlib.util.find_spec("onnxruntime") is not None
        )

    @staticmethod
    def check_config(cfg: Any, expected: dict[str, Any]) -> None:
        """Сверка итогового конфига движка с запрошенным: v6 не должен включиться молча."""
        for dotted, want in expected.items():
            section, _, field = dotted.partition(".")
            if section not in ("Det", "Rec"):
                continue
            node = cfg[section] if isinstance(cfg, dict) else getattr(cfg, section)
            got = node[field] if isinstance(node, dict) else getattr(node, field)
            got = getattr(got, "value", got)  # enum → строка
            want = getattr(want, "value", want)
            if str(got) != str(want):
                raise RapidOcrConfigError(f"{dotted}: запрошено {want!r}, в движке {got!r}")

    @staticmethod
    def _load_rapidocr() -> tuple[Any, Any]:
        from rapidocr import RapidOCR
        from rapidocr.utils import typings

        return RapidOCR, typings

    def _ensure_engine(self) -> Any:
        if self._engine is None:
            engine_cls, typings = self._load_rapidocr()
            params = self.engine_params()
            try:
                engine = engine_cls(params=self.to_enums(params, typings))
            except (TypeError, ValueError) as exc:
                if any(marker in str(exc) for marker in _CONFIG_ERROR_MARKERS):
                    raise RapidOcrConfigError(f"RapidOCR отверг конфигурацию: {exc}") from exc
                raise
            cfg = getattr(engine, "cfg", None)
            if cfg is None:
                raise RapidOcrConfigError("у движка нет cfg: версию модели не проверить")
            self.check_config(cfg, params)
            self._engine = engine
        return self._engine

    def read(self, image: np.ndarray, *, crop: CropName, budget_ms: int) -> Reading:
        """Бюджет не прерывает движок; `budget_ms` <= 0 — timeout без запуска."""
        ensure_rgb(image)
        t0 = time.perf_counter()
        params = self.params_hash
        if budget_ms <= 0:
            return make_reading(
                self, image, params=params, crop=crop, status="timeout", elapsed_ms=0
            )
        try:
            engine = self._ensure_engine()
        except RapidOcrConfigError:
            raise
        except Exception as exc:  # noqa: BLE001 — нет пакета или моделей
            return make_reading(
                self,
                image,
                params=params,
                crop=crop,
                status="unavailable",
                elapsed_ms=elapsed_ms_since(t0),
                raw=f"{type(exc).__name__}: {exc}",
            )
        try:
            output = engine(np.ascontiguousarray(image[:, :, ::-1]))
        except Exception as exc:  # noqa: BLE001
            return make_reading(
                self,
                image,
                params=params,
                crop=crop,
                status="error",
                elapsed_ms=elapsed_ms_since(t0),
                raw=f"{type(exc).__name__}: {exc}",
            )
        boxes = getattr(output, "boxes", None)
        txts = getattr(output, "txts", None) or ()
        scores = getattr(output, "scores", None) or ()
        h, w = image.shape[:2]
        lines: list[TextLine] = []
        if boxes is not None:
            for i, (quad, text, score) in enumerate(zip(boxes, txts, scores, strict=False)):
                if not str(text).strip():
                    continue
                lines.append(
                    TextLine(
                        id=i,
                        text=str(text).strip(),
                        box=box_from_quad(quad, w, h),
                        angle=quad_angle(quad),
                        conf=float(score),
                    )
                )
        lines = sort_reading_order(lines)
        return make_reading(
            self,
            image,
            params=params,
            crop=crop,
            status="ok" if lines else "empty",
            elapsed_ms=elapsed_ms_since(t0),
            lines=lines,
        )
