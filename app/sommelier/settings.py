"""Настройки сомелье из окружения `SVS_SOMM_*` (договор `docs/api-sommelier.md`, §1 и §6.7).

Единственное место, где читаются шесть переменных сомелье:

    SVS_SOMM_LIVE        живой голос; 0 — ответы листа только шаблоном (`reason: "off"`)
    SVS_SOMM_INPUT       вопрос текстом; 0 — `422 {"detail": "вопрос текстом выключен"}`
    SVS_SOMM_DIR         данные `data/somm/*.json`; пусто — `<SVS_DATA_DIR>/somm`
    SVS_SOMM_TIMEOUT_MS  предел живого голоса от запроса к Ollama до последнего токена
    SVS_SOMM_QUIET_S     тихое окно ворот после каждого `/v1/eval/predict`
    SVS_SOMM_SAFETY      смысловой слой барьера (`safety.py`, rubert-tiny2 с диска); по
                         умолчанию выключен: замер 24.09 не нашёл порога без лишних отказов

Настройки сервиса (`app.api.config.ServiceSettings.somm`) держат этот же объект: каталог данных
сомелье читают и «Сомелье у полки» слоя «после поиска», и маршруты сомелье, и ворота
видеокарты берут отсюда тихое окно. Флаги — `1/true/on/yes` и `0/false/off/no` без учёта
регистра, пусто — по умолчанию, иное — `SettingsError` (отказ старта, как у остальных `SVS_*`).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import REPO_ROOT
from app.sommelier.gate import DEFAULT_QUIET_S

TRUE_WORDS = frozenset({"1", "true", "on", "yes"})
FALSE_WORDS = frozenset({"0", "false", "off", "no"})

#: Предел живого голоса, мс (договор, §1 «Выключатели»).
DEFAULT_TIMEOUT_MS = 4000


class SettingsError(ValueError):
    """Переменная `SVS_SOMM_*` задана, но не разбирается или выходит за пределы."""


def _default_dir() -> Path:
    return REPO_ROOT / "data" / "somm"


def _flag(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = (env.get(name) or "").strip().lower()
    if not raw:
        return default
    if raw in TRUE_WORDS:
        return True
    if raw in FALSE_WORDS:
        return False
    raise SettingsError(f"{name}={raw!r}: ожидается 1/0, true/false, on/off или yes/no")


def _number(env: Mapping[str, str], name: str, default: float, cast: type) -> Any:
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError:
        raise SettingsError(f"{name}: ожидается число, получено {raw!r}") from None


@dataclass(frozen=True)
class SommSettings:
    """Выключатели и пределы сомелье. Пути — абсолютные или от текущего каталога."""

    live: bool = True
    input: bool = True
    data_dir: Path = field(default_factory=_default_dir)
    timeout_ms: int = DEFAULT_TIMEOUT_MS
    quiet_s: float = DEFAULT_QUIET_S
    safety: bool = False

    def __post_init__(self) -> None:
        if self.timeout_ms <= 0:
            raise SettingsError(f"SVS_SOMM_TIMEOUT_MS должен быть > 0, получено {self.timeout_ms}")
        if self.quiet_s < 0:
            raise SettingsError(f"SVS_SOMM_QUIET_S не может быть < 0, получено {self.quiet_s}")

    @property
    def timeout_s(self) -> float:
        return self.timeout_ms / 1000

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None, *, data_dir: Path | None = None
    ) -> SommSettings:
        """Настройки из окружения. `data_dir` — каталог данных сервиса (`SVS_DATA_DIR`); не задан —
        из той же переменной окружения, пусто — `data/` репозитория."""
        env = os.environ if environ is None else environ
        if data_dir is None:
            raw_data = (env.get("SVS_DATA_DIR") or "").strip()
            data_dir = Path(raw_data).expanduser().resolve() if raw_data else REPO_ROOT / "data"
        raw_dir = (env.get("SVS_SOMM_DIR") or "").strip()
        return cls(
            live=_flag(env, "SVS_SOMM_LIVE", cls.live),
            input=_flag(env, "SVS_SOMM_INPUT", cls.input),
            data_dir=Path(raw_dir).expanduser().resolve() if raw_dir else data_dir / "somm",
            timeout_ms=_number(env, "SVS_SOMM_TIMEOUT_MS", cls.timeout_ms, int),
            quiet_s=_number(env, "SVS_SOMM_QUIET_S", cls.quiet_s, float),
            safety=_flag(env, "SVS_SOMM_SAFETY", cls.safety),
        )

    def public(self) -> dict[str, Any]:
        """Для журнала и зондов: путь строкой."""
        return {
            "live": self.live,
            "input": self.input,
            "data_dir": str(self.data_dir),
            "timeout_ms": self.timeout_ms,
            "quiet_s": self.quiet_s,
            "safety": self.safety,
        }
