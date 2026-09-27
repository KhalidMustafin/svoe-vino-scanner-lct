"""Смысловой слой барьера: опасная тема, названная своими словами.

Перенос `Code/backend/app/nlu/safety.py` «Лозы» и её эталонов `data/reference/safety_anchors.json`
(`safety_anchors.json` рядом). Барьер (`barrier.py`) стоит на словах и фразах и ловит прямую
речь; окольную — «не могу остановиться на одном бокале», «голова тяжёлая после вчерашнего, каким
вином поправиться», «два бокала за вечер это много» — слов темы в ней нет, и списком её не
добрать. Здесь её узнаёт `cointegrated/rubert-tiny2`: вопрос сравнивается с эталонами тем по
смыслу, пять ближайших голосуют.

**Слой односторонний.** Он спрашивается, только когда правила промолчали (`Router._text`), и
умеет лишь добавить отказ, но никогда не снимает отказ правил. Ошибка модели стоит лишнего
отказа, обратная ошибка стоила бы совета выпить кормящей матери.

**По умолчанию слой выключен** (`SVS_SOMM_SAFETY=0`). Эталон голосует, только если он ближе
`near`, а тема побеждает с долей не меньше `min_share` (пороги «Лозы»: 0,75 и 0,55; у
rubert-tiny2 любые две русские фразы похожи примерно на 0,6 — это фон, а не сходство). Замер
24.09 сквозь весь маршрут (`research/2026-09-24_somm/safety_eval.py`, `safety_eval.json`): ни на
одном пороге модель не добавляет ни одного отказа, не добавив и лишнего. При 0,75 — +9 отказов
на 208 опасных фразах и +10 лишних на 594 законных (среди них «мне сорок, хочу разобраться в
вине» — липкий отказ по возрасту, «сколько стоит выдерживать в бокале» — отказ о цене); при 0,84
и выше — ни того ни другого. Окольную зависимость, трезвость и здоровье эталоны «Лозы» «без
себя» почти не ловят: при 0,75 — 3/18, 4/10 и 5/14 против 3/18, 4/10 и 3/14 у одних правил.

**Модель — только с диска.** Веса читаются из локального кэша Hugging Face
(`local_files_only=True`, как `HF_HUB_OFFLINE=1`): сервис ничего не скачивает. Нет весов,
`transformers` или `torch` — слой выключен, работают одни правила, а `/v1/health.somm.safety`
пишет `status: "missing"`. Загрузка ленивая: в фоне при старте сервиса (`start`), вопрос,
пришедший раньше, ждёт её не дольше `LOAD_WAIT_S`. Считается на CPU — видеокарта остаётся
скану; глобальные настройки `torch` (число потоков) слой не трогает.

Вопрос гостя слой не пишет никуда: в журнал идут только тема и доля голосов.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MODEL_NAME = "cointegrated/rubert-tiny2"
ANCHORS_PATH = Path(__file__).with_name("safety_anchors.json")
#: Классы эталонов «Лозы» → темы отказа договора (§4.4). «офтоп» и «ok» — не отказ: они нужны,
#: чтобы вопросу «что к борщу» не пришлось выбирать между темами риска.
TOPIC_OF_CLASS: dict[str, str] = {
    "медицина": "medical",
    "возраст": "minors",
    "вождение": "driving",
    "зависимость": "addiction",
    "трезвость": "sobriety",
    "здоровье": "health",
    "цена": "price",
}
NON_REFUSAL_CLASSES: tuple[str, ...] = ("офтоп", "ok")
CLASSES: tuple[str, ...] = (*TOPIC_OF_CLASS, *NON_REFUSAL_CLASSES)
#: Сколько ближайших эталонов голосуют — как у «Лозы».
NEIGHBOURS = 5
#: Доля голосов победившей темы — как у «Лозы».
MIN_SHARE = 0.55
#: Близость, ниже которой эталон не голосует — порог «Лозы». Замер `safety_eval.py`: при нём слой
#: добавляет и отказы, и лишние отказы, а без лишних (0,84 и выше) не добавляет ничего — поэтому
#: слой выключен по умолчанию, а не «включён с безопасным порогом» (см. докстринг модуля).
NEAR = 0.75
#: Длина вопроса в токенах — как у «Лозы».
MAX_TOKENS = 64
#: Сколько вопрос ждёт фоновой загрузки модели, прежде чем пройти по одним правилам.
LOAD_WAIT_S = 20.0

Encoder = Callable[[Sequence[str]], Any]  # тексты → матрица L2-нормированных векторов (numpy)


def load_anchors(path: Path = ANCHORS_PATH) -> dict[str, list[str]]:
    """Эталоны по классам; неизвестные классы файла пропускаются, пустых фраз нет."""
    raw = json.loads(path.read_text(encoding="utf-8"))["anchors"]
    return {
        name: [str(phrase).strip() for phrase in raw.get(name, ()) if str(phrase).strip()]
        for name in CLASSES
    }


def rubert_encoder(model_name: str = MODEL_NAME) -> Encoder:
    """Кодировщик rubert-tiny2 с диска: среднее по токенам, L2-норма, CPU, без скачивания."""
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True)
    model.eval()
    model.to("cpu")

    def encode(texts: Sequence[str]) -> Any:
        batch = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=MAX_TOKENS,
            return_tensors="pt",
        )
        with torch.inference_mode():
            out = model(**batch)
        mask = batch["attention_mask"].unsqueeze(-1).float()
        vectors = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        vectors = torch.nn.functional.normalize(vectors, dim=1)
        return np.asarray(vectors.numpy(), dtype=np.float32)

    return encode


class SafetyLayer:
    """Тема отказа по смыслу вопроса или `None`. Ошибки не бросает: без модели — `None`.

    `encoder` задан — модель не грузится (тесты, зонды); иначе `rubert_encoder` при `load`.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        anchors_path: Path = ANCHORS_PATH,
        near: float = NEAR,
        min_share: float = MIN_SHARE,
        neighbours: int = NEIGHBOURS,
        model_name: str = MODEL_NAME,
        encoder: Encoder | None = None,
        load_wait_s: float = LOAD_WAIT_S,
    ) -> None:
        self.enabled = enabled
        self.anchors_path = anchors_path
        self.near = near
        self.min_share = min_share
        self.neighbours = neighbours
        self.model_name = model_name
        self.load_wait_s = load_wait_s
        self._encoder = encoder
        self._labels: list[str] = []
        self._texts: list[str] = []
        self._vectors: Any = None
        self._status = "off" if not enabled else "idle"
        self._error: str | None = None
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._load_ms: int | None = None
        self._calls = 0
        self._refused: dict[str, int] = {}
        self._last_ms: float | None = None
        if not enabled:
            self._done.set()

    # -------------------------------------------------------------- загрузка
    def start(self) -> None:
        """Загрузить модель в фоне (старт сервиса): первый вопрос не ждёт её с нуля."""
        with self._lock:
            if self._status != "idle":
                return
            self._status = "loading"
        threading.Thread(target=self._load, name="somm-safety", daemon=True).start()

    def load(self) -> bool:
        """Загрузить в этом потоке (или дождаться уже начатой загрузки); готово — `True`."""
        with self._lock:
            mine = self._status == "idle"
            if mine:
                self._status = "loading"
        if mine:
            self._load()
        else:
            self._done.wait()
        return self._status == "ready"

    def _load(self) -> None:
        started = time.perf_counter()
        try:
            anchors = load_anchors(self.anchors_path)
            texts = [phrase for name in CLASSES for phrase in anchors[name]]
            labels = [name for name in CLASSES for _ in anchors[name]]
            if not any(labels.count(name) for name in TOPIC_OF_CLASS):
                raise ValueError(f"эталоны тем отказа пусты: {self.anchors_path}")
            encoder = self._encoder or rubert_encoder(self.model_name)
            vectors = encoder(texts)
        except (ImportError, OSError) as exc:
            self._fail("missing", exc)
            return
        except Exception as exc:  # noqa: BLE001 — слой не роняет сервис: остаются правила
            self._fail("error", exc)
            return
        with self._lock:
            self._encoder = encoder
            self._texts, self._labels, self._vectors = texts, labels, vectors
            self._load_ms = int((time.perf_counter() - started) * 1000)
            self._status = "ready"
        self._done.set()
        logger.info("Смысловой слой барьера: %d эталонов за %d мс", len(texts), self._load_ms)

    def _fail(self, status: str, exc: BaseException) -> None:
        with self._lock:
            self._status = status
            self._error = type(exc).__name__
        self._done.set()
        logger.warning(
            "Смысловой слой барьера недоступен (%s): отказы — только по правилам",
            type(exc).__name__,
        )

    @property
    def ready(self) -> bool:
        return self._status == "ready"

    # -------------------------------------------------------------- вопрос
    def classify(
        self, question: str, *, exclude: frozenset[str] = frozenset()
    ) -> tuple[str, float] | None:
        """Класс эталонов с долей голосов или `None` — риска не видно или слоя нет.

        `exclude` — эталоны, которые не голосуют (замер «без себя»: фраза набора, совпавшая с
        эталоном, не должна найти саму себя).
        """
        if not self.enabled or not question.strip():
            return None
        if not self._done.is_set():
            self.start()
            self._done.wait(self.load_wait_s)
        if self._status != "ready":
            return None
        import numpy as np

        started = time.perf_counter()
        try:
            vector = self._encoder([question])[0]  # type: ignore[misc]
            scores = self._vectors @ vector
        except Exception as exc:  # noqa: BLE001 — слой не роняет ответ
            logger.warning("Смысловой слой барьера сорвался (%s)", type(exc).__name__)
            return None
        order = np.argsort(-scores)
        votes: dict[str, float] = {}
        taken = 0
        for index in order:
            if taken == self.neighbours:
                break
            if self._texts[index] in exclude:
                continue
            taken += 1
            score = float(scores[index])
            if score < self.near:
                continue
            label = self._labels[index]
            votes[label] = votes.get(label, 0.0) + (score - self.near)
        with self._lock:
            self._calls += 1
            self._last_ms = round((time.perf_counter() - started) * 1000, 2)
        total = sum(votes.values())
        if total <= 0:
            return None
        name = max(votes, key=lambda key: votes[key])
        share = votes[name] / total
        if share < self.min_share:
            return None
        return name, share

    def refusal_topic(self, question: str) -> str | None:
        """Тема отказа договора (`medical`, `minors`…) или `None`. Вопрос в журнал не идёт."""
        found = self.classify(question)
        if found is None or found[0] not in TOPIC_OF_CLASS:
            return None
        topic = TOPIC_OF_CLASS[found[0]]
        with self._lock:
            self._refused[topic] = self._refused.get(topic, 0) + 1
        logger.info("somm safety: тема %s (доля %.2f) там, где правила молчали", topic, found[1])
        return topic

    def stats(self) -> Mapping[str, Any]:
        """Для `/v1/health.somm.safety`: включён ли слой, загружен ли, пороги и счёт отказов."""
        with self._lock:
            return {
                "enabled": self.enabled,
                "status": self._status,
                "error": self._error,
                "model": self.model_name,
                "anchors": len(self._texts),
                "near": self.near,
                "min_share": self.min_share,
                "neighbours": self.neighbours,
                "load_ms": self._load_ms,
                "calls": self._calls,
                "refused": dict(sorted(self._refused.items())),
                "last_ms": self._last_ms,
            }
