# syntax=docker/dockerfile:1.7
#
# Сервис сканера: `python -m app.api` на :8080 (POST /v1/eval/predict, /v1/scan, GET /v1/health).
#
#     docker build --build-arg TORCH=gpu -t svoe-vino-scanner:gpu .    # CUDA 12.8, torch cu128
#     docker build --build-arg TORCH=cpu -t svoe-vino-scanner:cpu .    # без видеокарты
#     docker build --build-arg TORCH=gpu --build-arg SIGLIP_AT_BUILD=1 -t svoe-vino-scanner:gpu .
#
# Окружение ставится строго по uv.lock (`uv sync --frozen`): те же колёса, на которых мерили.
# Экстры — только те, что нужны сервису: torch (gpu или cpu), cv (transformers для SigLIP) и api
# (FastAPI). `ocr` (EasyOCR, RapidOCR) сервису не нужен: он читает этикетку одной моделью через
# Ollama, а EasyOCR к тому же притянул бы второй torch с PyPI.
#
# Чего в образе нет и что приходит томами (compose.yml):
#   /data     пачка данных: индекс, карта адаптера, словарь, признаки каталога, фото выгрузки
#             (deploy/pack_data.sh); только чтение
#   /dataset  выгрузка организатора: strapi_output0709.csv — описания карточек; только чтение
#   /hf/hub   кэш Hugging Face с весами SigLIP (сервис их не скачивает: local_files_only)
# С `SIGLIP_AT_BUILD=1` веса SigLIP качаются при сборке в образ (+1,7 ГБ), и том /hf/hub не нужен.

ARG UV_VERSION=0.12.14
ARG PYTHON_IMAGE=python:3.12-slim
# gpu | cpu: какой torch ставить (`tool.uv.conflicts` в pyproject: в одном окружении один).
ARG TORCH=gpu
# 1 — веса SigLIP в образе; 0 — из тома /hf/hub.
ARG SIGLIP_AT_BUILD=0

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ---------------------------------------------------------------------------- база
FROM ${PYTHON_IMAGE} AS base
# /etc/mime.types: без него `mimetypes` в slim-образе не знает .webp, и фото каталога уходят
# как application/octet-stream (проверено на образе 23.09).
RUN apt-get update \
    && apt-get install -y --no-install-recommends media-types \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /uvx /usr/local/bin/
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    SVS_DATA_DIR=/data \
    SVS_DATASET_DIR=/dataset \
    SVS_HOST=0.0.0.0 \
    SVS_PORT=8080
WORKDIR /app

# ---------------------------------------------------------------------------- зависимости
# Отдельным слоем от кода: правка app/ не пересобирает torch. Проект сам не ставится
# (`--no-install-project`): сервис запускается из /app как `python -m app.api`, и REPO_ROOT
# (app/config.py:9) указывает на /app — там лежат configs/resolve с моделью слоя выбора.
FROM base AS deps-gpu
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=.python-version,target=.python-version \
    uv sync --frozen --no-install-project --extra gpu --extra cv --extra api
ENV SVS_DEVICE=cuda

FROM base AS deps-cpu
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=.python-version,target=.python-version \
    uv sync --frozen --no-install-project --extra cpu --extra cv --extra api
ENV SVS_DEVICE=cpu

# ---------------------------------------------------------------------------- веса SigLIP
# 0: весов в образе нет, HF_HUB_CACHE=/hf/hub — том.
FROM deps-${TORCH} AS siglip-0

# 1: ровно та ревизия, которой собран индекс (refs/main кэша на машине разработки), и от неё
# только зрительная башня: 4,5 ГБ чекпойнта → 1,7 ГБ. scripts/trim_vision_weights.py сверяет
# forward-проход обрезанной модели с полной — числа должны совпасть до бита, иначе сборка падает.
# safetensors пишет файл с правами 0600, а сервис работает не от root: веса открываются на чтение.
FROM deps-${TORCH} AS siglip-1
ARG SIGLIP_REVISION=e8e487298228002f3d8a82e0cd5c8ea9c567f57f
COPY scripts/trim_vision_weights.py /tmp/trim_vision_weights.py
RUN set -eu; \
    FULL=/tmp/hf-full/models--google--siglip2-so400m-patch14-384; \
    HF_HUB_OFFLINE=0 python -c "from huggingface_hub import snapshot_download; snapshot_download('google/siglip2-so400m-patch14-384', revision='${SIGLIP_REVISION}', cache_dir='/tmp/hf-full', allow_patterns=['config.json', 'preprocessor_config.json', 'model.safetensors'])"; \
    mkdir -p "$FULL/refs"; printf '%s' "${SIGLIP_REVISION}" > "$FULL/refs/main"; \
    python /tmp/trim_vision_weights.py --cache "$FULL" --out /opt/siglip/hub; \
    chmod -R a+rX /opt/siglip/hub; \
    rm -rf /tmp/hf-full /tmp/trim_vision_weights.py
ENV HF_HUB_CACHE=/opt/siglip/hub

# ---------------------------------------------------------------------------- сервис
FROM siglip-${SIGLIP_AT_BUILD} AS runtime
COPY app ./app
COPY configs ./configs
# Сервису нужен только каталог приложения; писать ему некуда и незачем: /data и /dataset
# монтируются только на чтение, полевой архив по умолчанию выключен (README, «Запуск в Docker»).
RUN useradd --uid 10001 --no-create-home --home-dir /app --shell /usr/sbin/nologin svs
USER svs
EXPOSE 8080
# Порт открывается только после прогрева (lifespan в app/api/main.py): ответ /v1/health — уже
# тёплый сервис. «ready» или «degraded» — смотреть в теле ответа.
HEALTHCHECK --interval=15s --timeout=5s --start-period=180s --retries=3 \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8080/v1/health', timeout=4)"]
CMD ["python", "-m", "app.api"]
