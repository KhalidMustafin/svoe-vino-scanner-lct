"""Файлы запуска (`.env.example`, `compose.yml`, `Dockerfile`) не расходятся с кодом.

Зачем. Умолчания живут в двух местах: в коде (`app/api/config.py`, `app/api/field.py`) и в
файлах запуска. 20.09 бюджеты подняли в полях класса, а `from_env` остался на 6000/2500 — и
сервис резал чтение (`app/api/config.py:187-188`). Здесь ловится та же беда между кодом и
Docker: новая переменная не попала в `.env.example` или compose, умолчание разъехалось,
в образ поехала лишняя экстра.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from app.api.config import ServiceSettings
from app.config import REPO_ROOT

ENV_EXAMPLE = REPO_ROOT / ".env.example"
COMPOSE = REPO_ROOT / "compose.yml"
DOCKERFILE = REPO_ROOT / "Dockerfile"

#: Имя переменной в строковом литерале кода: `env.get("SVS_FIELD_DIR")`, `text("SVS_ABSTAIN", …)`.
_CODE_VAR = re.compile(r"""["'](SVS_[A-Z0-9_]*[A-Z0-9])["']""")
#: Строка `.env.example` с переменной — заданной или закомментированной (`# SVS_HOST=…`).
_ENV_LINE = re.compile(r"^#?\s*(SVS_[A-Z0-9_]+)=(.*)$")
#: Подстановка compose с умолчанием: `${SVS_BUDGET_MS:-8000}`.
_COMPOSE_DEFAULT = re.compile(r"^\$\{(SVS_[A-Z0-9_]+):-(.*)\}$")

#: Переменные, которые сервис читает, но compose задаёт сам, а не из .env: пути внутри
#: контейнера, адрес и устройство по профилю. `SVS_CACHE_DIR` — кэш чтений стендов bench.
COMPOSE_FIXED = {"SVS_DATA_DIR", "SVS_DATASET_DIR", "SVS_HOST", "SVS_PORT", "SVS_DEVICE"}
NOT_FOR_SERVICE = {"SVS_CACHE_DIR"}


def code_variables() -> set[str]:
    names: set[str] = set()
    for path in (REPO_ROOT / "app").rglob("*.py"):
        names.update(_CODE_VAR.findall(path.read_text(encoding="utf-8")))
    return names


def env_example() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        match = _ENV_LINE.match(line.strip())
        if match and not line.lstrip().startswith("#"):
            out[match.group(1)] = match.group(2).strip().strip('"')
        elif match:
            out.setdefault(match.group(1), "")
    return out


def compose() -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))


def test_code_reads_variables_the_test_can_see():
    """Сам поиск по коду работает: иначе проверки ниже прошли бы на пустом множестве."""
    found = code_variables()
    assert {"SVS_BUDGET_MS", "SVS_DATASET_DIR", "SVS_CATALOG_CSV", "SVS_FIELD_TOKEN"} <= found


def test_env_example_lists_every_variable_the_code_reads():
    missing = sorted(code_variables() - set(env_example()))
    assert not missing, f"в .env.example нет переменных, которые читает код: {missing}"


def test_env_example_values_are_the_code_defaults():
    """Заданные в образце числа и модели — ровно умолчания кода, а не «примерно такие»."""
    defaults = ServiceSettings()
    expected = {
        "SVS_BUDGET_MS": str(defaults.budget_ms),
        "SVS_VLM_TIMEOUT_MS": str(defaults.vlm_timeout_ms),
        "SVS_TOP_K": str(defaults.top_k),
        "SVS_ABSTAIN": defaults.abstain,
        "SVS_VLM_MODEL": defaults.vlm_model,
        "SVS_CV_MODEL": defaults.cv_model,
        "SVS_VLM_KEEP_ALIVE": defaults.vlm_keep_alive,
        "SVS_MAX_UPLOAD_MB": str(defaults.max_upload_bytes // (1024 * 1024)),
    }
    values = env_example()
    assert {name: values.get(name) for name in expected} == expected


@pytest.mark.parametrize("service", ["scanner-gpu", "scanner-cpu"])
def test_compose_passes_every_service_variable(service):
    env = compose()["services"][service]["environment"]
    wanted = code_variables() - NOT_FOR_SERVICE
    missing = sorted(wanted - set(env))
    assert not missing, f"{service}: compose не передаёт {missing}"
    assert env["SVS_DATA_DIR"] == "/data"
    assert env["SVS_DATASET_DIR"] == "/dataset"
    assert env["SVS_DEVICE"] == {"scanner-gpu": "cuda", "scanner-cpu": "cpu"}[service]


def test_compose_defaults_are_the_code_defaults():
    env = compose()["services"]["scanner-gpu"]["environment"]
    defaults = env_example()
    for name, value in env.items():
        match = _COMPOSE_DEFAULT.match(str(value))
        if not match or name in COMPOSE_FIXED:
            continue
        example = defaults.get(name, "")
        if example:
            assert match.group(2) == example, f"{name}: compose и .env.example разошлись"


def test_compose_mounts_data_read_only_and_requires_the_dataset():
    """Данные только на чтение (кадры на диск не пишутся), без выгрузки compose не стартует."""
    volumes = compose()["services"]["scanner-gpu"]["volumes"]
    by_target = {v["target"]: v for v in volumes if isinstance(v, dict)}
    assert by_target["/data"]["read_only"] is True
    assert by_target["/dataset"]["read_only"] is True
    assert ":?" in by_target["/dataset"]["source"]


def test_dockerfile_installs_only_what_the_service_needs():
    extras = set(re.findall(r"--extra\s+(\w+)", DOCKERFILE.read_text(encoding="utf-8")))
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = set(pyproject["project"]["optional-dependencies"])
    assert extras <= declared
    assert extras == {"gpu", "cpu", "cv", "api"}
    assert "uv sync --frozen" in DOCKERFILE.read_text(encoding="utf-8")


def test_dockerignore_keeps_data_out_of_the_build_context():
    lines = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    rules = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    assert rules[0] == "*"
    assert not any(rule.startswith("!data") for rule in rules)


def test_env_example_is_not_a_real_env_file():
    """Образец в git, настоящий .env — нет (`.gitignore`)."""
    ignored = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in ignored and "!.env.example" in ignored
    assert Path(ENV_EXAMPLE).is_file()


def test_default_search_path_ships_its_adapter_and_documents_the_switch():
    """Путь по умолчанию (`SVS_CANDIDATE`, кандидат 2 с 26.09) стартует только с картой адаптера
    рядом с индексом: пачка данных везёт её как обязательный файл и сверяет sha1 индекса, а образец
    и compose называют выключатель — пусто = умолчание кода, `off` / `none` = прежний путь."""
    from app.api.config import (
        CANDIDATE_DEFAULT,
        CANDIDATE_OFF_WORDS,
        CV_ADAPTER_INDEX_SHA1,
        CV_ADAPTER_NAME,
    )

    script = (REPO_ROOT / "deploy" / "pack_data.sh").read_text(encoding="utf-8")
    required = next(line for line in script.splitlines() if line.startswith("REQUIRED=("))
    assert f"index/{CV_ADAPTER_NAME}" in required and "index/visual-s2so400m.npz" in required
    assert 'ITEMS=("${REQUIRED[@]}")' in script
    assert CV_ADAPTER_INDEX_SHA1 in script
    values = env_example()
    assert values["SVS_CANDIDATE"] == "" and values["SVS_CV_ADAPTER"] == ""
    example = ENV_EXAMPLE.read_text(encoding="utf-8")
    assert CANDIDATE_DEFAULT in example and CV_ADAPTER_NAME in example
    assert all(word in example for word in CANDIDATE_OFF_WORDS)
    assert ServiceSettings().candidate == CANDIDATE_DEFAULT
    for service in ("scanner-gpu", "scanner-cpu"):
        env = compose()["services"][service]["environment"]
        assert env["SVS_CANDIDATE"] == "${SVS_CANDIDATE:-}"
        assert env["SVS_CV_ADAPTER"] == "${SVS_CV_ADAPTER:-}"


def test_push_to_vps_ships_the_adapter_of_the_default_path():
    """Второй способ развёртывания — `deploy/push.sh` (рабочее дерево на VPS): он везёт те же
    обязательные файлы данных, что и пачка, иначе путь по умолчанию на сервере не стартует."""
    from app.api.config import CV_ADAPTER_NAME

    script = (REPO_ROOT / "deploy" / "push.sh").read_text(encoding="utf-8")
    lines = script.splitlines()
    required = next(line for line in lines if line.startswith("REQUIRED=("))
    assert f"index/{CV_ADAPTER_NAME}" in required and "index/visual-s2so400m.npz" in required
    assert 'for name in "${REQUIRED[@]}"; do' in lines
    assert any(line.startswith('tar czf - -C "$DATA_SRC" "${REQUIRED[@]}"') for line in lines)
