from __future__ import annotations

try:
    import _bootstrap  # noqa: F401
except ImportError:
    from . import _bootstrap  # noqa: F401

import os
import re
import shlex
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"

_ENV_REF_RE = re.compile(
    r"\$\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|\$(?P<bare>[A-Za-z_][A-Za-z0-9_]*)"
)


def _expand_env_refs(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group("braced") or match.group("bare")
        return os.environ.get(name, "")

    expanded = _ENV_REF_RE.sub(replace, value)
    if expanded == "~" or expanded.startswith(("~/", "~\\")):
        return str(Path(expanded).expanduser())
    return expanded


def load_dotenv(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        try:
            parts = shlex.split(line, comments=True, posix=True)
        except ValueError:
            continue

        if not parts:
            continue
        if parts[0] == "export":
            parts = parts[1:]

        for part in parts:
            if "=" not in part:
                continue
            name, value = part.split("=", 1)
            if not name:
                continue
            os.environ.setdefault(name, _expand_env_refs(value))


load_dotenv()

try:
    from utils.hf_http_compat import (
        configure_huggingface_http_client,
        patch_huggingface_http_backoff,
    )

    configure_huggingface_http_client()
    patch_huggingface_http_backoff()
except Exception:
    pass


def env_str(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def env_file_text(name: str, default: str = "") -> str:
    value = env_str(name)
    if not value:
        return default

    path = repo_path(value)
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"Could not read {name} file at {path}: {exc}") from exc


def env_int(name: str, default: int) -> int:
    return int(env_str(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(env_str(name, str(default)))


def env_list(name: str, default: list[str]) -> list[str]:
    value = os.environ.get(name)
    if value is None:
        return default
    return [part.strip() for part in value.split("|") if part.strip()]


def repo_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def env_path(name: str, default: str | Path) -> Path:
    return repo_path(env_str(name, str(default)))
