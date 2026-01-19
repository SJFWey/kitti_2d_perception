from __future__ import annotations

import configparser
import re
from pathlib import Path
from typing import Iterable, TypeVar, Callable, Optional

T = TypeVar("T")

DEFAULT_PUBLIC_CONFIG = Path("configs") / "public" / "default.ini"


def _read_ini(path: Path, parser: configparser.ConfigParser) -> None:
    content = path.read_text(encoding="utf-8")
    if not re.search(r"^\s*\[", content, flags=re.MULTILINE):
        content = "[default]\n" + content
    parser.read_string(content, source=str(path))


def _resolve_default_path(repo_root: Path, path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return repo_root / path


def load_config(
    repo_root: Path,
    config_path: str | Path | None = None,
    required: bool = False,
    default_paths: Optional[Iterable[str | Path]] = None,
) -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    parser.optionxform = str.lower

    explicit_path = Path(config_path) if config_path else None
    if default_paths is None:
        default_paths = [DEFAULT_PUBLIC_CONFIG]
    config_paths = [_resolve_default_path(repo_root, path) for path in default_paths]
    if explicit_path:
        config_paths.append(explicit_path)

    for path in config_paths:
        if not path.exists():
            if explicit_path and path == explicit_path and required:
                raise SystemExit(f"Config not found: {path}")
            continue
        _read_ini(path, parser)

    return parser


def load_public_config(
    repo_root: Path,
    config_path: str | Path | None = None,
    required: bool = False,
) -> configparser.ConfigParser:
    return load_config(
        repo_root,
        config_path=config_path,
        required=required,
        default_paths=[DEFAULT_PUBLIC_CONFIG],
    )


def cfg_get(
    cfg: configparser.ConfigParser,
    section: str,
    key: str,
    cast: Callable[[str], T],
    default: T,
    fallback_sections: Optional[Iterable[str]] = None,
) -> T:
    sections = [section]
    if fallback_sections:
        sections.extend(list(fallback_sections))

    for sec in sections:
        if cfg.has_option(sec, key):
            value = cfg.get(sec, key, fallback="").strip()
            if value == "":
                continue
            try:
                return cast(value)
            except (TypeError, ValueError) as exc:
                raise SystemExit(f"Invalid config value for {sec}.{key}: {value}") from exc

    return default


def cfg_get_bool(
    cfg: configparser.ConfigParser,
    section: str,
    key: str,
    default: bool,
    fallback_sections: Optional[Iterable[str]] = None,
) -> bool:
    value = cfg_get(cfg, section, key, str, "", fallback_sections)
    if value == "":
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise SystemExit(f"Invalid config value for {section}.{key}: {value}")


def cfg_get_list(
    cfg: configparser.ConfigParser,
    section: str,
    key: str,
    default: Iterable[str],
    fallback_sections: Optional[Iterable[str]] = None,
) -> list[str]:
    value = cfg_get(cfg, section, key, str, "", fallback_sections)
    if value == "":
        return list(default)
    if "," in value:
        parts = [p.strip() for p in value.split(",")]
    else:
        parts = value.split()
    return [p for p in parts if p]


def resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path
