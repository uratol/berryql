from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Callable, Dict, Iterable, Optional, Set, List, Union

NameConverter = Optional[Callable[[str], str]]

__all__ = [
    'NameConverter',
    'from_camel',
    'to_camel',
    'map_graphql_to_python',
    'build_name_candidates',
    'ensure_list',
    'fields_map_for',
]

_camel_to_snake_pattern = re.compile(r'(?<!^)(?=[A-Z])')


@lru_cache(maxsize=4096)
def _from_camel_cached(name: str) -> str:
    return _camel_to_snake_pattern.sub('_', name).lower()


@lru_cache(maxsize=4096)
def _to_camel_cached(name: str) -> str:
    parts = name.split('_')
    if not parts:
        return name
    return parts[0] + ''.join(p.capitalize() for p in parts[1:])


def from_camel(name: str) -> str:
    """Convert lower/upper camelCase to snake_case."""
    if not name:
        return name
    # Cache only hashable strings; coerce non-str to str once.
    s = name if isinstance(name, str) else str(name)
    return _from_camel_cached(s)


def to_camel(name: str) -> str:
    """Convert snake_case to lowerCamelCase."""
    if not name:
        return name
    s = name if isinstance(name, str) else str(name)
    return _to_camel_cached(s)


def map_graphql_to_python(
    name: str,
    fields_map: Dict[str, Any],
    *,
    auto_camel: bool,
    name_converter: NameConverter,
) -> str:
    """Map a GraphQL field name back to the Python/Berry field key."""
    if not name:
        return name
    if name in fields_map:
        return name
    if auto_camel:
        snake = from_camel(name)
        if snake in fields_map:
            return snake
    if callable(name_converter):
        for py_name in fields_map.keys():
            try:
                if name_converter(py_name) == name:
                    return py_name
            except Exception:
                continue
    return name


def build_name_candidates(
    base_name: str,
    *,
    auto_camel: bool,
    name_converter: NameConverter,
) -> Set[str]:
    """Return possible GraphQL representations for a python field name."""
    candidates: Set[str] = {base_name}
    if callable(name_converter):
        try:
            candidates.add(name_converter(base_name))
        except Exception:
            pass
    if auto_camel:
        candidates.add(to_camel(base_name))
    return candidates


def ensure_list(value: Any) -> Optional[List[Any]]:
    """Wrap scalars into a list, preserving list inputs."""
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def fields_map_for(btype: Any) -> Dict[str, Any]:
    """Safely fetch __berry_fields__ map for a Berry type."""
    try:
        return getattr(btype, '__berry_fields__', {}) or {}
    except Exception:
        return {}
