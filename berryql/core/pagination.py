from __future__ import annotations

import base64
import json
from typing import Any, Iterable, Tuple

from sqlalchemy import and_, or_

from .errors import InvalidOrderingError


def encode_cursor(*values: Any) -> str:
    """Encode deterministic ordering values as an opaque keyset cursor."""

    payload = json.dumps({"v": 1, "values": list(values)}, separators=(",", ":"), default=str).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def decode_cursor(cursor: str) -> Tuple[Any, ...]:
    if not isinstance(cursor, str) or not cursor:
        raise InvalidOrderingError("after cursor must be a non-empty string")
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
    except Exception as exc:
        raise InvalidOrderingError("Invalid after cursor") from exc
    if not isinstance(payload, dict) or payload.get("v") != 1 or not isinstance(payload.get("values"), list):
        raise InvalidOrderingError("Invalid after cursor payload")
    return tuple(payload["values"])


def keyset_predicate(expressions: Iterable[Any], directions: Iterable[str], values: Iterable[Any]) -> Any:
    expressions = tuple(expressions)
    directions = tuple(directions)
    values = tuple(values)
    if not expressions or len(expressions) != len(values) or len(expressions) != len(directions):
        raise InvalidOrderingError("Cursor value count does not match deterministic ordering")
    branches = []
    for index, (expression, direction, value) in enumerate(zip(expressions, directions, values)):
        if value is None:
            raise InvalidOrderingError("Null keyset cursor values are not supported")
        prefix = [expressions[prefix_index] == values[prefix_index] for prefix_index in range(index)]
        comparison = expression < value if direction == "desc" else expression > value
        branches.append(and_(*prefix, comparison))
    return or_(*branches)


__all__ = ["decode_cursor", "encode_cursor", "keyset_predicate"]
