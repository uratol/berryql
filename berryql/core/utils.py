from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Dict, List
import strawberry
from sqlalchemy.sql.sqltypes import Integer, String, Boolean, DateTime
from sqlalchemy import and_ as _and
from .filters import OPERATOR_REGISTRY

class _DirectionEnum(Enum):
    asc = 'asc'
    desc = 'desc'

Direction = strawberry.enum(_DirectionEnum, name="Direction")  # type: ignore

def dir_value(order_dir: Any) -> str:
    if order_dir is None:
        return 'asc'
    try:
        val = getattr(order_dir, 'value', order_dir)
        return str(val).lower()
    except Exception:
        return 'asc'

def coerce_where_value(col, val):
    try:
        from sqlalchemy.sql.sqltypes import Integer as _I, Float as _F, Boolean as _B, DateTime as _DT, Numeric as _N
    except Exception:
        _I = Integer; _F = None; _B = Boolean; _DT = DateTime; _N = None
    if isinstance(val, (list, tuple)):
        return [ coerce_where_value(col, v) for v in val ]
    ctype = getattr(col, 'type', None)
    if ctype is None:
        return val
    try:
        if isinstance(ctype, _DT):
            if isinstance(val, str):
                s = val.replace('Z', '+00:00') if 'Z' in val else val
                try:
                    dv = datetime.fromisoformat(s)
                    try:
                        if getattr(ctype, 'timezone', False) is False and getattr(dv, 'tzinfo', None) is not None:
                            dv = dv.replace(tzinfo=None)
                    except Exception:
                        pass
                    return dv
                except Exception:
                    return val
            return val
        if isinstance(ctype, _I):
            try:
                return int(val) if isinstance(val, str) else val
            except Exception:
                return val
        if _N is not None and isinstance(ctype, _N):
            try:
                return float(val) if isinstance(val, str) else val
            except Exception:
                return val
        if _F is not None and isinstance(ctype, _F):
            try:
                return float(val) if isinstance(val, str) else val
            except Exception:
                return val
        if isinstance(ctype, _B):
            if isinstance(val, str):
                lv = val.strip().lower()
                if lv in ('true','t','1','yes','y'):
                    return True
                if lv in ('false','f','0','no','n'):
                    return False
            return bool(val)
    except Exception:
        return val
    return val

# --- DRY helpers used across registry/builders ---
def coerce_literal(v: Any) -> Any:
    """Best-effort coercion of GraphQL AST-like values to plain Python types.

    Handles lists, dicts, objects exposing `.values`/`.fields`/`.value`.
    """
    try:
        # list-like
        if isinstance(v, list):
            return [coerce_literal(x) for x in v]
        # dict-like
        if isinstance(v, dict):
            return {k: coerce_literal(x) for k, x in v.items()}
        # GraphQL AST nodes often have `.values`
        if hasattr(v, 'values'):
            try:
                return [coerce_literal(x) for x in getattr(v, 'values', []) or []]
            except Exception:
                return v
        # GraphQL AST object nodes expose `.fields`
        if hasattr(v, 'fields'):
            try:
                out: Dict[str, Any] = {}
                for f in getattr(v, 'fields', []) or []:
                    k = getattr(getattr(f, 'name', None), 'value', None) or getattr(f, 'name', None)
                    out[k] = coerce_literal(getattr(f, 'value', None))
                return out
            except Exception:
                return v
        # Generic scalar value node
        if hasattr(v, 'value'):
            return getattr(v, 'value')
    except Exception:
        return v
    return v

def normalize_relation_cfg(cfg: Dict[str, Any]) -> None:
    """Normalize a relation selection config in-place.

    - Coerce literals for common keys
    - Normalize order_multi/fields to List[str]
    - Coerce filter_args values
    - Recurse into nested configs
    """
    if not isinstance(cfg, dict):
        return
    for key in ('limit', 'offset', 'order_by', 'order_dir', 'where', 'default_where'):
        if key in cfg:
            val = coerce_literal(cfg.get(key))
            # Best-effort: if 'where' or 'default_where' is a JSON string, parse into dict
            if key in ('where', 'default_where'):
                # unwrap single-item containers
                try:
                    if isinstance(val, (list, tuple)) and len(val) == 1:
                        val = val[0]
                except Exception:
                    pass
                # decode bytes
                if isinstance(val, (bytes, bytearray)):
                    try:
                        val = val.decode('utf-8')
                    except Exception:
                        val = str(val)
                if isinstance(val, str):
                    s = val.strip()
                    if (s.startswith('{') and s.endswith('}')) or (s.startswith('"{') and s.endswith('}"')):
                        import json as _json
                        import ast as _ast
                        try:
                            parsed = _json.loads(s)
                            # Unwrap if still a JSON string up to two times
                            unwrap = 0
                            while isinstance(parsed, str) and unwrap < 2:
                                try:
                                    parsed = _json.loads(parsed)
                                except Exception:
                                    break
                                unwrap += 1
                            val = parsed
                        except Exception:
                            # Try Python literal_eval as last resort (e.g., single-quoted dicts)
                            try:
                                val = _ast.literal_eval(s)
                            except Exception:
                                # keep original to let builders raise a consistent error later
                                pass
            cfg[key] = val
    # order_multi
    if 'order_multi' in cfg and cfg.get('order_multi') is not None:
        try:
            om = coerce_literal(cfg.get('order_multi'))
            if hasattr(om, 'values'):
                try:
                    om = [coerce_literal(x) for x in getattr(om, 'values', []) or []]
                except Exception:
                    pass
            if not isinstance(om, list):
                om = [om]
            cfg['order_multi'] = [str(coerce_literal(x)) for x in (om or [])]
        except Exception:
            cfg['order_multi'] = [str(cfg.get('order_multi'))] if cfg.get('order_multi') is not None else []
    # fields
    if 'fields' in cfg and cfg.get('fields') is not None:
        try:
            fl = coerce_literal(cfg.get('fields'))
            if hasattr(fl, 'values'):
                try:
                    fl = [coerce_literal(x) for x in getattr(fl, 'values', []) or []]
                except Exception:
                    pass
            if not isinstance(fl, list):
                fl = [fl]
            cfg['fields'] = [str(coerce_literal(x)) for x in (fl or [])]
        except Exception:
            cfg['fields'] = [str(cfg.get('fields'))] if cfg.get('fields') is not None else []
    # filter args
    if 'filter_args' in cfg and isinstance(cfg.get('filter_args'), dict):
        fa = cfg.get('filter_args') or {}
        for k in list(fa.keys()):
            fa[k] = coerce_literal(fa[k])
    # nested
    for n in list((cfg.get('nested') or {}).values()):
        normalize_relation_cfg(n)

def expr_from_where_dict(model_cls, wdict: Dict[str, Any]):
    """Build a SQLAlchemy conjunction from simple where dict: {col: {op: val}}."""
    from .utils import coerce_where_value  # local import to avoid cycles
    exprs: List[Any] = []
    for col_name, op_map in (wdict or {}).items():
        try:
            col = model_cls.__table__.c.get(col_name)
        except Exception:
            col = None
        if col is None:
            raise ValueError(f"Unknown where column: {col_name}")
        for op_name, val in (op_map or {}).items():
            op_fn = OPERATOR_REGISTRY.get(op_name)
            if not op_fn:
                raise ValueError(f"Unknown where operator: {op_name}")
            # Coerce values to match column types
            if op_name in ('in', 'between') and isinstance(val, (list, tuple)):
                val = [coerce_where_value(col, v) for v in val]
            else:
                val = coerce_where_value(col, val)
            exprs.append(op_fn(col, val))
    if not exprs:
        return None
    return _and(*exprs)

# --- Context helpers ---
def get_db_session(info_or_ctx: Any) -> Any | None:
    """Best-effort extraction of an AsyncSession-like object from context.

    Accepts either a Strawberry ``Info`` or a plain context object/dict. Tries
    common keys/attributes in order: ``db_session``, ``db``, ``session``,
    ``async_session``.

    Returns:
        The session object if found; otherwise ``None``.
    """
    if info_or_ctx is None:
        return None
    # If a Strawberry Info is passed, use its .context
    ctx = getattr(info_or_ctx, 'context', info_or_ctx)
    if ctx is None:
        return None
    candidates = ('db_session', 'db', 'session', 'async_session')
    # Mapping-like access with .get
    try:
        get = getattr(ctx, 'get', None)
        if callable(get):
            for k in candidates:
                try:
                    v = get(k, None)
                except Exception:
                    v = None
                if v is not None:
                    return v
    except Exception:
        pass
    # Mapping access via __getitem__
    try:
        for k in candidates:
            try:
                v = ctx[k]  # type: ignore[index]
            except Exception:
                v = None
            if v is not None:
                return v
    except Exception:
        pass
    # Attribute access
    try:
        for k in candidates:
            try:
                v = getattr(ctx, k)
            except Exception:
                v = None
            if v is not None:
                return v
    except Exception:
        pass
    return None
