from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import strawberry
from sqlalchemy.sql.sqltypes import Integer, String, Boolean, DateTime

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
