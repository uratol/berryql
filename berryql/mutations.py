from __future__ import annotations
import re
import logging
import uuid
from typing import Any, Dict, Optional, Type, List, get_args, get_origin, cast
import strawberry
from typing import TYPE_CHECKING
from .core.utils import get_db_session as _get_db
from sqlalchemy import select
from sqlalchemy.sql.elements import ColumnElement

if TYPE_CHECKING:  # pragma: no cover
    from .registry import BerrySchema, BerryType, BerryDomain
    from strawberry.types import Info as StrawberryInfo
else:
    try:
        from strawberry.types import Info as StrawberryInfo  # type: ignore
    except Exception:  # pragma: no cover
        class StrawberryInfo:  # type: ignore
            ...

_logger = logging.getLogger("berryql")

# --- Merge builder -------------------------------------------------------

def build_merge_resolver_for_type(
    schema: 'BerrySchema',
    btype_cls: Type['BerryType'],
    *,
    field_name: Optional[str] = None,
    relation_scope: Any | None = None,
    pre_callback: Any | None = None,
    post_callback: Any | None = None,
):
    """Return (fname, strawberry.field, st_return) implementing recursive merge for a BerryType.

    This mirrors the previous inline version in registry, but lives here for clarity.
    """
    type_name = getattr(btype_cls, '__name__', 'Type')
    st_return = schema._st_types.get(type_name)
    input_type = schema._ensure_input_type(btype_cls)
    model_cls = getattr(btype_cls, 'model', None)
    if st_return is None or model_cls is None:
        return None
    try:
        pk_name = schema._get_pk_name(model_cls)
    except Exception:
        pk_name = 'id'

    async def _merge_impl(info: StrawberryInfo, payload):
        session = _get_db(info)
        if session is None:
            raise ValueError("No db_session in context")
        data = schema._input_to_dict(payload)

        # Helpers to compile and enforce scope for a given model instance
        from .core.utils import to_where_dict as _to_where_dict, expr_from_where_dict as _expr_from_where_dict

        # --- Callback helpers -------------------------------------------------
        import inspect

        def _norm_callable(cb):
            return cb if callable(cb) else None

        # Type-level hooks on BerryType, best-effort
        try:
            type_pre_cb = _norm_callable(getattr(btype_cls, 'pre_merge', None) or getattr(btype_cls, 'pre_upsert', None) or getattr(btype_cls, '__pre_upsert__', None))
        except Exception:
            type_pre_cb = None
        try:
            type_post_cb = _norm_callable(getattr(btype_cls, 'post_merge', None) or getattr(btype_cls, 'post_upsert', None) or getattr(btype_cls, '__post_upsert__', None))
        except Exception:
            type_post_cb = None

        async def _maybe_call_pre(cb, model_cls_local, info_local, data_local, context_local):
            if not callable(cb):
                return data_local
            try:
                sig = inspect.signature(cb)
                params = list(sig.parameters.keys())
            except Exception:
                params = []
            try:
                if len(params) >= 4:
                    res = cb(model_cls_local, info_local, data_local, context_local)
                elif len(params) == 3:
                    res = cb(model_cls_local, info_local, data_local)
                elif len(params) == 2:
                    res = cb(data_local, info_local)
                else:
                    res = cb(data_local)
                if inspect.isawaitable(res):
                    res = await res  # type: ignore
                return schema._input_to_dict(res) if res is not None else data_local
            except Exception:
                raise

        async def _maybe_call_post(cb, model_cls_local, info_local, instance_local, created_local, context_local):
            if not callable(cb):
                return None
            try:
                sig = inspect.signature(cb)
                params = list(sig.parameters.keys())
            except Exception:
                params = []
            try:
                if len(params) >= 5:
                    res = cb(model_cls_local, info_local, instance_local, created_local, context_local)
                elif len(params) == 4:
                    res = cb(model_cls_local, info_local, instance_local, created_local)
                elif len(params) == 3:
                    res = cb(instance_local, info_local, created_local)
                elif len(params) == 2:
                    res = cb(instance_local, info_local)
                else:
                    res = cb(instance_local)
                if inspect.isawaitable(res):
                    await res
            except Exception:
                raise

        def _resolve_scope_value(value):
            v = value
            try:
                # GraphQL VariableNode-like
                if not isinstance(v, (dict, str)) and hasattr(v, 'name'):
                    vname = getattr(getattr(v, 'name', None), 'value', None) or getattr(v, 'name', None)
                    var_vals = getattr(info, 'variable_values', None)
                    if var_vals is None:
                        raw_info = getattr(info, '_raw_info', None)
                        var_vals = getattr(raw_info, 'variable_values', None) if raw_info is not None else None
                    if isinstance(var_vals, dict) and vname in var_vals:
                        v = var_vals[vname]
                if not isinstance(v, (dict, str)) and hasattr(v, 'value'):
                    v = getattr(v, 'value')
            except Exception:
                pass
            return v

        def _safe_attrs_dump(model_cls_local, instance_local):
            try:
                out: Dict[str, Any] = {}
                try:
                    cols = [c.name for c in getattr(getattr(model_cls_local, '__table__', None), 'columns', [])]
                except Exception:
                    cols = []
                for cn in cols or []:
                    try:
                        out[cn] = getattr(instance_local, cn, None)
                    except Exception:
                        out[cn] = None
                if not out:
                    try:
                        for k, v in (getattr(instance_local, '__dict__', {}) or {}).items():
                            if not str(k).startswith('_'):
                                out[k] = v
                    except Exception:
                        pass
                if not out:
                    return {'repr': repr(instance_local)}
                return out
            except Exception:
                return {'repr': repr(instance_local)}

        async def _enforce_scope(model_cls_local, instance_local, scope_raw: Any | None):
            if scope_raw is None:
                return True
            try:
                pk_name_local = schema._get_pk_name(model_cls_local)
            except Exception:
                pk_name_local = 'id'
            try:
                pk_val_local = getattr(instance_local, pk_name_local)
            except Exception:
                pk_val_local = None
            if pk_val_local is None:
                return True
            v = _resolve_scope_value(scope_raw)
            expr: Optional[ColumnElement[bool]] = None
            if isinstance(v, (dict, str)):
                try:
                    wdict = _to_where_dict(v, strict=True)
                    if wdict:
                        maybe_expr = _expr_from_where_dict(model_cls_local, wdict, strict=True)
                        if maybe_expr is not None:
                            expr = cast(ColumnElement[bool], maybe_expr)
                except Exception:
                    expr = None
            else:
                try:
                    res = v(model_cls_local, info) if callable(v) else v
                    if res is not None:
                        expr = cast(ColumnElement[bool], res)
                except Exception:
                    expr = None
            if expr is None:
                return True
            stmt = select(model_cls_local.__table__.c[pk_name_local]).select_from(model_cls_local).where(
                getattr(model_cls_local.__table__.c, pk_name_local) == pk_val_local
            ).where(expr)
            res = await session.execute(stmt)
            row = res.first()
            return bool(row)

        async def _merge_single(model_cls_local, btype_local, data_local, parent_ctx=None, rel_name_from_parent: Optional[str] = None):
            # Compute effective relation-level callbacks (from parent relation meta or top-level builder args)
            eff_pre_cb = None
            eff_post_cb = None
            try:
                if rel_name_from_parent and parent_ctx is not None:
                    parent_model_cls = parent_ctx.get('parent_model')
                    parent_btype = None
                    try:
                        for _n, _bt in (schema.types or {}).items():
                            if getattr(_bt, 'model', None) is parent_model_cls:
                                parent_btype = _bt
                                break
                    except Exception:
                        parent_btype = None
                    if parent_btype is not None:
                        try:
                            rdef = getattr(parent_btype, '__berry_fields__', {}).get(rel_name_from_parent)
                            if rdef is not None and getattr(rdef, 'kind', None) == 'relation':
                                meta = (getattr(rdef, 'meta', {}) or {})
                                eff_pre_cb = _norm_callable(meta.get('pre') or meta.get('pre_merge') or meta.get('pre_upsert'))
                                eff_post_cb = _norm_callable(meta.get('post') or meta.get('post_merge') or meta.get('post_upsert'))
                        except Exception:
                            eff_pre_cb = eff_pre_cb or None
                            eff_post_cb = eff_post_cb or None
                if eff_pre_cb is None:
                    eff_pre_cb = _norm_callable(pre_callback)
                if eff_post_cb is None:
                    eff_post_cb = _norm_callable(post_callback)
            except Exception:
                eff_pre_cb = _norm_callable(pre_callback)
                eff_post_cb = _norm_callable(post_callback)

            # Determine applicable scope for this level (needed early for delete path)
            eff_scope = None
            try:
                if rel_name_from_parent and parent_ctx is not None:
                    parent_model_cls = parent_ctx.get('parent_model')
                    parent_btype = None
                    try:
                        for _n, _bt in (schema.types or {}).items():
                            if getattr(_bt, 'model', None) is parent_model_cls:
                                parent_btype = _bt
                                break
                    except Exception:
                        parent_btype = None
                    if parent_btype is not None:
                        try:
                            rdef = getattr(parent_btype, '__berry_fields__', {}).get(rel_name_from_parent)
                            if rdef is not None and getattr(rdef, 'kind', None) == 'relation':
                                eff_scope = (getattr(rdef, 'meta', {}) or {}).get('scope')
                        except Exception:
                            eff_scope = None
                if eff_scope is None:
                    eff_scope = relation_scope
            except Exception:
                eff_scope = relation_scope

            # Early delete handling
            try:
                if isinstance(data_local, dict):
                    delete_flag = bool(data_local.get('_Delete'))
                else:
                    delete_flag = False
            except Exception:
                delete_flag = False
            if delete_flag:
                try:
                    pk_name_local = schema._get_pk_name(model_cls_local)
                except Exception:
                    pk_name_local = 'id'
                pk_val_local = None
                try:
                    if isinstance(data_local, dict):
                        pk_val_local = data_local.get(pk_name_local)
                except Exception:
                    pk_val_local = None
                if pk_val_local in (None, 0, ''):
                    raise ValueError(f"Missing primary key for delete on {getattr(model_cls_local,'__name__',model_cls_local)}")
                instance_for_delete = await session.get(model_cls_local, pk_val_local)
                if instance_for_delete is None:
                    raise ValueError(f"Instance not found for delete; {getattr(model_cls_local,'__name__',model_cls_local)}.{pk_name_local}={pk_val_local}")
                if eff_scope is not None:
                    ok = await _enforce_scope(model_cls_local, instance_for_delete, eff_scope)
                    if not ok:
                        _cls = getattr(model_cls_local, '__name__', str(model_cls_local))
                        _attrs = _safe_attrs_dump(model_cls_local, instance_for_delete)
                        raise PermissionError(f"Mutation out of scope for delete; model={_cls}; attrs={_attrs}")
                # Application-level cascade: delete dependent rows first (handles MSSQL FK constraints)
                from sqlalchemy import select as _sa_select, delete as _sa_delete
                async def _cascade_delete_children(parent_model_cls: Any, parent_btype_cls: Any, parent_pk_val: Any):
                    try:
                        bfields = getattr(parent_btype_cls, '__berry_fields__', {}) or {}
                    except Exception:
                        bfields = {}
                    for rname, rdef in bfields.items():
                        try:
                            if getattr(rdef, 'kind', None) != 'relation':
                                continue
                            meta_r = getattr(rdef, 'meta', {}) or {}
                            target_name = meta_r.get('target')
                            if not target_name:
                                continue
                            child_btype = schema.types.get(target_name)
                            child_model_cls = getattr(child_btype, 'model', None) if child_btype is not None else None
                            if child_model_cls is None:
                                continue
                            explicit_child_fk_name = meta_r.get('fk_column_name')
                            # Find FK on child referencing parent
                            try:
                                fk_col = schema._find_child_fk_column(parent_model_cls, child_model_cls, explicit_child_fk_name)  # type: ignore[attr-defined]
                            except Exception:
                                fk_col = None
                            if fk_col is None:
                                continue
                            # Recurse to grandchildren first by selecting child PKs
                            try:
                                child_pk_name = schema._get_pk_name(child_model_cls)
                            except Exception:
                                child_pk_name = 'id'
                            try:
                                child_pk_col = getattr(getattr(child_model_cls, '__table__', None).c, child_pk_name)
                            except Exception:
                                child_pk_col = None
                            # Fetch child ids to recurse
                            try:
                                if child_pk_col is not None:
                                    ids_res = await session.execute(_sa_select(child_pk_col).where(fk_col == parent_pk_val))
                                    child_ids = [row[0] for row in ids_res.fetchall()]
                                else:
                                    ids_res = await session.execute(_sa_select(getattr(child_model_cls, child_pk_name)).where(fk_col == parent_pk_val))
                                    child_ids = [row[0] for row in ids_res.fetchall()]
                            except Exception:
                                child_ids = []
                            for cid in child_ids:
                                await _cascade_delete_children(child_model_cls, child_btype, cid)
                            # Delete children referencing the parent
                            try:
                                await session.execute(_sa_delete(child_model_cls).where(fk_col == parent_pk_val))
                            except Exception:
                                # Fall back to ORM delete one-by-one
                                for cid in child_ids:
                                    inst_c = await session.get(child_model_cls, cid)
                                    if inst_c is not None:
                                        await session.delete(inst_c)
                        except Exception:
                            continue
                await _cascade_delete_children(model_cls_local, btype_local, pk_val_local)
                if eff_pre_cb is not None:
                    await _maybe_call_pre(eff_pre_cb, model_cls_local, info, data_local, {'parent': parent_ctx, 'relation': rel_name_from_parent, 'delete': True})
                if type_pre_cb is not None:
                    await _maybe_call_pre(type_pre_cb, model_cls_local, info, data_local, {'parent': parent_ctx, 'relation': rel_name_from_parent, 'delete': True})
                # Use bulk DELETE to avoid ORM trying to NULL-out child FKs (which may be NOT NULL)
                try:
                    from sqlalchemy import delete as _sa_delete
                    try:
                        pk_col = getattr(getattr(model_cls_local, '__table__', None).c, pk_name_local)
                    except Exception:
                        pk_col = None
                    if pk_col is not None:
                        await session.execute(_sa_delete(model_cls_local).where(pk_col == pk_val_local))
                    else:
                        # Fallback to ORM delete if we can't access column object
                        await session.delete(instance_for_delete)
                except Exception:
                    # As a last resort, perform ORM delete
                    await session.delete(instance_for_delete)
                await session.flush()
                if eff_post_cb is not None:
                    await _maybe_call_post(eff_post_cb, model_cls_local, info, instance_for_delete, False, {'parent': parent_ctx, 'relation': rel_name_from_parent, 'delete': True})
                if type_post_cb is not None:
                    await _maybe_call_post(type_post_cb, model_cls_local, info, instance_for_delete, False, {'parent': parent_ctx, 'relation': rel_name_from_parent, 'delete': True})
                return instance_for_delete

            # Call pre-callbacks before any processing
            if eff_pre_cb is not None:
                data_local = await _maybe_call_pre(eff_pre_cb, model_cls_local, info, data_local, {'parent': parent_ctx, 'relation': rel_name_from_parent})
            if type_pre_cb is not None:
                data_local = await _maybe_call_pre(type_pre_cb, model_cls_local, info, data_local, {'parent': parent_ctx, 'relation': rel_name_from_parent})

            # Split scalars and relations
            scalar_vals: Dict[str, Any] = {}
            relation_vals: Dict[str, Any] = {}
            for k, v in list((data_local or {}).items()):
                fdef = (getattr(btype_local, '__berry_fields__', {}) or {}).get(k)
                if fdef is None:
                    scalar_vals[k] = v
                elif fdef.kind == 'scalar':
                    scalar_vals[k] = v
                elif fdef.kind == 'relation':
                    relation_vals[k] = v

            # Helper: MSSQL/GUID
            def _is_mssql(session_local) -> bool:
                try:
                    bind = getattr(session_local, 'bind', None) or session_local.get_bind()
                    dialect = getattr(bind, 'dialect', None) or getattr(getattr(bind, 'sync_engine', None), 'dialect', None)
                    name = getattr(dialect, 'name', '')
                    return str(name).lower() == 'mssql'
                except Exception:
                    return False

            def _has_guid_pk(model_cls_inner, pk_name_inner: str) -> bool:
                try:
                    col = getattr(getattr(model_cls_inner, '__table__', None).c, pk_name_inner)
                except Exception:
                    col = None
                try:
                    if col is not None:
                        t = getattr(col, 'type', None)
                        tname = getattr(getattr(t, '__class__', object), '__name__', '')
                        if 'GUID' in str(tname):
                            return True
                        if 'UNIQUEIDENTIFIER' in str(t).upper():
                            return True
                except Exception:
                    pass
                return False

            # Pre-create single child relations when FK is on parent
            precreated_children: Dict[str, Any] = {}
            for rel_key, rel_value in list(relation_vals.items()):
                try:
                    rel_def = btype_local.__berry_fields__.get(rel_key)
                    if rel_def is None or getattr(rel_def, 'kind', None) != 'relation':
                        continue
                    meta = getattr(rel_def, 'meta', {}) or {}
                    is_single = bool(meta.get('single'))
                    if not is_single:
                        continue
                    if rel_value is None:
                        continue
                    target_name = meta.get('target')
                    if not target_name:
                        continue
                    child_btype = schema.types.get(target_name)
                    child_model = getattr(child_btype, 'model', None) if child_btype else None
                    if not child_btype or not child_model:
                        continue
                    fk_on_parent = None
                    try:
                        fk_on_parent = schema._find_parent_fk_column_name(model_cls_local, child_model, rel_key)
                    except Exception:
                        fk_on_parent = None
                    if not fk_on_parent:
                        continue
                    child_data = schema._input_to_dict(rel_value)
                    if not isinstance(child_data, dict):
                        continue
                    try:
                        child_pk = schema._get_pk_name(child_model)
                    except Exception:
                        child_pk = 'id'
                    has_substantial = False
                    try:
                        for cc in getattr(getattr(child_model, '__table__', None), 'columns', []) or []:
                            if getattr(cc, 'nullable', True):
                                continue
                            if cc.name == child_pk:
                                continue
                            if child_data.get(cc.name) is not None:
                                has_substantial = True
                                break
                    except Exception:
                        has_substantial = any(v is not None for v in child_data.values())
                    if not has_substantial:
                        continue
                    child_inst = await _merge_single(child_model, child_btype, child_data)
                    try:
                        child_pk = schema._get_pk_name(child_model)
                    except Exception:
                        child_pk = 'id'
                    try:
                        child_id = getattr(child_inst, child_pk, None)
                    except Exception:
                        child_id = None
                    if child_id is not None and scalar_vals.get(fk_on_parent) in (None, 0):
                        scalar_vals[fk_on_parent] = child_id
                    precreated_children[rel_key] = child_inst
                    del relation_vals[rel_key]
                except Exception:
                    continue

            # Parent context: ensure child FK to parent is set on payload
            child_rel_to_parent_name: Optional[str] = None
            if parent_ctx is not None:
                try:
                    parent_model_cls = parent_ctx.get('parent_model')
                    parent_inst = parent_ctx.get('parent_inst')
                except Exception:
                    parent_model_cls = None
                    parent_inst = None
                if parent_model_cls is not None and parent_inst is not None:
                    # Determine child->parent FK column name
                    child_fk_to_parent = None
                    try:
                        if hasattr(model_cls_local, '__table__') and hasattr(parent_model_cls, '__table__'):
                            for cc in model_cls_local.__table__.columns:
                                for fk in getattr(cc, 'foreign_keys', []) or []:
                                    if fk.column.table.name == parent_model_cls.__table__.name:
                                        child_fk_to_parent = cc.name
                                        break
                                if child_fk_to_parent is not None:
                                    break
                    except Exception:
                        child_fk_to_parent = None
                    # Also relationship attribute on child targeting parent
                    try:
                        rel_name_found = None
                        mapper = getattr(model_cls_local, '__mapper__', None)
                        if mapper is not None:
                            for rel in getattr(mapper, 'relationships', []) or []:
                                try:
                                    target_cls = getattr(rel, 'mapper', None)
                                    target_cls = getattr(target_cls, 'class_', None) if target_cls is not None else None
                                    if target_cls is parent_model_cls:
                                        rel_name_found = getattr(rel, 'key', None)
                                        break
                                except Exception:
                                    continue
                        child_rel_to_parent_name = rel_name_found or child_rel_to_parent_name
                    except Exception:
                        child_rel_to_parent_name = child_rel_to_parent_name or None
                    # Conventional fallback for fk name
                    if child_fk_to_parent is None:
                        try:
                            conv_name = f"{parent_model_cls.__table__.name.rstrip('s')}_id"
                            if hasattr(model_cls_local, '__table__') and any(c.name == conv_name for c in model_cls_local.__table__.columns):
                                child_fk_to_parent = conv_name
                        except Exception:
                            pass
                    if child_fk_to_parent:
                        try:
                            parent_pk_name_local = schema._get_pk_name(parent_model_cls)
                        except Exception:
                            parent_pk_name_local = 'id'
                        try:
                            parent_pk_val_local = getattr(parent_inst, parent_pk_name_local)
                        except Exception:
                            parent_pk_val_local = None
                        if parent_pk_val_local is None:
                            try:
                                await session.flush()
                            except Exception:
                                pass
                            try:
                                await session.refresh(parent_inst)
                            except Exception:
                                pass
                            try:
                                parent_pk_val_local = getattr(parent_inst, parent_pk_name_local)
                            except Exception:
                                parent_pk_val_local = None
                        if scalar_vals.get(child_fk_to_parent) in (None, 0):
                            scalar_vals[child_fk_to_parent] = parent_pk_val_local

            # Determine instance (update or create)
            instance = None
            pk_val = scalar_vals.get(pk_name) or (data_local.get(pk_name) if isinstance(data_local, dict) else None)
            if pk_val is not None:
                try:
                    instance = await session.get(model_cls_local, pk_val)
                except Exception:
                    instance = None

            # Enforce scope before update
            if pk_val is not None and instance is not None and eff_scope is not None:
                ok = await _enforce_scope(model_cls_local, instance, eff_scope)
                if not ok:
                    _cls = getattr(model_cls_local, '__name__', str(model_cls_local))
                    _attrs = _safe_attrs_dump(model_cls_local, instance)
                    raise PermissionError(f"Mutation out of scope for update; model={_cls}; attrs={_attrs}")

            created_now = False
            if instance is None:
                instance = model_cls_local()
                if pk_val is not None:
                    try:
                        setattr(instance, pk_name, pk_val)
                    except Exception:
                        pass
                created_now = True
                try:
                    current_pk = getattr(instance, pk_name, None)
                except Exception:
                    current_pk = None
                try:
                    if current_pk in (None, '') and _is_mssql(session) and _has_guid_pk(model_cls_local, pk_name):
                        setattr(instance, pk_name, uuid.uuid4())
                except Exception:
                    pass
                session.add(instance)

            # Assign scalar fields (ignore None)
            for k, v in list(scalar_vals.items()):
                if v is None:
                    continue
                try:
                    setattr(instance, k, v)
                except Exception:
                    try:
                        src_col = (getattr(btype_local.__berry_fields__.get(k), 'meta', {}) or {}).get('column')
                        if src_col:
                            setattr(instance, src_col, v)
                    except Exception:
                        pass

            # Robust fallback for MSSQL: ensure child FK set directly on instance
            if parent_ctx is not None:
                try:
                    parent_model_cls = parent_ctx.get('parent_model')
                    parent_inst = parent_ctx.get('parent_inst')
                except Exception:
                    parent_model_cls = None
                    parent_inst = None
                if parent_model_cls is not None and parent_inst is not None:
                    try:
                        parent_pk_name_local = schema._get_pk_name(parent_model_cls)
                    except Exception:
                        parent_pk_name_local = 'id'
                    try:
                        parent_pk_val_local = getattr(parent_inst, parent_pk_name_local, None)
                    except Exception:
                        parent_pk_val_local = None
                    if parent_pk_val_local is None:
                        try:
                            await session.flush()
                        except Exception:
                            pass
                        try:
                            await session.refresh(parent_inst)
                        except Exception:
                            pass
                        try:
                            parent_pk_val_local = getattr(parent_inst, parent_pk_name_local, None)
                        except Exception:
                            parent_pk_val_local = None
                    # Determine fk name
                    try:
                        _fk_name = None
                        if hasattr(model_cls_local, '__table__') and hasattr(parent_model_cls, '__table__'):
                            for cc in model_cls_local.__table__.columns:
                                for fk in getattr(cc, 'foreign_keys', []) or []:
                                    if fk.column.table.name == parent_model_cls.__table__.name:
                                        _fk_name = cc.name
                                        break
                                if _fk_name is not None:
                                    break
                    except Exception:
                        _fk_name = None
                    if _fk_name and parent_pk_val_local is not None:
                        try:
                            cur = getattr(instance, _fk_name, None)
                        except Exception:
                            cur = None
                        if cur in (None, 0):
                            try:
                                setattr(instance, _fk_name, parent_pk_val_local)
                            except Exception:
                                pass

            # Also set relationship attribute to parent instance if available
            if parent_ctx is not None and child_rel_to_parent_name and 'parent_inst' in parent_ctx:
                try:
                    cur_val = getattr(instance, child_rel_to_parent_name, None)
                except Exception:
                    cur_val = None
                try:
                    if cur_val is None:
                        setattr(instance, child_rel_to_parent_name, parent_ctx.get('parent_inst'))
                except Exception:
                    pass

            # If we pre-created a single child whose FK is on parent, also set relationship attribute when possible
            for _rk, _child in list(precreated_children.items()):
                try:
                    if getattr(instance, _rk, None) is None:
                        setattr(instance, _rk, _child)
                except Exception:
                    pass

            # Flush to materialize identities
            await session.flush()

            # Enforce scope AFTER changes
            if eff_scope is not None:
                ok2 = await _enforce_scope(model_cls_local, instance, eff_scope)
                if not ok2:
                    try:
                        await session.rollback()
                    except Exception:
                        pass
                    _cls = getattr(model_cls_local, '__name__', str(model_cls_local))
                    _attrs = _safe_attrs_dump(model_cls_local, instance)
                    raise PermissionError(f"Mutation out of scope; model={_cls}; attrs={_attrs}")

            # Process relations
            for rel_key, rel_value in list(relation_vals.items()):
                if rel_value is None:
                    continue
                rel_def = btype_local.__berry_fields__.get(rel_key)
                if rel_def is None or rel_def.kind != 'relation':
                    continue
                target_name = rel_def.meta.get('target') if isinstance(rel_def.meta, dict) else None
                is_single = bool(rel_def.meta.get('single')) if isinstance(rel_def.meta, dict) else False
                if not target_name:
                    continue
                child_btype = schema.types.get(target_name)
                child_model = getattr(child_btype, 'model', None) if child_btype else None
                if not child_btype or not child_model:
                    continue
                try:
                    child_pk_name = schema._get_pk_name(child_model)
                except Exception:
                    child_pk_name = 'id'
                if is_single:
                    child_data = schema._input_to_dict(rel_value)
                    child_inst = await _merge_single(child_model, child_btype, child_data, parent_ctx={'parent_model': model_cls_local, 'parent_inst': instance}, rel_name_from_parent=rel_key)
                    try:
                        fk_name = schema._find_parent_fk_column_name(model_cls_local, child_model, rel_key)
                    except Exception:
                        fk_name = None
                    if fk_name and getattr(instance, fk_name, None) in (None, 0):
                        try:
                            setattr(instance, fk_name, getattr(child_inst, child_pk_name))
                        except Exception:
                            pass
                else:
                    items = schema._input_to_dict(rel_value)
                    if not isinstance(items, list):
                        items = [items]
                    child_fk_col_name = None
                    try:
                        for cc in child_model.__table__.columns:
                            for fk in cc.foreign_keys:
                                if fk.column.table.name == model_cls_local.__table__.name:
                                    child_fk_col_name = cc.name
                                    break
                            if child_fk_col_name is not None:
                                break
                    except Exception:
                        child_fk_col_name = None
                    if child_fk_col_name is None:
                        try:
                            conv_name = f"{model_cls_local.__table__.name.rstrip('s')}_id"
                            if any(c.name == conv_name for c in child_model.__table__.columns):
                                child_fk_col_name = conv_name
                        except Exception:
                            pass
                    try:
                        parent_pk_name_local = schema._get_pk_name(model_cls_local)
                    except Exception:
                        parent_pk_name_local = 'id'
                    try:
                        parent_pk_val = getattr(instance, parent_pk_name_local)
                    except Exception:
                        parent_pk_val = None
                    if parent_pk_val is None:
                        try:
                            await session.flush()
                        except Exception:
                            pass
                        try:
                            await session.refresh(instance)
                        except Exception:
                            pass
                        try:
                            parent_pk_val = getattr(instance, parent_pk_name_local)
                        except Exception:
                            parent_pk_val = None
                    for item in items:
                        if child_fk_col_name and parent_pk_val is not None:
                            item = dict(item or {})
                            if item.get(child_fk_col_name) in (None, 0):
                                item[child_fk_col_name] = parent_pk_val
                        await _merge_single(child_model, child_btype, item, parent_ctx={'parent_model': model_cls_local, 'parent_inst': instance}, rel_name_from_parent=rel_key)

            await session.flush()
            # Fire post-callbacks
            if eff_post_cb is not None:
                await _maybe_call_post(eff_post_cb, model_cls_local, info, instance, created_now, {'parent': parent_ctx, 'relation': rel_name_from_parent})
            if type_post_cb is not None:
                await _maybe_call_post(type_post_cb, model_cls_local, info, instance, created_now, {'parent': parent_ctx, 'relation': rel_name_from_parent})
            return instance

        inst = await _merge_single(model_cls, btype_cls, data)
        await session.commit()
        return schema.from_model(type_name, inst)

    fname = field_name or (f"merge_{type_name[:-2].lower()}" if type_name.endswith('QL') else f"merge_{type_name.lower()}")
    ann: Dict[str, Any] = {'info': StrawberryInfo, 'payload': input_type, 'return': st_return}

    async def _resolver(self, info: StrawberryInfo, payload):  # noqa: D401
        return await _merge_impl(info, payload)

    _resolver.__name__ = fname
    _resolver.__annotations__ = ann
    return fname, strawberry.field(resolver=_resolver), st_return

# --- Domain mutation support ---------------------------------------------

def _compose_mut_name(schema: 'BerrySchema', rel_attr: str) -> str:
    if getattr(schema, '_auto_camel_case', False):
        return f"merge{rel_attr[:1].upper()}{rel_attr[1:]}"
    out = re.sub(r"([A-Z])", r"_\1", rel_attr).lower().strip('_')
    return f"merge_{out}"

# Backward-compatibility aliases (no cover)
def build_upsert_resolver_for_type(*args, **kwargs):  # pragma: no cover
    return build_merge_resolver_for_type(*args, **kwargs)

def add_top_level_upserts(schema: 'BerrySchema', MPlain: type, anns_m: Dict[str, Any]) -> None:  # pragma: no cover
    return add_top_level_merges(schema, MPlain, anns_m)


def ensure_mutation_domain_type(schema: 'BerrySchema', dom_cls: Type['BerryDomain']):
    """Build or return a Strawberry type for domain-scoped mutation container.

    Adds auto-generated merge_* fields for relations on the domain with meta['mutation'] True.
    """
    # Cache key/type name
    type_name = f"{getattr(dom_cls, '__name__', 'Domain')}MutType"
    # If cached, ensure it's not an empty/stale type without fields; if so, drop it and rebuild/skip
    if type_name in schema._st_types:
        cached = schema._st_types.get(type_name)
        try:
            has_fields = False
            for attr_name in dir(cached):
                try:
                    v = getattr(cached, attr_name)
                except Exception:
                    continue
                mod = getattr(getattr(v, "__class__", object), "__module__", "") or ""
                # Heuristic: strawberry field/mutation objects live under the strawberry.* modules or expose a resolver
                if mod.startswith("strawberry") or hasattr(v, "resolver") or hasattr(v, "base_resolver"):
                    has_fields = True
                    break
            if not has_fields:
                # Stale/empty cached type; remove and treat as not present
                try:
                    del schema._st_types[type_name]
                except Exception:
                    pass
            else:
                return cached
        except Exception:
            # If inspection fails, fall through and rebuild
            pass
    DomSt_local = type(type_name, (), {'__doc__': f'Mutation domain container for {getattr(dom_cls, "__name__", type_name)}'})
    DomSt_local.__module__ = schema.__class__.__module__
    ann_local: Dict[str, Any] = {}
    # Copy callable/strawberry fields from domain class (mutations authored by user)
    from .core.fields import FieldDescriptor as _FldDesc, DomainDescriptor as _DomDesc
    for fname, fval in list(vars(dom_cls).items()):
        if isinstance(fval, _FldDesc) or isinstance(fval, _DomDesc):
            continue
        if fname.startswith('__') or fname.startswith('_'):
            continue
        looks_strawberry = False
        try:
            mod = getattr(getattr(fval, "__class__", object), "__module__", "") or ""
            looks_strawberry = (
                mod.startswith("strawberry") or hasattr(fval, "resolver") or hasattr(fval, "base_resolver")
            )
        except Exception:
            looks_strawberry = False
        try:
            if looks_strawberry:
                setattr(DomSt_local, fname, fval)
                fn = getattr(fval, 'resolver', None) or getattr(getattr(fval, 'base_resolver', None), 'func', None) or getattr(fval, 'func', None)
                ret_ann = getattr(fn, '__annotations__', {}).get('return') if callable(fn) else None
            elif callable(fval):
                setattr(DomSt_local, fname, strawberry.field(resolver=fval))
                ret_ann = getattr(fval, '__annotations__', {}).get('return')
            else:
                continue
            # Map BerryType return to runtime Strawberry type
            mapped_type = None
            try:
                if get_origin(ret_ann) is getattr(__import__('typing'), 'Annotated', None):
                    args = list(get_args(ret_ann) or [])
                    if args:
                        ret_ann = args[0]
                if isinstance(ret_ann, str) and ret_ann in schema.types:
                    mapped_type = schema._st_types.get(ret_ann)
                if ret_ann in schema.types.values():
                    nm = getattr(ret_ann, '__name__', None)
                    if nm:
                        mapped_type = schema._st_types.get(nm)
            except Exception:
                mapped_type = None
            ann_local[fname] = mapped_type or ret_ann or Any
        except Exception:
            pass
    # Nested mutation domains
    for fname, fval in list(vars(dom_cls).items()):
        if isinstance(fval, _DomDesc):
            child_cls = fval.domain_cls
            child_dom_st = ensure_mutation_domain_type(schema, child_cls)
            if child_dom_st is None:
                continue
            ann_local[fname] = child_dom_st  # type: ignore
            def _make_nested_resolver(ChildSt):
                async def _resolver(self, info: StrawberryInfo):  # noqa: D401
                    inst = ChildSt()
                    setattr(inst, '__berry_registry__', getattr(self, '__berry_registry__', self))
                    return inst
                return _resolver
            setattr(DomSt_local, fname, strawberry.field(resolver=_make_nested_resolver(child_dom_st)))
    # Auto-generate merge mutations for relations on domain with mutation=True
    try:
        for rf, rval in list(vars(dom_cls).items()):
            if isinstance(rval, _FldDesc):
                rdef = rval.build(getattr(dom_cls, '__name__', 'Domain'))
                if rdef.kind != 'relation':
                    continue
                if not bool((rdef.meta or {}).get('mutation')):
                    continue
                target_name = rdef.meta.get('target') if isinstance(rdef.meta, dict) else None
                if not target_name:
                    continue
                btype_t = schema.types.get(target_name)
                if not btype_t:
                    continue
                # Determine effective scope: relation scope or domain guard
                eff_scope = None
                eff_pre_cb = None
                eff_post_cb = None
                try:
                    eff_scope = (getattr(rdef, 'meta', {}) or {}).get('scope')
                    eff_pre_cb = (getattr(rdef, 'meta', {}) or {}).get('pre') or (getattr(rdef, 'meta', {}) or {}).get('pre_merge') or (getattr(rdef, 'meta', {}) or {}).get('pre_upsert')
                    eff_post_cb = (getattr(rdef, 'meta', {}) or {}).get('post') or (getattr(rdef, 'meta', {}) or {}).get('post_merge') or (getattr(rdef, 'meta', {}) or {}).get('post_upsert')
                    if eff_scope is None:
                        eff_scope = getattr(dom_cls, '__domain_guard__', None)
                except Exception:
                    eff_scope = getattr(dom_cls, '__domain_guard__', None)
                    eff_pre_cb = None
                    eff_post_cb = None
                triplet = None
                try:
                    triplet = build_merge_resolver_for_type(
                        schema,
                        btype_t,
                        field_name=_compose_mut_name(schema, rf),
                        relation_scope=eff_scope,
                        pre_callback=eff_pre_cb,
                        post_callback=eff_post_cb,
                    )
                except Exception:
                    triplet = None
                if triplet is not None:
                    fname_u, field_obj, st_ret = triplet
                    try:
                        setattr(DomSt_local, fname_u, field_obj)
                        ann_local[fname_u] = st_ret  # type: ignore
                    except Exception:
                        pass
    except Exception:
        pass
    # If no fields were collected, or DomSt_local ended up without any strawberry field/mutation attrs, skip creating the type entirely
    if not ann_local:
        try:
            _logger.warning("berryql.mutations: skip %s (no fields)", type_name)
        except Exception:
            pass
        return None
    try:
        any_st_fields = False
        for attr_name in dir(DomSt_local):
            try:
                v = getattr(DomSt_local, attr_name)
            except Exception:
                continue
            mod = getattr(getattr(v, "__class__", object), "__module__", "") or ""
            if mod.startswith("strawberry") or hasattr(v, "resolver") or hasattr(v, "base_resolver"):
                any_st_fields = True
                break
        if not any_st_fields:
            try:
                _logger.warning("berryql.mutations: skip %s (no strawberry fields)", type_name)
            except Exception:
                pass
            return None
    except Exception:
        # best-effort: proceed with annotations gate alone
        pass
    DomSt_local.__annotations__ = ann_local
    try:
        _logger.warning("berryql.mutations: create %s with fields=%s", type_name, sorted(list(ann_local.keys())))
    except Exception:
        pass
    schema._st_types[type_name] = strawberry.type(DomSt_local)  # type: ignore
    try:
        globals()[type_name] = schema._st_types[type_name]
    except Exception:
        pass
    return schema._st_types[type_name]


def add_mutation_domains(schema: 'BerrySchema', MPlain: type, anns_m: Dict[str, Any]) -> None:
    """Attach domain fields declared on user Mutation, using mutation domain types, and add merges inside them."""
    from .core.fields import DomainDescriptor as _DomDesc
    if getattr(schema, '_user_mutation_cls', None) is None:
        return
    for uf, val in list(vars(schema._user_mutation_cls).items()):
        if isinstance(val, _DomDesc):
            dom_cls = val.domain_cls
            DomSt = ensure_mutation_domain_type(schema, dom_cls)
            if DomSt is None:
                continue
            def _make_domain_resolver(DomSt_local):
                async def _resolver(self, info: StrawberryInfo):  # noqa: D401
                    inst = DomSt_local()
                    setattr(inst, '__berry_registry__', schema)
                    return inst
                return _resolver
            anns_m[uf] = DomSt  # type: ignore
            setattr(MPlain, uf, strawberry.field(resolver=_make_domain_resolver(DomSt)))
            # Best-effort: ensure merge_* are present on the decorated runtime type too
            try:
                DomRuntime = schema._st_types.get(f"{getattr(dom_cls,'__name__','Domain')}MutType")
                if DomRuntime is not None:
                    from .core.fields import FieldDescriptor as _FldDesc
                    updated = False
                    for rf, rval in list(vars(dom_cls).items()):
                        if isinstance(rval, _FldDesc):
                            rdef = rval.build(getattr(dom_cls, '__name__', 'Domain'))
                            if rdef.kind != 'relation':
                                continue
                            if not bool((rdef.meta or {}).get('mutation')):
                                continue
                            target_name = rdef.meta.get('target') if isinstance(rdef.meta, dict) else None
                            if not target_name:
                                continue
                            btype_t = schema.types.get(target_name)
                            if not btype_t:
                                continue
                            # Determine scope as in builder above
                            eff_scope = None
                            eff_pre_cb = None
                            eff_post_cb = None
                            try:
                                eff_scope = (getattr(rdef, 'meta', {}) or {}).get('scope')
                                eff_pre_cb = (getattr(rdef, 'meta', {}) or {}).get('pre') or (getattr(rdef, 'meta', {}) or {}).get('pre_merge') or (getattr(rdef, 'meta', {}) or {}).get('pre_upsert')
                                eff_post_cb = (getattr(rdef, 'meta', {}) or {}).get('post') or (getattr(rdef, 'meta', {}) or {}).get('post_merge') or (getattr(rdef, 'meta', {}) or {}).get('post_upsert')
                                if eff_scope is None:
                                    eff_scope = getattr(dom_cls, '__domain_guard__', None)
                            except Exception:
                                eff_scope = getattr(dom_cls, '__domain_guard__', None)
                                eff_pre_cb = None
                                eff_post_cb = None
                            triplet = build_merge_resolver_for_type(
                                schema,
                                btype_t,
                                field_name=_compose_mut_name(schema, rf),
                                relation_scope=eff_scope,
                                pre_callback=eff_pre_cb,
                                post_callback=eff_post_cb,
                            )
                            if triplet is None:
                                continue
                            fname_u, field_obj, st_ret = triplet
                            if not hasattr(DomRuntime, fname_u):
                                try:
                                    setattr(DomRuntime, fname_u, field_obj)
                                    ann_dom = getattr(DomRuntime, '__annotations__', None)
                                    if isinstance(ann_dom, dict):
                                        ann_dom[fname_u] = st_ret
                                    updated = True
                                except Exception:
                                    pass
                    if updated:
                        schema._st_types[f"{getattr(dom_cls,'__name__','Domain')}MutType"] = DomRuntime
            except Exception:
                pass


# --- Top-level merges ----------------------------------------------------

def add_top_level_merges(schema: 'BerrySchema', MPlain: type, anns_m: Dict[str, Any]) -> None:
    """Attach auto-generated merge mutations at root, gated by Query relations with mutation=True."""
    # Compute allow-list of target Berry types from root Query relations with mutation=True
    allowed_targets: Optional[set[str]] = None
    scope_by_target: Dict[str, Any] = {}
    try:
        if isinstance(getattr(schema, '_root_query_fields', None), dict):
            allowed_targets = set()
            for fdef in schema._root_query_fields.values():  # type: ignore[attr-defined]
                try:
                    if getattr(fdef, 'kind', None) != 'relation':
                        continue
                    meta = getattr(fdef, 'meta', {}) or {}
                    if not bool(meta.get('mutation')):
                        continue
                    tgt = meta.get('target')
                    if not tgt:
                        continue
                    allowed_targets.add(tgt)
                    if tgt not in scope_by_target:
                        scope_by_target[tgt] = meta.get('scope')
                    # Also stash callbacks per target when declared on root
                    try:
                        _pre = meta.get('pre') or meta.get('pre_merge') or meta.get('pre_upsert')
                        _post = meta.get('post') or meta.get('post_merge') or meta.get('post_upsert')
                    except Exception:
                        _pre = None
                        _post = None
                    if _pre is not None:
                        scope_by_target[f"__pre__::{tgt}"] = _pre
                    if _post is not None:
                        scope_by_target[f"__post__::{tgt}"] = _post
                except Exception:
                    continue
    except Exception:
        allowed_targets = None
        scope_by_target = {}
    for tname, btype_cls in list(schema.types.items()):
        if allowed_targets is not None and (not tname or tname not in allowed_targets):
            continue
        triplet = build_merge_resolver_for_type(
            schema,
            btype_cls,
            relation_scope=scope_by_target.get(tname),
            pre_callback=scope_by_target.get(f"__pre__::{tname}"),
            post_callback=scope_by_target.get(f"__post__::{tname}"),
        )
        if triplet is None:
            continue
        fname_u, field_obj, st_ret = triplet
        if hasattr(MPlain, fname_u):
            continue
        setattr(MPlain, fname_u, field_obj)
        anns_m[fname_u] = st_ret  # type: ignore
