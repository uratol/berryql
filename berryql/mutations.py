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
from .sql.builders import RelationSQLBuilders
from .core.enum_utils import get_model_enum_cls, coerce_input_to_storage_value, normalize_instance_enums

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

# --- Shared helpers (module-level) ---------------------------------------

def compose_scope_with_guard(dom_cls: Any, desc_scope: Any):
    """Compose descriptor-level scope with domain guard into a single async callable.

    Returns an async function (model_cls, info) -> scope_value suitable for _apply_where_common.
    """
    async def _inner(model_cls_local, info_local):
        import inspect as _ins
        ds_val = desc_scope
        if callable(ds_val):
            ds_val = ds_val(model_cls_local, info_local)
            if _ins.isawaitable(ds_val):
                ds_val = await ds_val  # type: ignore
        guard = getattr(dom_cls, '__domain_guard__', None)
        if callable(guard):
            g_val = guard(model_cls_local, info_local)
            if _ins.isawaitable(g_val):
                g_val = await g_val  # type: ignore
        else:
            g_val = guard
        if ds_val is None:
            return g_val
        if g_val is None:
            return ds_val
        return (ds_val, g_val)
    return _inner

# --- Merge builder -------------------------------------------------------

def build_merge_resolver_for_type(
    schema: 'BerrySchema',
    btype_cls: Type['BerryType'],
    *,
    field_name: Optional[str] = None,
    relation_scope: Any | None = None,
    pre_callback: Any | None = None,
    post_callback: Any | None = None,
    payload_is_list: bool | None = None,
    return_is_list: bool | None = None,
    description: str | None = None,
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


        async def _maybe_call_pre(cb, model_cls_local, info_local, data_local, context_local):
            if not callable(cb):
                return data_local
            try:
                sig = inspect.signature(cb)
                params = list(sig.parameters.keys())
            except Exception:
                params = []
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

        async def _maybe_call_post(cb, model_cls_local, info_local, instance_local, created_local, context_local):
            if not callable(cb):
                return None
            try:
                sig = inspect.signature(cb)
                params = list(sig.parameters.keys())
            except Exception:
                params = []
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

        # Multi-callback helpers
        def _collect_cbs(cbs_or_one):
            out: List[Any] = []
            try:
                if cbs_or_one is None:
                    return out
                if isinstance(cbs_or_one, (list, tuple)):
                    for it in cbs_or_one:
                        if callable(it):
                            out.append(it)
                    return out
                if callable(cbs_or_one):
                    return [cbs_or_one]
            except Exception:
                return out
            return out

        async def _maybe_call_pre_many(cbs, model_cls_local, info_local, data_local, context_local):
            current = data_local
            for cb in _collect_cbs(cbs):
                current = await _maybe_call_pre(cb, model_cls_local, info_local, current, context_local)
            return current

        async def _maybe_call_post_many(cbs, model_cls_local, info_local, instance_local, created_local, context_local):
            for cb in _collect_cbs(cbs):
                await _maybe_call_post(cb, model_cls_local, info_local, instance_local, created_local, context_local)

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

        # Local DRY helpers
        def _get_pk_name_safe(model_cls_local) -> str:
            try:
                return schema._get_pk_name(model_cls_local)
            except Exception:
                return 'id'

        async def _get_pk_value(instance_local, model_cls_local):
            pk_name_local = _get_pk_name_safe(model_cls_local)
            try:
                val = getattr(instance_local, pk_name_local, None)
            except Exception:
                val = None
            if val not in (None, ''):
                return val
            try:
                await session.flush()
            except Exception:
                pass
            try:
                await session.refresh(instance_local)
            except Exception:
                pass
            try:
                return getattr(instance_local, pk_name_local, None)
            except Exception:
                return None

        def _get_parent_btype_for_model(parent_model_cls):
            try:
                for _n, _bt in (schema.types or {}).items():
                    if getattr(_bt, 'model', None) is parent_model_cls:
                        return _bt
            except Exception:
                return None
            return None

        def _get_parent_relation_meta(parent_ctx_local, rel_name_local):
            if not parent_ctx_local or not rel_name_local:
                return {}
            try:
                parent_model_cls = parent_ctx_local.get('parent_model')
                parent_btype = _get_parent_btype_for_model(parent_model_cls)
                if parent_btype is None:
                    return {}
                rdef = getattr(parent_btype, '__berry_fields__', {}).get(rel_name_local)
                if rdef is not None and getattr(rdef, 'kind', None) == 'relation':
                    return getattr(rdef, 'meta', {}) or {}
            except Exception:
                return {}
            return {}

        def _detect_child_fk_column(parent_model_cls, child_model_cls, override: Optional[str] = None) -> Optional[str]:
            if override:
                return override
            try:
                if hasattr(child_model_cls, '__table__') and hasattr(parent_model_cls, '__table__'):
                    for cc in child_model_cls.__table__.columns:
                        for fk in getattr(cc, 'foreign_keys', []) or []:
                            try:
                                if fk.column.table.name == parent_model_cls.__table__.name:
                                    return cc.name
                            except Exception:
                                continue
            except Exception:
                pass
            try:
                conv_name = f"{parent_model_cls.__table__.name.rstrip('s')}_id"
                if hasattr(child_model_cls, '__table__') and any(c.name == conv_name for c in child_model_cls.__table__.columns):
                    return conv_name
            except Exception:
                pass
            return None

        async def _enforce_scope(model_cls_local, instance_local, scope_raw: Any | None):
            """Enforce mutation scope using the same where builder as queries.

            Accepts dict/JSON string/callable/SQLA expression. Callable may return any of those.
            Supports ScalarSelect inside where dicts (e.g., {'id': {'in': select(...).scalar_subquery()}}).
            """
            if scope_raw is None:
                return True
            pk_name_local = _get_pk_name_safe(model_cls_local)
            try:
                pk_val_local = getattr(instance_local, pk_name_local)
            except Exception:
                pk_val_local = None
            if pk_val_local is None:
                return True
            # Resolve potential GraphQL variables/literals
            v = _resolve_scope_value(scope_raw)
            # Match query path: resolve async callables (like apply_root_filters does)
            if callable(v):
                try:
                    v_res = v(model_cls_local, info)
                    import inspect as _ins
                    if _ins.isawaitable(v_res):
                        v_res = await v_res  # type: ignore
                    v = v_res
                except Exception:
                    # Do not suppress scope resolution errors
                    raise
            # Base statement: select PK for this instance
            stmt = select(getattr(model_cls_local.__table__.c, pk_name_local)).select_from(model_cls_local).where(
                getattr(model_cls_local.__table__.c, pk_name_local) == pk_val_local
            )
            # Reuse common builder to apply where regardless of type (dict/str/expr)
            builder = RelationSQLBuilders(schema)
            stmt = builder._apply_where_common(
                stmt,
                model_cls_local,
                v,
                strict=True,
                to_where_dict=_to_where_dict,
                expr_from_where_dict=_expr_from_where_dict,
                info=info,
            )
            res = await session.execute(stmt)
            row = res.first()
            return bool(row)

        async def _merge_single(model_cls_local, btype_local, data_local, parent_ctx=None, rel_name_from_parent: Optional[str] = None):
            # Compute effective callbacks: combine optional explicit ones with type-level decorators
            eff_pre_cbs: List[Any] = []
            eff_post_cbs: List[Any] = []
            try:
                meta = _get_parent_relation_meta(parent_ctx, rel_name_from_parent)
                if meta:
                    # Only support type-level decorators now; relation meta pre/post removed.
                    pass
            except Exception:
                pass

            # Resolve type-level callbacks for the current (possibly nested) BerryType
            local_type_pre_cbs = list(getattr(btype_local, '__merge_pre_cbs__', ()) or ())
            local_type_post_cbs = list(getattr(btype_local, '__merge_post_cbs__', ()) or ())
            # Prepend/append explicit callbacks if provided at builder time
            if pre_callback is not None:
                try:
                    if isinstance(pre_callback, (list, tuple)):
                        eff_pre_cbs.extend([cb for cb in pre_callback if callable(cb)])
                    elif callable(pre_callback):
                        eff_pre_cbs.append(pre_callback)
                except Exception:
                    pass
            if post_callback is not None:
                try:
                    if isinstance(post_callback, (list, tuple)):
                        eff_post_cbs.extend([cb for cb in post_callback if callable(cb)])
                    elif callable(post_callback):
                        eff_post_cbs.append(post_callback)
                except Exception:
                    pass
            # Final effective callback lists
            eff_pre_cbs.extend(local_type_pre_cbs)
            eff_post_cbs.extend(local_type_post_cbs)

            # Determine applicable scope for this level (needed early for delete path)
            eff_scope = None
            eff_scope_inherited_from_parent = False  # when True, enforce on parent model, not on child
            try:
                meta = _get_parent_relation_meta(parent_ctx, rel_name_from_parent)
                eff_scope = meta.get('scope') if isinstance(meta, dict) else None
                if eff_scope is None:
                    eff_scope = relation_scope
                    # If we're nested under a parent and there's no relation-specific scope,
                    # the effective scope comes from the domain. In that case enforce it
                    # against the parent instance rather than the child rows.
                    if parent_ctx is not None and relation_scope is not None:
                        eff_scope_inherited_from_parent = True
            except Exception:
                eff_scope = relation_scope
                if parent_ctx is not None and relation_scope is not None:
                    eff_scope_inherited_from_parent = True

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
                # If this delete is occurring as a child operation under a parent instance,
                # validate that the child belongs to the parent (by FK) and prepare data for parent scope enforcement.
                _fk_name_ctx = (parent_ctx or {}).get('child_fk_col_name')
                _parent_scope = (parent_ctx or {}).get('parent_scope')
                _parent_inst = (parent_ctx or {}).get('parent_inst')
                _child_fk_val = None
                try:
                    if _fk_name_ctx and _parent_inst is not None:
                        try:
                            _parent_pk_name = schema._get_pk_name(parent_ctx.get('parent_model')) if parent_ctx else 'id'
                        except Exception:
                            _parent_pk_name = 'id'
                        try:
                            _parent_pk_val = getattr(_parent_inst, _parent_pk_name, None)
                        except Exception:
                            _parent_pk_val = None
                        try:
                            _child_fk_val = getattr(instance_for_delete, _fk_name_ctx, None)
                        except Exception:
                            _child_fk_val = None
                        if _parent_pk_val is not None and _child_fk_val is not None and str(_child_fk_val) != str(_parent_pk_val):
                            raise PermissionError("Mutation out of scope for delete; child does not belong to parent")
                except PermissionError:
                    raise
                except Exception:
                    # Non-fatal validation issues shouldn't prevent subsequent scope enforcement
                    pass
                # Enforce the parent scope against the actual parent row id from the child FK.
                if _parent_scope is not None and _fk_name_ctx:
                    # If we can map back to a parent model, check that parent id under scope exists.
                    parent_model_cls_local = (parent_ctx or {}).get('parent_model') if parent_ctx else None
                    parent_pk_name_scope = schema._get_pk_name(parent_model_cls_local) if parent_model_cls_local else 'id'
                    class _StubParent:
                        pass
                    _stub = _StubParent()
                    setattr(_stub, parent_pk_name_scope, _child_fk_val)
                    ok_parent = await _enforce_scope(parent_model_cls_local, _stub, _parent_scope)
                    if not ok_parent:
                        raise PermissionError("Mutation out of scope for delete; parent not within scope")
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
                # Type-level pre callbacks
                data_local = await _maybe_call_pre_many(local_type_pre_cbs, model_cls_local, info, data_local, {'parent': parent_ctx, 'relation': rel_name_from_parent, 'delete': True})
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
                await _maybe_call_post_many(local_type_post_cbs, model_cls_local, info, instance_for_delete, False, {'parent': parent_ctx, 'relation': rel_name_from_parent, 'delete': True})
                return instance_for_delete

            # Call pre-callbacks before any processing
            data_local = await _maybe_call_pre_many(local_type_pre_cbs, model_cls_local, info, data_local, {'parent': parent_ctx, 'relation': rel_name_from_parent})

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

            # If relation/domain scope provides fixed equality constraints, use them as defaults
            # for missing scalar fields on this level (exclude primary key).
            pk_name_local = _get_pk_name_safe(model_cls_local)
            if eff_scope is not None:
                try:
                    v_scope = _resolve_scope_value(eff_scope)
                    # Only handle dict/JSON forms for defaults; callables/expressions enforced below
                    from .core.utils import to_where_dict as _to_where_dict
                    wdict = _to_where_dict(v_scope, strict=False) if not isinstance(v_scope, dict) else v_scope
                except Exception:
                    wdict = None
                if isinstance(wdict, dict):
                    try:
                        # Iterate simple equality constraints and apply missing defaults
                        for _col, _ops in (wdict or {}).items():
                            try:
                                if _col == pk_name_local:
                                    continue
                                if not isinstance(_ops, dict):
                                    continue
                                if 'eq' not in _ops:
                                    continue
                                if scalar_vals.get(_col) is None:
                                    scalar_vals[_col] = _ops.get('eq')
                            except Exception:
                                continue
                    except Exception:
                        pass

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
                    # Determine child->parent FK column name (meta override > inspection > conventional)
                    meta = _get_parent_relation_meta(parent_ctx, rel_name_from_parent)
                    child_fk_to_parent = _detect_child_fk_column(parent_model_cls, model_cls_local, (meta or {}).get('fk_column_name'))
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
                    # child_fk_to_parent already includes conventional fallback
                    if child_fk_to_parent:
                        parent_pk_val_local = await _get_pk_value(parent_inst, parent_model_cls)
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

            # Enforce parent ownership constraint for child update when invoked under a parent
            if pk_val is not None and instance is not None and parent_ctx is not None:
                _fk_name_ctx = parent_ctx.get('child_fk_col_name')
                _parent_inst_ctx = parent_ctx.get('parent_inst')
                if _fk_name_ctx and _parent_inst_ctx is not None:
                    try:
                        _parent_pk_name_ctx = schema._get_pk_name(parent_ctx.get('parent_model')) if parent_ctx.get('parent_model') else 'id'
                    except Exception:
                        _parent_pk_name_ctx = 'id'
                    try:
                        _parent_pk_val_ctx = getattr(_parent_inst_ctx, _parent_pk_name_ctx, None)
                    except Exception:
                        _parent_pk_val_ctx = None
                    try:
                        _child_fk_val_ctx = getattr(instance, _fk_name_ctx, None)
                    except Exception:
                        _child_fk_val_ctx = None
                    if _parent_pk_val_ctx is not None and _child_fk_val_ctx is not None and str(_child_fk_val_ctx) != str(_parent_pk_val_ctx):
                        raise PermissionError("Mutation out of scope for update; child does not belong to parent")
            # Enforce relation scope before update
            if pk_val is not None and instance is not None and eff_scope is not None:
                if parent_ctx is not None and eff_scope_inherited_from_parent:
                    # Enforce the inherited domain scope on the parent instance instead of the child
                    parent_model_cls_local = parent_ctx.get('parent_model') if isinstance(parent_ctx, dict) else None
                    parent_inst_local = parent_ctx.get('parent_inst') if isinstance(parent_ctx, dict) else None
                    if parent_model_cls_local is not None and parent_inst_local is not None:
                        ok_parent = await _enforce_scope(parent_model_cls_local, parent_inst_local, parent_ctx.get('parent_scope'))
                        if not ok_parent:
                            raise PermissionError("Mutation out of scope for update; parent not within scope")
                else:
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
                # If invoked as child under a parent and child FK is present in scalar_vals but mismatched, reject
                if parent_ctx is not None:
                    _fk_name_ctx = parent_ctx.get('child_fk_col_name')
                    _parent_inst_ctx = parent_ctx.get('parent_inst')
                    if _fk_name_ctx and _parent_inst_ctx is not None:
                        try:
                            _parent_pk_name_ctx = schema._get_pk_name(parent_ctx.get('parent_model')) if parent_ctx.get('parent_model') else 'id'
                        except Exception:
                            _parent_pk_name_ctx = 'id'
                        try:
                            _parent_pk_val_ctx = getattr(_parent_inst_ctx, _parent_pk_name_ctx, None)
                        except Exception:
                            _parent_pk_val_ctx = None
                        if _parent_pk_val_ctx is not None and scalar_vals.get(_fk_name_ctx) not in (None, 0):
                            if str(scalar_vals.get(_fk_name_ctx)) != str(_parent_pk_val_ctx):
                                raise PermissionError("Mutation out of scope for create; child does not belong to parent")
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

            # Assign scalar fields; allow explicit None to clear nullable fields.
            # Do not overwrite primary key with None/empty.
            # Enum-aware coercion via SAEnum: accept NAME strings and coerce to enum.value before assignment.
            for k, v in list(scalar_vals.items()):
                try:
                    # Primary key guard
                    if k == pk_name and (v is None or v == ''):
                        continue
                    # Coerce enum NAME -> value when applicable
                    try:
                        col = getattr(getattr(model_cls_local, '__table__', None).c, k)
                    except Exception:
                        col = None
                    if col is not None:
                        enum_cls = get_model_enum_cls(model_cls_local, k)
                        if enum_cls is not None:
                            v = coerce_input_to_storage_value(enum_cls, v)
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
                    # Determine fk name using helper
                    _fk_name = _detect_child_fk_column(parent_model_cls, model_cls_local)
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

            # Flush to materialize identities; ensure enum fields are normalized
            try:
                normalize_instance_enums(model_cls_local, instance)
            except Exception:
                pass
            await session.flush()

            # Enforce scope AFTER changes
            if eff_scope is not None:
                if parent_ctx is not None and eff_scope_inherited_from_parent:
                    parent_model_cls_local = parent_ctx.get('parent_model') if isinstance(parent_ctx, dict) else None
                    parent_inst_local = parent_ctx.get('parent_inst') if isinstance(parent_ctx, dict) else None
                    if parent_model_cls_local is not None and parent_inst_local is not None:
                        ok2p = await _enforce_scope(parent_model_cls_local, parent_inst_local, parent_ctx.get('parent_scope'))
                        if not ok2p:
                            try:
                                await session.rollback()
                            except Exception:
                                pass
                            raise PermissionError("Mutation out of scope; parent not within scope")
                else:
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
                    # Determine child FK column name for list relations using helper (override > inspection > conventional)
                    _meta = getattr(rel_def, 'meta', {}) or {}
                    child_fk_col_name = _detect_child_fk_column(model_cls_local, child_model, _meta.get('fk_column_name'))
                    try:
                        parent_pk_name_local = schema._get_pk_name(model_cls_local)
                    except Exception:
                        parent_pk_name_local = 'id'
                    try:
                        parent_pk_val = getattr(instance, parent_pk_name_local)
                    except Exception:
                        parent_pk_val = None
                    if parent_pk_val is None:
                        parent_pk_val = await _get_pk_value(instance, model_cls_local)
                    for item in items:
                        if child_fk_col_name and parent_pk_val is not None:
                            item = dict(item or {})
                            if item.get(child_fk_col_name) in (None, 0):
                                item[child_fk_col_name] = parent_pk_val
                        await _merge_single(
                            child_model,
                            child_btype,
                            item,
                            parent_ctx={
                                'parent_model': model_cls_local,
                                'parent_inst': instance,
                                'child_fk_col_name': child_fk_col_name,
                                'parent_scope': eff_scope,
                            },
                            rel_name_from_parent=rel_key,
                        )

            await session.flush()
            # Fire post-callbacks
            await _maybe_call_post_many(eff_post_cbs, model_cls_local, info, instance, created_now, {'parent': parent_ctx, 'relation': rel_name_from_parent})
            return instance

        # Support list payloads when configured by caller
        pl_is_list = bool(payload_is_list)
        rt_is_list = bool(return_is_list)
        if pl_is_list:
            items = data if isinstance(data, list) else ([data] if data is not None else [])
            out_models = []
            for item in items:
                inst = await _merge_single(model_cls, btype_cls, item)
                out_models.append(inst)
            await session.commit()
            if rt_is_list:
                return [schema.from_model(type_name, m) for m in out_models]
            # Fallback: return last item if single return requested
            last = out_models[-1] if out_models else None
            return schema.from_model(type_name, last) if last is not None else None
        else:
            inst = await _merge_single(model_cls, btype_cls, data)
            await session.commit()
            return schema.from_model(type_name, inst)

    fname = field_name or (f"merge_{type_name[:-2].lower()}" if type_name.endswith('QL') else f"merge_{type_name.lower()}")
    # Decide arg/return annotations (list vs single)
    ann_payload = input_type
    ann_return = st_return
    if bool(payload_is_list):
        from typing import List as _ListType  # avoid shadowing
        ann_payload = _ListType[input_type]  # type: ignore[index]
    if bool(return_is_list):
        from typing import List as _ListType  # avoid shadowing
        ann_return = _ListType[st_return]  # type: ignore[index]
    ann: Dict[str, Any] = {'info': StrawberryInfo, 'payload': ann_payload, 'return': ann_return}

    async def _resolver(self, info: StrawberryInfo, payload):  # noqa: D401
        return await _merge_impl(info, payload)

    _resolver.__name__ = fname
    _resolver.__annotations__ = ann
    if description:
        return fname, strawberry.field(resolver=_resolver, description=str(description)), st_return
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
    # Important: set the module to the domain class' module so that string annotations
    # (from "from __future__ import annotations") resolve in the correct global scope.
    # This allows Strawberry to find helper types like S3DownloadResponseType defined
    # next to the domain in its own module.
    try:
        DomSt_local.__module__ = getattr(dom_cls, '__module__', schema.__class__.__module__)
    except Exception:
        DomSt_local.__module__ = schema.__class__.__module__
    ann_local: Dict[str, Any] = {}
    # Keep track of effective annotations to enforce post-decoration
    _forced_type_anns: Dict[str, Any] = {}
    # Copy only strawberry mutation fields from domain class (mutations authored by user)
    from .core.fields import FieldDescriptor as _FldDesc, DomainDescriptor as _DomDesc
    for fname, fval in list(vars(dom_cls).items()):
        if isinstance(fval, _FldDesc) or isinstance(fval, _DomDesc):
            continue
        if fname.startswith('__') or fname.startswith('_'):
            continue
        # Expose Strawberry-decorated fields (including @strawberry.mutation). Skip plain callables.
        looks_strawberry = False
        try:
            mod = getattr(getattr(fval, "__class__", object), "__module__", "") or ""
            looks_strawberry = (
                mod.startswith("strawberry")
                or hasattr(fval, "resolver")
                or hasattr(fval, "base_resolver")
            )
        except Exception:
            looks_strawberry = False
        if not looks_strawberry:
            continue
    # Convert to a regular field using the underlying resolver to fit inside an object type
        try:
            fn = getattr(fval, 'resolver', None)
            br = getattr(fval, 'base_resolver', None)
            if fn is None and br is not None:
                fn = getattr(br, 'wrapped_func', None) or getattr(br, 'func', None)
            if fn is None:
                fn = getattr(fval, 'func', None)
            # Decide mutation using only explicit Strawberry flags (no heuristics)
            is_mut_like = False
            try:
                # Stable custom marker (set in berryql.__init__) takes priority
                if bool(getattr(fval, '__berry_is_mutation__', False)):
                    is_mut_like = True
                elif br is not None and bool(getattr(br, '__berry_is_mutation__', False)):
                    is_mut_like = True
                else:
                    # Fallback to Strawberry's flags when available
                    if bool(getattr(fval, 'is_mutation', False)):
                        is_mut_like = True
                    elif br is not None and bool(getattr(br, 'is_mutation', False)):
                        is_mut_like = True
                    else:
                        _fd = getattr(fval, 'field_definition', None)
                        if _fd is not None and bool(getattr(_fd, 'is_mutation', False)):
                            is_mut_like = True
            except Exception:
                is_mut_like = False
            if not is_mut_like:
                # Not a mutation; skip exposing on mutation domain
                continue
            # Determine return type from resolver annotation and map/resolve to a concrete Strawberry type
            ret_ann = None
            # Prefer original wrapped function annotations for accuracy
            _ann_source = None
            try:
                if br is not None:
                    _ann_source = getattr(br, 'wrapped_func', None) or getattr(br, 'func', None)
            except Exception:
                _ann_source = None
            if _ann_source is None:
                _ann_source = fn if callable(fn) else None
            if callable(_ann_source):
                try:
                    ret_ann = getattr(_ann_source, '__annotations__', {}).get('return')
                except Exception:
                    ret_ann = None
            mapped_type = None
            try:
                _ret_base = ret_ann
                from typing import Annotated as _Annotated, ForwardRef as _FwdRef  # type: ignore
                if get_origin(_ret_base) is _Annotated:
                    args = list(get_args(_ret_base) or [])
                    if args:
                        _ret_base = args[0]
                # Debug: dump initial state for specific known failing case
                try:
                    if getattr(dom_cls, '__name__', '') == 'BlogDomain' and fname == 'create_post_mut':
                        _logger.warning("mut-debug pre-map: ret_ann=%r base=%r st_types_has_PostQL=%s", ret_ann, _ret_base, 'PostQL' in (schema._st_types or {}))
                except Exception:
                    pass
                # If it's a bare ForwardRef (e.g., ForwardRef('PostQL')), resolve it in the domain module
                try:
                    if isinstance(_ret_base, _FwdRef):
                        _modname = getattr(dom_cls, '__module__', DomSt_local.__module__)
                        import importlib as _importlib
                        _mod = _importlib.import_module(_modname)
                        _name = getattr(_ret_base, '__forward_arg__', None)
                        if _name:
                            _evaled = eval(_name, vars(_mod))
                            # If Annotated after eval, unwrap to inner target
                            if get_origin(_evaled) is _Annotated:
                                _args2 = list(get_args(_evaled) or [])
                                if _args2:
                                    _evaled = _args2[0]
                            _ret_base = _evaled
                except Exception:
                    pass
                if isinstance(_ret_base, str) and _ret_base in schema.types:
                    mapped_type = schema._st_types.get(_ret_base)
                elif _ret_base in schema.types.values():
                    nm = getattr(_ret_base, '__name__', None)
                    if nm:
                        mapped_type = schema._st_types.get(nm)
                # Best-effort: resolve forward-ref strings in the domain module for non-Berry types
                if mapped_type is None and isinstance(_ret_base, str):
                    import importlib as _importlib
                    try:
                        _mod = _importlib.import_module(getattr(dom_cls, '__module__', DomSt_local.__module__))
                        _resolved = eval(_ret_base, vars(_mod))
                        # If we evaluated to Annotated, unwrap to the inner target
                        if get_origin(_resolved) is _Annotated:
                            _args = list(get_args(_resolved) or [])
                            if _args:
                                _target = _args[0]
                                if isinstance(_target, _FwdRef):
                                    _target_name = getattr(_target, '__forward_arg__', None)
                                    if _target_name:
                                        try:
                                            mapped_type = eval(_target_name, vars(_mod))
                                        except Exception:
                                            mapped_type = None
                                else:
                                    mapped_type = _target
                        else:
                            mapped_type = _resolved
                    except Exception:
                        mapped_type = None
                # Targeted debug for stubborn field
                try:
                    if getattr(dom_cls, '__name__', '') == 'RefDomain' and fname == 'get_s3_download_url':
                        _dbg_resolved_name = None
                        try:
                            import importlib as __importlib
                            __mod = __importlib.import_module(getattr(dom_cls, '__module__', DomSt_local.__module__))
                            _dbg_resolved_name = getattr(__mod, 'S3DownloadResponseType', None)
                        except Exception:
                            _dbg_resolved_name = None
                        _logger.warning(
                            "berryql.mutations: debug %s ret_ann=%r base=%r mapped=%r mod=%s has_strawberry_def=%s",
                            fname,
                            ret_ann,
                            _ret_base,
                            mapped_type,
                            getattr(dom_cls, '__module__', DomSt_local.__module__),
                            bool(getattr(_dbg_resolved_name, '__strawberry_definition__', None) or getattr(_dbg_resolved_name, '__is_strawberry_type__', False)),
                        )
                except Exception:
                    pass
            except Exception:
                mapped_type = None
            # Helper: robustly map any annotation (including Optional/List/Annotated) to runtime Strawberry types
            def _map_annotation_to_runtime(t: Any) -> Any:
                try:
                    if t is None:
                        return None
                    origin = get_origin(t)
                    args = list(get_args(t) or [])
                    from typing import Annotated as _Ann  # type: ignore
                    # Unwrap Annotated[T, ...]
                    if origin is _Ann:
                        return _map_annotation_to_runtime(args[0]) if args else None
                    # Optional[T] as Union[T, None]
                    try:
                        from typing import Union as _Union  # type: ignore
                    except Exception:
                        _Union = None  # type: ignore
                    if _Union is not None and origin is _Union:
                        non_none = [a for a in args if a is not type(None)]  # noqa: E721
                        if len(non_none) == 1 and len(args) == 2:
                            inner = _map_annotation_to_runtime(non_none[0])
                            from typing import Optional as _Optional  # type: ignore
                            try:
                                return _Optional[inner]  # type: ignore[index]
                            except Exception:
                                return inner
                        try:
                            return _Union[tuple(_map_annotation_to_runtime(a) for a in args)]  # type: ignore[index]
                        except Exception:
                            return t
                    # List[T]
                    if origin in (list, List):
                        inner = _map_annotation_to_runtime(args[0]) if args else Any
                        try:
                            return List[inner]  # type: ignore[index]
                        except Exception:
                            return list
                    # Plain string name of a Berry type
                    if isinstance(t, str) and t in (schema.types or {}):
                        rt = schema._st_types.get(t)
                        return rt or t
                    # BerryType class object
                    try:
                        if t in (schema.types or {}).values():
                            nm = getattr(t, '__name__', None)
                            if nm and nm in (schema._st_types or {}):
                                return schema._st_types[nm]
                    except Exception:
                        pass
                    return t
                except Exception:
                    return t

            # Build a fresh strawberry.field using the resolver and compute effective return type
            ann_effective = mapped_type if mapped_type is not None else ret_ann
            try:
                if getattr(dom_cls, '__name__', '') == 'BlogDomain' and fname == 'create_post_mut':
                    _logger.warning("mut-debug mapped: mapped=%r ann_effective=%r", mapped_type, ann_effective)
            except Exception:
                pass
            ann_effective = _map_annotation_to_runtime(ann_effective)
            # Also patch the resolver function's return annotation so Strawberry won't try
            # to resolve the original forward ref/lazy annotation attached on the class method.
            try:
                # Patch both original and effective resolver with the mapped return type
                if ann_effective is not None:
                    for _cand in (fn, getattr(br, 'wrapped_func', None), getattr(br, 'func', None), getattr(fval, 'func', None)):
                        if callable(_cand):
                            _anns = getattr(_cand, '__annotations__', None)
                            if isinstance(_anns, dict):
                                _anns['return'] = ann_effective
            except Exception:
                pass
            # If the effective return type is a Strawberry-decorated class living outside the
            # Berry registry, pre-register it into schema._st_types so the schema builder can
            # discover it when resolving this field.
            try:
                _rt = ann_effective
                # Unwrap Annotated once more if present
                if _rt is not None and get_origin(_rt) is not None:
                    from typing import Annotated as _Annotated
                    if get_origin(_rt) is _Annotated:
                        _rt_args = list(get_args(_rt) or [])
                        if _rt_args:
                            _rt = _rt_args[0]
                if isinstance(_rt, type):
                    # Only register if it looks like a strawberry type (avoid deprecated attribute access)
                    has_def = bool(getattr(_rt, "__strawberry_definition__", None) or getattr(_rt, "__is_strawberry_type__", False))
                    if has_def:
                        _nm = getattr(_rt, "__name__", None)
                        if _nm and schema._st_types.get(_nm) is None:
                            schema._st_types[_nm] = _rt  # type: ignore[assignment]
            except Exception:
                pass
            # Always attach a fresh Strawberry field with explicit resolver to avoid
            # leaking the original field's unresolved annotation into this container.
            try:
                # Reuse description if present on the original field
                _desc = getattr(getattr(fval, 'field_definition', None), 'description', None)
            except Exception:
                _desc = None
            # Resolve callable to the underlying plain function for Strawberry
            try:
                _fn_eff = fn
                if _fn_eff is None and br is not None:
                    _fn_eff = getattr(br, 'wrapped_func', None) or getattr(br, 'func', None)
                if _fn_eff is None:
                    _fn_eff = getattr(fval, 'func', None)
            except Exception:
                _fn_eff = fn
            try:
                fld = strawberry.field(resolver=_fn_eff, description=str(_desc)) if _desc else strawberry.field(resolver=_fn_eff)
                setattr(DomSt_local, fname, fld)
            except Exception:
                setattr(DomSt_local, fname, strawberry.field(resolver=_fn_eff))
            # If we could compute an effective return type, force the strawberry field
            # to use it by overriding its type_annotation. This prevents Strawberry from
            # failing to resolve forward-ref/lazy annotations on dynamically attached fields.
            try:
                if ann_effective is not None:
                    from strawberry.annotation import StrawberryAnnotation as _StrawberryAnnotation  # type: ignore
                    _fld_eff = getattr(DomSt_local, fname, None)
                    if _fld_eff is not None:
                        _fld_eff.type_annotation = _StrawberryAnnotation.from_annotation(ann_effective)
                        # If resolver exists on the field object, patch its return annotation too
                        try:
                            _res = getattr(_fld_eff, 'resolver', None)
                            if callable(_res):
                                anns = getattr(_res, '__annotations__', None)
                                if isinstance(anns, dict):
                                    anns['return'] = ann_effective
                        except Exception:
                            pass
            except Exception:
                pass
            # Targeted inspection of Strawberry internals for stubborn field
            try:
                if getattr(dom_cls, '__name__', '') == 'RefDomain' and fname == 'get_s3_download_url':
                    _fld = getattr(DomSt_local, fname, None)
                    _ta = getattr(_fld, 'type_annotation', None)
                    _ta_ann = getattr(_ta, 'annotation', None) if _ta is not None else None
                    _ta_resolved = None
                    try:
                        _ta_resolved = getattr(_ta, 'resolve', None)
                        if callable(_ta_resolved):
                            _ta_resolved = _ta.resolve()
                    except Exception:
                        _ta_resolved = None
                    _logger.warning("berryql.mutations: field=%s type_ann=%r ann=%r resolved=%r", fname, _ta, _ta_ann, _ta_resolved)
            except Exception:
                pass
            # Set class annotations accordingly so the decorated type sees a concrete return type
            if ann_effective is not None:
                ann_local[fname] = ann_effective  # type: ignore
                _forced_type_anns[fname] = ann_effective
        except Exception:
            pass
    for fname, fval in list(vars(dom_cls).items()):
        if isinstance(fval, _DomDesc):
            child_cls = fval.domain_cls
            child_dom_st = ensure_mutation_domain_type(schema, child_cls)
            if child_dom_st is None:
                continue
            ann_local[fname] = child_dom_st  # type: ignore
            _forced_type_anns[fname] = child_dom_st
            def _make_nested_resolver(ChildSt):
                async def _resolver(self, info: StrawberryInfo):  # noqa: D401
                    inst = ChildSt()
                    setattr(inst, '__berry_registry__', getattr(self, '__berry_registry__', self))
                    return inst
                return _resolver
            setattr(DomSt_local, fname, strawberry.field(resolver=_make_nested_resolver(child_dom_st)))
    # Auto-generate merge mutations declared explicitly via MutationDescriptor on the domain
    try:
        from .core.fields import MutationDescriptor as _MutDesc
        for attr_name, desc in list(vars(dom_cls).items()):
            if not isinstance(desc, _MutDesc):
                continue
            meta = getattr(desc, 'meta', {}) or {}
            target_name = meta.get('target')
            if not target_name:
                continue
            btype_t = schema.types.get(target_name)
            if not btype_t:
                continue
            # Build effective scope: optional descriptor scope AND domain guard
            eff_scope = compose_scope_with_guard(dom_cls, meta.get('scope'))
            is_single = bool(meta.get('single'))
            # Use attribute name as GraphQL mutation field name directly
            triplet = build_merge_resolver_for_type(
                schema,
                btype_t,
                field_name=attr_name,
                relation_scope=eff_scope,
                description=(meta.get('comment') if isinstance(meta, dict) else None),
                payload_is_list=(not is_single),
                return_is_list=False,
            )
            if triplet is None:
                continue
            fname_u, field_obj, st_ret = triplet
            try:
                setattr(DomSt_local, fname_u, field_obj)
                ann_local[fname_u] = st_ret  # type: ignore
                _forced_type_anns[fname_u] = st_ret
            except Exception:
                pass
    except Exception:
        pass
    # If DomSt_local ended up without any strawberry field/mutation attrs, skip creating the type entirely
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
    # Set annotations last (may be empty when only StrawberryField attributes are present)
    DomSt_local.__annotations__ = ann_local
    try:
        _logger.info("berryql.mutations: create %s with fields=%s", type_name, sorted(list(ann_local.keys())))
    except Exception:
        pass
    st_type = strawberry.type(DomSt_local)  # type: ignore
    # Enforce type annotations on the decorated Strawberry field definitions
    try:
        sd = getattr(st_type, '__strawberry_definition__', None)
        if sd is not None and _forced_type_anns:
            fields = list(getattr(sd, 'fields', []) or [])
            from strawberry.annotation import StrawberryAnnotation as _SA  # type: ignore
            for _f in fields:
                try:
                    _name = getattr(_f, 'name', None)
                    if not _name:
                        continue
                    if _name in _forced_type_anns:
                        _ann = _forced_type_anns[_name]
                        try:
                            _f.type_annotation = _SA.from_annotation(_ann)
                        except Exception:
                            pass
                except Exception:
                    continue
    except Exception:
        pass
    try:
        st_type.__module__ = DomSt_local.__module__
    except Exception:
        pass
    schema._st_types[type_name] = st_type
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
                    from .core.fields import MutationDescriptor as _MutDesc
                    updated = False
                    for attr_name, desc in list(vars(dom_cls).items()):
                        if not isinstance(desc, _MutDesc):
                            continue
                        meta = getattr(desc, 'meta', {}) or {}
                        target_name = meta.get('target')
                        if not target_name:
                            continue
                        btype_t = schema.types.get(target_name)
                        if not btype_t:
                            continue
                        # Compose descriptor-level scope with domain guard at runtime
                        _runtime_scope = compose_scope_with_guard(dom_cls, meta.get('scope'))
                        triplet = build_merge_resolver_for_type(
                            schema,
                            btype_t,
                            field_name=attr_name,
                            relation_scope=_runtime_scope,
                            pre_callback=meta.get('pre'),
                            post_callback=meta.get('post'),
                            description=(meta.get('comment') if isinstance(meta, dict) else None),
                            payload_is_list=(not bool(meta.get('single'))),
                            return_is_list=False,
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
    """Attach explicitly declared merge mutations at the root Mutation class.

    Uses MutationDescriptor entries on the user-declared Mutation class.
    """
    user_mut = getattr(schema, '_user_mutation_cls', None)
    if user_mut is None:
        return
    try:
        from .core.fields import MutationDescriptor as _MutDesc
        for attr_name, desc in list(vars(user_mut).items()):
            if not isinstance(desc, _MutDesc):
                continue
            meta = getattr(desc, 'meta', {}) or {}
            target_name = meta.get('target')
            if not target_name:
                continue
            btype_cls = schema.types.get(target_name)
            if not btype_cls:
                continue
            is_single = bool(meta.get('single'))
            eff_scope = meta.get('scope')
            triplet = build_merge_resolver_for_type(
                schema,
                btype_cls,
                field_name=attr_name,
                relation_scope=eff_scope,
                pre_callback=meta.get('pre'),
                post_callback=meta.get('post'),
                description=(meta.get('comment') if isinstance(meta, dict) else None),
                payload_is_list=(not is_single),
                return_is_list=False,
            )
            if triplet is None:
                continue
            an, field_obj, st_ret = triplet
            setattr(MPlain, an, field_obj)
            anns_m[an] = st_ret  # type: ignore
    except Exception:
        return
