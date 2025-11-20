from __future__ import annotations
from typing import Any, Dict, Optional

from .naming import (
    build_name_candidates,
    ensure_list,
    fields_map_for,
    map_graphql_to_python,
)

_REL_ARGS = {'limit', 'offset', 'order_by', 'order_dir', 'order_multi', 'where'}
_ORDER_FLAGS = {
    'order_by': '_has_explicit_order_by',
    'order_dir': '_has_explicit_order_dir',
    'order_multi': '_has_explicit_order_multi',
}


def _assign_rel_arg(cfg: Dict[str, Any], arg_name: Optional[str], value: Any) -> None:
    if not arg_name:
        return
    if arg_name == 'order_multi':
        value = ensure_list(value)
    if arg_name in _REL_ARGS:
        cfg[arg_name] = value
        flag = _ORDER_FLAGS.get(arg_name)
        if flag:
            cfg[flag] = True
    else:
        cfg['filter_args'][arg_name] = value

class RelationSelectionExtractor:
    def __init__(self, registry: Any | None = None):
        self.registry = registry or getattr(self, 'registry', None)
        self._name_converter = getattr(getattr(self.registry, '_name_converter', None), 'apply_naming_config', None)
        self._auto_camel = bool(getattr(self.registry, '_auto_camel_case', False))

    def _map_name(self, name: str, fields_map: Dict[str, Any]) -> str:
        return map_graphql_to_python(
            str(name),
            fields_map,
            auto_camel=self._auto_camel,
            name_converter=self._name_converter,
        )

    def _init_rel_cfg(self, fdef: Any) -> Dict[str, Any]:
        single = bool(fdef.meta.get('single') or (fdef.meta.get('mode') == 'single'))
        target = fdef.meta.get('target')
        fk_col_name = fdef.meta.get('fk_column_name')
        # Default ordering values come from relation field meta; query args will override later in extract
        def_ob = fdef.meta.get('order_by') if fdef.meta.get('order_by') is not None else None
        def_od = fdef.meta.get('order_dir') if fdef.meta.get('order_dir') is not None else None
        def_om = fdef.meta.get('order_multi') if fdef.meta.get('order_multi') is not None else []
        # Type-level scope (on target BerryType) is combined with relation-level scope
        type_default_where = None
        try:
            if target and getattr(self, 'registry', None) is not None:
                tb = getattr(self.registry.types, 'get', lambda *_: None)(target)
                if tb is None:
                    tb = (self.registry.types or {}).get(target)
                if tb is not None:
                    # Support either explicit __type_scope__ or plain 'scope' on the type class
                    t_scope = getattr(tb, '__type_scope__', None)
                    if t_scope is None:
                        t_scope = getattr(tb, 'scope', None)
                    # Normalize: unwrap single-element list/tuple; drop empty containers
                    if isinstance(t_scope, (list, tuple)):
                        if len(t_scope) == 0:
                            type_default_where = None
                        elif len(t_scope) == 1:
                            type_default_where = t_scope[0]
                        else:
                            type_default_where = list(t_scope)
                    else:
                        type_default_where = t_scope
                    # Avoid passing empty dict as a filter (invalid for SA .where)
                    if isinstance(type_default_where, dict) and not type_default_where:
                        type_default_where = None
        except Exception:
            type_default_where = None
        return {
            'fields': [], 'limit': None, 'offset': None,
            'order_by': def_ob, 'order_dir': def_od, 'order_multi': list(def_om) if isinstance(def_om, (list, tuple)) else ([def_om] if def_om else []),
            'where': None,
            'default_where': fdef.meta.get('scope') if fdef.meta.get('scope') is not None else None,
            'type_default_where': type_default_where,
            'single': single, 'target': target, 'nested': {}, 'skip_pushdown': False,
            'fk_column_name': fk_col_name,
            'filter_args': {}, 'arg_specs': fdef.meta.get('arguments') if fdef.meta.get('arguments') is not None else None,
            # Flags to distinguish explicit args from defaults (used for precedence rules)
            '_has_explicit_order_by': False,
            '_has_explicit_order_dir': False,
            '_has_explicit_order_multi': False,
        }

    def _children(self, sel: Any) -> list[Any]:
        kids = getattr(sel, 'selections', None) or getattr(sel, 'children', None)
        return list(kids or [])

    def _iter_arguments(self, raw_args: Any):
        info = getattr(self, '_info', None)
        if raw_args is None:
            return
        if isinstance(raw_args, dict):
            for arg_name, raw_val in raw_args.items():
                yield arg_name, self._ast_value(raw_val, info)
            return
        for arg in (raw_args or []):
            arg_name = getattr(getattr(arg, 'name', None), 'value', None) or getattr(arg, 'name', None)
            raw_val = getattr(arg, 'value', None)
            yield arg_name, self._ast_value(raw_val, info)

    def _ast_value(self, node: Any, info: Any) -> Any:
        try:
            if node is None:
                return None
            # Resolve GraphQL variable references to concrete values using info.variable_values
            try:
                kind = getattr(node, 'kind', None)
                is_var = bool(kind and 'Variable' in str(kind))
            except Exception:
                is_var = False
            if is_var:
                try:
                    name_node = getattr(node, 'name', None)
                    var_name = getattr(name_node, 'value', None) if name_node is not None else None
                except Exception:
                    var_name = None
                if var_name:
                    var_vals = None
                    try:
                        var_vals = getattr(info, 'variable_values', None)
                    except Exception:
                        var_vals = None
                    if not isinstance(var_vals, dict):
                        try:
                            raw = getattr(info, '_raw_info', None)
                            var_vals = getattr(raw, 'variable_values', None) if raw is not None else None
                        except Exception:
                            var_vals = None
                    if isinstance(var_vals, dict) and var_name in var_vals:
                        return var_vals[var_name]
            vals = getattr(node, 'values', None)
            if vals is not None:
                return [self._ast_value(v, info) for v in vals]
            fields = getattr(node, 'fields', None)
            if fields is not None:
                out: dict[str, Any] = {}
                for f in fields:
                    k = getattr(getattr(f, 'name', None), 'value', None) or getattr(f, 'name', None)
                    out[k] = self._ast_value(getattr(f, 'value', None), info)
                return out
            if hasattr(node, 'value'):
                return getattr(node, 'value')
        except Exception:
            return node
        return node

    def _walk_selected(self, sel: Any, btype: Any, out: Dict[str, Dict[str, Any]]):
        # Helper: recursively populate nested config for a given relation node
        def _collect_nested(sub_node: Any, parent_btype: Any, cfg: Dict[str, Any]):
            if not sub_node:
                return
            sub_children = self._children(sub_node)
            if not sub_children:
                return
            # Infer the btype from cfg when available; fallback to scanning by names at runtime
            # We accept that fields list is filled as we walk
            for sub in sub_children:
                sub_name = getattr(getattr(sub, 'name', None), 'value', None) or getattr(sub, 'name', None)
                if not sub_name or str(sub_name).startswith('__'):
                    continue
                try:
                    target_name = cfg.get('target') or None
                    tgt_b2 = self.registry.types.get(target_name) if target_name else None
                except Exception:
                    tgt_b2 = None
                sub_fields_map = fields_map_for(tgt_b2) if tgt_b2 else {}
                py_sub = self._map_name(str(sub_name), sub_fields_map) if sub_fields_map else sub_name
                sdef = sub_fields_map.get(py_sub) if sub_fields_map else None
                if sdef:
                    sub_name = py_sub
                if not sdef or sdef.kind == 'scalar':
                    if sub_name not in cfg['fields']:
                        cfg['fields'].append(sub_name)
                else:
                    # nested relation inside cfg
                    ncfg = cfg['nested'].setdefault(sub_name, self._init_rel_cfg(sdef))
                    # record its target for downstream builders
                    ncfg['target'] = sdef.meta.get('target')
                    # parse arguments for this nested sub-relation
                    try:
                        for narg_name, nval in self._iter_arguments(getattr(sub, 'arguments', None)):
                            _assign_rel_arg(ncfg, narg_name, nval)
                    except Exception:
                        pass
                    # recurse deeper
                    _collect_nested(sub, ncfg.get('target') and self.registry.types.get(ncfg.get('target')), ncfg)
        for child in self._children(sel):
            name = getattr(getattr(child, 'name', None), 'value', None) or getattr(child, 'name', None)
            if not name or name.startswith('__'):
                continue
            # Support camelCase field names when schema is configured with auto_camel_case
            fields_map = fields_map_for(btype)
            # Normalize name using config-aware converter
            py_name = self._map_name(str(name), fields_map)
            fdef = fields_map.get(py_name)
            if fdef:
                name = py_name
            if not fdef or fdef.kind != 'relation':
                continue
            rel_cfg = out.setdefault(name, self._init_rel_cfg(fdef))
            try:
                for arg_name, val in self._iter_arguments(getattr(child, 'arguments', None)):
                    _assign_rel_arg(rel_cfg, arg_name, val)
            except Exception:
                pass
            # Recursively collect nested under this relation
            try:
                _collect_nested(child, self.registry.types.get(fdef.meta.get('target')) if fdef.meta.get('target') else None, rel_cfg)
            except Exception:
                pass

    def extract(self, info: Any, root_field_name: str, btype: Any) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        # Stash info for use during selected_fields traversal
        try:
            setattr(self, '_info', info)
        except Exception:
            pass
        # Build candidate root field names to handle auto camelCase/name converters
        candidates = build_name_candidates(
            root_field_name,
            auto_camel=self._auto_camel,
            name_converter=self._name_converter,
        )
        try:
            fields = getattr(info, 'selected_fields', None)
            for f in (fields or []):
                if getattr(f, 'name', None) in candidates:
                    fake = type('Sel2', (), {})()
                    setattr(fake, 'selections', getattr(f, 'selections', []) or getattr(f, 'children', []))
                    self._walk_selected(fake, btype, out)
                    break
        except Exception:
            pass
        def _children_ast(node: Any) -> list[Any]:
            try:
                selset = getattr(node, 'selection_set', None)
                if selset is not None and getattr(selset, 'selections', None) is not None:
                    return list(selset.selections) or []
            except Exception:
                pass
            return []
        def _name_ast(node: Any) -> Optional[str]:
            try:
                n = getattr(node, 'name', None)
                if hasattr(n, 'value'):
                    return getattr(n, 'value', None)
                return n
            except Exception:
                return None
        def _merge_rel_args(cfg_dst: Dict[str, Any], args_dict: Dict[str, Any]):
            if not isinstance(cfg_dst, dict) or not isinstance(args_dict, dict):
                return
            for k, v in args_dict.items():
                _assign_rel_arg(cfg_dst, k, v)
        try:
            raw_info = getattr(info, '_raw_info', None)
            field_nodes = getattr(raw_info, 'field_nodes', None) if raw_info is not None else None
        except Exception:
            field_nodes = None
        if field_nodes:
            try:
                root_nodes = [n for n in field_nodes if _name_ast(n) in candidates]
                if root_nodes:
                    root_node = root_nodes[0]
                    for rel_node in _children_ast(root_node):
                        rname = _name_ast(rel_node)
                        if not rname or rname.startswith('__'):
                            continue
                        # root relation camelCase support
                        fields_map = fields_map_for(btype)
                        py_r = self._map_name(str(rname), fields_map)
                        fdef = fields_map.get(py_r)
                        if fdef:
                            rname = py_r
                        if not fdef or fdef.kind != 'relation':
                            continue
                        ast_args = {}
                        try:
                            for a in getattr(rel_node, 'arguments', []) or []:
                                an = _name_ast(a)
                                av = self._ast_value(getattr(a, 'value', None), info)
                                ast_args[an] = av
                        except Exception:
                            pass
                        rel_cfg = out.setdefault(rname, self._init_rel_cfg(fdef))
                        _merge_rel_args(rel_cfg, ast_args)
                        try:
                            tgt_b = self.registry.types.get(fdef.meta.get('target')) if fdef.meta.get('target') else None
                        except Exception:
                            tgt_b = None
                        try:
                            for sub in _children_ast(rel_node):
                                sub_name = _name_ast(sub)
                                if not sub_name or sub_name.startswith('__'):
                                    continue
                                sub_fields = fields_map_for(tgt_b) if tgt_b else {}
                                py_sub = self._map_name(str(sub_name), sub_fields) if tgt_b else sub_name
                                sdef = sub_fields.get(py_sub) if tgt_b else None
                                if sdef:
                                    sub_name = py_sub
                                if not sdef or sdef.kind == 'scalar':
                                    if sub_name not in rel_cfg['fields']:
                                        rel_cfg['fields'].append(sub_name)
                        except Exception:
                            pass
                        tgt_b = self.registry.types.get(fdef.meta.get('target')) if fdef.meta.get('target') else None
                        # Recursive AST nested collection
                        def _collect_nested_ast(parent_node: Any, parent_b: Any, parent_cfg: Dict[str, Any]):
                            for sub in _children_ast(parent_node):
                                sub_name = _name_ast(sub)
                                if not sub_name or sub_name.startswith('__'):
                                    continue
                                sub_fields = fields_map_for(parent_b) if parent_b else {}
                                py_sub = self._map_name(str(sub_name), sub_fields) if parent_b else sub_name
                                sub_def = sub_fields.get(py_sub) if parent_b else None
                                if sub_def:
                                    sub_name = py_sub
                                if not sub_def or sub_def.kind != 'relation':
                                    # treat as scalar
                                    if sub_name not in parent_cfg['fields']:
                                        parent_cfg['fields'].append(sub_name)
                                    continue
                                n_args = {}
                                try:
                                    for na in getattr(sub, 'arguments', []) or []:
                                        an = _name_ast(na)
                                        av = self._ast_value(getattr(na, 'value', None), info)
                                        n_args[an] = av
                                except Exception:
                                    pass
                                ncfg = parent_cfg['nested'].setdefault(sub_name, self._init_rel_cfg(sub_def))
                                _merge_rel_args(ncfg, n_args)
                                # ensure target stored
                                ncfg['target'] = sub_def.meta.get('target')
                                # recurse deeper
                                nxt_b = self.registry.types.get(sub_def.meta.get('target')) if sub_def.meta.get('target') else None
                                _collect_nested_ast(sub, nxt_b, ncfg)
                        _collect_nested_ast(rel_node, tgt_b, rel_cfg)
            except Exception:
                pass
        return out

class RootSelectionExtractor:
    def __init__(self, registry: Any | None = None):
        # Keep reference to registry for naming conversion (camelCase <-> snake_case)
        self.registry = registry or getattr(self, 'registry', None)
        self._name_converter = getattr(getattr(self.registry, '_name_converter', None), 'apply_naming_config', None)
        self._auto_camel = bool(getattr(self.registry, '_auto_camel_case', False))

    def _map_name(self, name: str, fields_map: Dict[str, Any]) -> str:
        return map_graphql_to_python(
            str(name),
            fields_map,
            auto_camel=self._auto_camel,
            name_converter=self._name_converter,
        )
    def _children(self, node: Any) -> list[Any]:
        selset = getattr(node, 'selection_set', None)
        if selset is not None and getattr(selset, 'selections', None) is not None:
            return list(selset.selections) or []
        sels = getattr(node, 'selections', None)
        if sels is not None:
            return list(sels) or []
        return []

    def _name_of(self, node: Any) -> Optional[str]:
        n = getattr(node, 'name', None)
        if hasattr(n, 'value'):
            return getattr(n, 'value', None)
        return n

    def _walk(self, sel_node: Any, out: dict[str, set[str]], btype: Any, fragments: dict | None):
        for child in self._children(sel_node):
            kind = getattr(child, 'kind', None)
            if kind and 'InlineFragment' in str(kind):
                self._walk(child, out, btype, fragments)
                continue
            if kind and 'FragmentSpread' in str(kind):
                frag_name = self._name_of(child)
                if frag_name and fragments and frag_name in fragments:
                    frag_def = fragments[frag_name]
                    self._walk(frag_def, out, btype, fragments)
                continue
            name = self._name_of(child)
            if not name or name.startswith('__'):
                continue
            # Map GraphQL field to python name when auto-camel-case is used
            try:
                fields_map = fields_map_for(btype)
                py_name = self._map_name(str(name), fields_map)
                fdef = fields_map.get(py_name)
                if fdef:
                    name = py_name
            except Exception:
                fdef = None
            if not fdef:
                # Unknown to Berry fields -> consider as non-Berry (Strawberry) field
                out['other'].add(str(name))
                continue
            k = fdef.kind
            if k == 'scalar':
                out['scalars'].add(name)
            elif k == 'relation':
                out['relations'].add(name)
            elif k == 'custom':
                out['custom'].add(name)
            elif k == 'custom_object':
                out['custom_object'].add(name)
            elif k == 'aggregate':
                out['aggregate'].add(name)

    def extract(self, info: Any, root_field_name: str, btype: Any) -> dict[str, set[str]]:
        out = {'scalars': set(), 'relations': set(), 'custom': set(), 'custom_object': set(), 'aggregate': set(), 'other': set()}
        if info is None:
            return out
        # Build candidate root field names to handle auto camelCase
        candidates = build_name_candidates(
            root_field_name,
            auto_camel=self._auto_camel,
            name_converter=self._name_converter,
        )
        try:
            fields = getattr(info, 'selected_fields', None)
            frags = None
            try:
                frags = getattr(info, 'fragments', None) or {}
            except Exception:
                frags = None
            for f in (fields or []):
                if getattr(f, 'name', None) in candidates:
                    fake = type('Sel', (), {})()
                    setattr(fake, 'selections', getattr(f, 'selections', []) or getattr(f, 'children', []))
                    self._walk(fake, out, btype, frags)
                    break
        except Exception:
            pass
        if not any(out.values()):
            try:
                def _children_ast(node: Any) -> list[Any]:
                    try:
                        selset = getattr(node, 'selection_set', None)
                        if selset is not None and getattr(selset, 'selections', None) is not None:
                            return list(selset.selections) or []
                    except Exception:
                        pass
                    return []
                def _name_ast(node: Any) -> Optional[str]:
                    try:
                        n = getattr(node, 'name', None)
                        if hasattr(n, 'value'):
                            return getattr(n, 'value', None)
                        return n
                    except Exception:
                        return None
                raw_info = getattr(info, '_raw_info', None)
                field_nodes = getattr(raw_info, 'field_nodes', None) if raw_info is not None else None
                if field_nodes:
                    root_nodes = [n for n in field_nodes if _name_ast(n) in candidates]
                    if root_nodes:
                        root_node = root_nodes[0]
                        for child in _children_ast(root_node):
                            name = _name_ast(child)
                            if not name or name.startswith('__'):
                                continue
                            try:
                                fields_map = fields_map_for(btype)
                                py_name = self._map_name(str(name), fields_map)
                                fdef = fields_map.get(py_name)
                                if fdef:
                                    name = py_name
                            except Exception:
                                fdef = None
                            if not fdef:
                                out['other'].add(str(name))
                                continue
                            k = fdef.kind
                            if k == 'scalar':
                                out['scalars'].add(name)
                            elif k == 'relation':
                                out['relations'].add(name)
                            elif k == 'custom':
                                out['custom'].add(name)
                            elif k == 'custom_object':
                                out['custom_object'].add(name)
                            elif k == 'aggregate':
                                out['aggregate'].add(name)
            except Exception:
                pass
        return out
