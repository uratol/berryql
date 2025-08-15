from __future__ import annotations
from typing import Any, Dict, Optional

def _from_camel(name: str) -> str:
    # Convert lowerCamelCase or UpperCamelCase to snake_case
    out = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0 and name[i-1] != '_':
            out.append('_')
        out.append(ch.lower())
    return ''.join(out)

class RelationSelectionExtractor:
    def __init__(self, registry: Any | None = None):
        self.registry = registry or getattr(self, 'registry', None)
        # Extract a name normalizer function from registry.name_converter if present
        self._name_converter = getattr(getattr(self.registry, '_name_converter', None), 'apply_naming_config', None)
        # Precompute a mapping function from GraphQL -> Python name
        # If a name_converter is present, reverse apply_naming_config by comparing candidates; fallback to camel->snake.
        def _to_python(name: str, fields_map: Dict[str, Any]) -> str:
            # Fast path: direct match
            if name in fields_map:
                return name
            # Try auto camel-case hint
            if getattr(self.registry, '_auto_camel_case', False):
                snake = _from_camel(str(name))
                if snake in fields_map:
                    return snake
            # If we have a converter, try to reverse map by applying converter to known python names
            if callable(self._name_converter):
                for py_name in fields_map.keys():
                    try:
                        if self._name_converter(py_name) == name:
                            return py_name
                    except Exception:
                        continue
            return name
        self._to_python = _to_python

    def _init_rel_cfg(self, fdef: Any) -> Dict[str, Any]:
        single = bool(fdef.meta.get('single') or (fdef.meta.get('mode') == 'single'))
        target = fdef.meta.get('target')
        return {
            'fields': [], 'limit': None, 'offset': None,
            'order_by': None, 'order_dir': None, 'order_multi': [],
            'where': None, 'default_where': fdef.meta.get('where') if fdef.meta.get('where') is not None else None,
            'single': single, 'target': target, 'nested': {}, 'skip_pushdown': False,
            'filter_args': {}, 'arg_specs': fdef.meta.get('arguments') if fdef.meta.get('arguments') is not None else None
        }

    def _children(self, sel: Any) -> list[Any]:
        kids = getattr(sel, 'selections', None) or getattr(sel, 'children', None)
        return list(kids or [])

    def _ast_value(self, node: Any, info: Any) -> Any:
        try:
            if node is None:
                return None
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
        for child in self._children(sel):
            name = getattr(getattr(child, 'name', None), 'value', None) or getattr(child, 'name', None)
            if not name or name.startswith('__'):
                continue
            # Support camelCase field names when schema is configured with auto_camel_case
            fields_map = getattr(btype, '__berry_fields__', {}) or {}
            # Normalize name using config-aware converter
            py_name = self._to_python(str(name), fields_map)
            fdef = fields_map.get(py_name)
            if fdef:
                name = py_name
            if not fdef or fdef.kind != 'relation':
                continue
            rel_cfg = out.setdefault(name, self._init_rel_cfg(fdef))
            try:
                _args = getattr(child, 'arguments', None)
                if isinstance(_args, dict):
                    for arg_name, val in _args.items():
                        if arg_name in ('limit','offset','order_by','order_dir','order_multi','where'):
                            rel_cfg[arg_name] = val
                        else:
                            rel_cfg['filter_args'][arg_name] = val
                else:
                    for arg in (_args or []):
                        arg_name = getattr(getattr(arg, 'name', None), 'value', None) or getattr(arg, 'name', None)
                        raw_val = getattr(arg, 'value', None)
                        val = self._ast_value(raw_val, getattr(sel, 'info', None))
                        if arg_name in ('limit','offset','order_by','order_dir','order_multi','where'):
                            rel_cfg[arg_name] = val
                        else:
                            rel_cfg['filter_args'][arg_name] = val
            except Exception:
                pass
            sub_children = self._children(child)
            if sub_children:
                tgt_b = self.registry.types.get(fdef.meta.get('target')) if fdef.meta.get('target') else None
                for sub in sub_children:
                    sub_name = getattr(getattr(sub, 'name', None), 'value', None) or getattr(sub, 'name', None)
                    if not sub_name or sub_name.startswith('__'):
                        continue
                    # target sub-field camelCase support
                    sub_fields = getattr(tgt_b, '__berry_fields__', {}) if tgt_b else {}
                    # Normalize nested field name
                    py_sub = self._to_python(str(sub_name), sub_fields) if tgt_b else sub_name
                    sub_def = sub_fields.get(py_sub) if tgt_b else None
                    if sub_def:
                        sub_name = py_sub
                    if not sub_def or sub_def.kind == 'scalar':
                        if sub_name not in rel_cfg['fields']:
                            rel_cfg['fields'].append(sub_name)
                    elif sub_def and sub_def.kind == 'relation':
                        ncfg = rel_cfg['nested'].setdefault(sub_name, self._init_rel_cfg(sub_def))
                        try:
                            _nargs = getattr(sub, 'arguments', None)
                            if isinstance(_nargs, dict):
                                for narg_name, nval in _nargs.items():
                                    if narg_name in ('limit','offset','order_by','order_dir','order_multi','where'):
                                        ncfg[narg_name] = nval
                                    else:
                                        ncfg['filter_args'][narg_name] = nval
                            else:
                                for narg in (_nargs or []):
                                    narg_name = getattr(getattr(narg, 'name', None), 'value', None) or getattr(narg, 'name', None)
                                    nraw = getattr(narg, 'value', None)
                                    nval = self._ast_value(nraw, getattr(sel, 'info', None))
                                    if narg_name in ('limit','offset','order_by','order_dir','order_multi','where'):
                                        ncfg[narg_name] = nval
                                    else:
                                        ncfg['filter_args'][narg_name] = nval
                        except Exception:
                            pass
                        try:
                            sub2_children = self._children(sub)
                            if sub2_children:
                                tgt_b2 = self.registry.types.get(sub_def.meta.get('target')) if sub_def.meta.get('target') else None
                                for sub2 in sub2_children:
                                    nname2 = getattr(getattr(sub2, 'name', None), 'value', None) or getattr(sub2, 'name', None)
                                    if nname2 and not nname2.startswith('__'):
                                        sfields2 = getattr(tgt_b2, '__berry_fields__', {}) if tgt_b2 else {}
                                        py2 = self._to_python(str(nname2), sfields2) if tgt_b2 else nname2
                                        sdef2 = sfields2.get(py2) if tgt_b2 else None
                                        if sdef2:
                                            nname2 = py2
                                        if not sdef2 or sdef2.kind == 'scalar':
                                            if nname2 not in ncfg['fields']:
                                                ncfg['fields'].append(nname2)
                        except Exception:
                            pass
            try:
                tgt = self.registry.types.get(fdef.meta.get('target')) if fdef.meta.get('target') else None
                self._walk_selected(child, tgt, out)
            except Exception:
                pass

    def extract(self, info: Any, root_field_name: str, btype: Any) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        try:
            fields = getattr(info, 'selected_fields', None)
            for f in (fields or []):
                if getattr(f, 'name', None) == root_field_name:
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
            if not isinstance(cfg_dst, dict):
                return
            for k in ('limit','offset','order_by','order_dir','order_multi','where'):
                if k in args_dict and args_dict[k] is not None:
                    cfg_dst[k] = args_dict[k]
        try:
            raw_info = getattr(info, '_raw_info', None)
            field_nodes = getattr(raw_info, 'field_nodes', None) if raw_info is not None else None
        except Exception:
            field_nodes = None
        if field_nodes:
            try:
                root_nodes = [n for n in field_nodes if _name_ast(n) == root_field_name]
                if root_nodes:
                    root_node = root_nodes[0]
                    for rel_node in _children_ast(root_node):
                        rname = _name_ast(rel_node)
                        if not rname or rname.startswith('__'):
                            continue
                        # root relation camelCase support
                        fields_map = getattr(btype, '__berry_fields__', {}) or {}
                        py_r = self._to_python(str(rname), fields_map)
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
                                if an == 'order_multi' and av is not None and not isinstance(av, list):
                                    av = [av]
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
                                sub_fields = getattr(tgt_b, '__berry_fields__', {}) if tgt_b else {}
                                py_sub = self._to_python(str(sub_name), sub_fields) if tgt_b else sub_name
                                sdef = sub_fields.get(py_sub) if tgt_b else None
                                if sdef:
                                    sub_name = py_sub
                                if not sdef or sdef.kind == 'scalar':
                                    if sub_name not in rel_cfg['fields']:
                                        rel_cfg['fields'].append(sub_name)
                        except Exception:
                            pass
                        tgt_b = self.registry.types.get(fdef.meta.get('target')) if fdef.meta.get('target') else None
                        for sub in _children_ast(rel_node):
                            sub_name = _name_ast(sub)
                            if not sub_name or sub_name.startswith('__'):
                                continue
                            sub_fields = getattr(tgt_b, '__berry_fields__', {}) if tgt_b else {}
                            py_sub = self._to_python(str(sub_name), sub_fields) if tgt_b else sub_name
                            sub_def = sub_fields.get(py_sub) if tgt_b else None
                            if sub_def:
                                sub_name = py_sub
                            if not sub_def or sub_def.kind != 'relation':
                                continue
                            n_args = {}
                            try:
                                for na in getattr(sub, 'arguments', []) or []:
                                    an = _name_ast(na)
                                    av = self._ast_value(getattr(na, 'value', None), info)
                                    if an == 'order_multi' and av is not None and not isinstance(av, list):
                                        av = [av]
                                    n_args[an] = av
                            except Exception:
                                pass
                            ncfg = rel_cfg['nested'].setdefault(sub_name, self._init_rel_cfg(sub_def))
                            _merge_rel_args(ncfg, n_args)
            except Exception:
                pass
        return out

class RootSelectionExtractor:
    def __init__(self, registry: Any | None = None):
        # Keep reference to registry for naming conversion (camelCase <-> snake_case)
        self.registry = registry or getattr(self, 'registry', None)

    def _to_camel(self, name: str) -> str:
        # Convert snake_case to lowerCamelCase
        if not name:
            return name
        parts = name.split('_')
        if not parts:
            return name
        head = parts[0]
        tail = ''.join(p.capitalize() for p in parts[1:])
        return head + tail
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
            try:
                fdef = getattr(btype, '__berry_fields__', {}).get(name)
            except Exception:
                fdef = None
            if not fdef:
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
        out = {'scalars': set(), 'relations': set(), 'custom': set(), 'custom_object': set(), 'aggregate': set()}
        if info is None:
            return out
        # Build candidate root field names to handle auto camelCase
        candidates = {root_field_name}
        try:
            # Prefer explicit name converter if available
            name_conv = getattr(getattr(self.registry, '_name_converter', None), 'apply_naming_config', None)
            if callable(name_conv):
                try:
                    candidates.add(name_conv(root_field_name))
                except Exception:
                    pass
            # Fallback to naive snake->camel when auto_camel_case is enabled
            if getattr(self.registry, '_auto_camel_case', False):
                candidates.add(self._to_camel(root_field_name))
        except Exception:
            # best-effort; keep original only
            pass
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
                                fdef = getattr(btype, '__berry_fields__', {}).get(name)
                            except Exception:
                                fdef = None
                            if not fdef:
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
