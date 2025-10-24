from tests.schema import schema as berry_strawberry_schema

# Helper to get argument description from schema

def _get_arg_desc(type_name: str, field_name: str, arg_name: str) -> str | None:
    # Prefer underlying GraphQL-core schema to access fields/args reliably
    core_schema = getattr(berry_strawberry_schema, '_schema', None)
    if core_schema is not None and hasattr(core_schema, 'get_type'):
        t = core_schema.get_type(type_name)
    else:
        t = berry_strawberry_schema.get_type_by_name(type_name)
    assert t is not None, f"Type {type_name} not found in schema"
    # GraphQL-core exposes .fields as a mapping (may be a special mapping, not a dict)
    flds = getattr(t, 'fields', None)
    if callable(flds):  # legacy/lazy fallback
        flds = flds()
    field = None
    if flds is not None:
        getter = getattr(flds, 'get', None)
        if callable(getter):
            field = getter(field_name, None)
        if field is None:
            try:
                field = flds[field_name]
            except Exception:
                field = None
    assert field is not None, f"Field {type_name}.{field_name} not found"
    args = getattr(field, 'args', {}) or {}
    arg = args.get(arg_name)
    assert arg is not None, f"Arg {arg_name} not found on {type_name}.{field_name}"
    return getattr(arg, 'description', None)


def test_root_posts_default_args_have_descriptions():
    # Root Query.posts list field
    ob = _get_arg_desc('Query', 'posts', 'order_by')
    od = _get_arg_desc('Query', 'posts', 'order_dir')
    om = _get_arg_desc('Query', 'posts', 'order_multi')
    wh = _get_arg_desc('Query', 'posts', 'where')
    assert ob is not None and 'order' in ob.lower() and 'example' in ob.lower()
    assert od is not None and 'asc' in od.lower() and 'desc' in od.lower()
    assert om is not None and ("col:dir" in om.lower() or 'id:asc' in om.lower())
    assert wh is not None and 'json' in wh.lower() and ('example' in wh.lower() or 'examples' in wh.lower())


def test_user_posts_default_args_have_descriptions():
    # Nested relation UserQL.posts list field
    ob = _get_arg_desc('UserQL', 'posts', 'order_by')
    od = _get_arg_desc('UserQL', 'posts', 'order_dir')
    om = _get_arg_desc('UserQL', 'posts', 'order_multi')
    wh = _get_arg_desc('UserQL', 'posts', 'where')
    assert ob is not None and 'order' in ob.lower() and 'example' in ob.lower()
    assert od is not None and 'asc' in od.lower() and 'desc' in od.lower()
    assert om is not None and ("col:dir" in om.lower() or 'id:asc' in om.lower())
    assert wh is not None and 'json' in wh.lower() and ('example' in wh.lower() or 'examples' in wh.lower())
