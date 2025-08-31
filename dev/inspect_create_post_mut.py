import ast
import sys
import types
from pathlib import Path

# Resolve path to tests/schema.py
ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "tests" / "schema.py"

src = SCHEMA_PATH.read_text(encoding="utf-8")
# Parse and drop the terminal schema = berry_schema.to_strawberry() assignment to avoid building
mod = ast.parse(src, filename=str(SCHEMA_PATH))
new_body = []
for node in mod.body:
    drop = False
    try:
        # Remove lines that call to_strawberry at module level (e.g., schema = berry_schema.to_strawberry())
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Attribute) and func.attr == "to_strawberry":
                drop = True
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Attribute) and func.attr == "to_strawberry":
                drop = True
    except Exception:
        drop = False
    if not drop:
        new_body.append(node)
mod.body = new_body
ast.fix_missing_locations(mod)

code = compile(mod, str(SCHEMA_PATH), "exec")

# Prepare a module namespace with name 'tests.schema' so strawberry.lazy('tests.schema') points here
mname = "tests.schema"
module = types.ModuleType(mname)
module.__file__ = str(SCHEMA_PATH)
# Preload berryql to apply our patch on strawberry.mutation
import berryql as _berryql  # noqa: F401

# Ensure parent package exists in sys.modules
pkg_name = "tests"
if pkg_name not in sys.modules:
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(ROOT / "tests")]  # type: ignore[attr-defined]
    sys.modules[pkg_name] = pkg

sys.modules[mname] = module

# Execute the modified schema module
exec(code, module.__dict__)

# Fetch BlogDomain and its strawberry mutation field
BlogDomain = getattr(module, "BlogDomain", None)
berry_schema = getattr(module, "berry_schema", None)
print("BlogDomain:", BlogDomain)
print("berry_schema:", type(berry_schema))

create_post_mut = getattr(BlogDomain, "create_post_mut", None)
print("create_post_mut type:", type(create_post_mut))
print("attr __berry_is_mutation__:", getattr(create_post_mut, "__berry_is_mutation__", None))
print("attr is_mutation:", getattr(create_post_mut, "is_mutation", None))

resolver = getattr(create_post_mut, "resolver", None)
base_resolver = getattr(create_post_mut, "base_resolver", None)
fd = getattr(create_post_mut, "field_definition", None)
print("resolver:", resolver)
print("resolver.__berry_is_mutation__:", getattr(resolver, "__berry_is_mutation__", None))
print("base_resolver:", base_resolver)
print("base_resolver.__berry_is_mutation__:", getattr(base_resolver, "__berry_is_mutation__", None))
print("field_definition.is_mutation:", getattr(fd, "is_mutation", None))
print("field_definition.__berry_is_mutation__:", getattr(fd, "__berry_is_mutation__", None))
try:
    ta_attr = getattr(create_post_mut, "type_annotation", None)
    print("attr.type_annotation:", ta_attr)
    if ta_attr is not None:
        try:
            print("attr.type_annotation.annotation:", getattr(ta_attr, "annotation", None))
        except Exception:
            pass
        try:
            resolved = ta_attr.resolve() if hasattr(ta_attr, "resolve") else None
        except Exception as e_res:
            resolved = f"ERROR: {e_res!r}"
        print("attr.type_annotation.resolve():", resolved)
except Exception:
    pass

# Try building mutation domain type and inspect the field type annotation
try:
    from berryql.mutations import ensure_mutation_domain_type
    DomMut = ensure_mutation_domain_type(berry_schema, BlogDomain)
    print("DomMut:", DomMut)
    f = getattr(DomMut, "create_post_mut", None)
    print("DomMut.create_post_mut (attr):", f)
    try:
        sd = getattr(DomMut, "__strawberry_definition__", None)
        fields = list(getattr(sd, "fields", []) or [])
        print("__strawberry_definition__ present:", bool(sd))
        try:
            print("definition attrs:", sorted([k for k in dir(sd) if not k.startswith('_')]))
        except Exception:
            pass
        for fld in fields:
            try:
                if getattr(fld, "name", None) == "create_post_mut":
                    print("- field:", fld)
                    ta = getattr(fld, "type_annotation", None)
                    print("  type_annotation:", ta)
                    try:
                        ann = getattr(ta, "annotation", None)
                    except Exception:
                        ann = None
                    print("  type_annotation.annotation:", ann)
                    # Try resolving the type annotation
                    try:
                        resolved = ta.resolve() if hasattr(ta, "resolve") else None
                    except Exception as e_res:
                        resolved = f"ERROR: {e_res!r}"
                    print("  resolved:", resolved)
            except Exception:
                continue
        try:
            gf = sd.get_field("create_post_mut") if hasattr(sd, "get_field") else None
            print("get_field('create_post_mut'):", gf)
            if gf is not None:
                gta = getattr(gf, "type_annotation", None)
                print("  gf.type_annotation:", gta)
                try:
                    print("  gf.type_annotation.annotation:", getattr(gta, "annotation", None))
                except Exception:
                    pass
        except Exception as e2:
            print("get_field inspection failed:", repr(e2))
    except Exception as e:
        print("failed to inspect __strawberry_definition__:", repr(e))
    # Print mapping presence for PostQL
    try:
        st_types = getattr(berry_schema, "_st_types", {})
        print("_st_types has PostQL:", "PostQL" in (st_types or {}))
        if "PostQL" in (st_types or {}):
            print("  PostQL runtime type:", st_types.get("PostQL"))
    except Exception:
        pass
except Exception as e:
    print("ERROR while ensure_mutation_domain_type:", repr(e))
