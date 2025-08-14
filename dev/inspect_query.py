import importlib
import sys
from pathlib import Path


def main():
    # Ensure repository root is on sys.path so 'tests' can be imported
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    # Import schema from tests
    try:
        ts = importlib.import_module('tests.schema')
    except Exception as e:
        print('Failed to import tests.schema:', e)
        sys.exit(1)
    s = getattr(ts, 'schema', None)
    if s is None:
        print('tests.schema.schema not found')
        sys.exit(2)
    # Get Strawberry internal schema converter
    # Try strawberry schema converter path
    sc = getattr(s, 'schema_converter', None) or getattr(getattr(s, '_schema', None), 'schema_converter', None)
    qt = None
    if sc is not None:
        qt = sc.type_map.get('Query')
        if qt is not None:
            fields = getattr(qt, 'fields', [])
            print('Strawberry Query fields:', [getattr(f, 'name', '?') for f in fields])
            out = []
            for f in fields:
                br = getattr(f, 'base_resolver', None)
                fn = getattr(br, '_func', None) or getattr(br, 'func', None) or br
                out.append((f.name, bool(br), getattr(fn, '__qualname__', None), getattr(fn, '__module__', None)))
            print('Strawberry resolver info:', out)
    # Also try graphql-core schema directly
    gqs = getattr(s, '_schema', None) or getattr(s, 'as_graphql_schema', lambda: None)()
    if gqs is None and hasattr(s, 'get_graphql_schema'):
        gqs = s.get_graphql_schema()
    if gqs is None and hasattr(s, 'graphql_schema'):
        gqs = s.graphql_schema
    if gqs is not None:
        qt2 = getattr(gqs, 'query_type', None)
        if qt2 is not None:
            fdict = getattr(qt2, 'fields', {}) or {}
            print('GraphQL-core Query fields:', list(fdict.keys()))
            try:
                infos = []
                for name, field in fdict.items():
                    resolver = getattr(field, 'resolve', None)
                    rname = getattr(getattr(resolver, '__func__', resolver), '__qualname__', None)
                    rmod = getattr(getattr(resolver, '__func__', resolver), '__module__', None)
                    infos.append((name, bool(resolver), rname, rmod))
                print('GraphQL-core resolvers:', infos)
            except Exception as e:
                print('Failed to inspect core resolvers:', e)
        else:
            print('GraphQL-core query_type is None')


if __name__ == '__main__':
    main()
