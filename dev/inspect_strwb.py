import importlib
import inspect

def dump_attr(label, v):
    print(f"--- {label} ---")
    try:
        print("type:", type(v))
        print("type.module:", getattr(type(v), "__module__", ""))
        print("dir(sample):", [a for a in dir(v) if not a.startswith("__")][:20])
        print("has .resolver:", hasattr(v, "resolver"))
        print("has .base_resolver:", hasattr(v, "base_resolver"))
        print("has .field_definition:", hasattr(v, "field_definition"))
        print("is_mutation on v:", getattr(v, "is_mutation", None))
        br = getattr(v, "base_resolver", None)
        print("is_mutation on base_resolver:", getattr(br, "is_mutation", None))
        fd = getattr(v, "field_definition", None)
        print("type(field_definition):", type(fd))
        print("is_mutation on field_definition:", getattr(fd, "is_mutation", None))
        fn = getattr(v, "resolver", None) or getattr(br, "func", None) or getattr(v, "func", None)
        print("callable resolver?", callable(fn))
        if callable(fn):
            print("resolver name:", getattr(fn, "__name__", None))
            print("return ann:", getattr(fn, "__annotations__", {}).get("return"))
    except Exception as e:
        print("dump error:", e)

def main():
    m = importlib.import_module('tests.schema')
    Mut = getattr(m, 'Mutation')
    Dom = getattr(m, 'BlogDomain')
    dump_attr('Mutation.create_post', getattr(Mut, 'create_post', None))
    dump_attr('Mutation.create_post_id', getattr(Mut, 'create_post_id', None))
    dump_attr('BlogDomain.create_post_mut', getattr(Dom, 'create_post_mut', None))

if __name__ == '__main__':
    main()
