import strawberry

class Demo:
    @strawberry.mutation
    async def m(self) -> int:
        return 1

    @strawberry.field
    def f(self) -> int:
        return 1

def dump(label, v):
    print(f"-- {label} --")
    print("type:", type(v), "module:", type(v).__module__)
    for attr in [
        'resolver', 'base_resolver', 'is_mutation', 'field_definition',
        'function', 'func', 'wrapped_func', 'implementation',
        'type', 'type_annotation', 'python_name', 'graphql_name',
    ]:
        val = getattr(v, attr, None)
        print(attr, type(val), bool(val))
    br = getattr(v, 'base_resolver', None)
    if br is not None:
        for attr in ['func', 'wrapped_func', 'resolver', 'is_mutation']:
            val = getattr(br, attr, None)
            print(' base.', attr, type(val), bool(val))

def main():
    dump('Demo.m', getattr(Demo, 'm'))
    dump('Demo.f', getattr(Demo, 'f'))

if __name__ == '__main__':
    main()
