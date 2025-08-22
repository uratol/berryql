import strawberry
from strawberry.schema.config import StrawberryConfig as C

@strawberry.type
class Q:
    a: int = 1

s = strawberry.Schema(Q, config=C(auto_camel_case=False))
print('auto?', s.config.name_converter.auto_camel_case)
print(s.as_str())
