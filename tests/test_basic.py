import pytest
from berryql import BerrySchema, BerryType, field, relation, count

schema = BerrySchema()

@schema.type(model=None)
class UserQL(BerryType):
    id = field()
    name = field()
    posts = relation('PostQL')
    posts_count = count('posts')

@schema.type(model=None)
class PostQL(BerryType):
    id = field()
    title = field()


def test_registry_collects_fields():
    assert 'UserQL' in schema.types
    assert set(UserQL.__berry_fields__.keys()) == {'id', 'name', 'posts', 'posts_count'}
    assert UserQL.__berry_fields__['posts'].kind == 'relation'
    assert UserQL.__berry_fields__['posts_count'].kind == 'aggregate'


def test_schema_to_strawberry_builds():
    s = schema.to_strawberry()
    assert s is not None
    # ensure internal query field exists
    assert '_ping' in [f.name for f in s.get_type_by_name('Query').fields]
