import pytest

from tests.schema import schema

pytestmark = pytest.mark.asyncio


async def test_custom_fields_excluded_by_default_from_mutation_inputs():
    # The default custom field comment_text_len should not appear in PostQLInput
    q = '{ __type(name: "PostQLInput") { inputFields { name } } }'
    res = await schema.execute(q)
    assert res.errors is None, res.errors
    names = {f['name'] for f in res.data['__type']['inputFields']}
    assert 'comment_text_len' not in names


async def test_custom_fields_included_when_read_only_false():
    # The writable custom field title_len_custom_input should be present
    q = '{ __type(name: "PostQLInput") { inputFields { name } } }'
    res = await schema.execute(q)
    assert res.errors is None, res.errors
    names = {f['name'] for f in res.data['__type']['inputFields']}
    assert 'title_len_custom_input' in names
