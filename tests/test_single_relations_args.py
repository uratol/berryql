import pytest
from tests.schema import schema as berry_strawberry_schema


async def _get_field_args(type_name: str, field_name: str) -> set[str]:
        # Use GraphQL introspection to get argument names in a stable way across library versions
        q = """
        query Introspect($type: String!) {
            __type(name: $type) {
                fields { name args { name } }
            }
        }
        """
        res = await berry_strawberry_schema.execute(q, variable_values={"type": type_name})
        assert res.errors is None, res.errors
        t = res.data["__type"]
        assert t is not None, f"Type {type_name} not found"
        fields = t["fields"] or []
        for f in fields:
                if f["name"] == field_name:
                        return set(a["name"] for a in (f.get("args") or []))
        assert False, f"Field {field_name} not found on {type_name}"


def _assert_no_std_collection_args(arg_names: set[str]):
    banned = {"limit", "offset", "order_by", "order_dir"}
    assert banned.isdisjoint(arg_names), f"Unexpected standard args on single relation: {banned & arg_names}"


@pytest.mark.asyncio
async def test_root_single_relation_args_shape():
    # Query.userById is single: should only have where + filter args (id), no limit/offset/order*
    arg_names = await _get_field_args('Query', 'userById')
    assert 'where' in arg_names
    assert 'id' in arg_names
    _assert_no_std_collection_args(arg_names)

@pytest.mark.asyncio
async def test_nested_single_relation_args_shape():
    # PostQL.author is single: should only have where + declared filter args, no limit/offset/order*
    arg_names = await _get_field_args('PostQL', 'author')
    for expected in {'where', 'name_ilike', 'created_at_between', 'is_admin_eq'}:
        assert expected in arg_names
    _assert_no_std_collection_args(arg_names)
