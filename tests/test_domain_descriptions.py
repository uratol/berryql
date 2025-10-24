import pytest

from tests.schema import schema as berry_strawberry_schema


async def _introspect_type(type_name: str) -> dict:
    q = """
    query Introspect($type: String!) {
      __type(name: $type) {
        name
        description
        fields { name description type { name } }
      }
    }
    """
    res = await berry_strawberry_schema.execute(q, variable_values={"type": type_name})
    assert res.errors is None, res.errors
    t = res.data["__type"]
    assert t is not None, f"Type {type_name} not found"
    return t


def _fields_dict(t: dict) -> dict[str, tuple[str | None, str | None]]:
    return {f["name"]: (f.get("description"), (f.get("type") or {}).get("name")) for f in (t.get("fields") or [])}


@pytest.mark.asyncio
async def test_query_domain_field_and_type_descriptions():
    # Query root should expose domain fields with descriptions from domain docstrings
    t_query = await _introspect_type("Query")
    f = _fields_dict(t_query)
    assert f.get("blogDomain")[0] == "Blog operations domain"
    assert f.get("userDomain")[0] == "User operations domain"
    # Domain container type should carry the same description
    t_dom = await _introspect_type("BlogDomainType")
    assert t_dom["description"] == "Blog operations domain"


@pytest.mark.asyncio
async def test_mutation_domain_field_description():
    q = """
    query { __schema { mutationType { name fields { name description type { name } } } } }
    """
    res = await berry_strawberry_schema.execute(q)
    assert res.errors is None, res.errors
    m = res.data["__schema"]["mutationType"]
    assert m is not None
    fields = {f["name"]: (f.get("description"), (f.get("type") or {}).get("name")) for f in (m.get("fields") or [])}
    # blogDomain field on Mutation should have domain docstring description
    desc, tname = fields.get("blogDomain")
    assert desc == "Blog operations domain"
    # Note: GraphQL wraps field types (NON_NULL/OBJECT); type.name may be null here depending on wrapper
