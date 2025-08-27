import pytest

from tests.schema import schema as berry_strawberry_schema


@pytest.mark.asyncio
async def test_mutation_field_descriptions_from_comments():
    q = """
    query Introspect { __schema { mutationType { name fields { name description } } } }
    """
    res = await berry_strawberry_schema.execute(q)
    assert res.errors is None, res.errors
    m = res.data["__schema"]["mutationType"]
    assert m is not None
    f = {f["name"]: f.get("description") for f in (m.get("fields") or [])}
    # Descriptions come from explicit comments passed to mutation() descriptors
    assert f.get("merge_posts") == "Create or update posts"
    assert f.get("merge_post") == "Create or update a single post"
    assert f.get("merge_posts_scoped") == "Create or update posts (only author_id==1)"

