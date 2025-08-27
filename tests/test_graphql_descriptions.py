import pytest

from tests.schema import schema as berry_strawberry_schema


async def _introspect_type(type_name: str) -> dict:
    q = """
    query Introspect($type: String!) {
      __type(name: $type) {
        name
        description
        fields { name description }
      }
    }
    """
    res = await berry_strawberry_schema.execute(q, variable_values={"type": type_name})
    assert res.errors is None, res.errors
    t = res.data["__type"]
    assert t is not None, f"Type {type_name} not found"
    return t


def _fields_dict(t: dict) -> dict[str, str | None]:
    return {f["name"]: f.get("description") for f in (t.get("fields") or [])}


@pytest.mark.asyncio
async def test_userql_descriptions_from_comments():
    t = await _introspect_type("UserQL")
    # Type description comes from SQLAlchemy table comment
    assert t["description"] == "Application users"
    f = _fields_dict(t)
    # Field descriptions come from Column.comment
    assert f.get("email") == "Unique login email"
    assert f.get("is_admin") == "Administrative flag"
    assert f.get("created_at") == "Creation timestamp (UTC)"


@pytest.mark.asyncio
async def test_postql_descriptions_from_comments():
    t = await _introspect_type("PostQL")
    assert t["description"] == "Blog posts"
    f = _fields_dict(t)
    assert f.get("title") == "Post title"
    assert f.get("author_id") == "Author FK to users.id"


@pytest.mark.asyncio
async def test_postcommentql_descriptions_from_comments():
    t = await _introspect_type("PostCommentQL")
    assert t["description"] == "User comments on posts"
    f = _fields_dict(t)
    assert f.get("content") == "Comment text"
