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
    # Type description prefers model docstring over table comment
    assert t["description"] == "Application users (docstring)"
    f = _fields_dict(t)
    # Field descriptions come from Column.comment
    assert f.get("email") == "Unique login email"
    assert f.get("is_admin") == "Administrative flag"
    assert f.get("created_at") == "Creation timestamp (UTC)"
    # Scalar field without explicit comment inherits from SA column
    assert f.get("name") == "Public display name"
    # Relation without explicit comment prefers target model's docstring, then table comment
    assert f.get("posts") == "Blog posts (docstring)"
    assert f.get("post_comments") == "User comments on posts"


@pytest.mark.asyncio
async def test_postql_descriptions_from_comments():
    t = await _introspect_type("PostQL")
    assert t["description"] == "Blog posts (docstring)"
    f = _fields_dict(t)
    assert f.get("title") == "Post title"
    assert f.get("author_id") == "Author FK to users.id"
    # Relation without explicit comment prefers target model's docstring
    assert f.get("author") == "Application users (docstring)"
    assert f.get("post_comments") == "User comments on posts"
    assert f.get("views") == "Polymorphic views on posts and comments"
    # Enum field should include a values list in its description
    status_desc = f.get("status") or ""
    assert "Values: draft, published, archived" in status_desc


@pytest.mark.asyncio
async def test_postcommentql_descriptions_from_comments():
    t = await _introspect_type("PostCommentQL")
    assert t["description"] == "User comments on posts"
    f = _fields_dict(t)
    assert f.get("content") == "Comment text"
    # Scalar fallback from SA column
    assert f.get("rate") == "Simple rating value"
