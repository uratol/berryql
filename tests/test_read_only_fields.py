import pytest

from tests.schema import schema

pytestmark = pytest.mark.asyncio


async def test_read_only_fields_excluded_from_mutation_inputs():
    # Introspect PostQLInput and ensure read-only fields are not present
    sdl_query = """
    { __type(name: "PostQLInput") { name inputFields { name } } }
    """
    res = await schema.execute(sdl_query)
    assert res.errors is None, res.errors
    fields = {f["name"] for f in res.data["__type"]["inputFields"]}
    # created_at and content_length were marked read_only
    assert "created_at" not in fields
    assert "content_length" not in fields
    # writable fields still present
    assert "title" in fields and "content" in fields and "author_id" in fields


async def test_read_only_fields_still_queryable(db_session, populated_db):
    # Ensure read-only fields remain on query output types
    q = """
    query {
      posts { id created_at content_length }
    }
    """
    res = await schema.execute(q, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    rows = res.data["posts"]
    assert isinstance(rows, list)
    # All rows should have created_at, and content_length may be int (>=0)
    assert all("created_at" in r for r in rows)


async def test_domain_mutation_uses_input_without_read_only_fields():
  # Discover the domain mutation container type from root Mutation
  q_dom = """
  { __type(name: "Mutation") { fields { name type { name ofType { name } } } } }
  """
  res1 = await schema.execute(q_dom)
  assert res1.errors is None, res1.errors
  fields = res1.data["__type"]["fields"]
  blog_field = next((f for f in fields if f["name"] == "blogDomain"), None)
  assert blog_field is not None
  dom_type = blog_field["type"]["name"] or (blog_field["type"].get("ofType", {}) or {}).get("name")
  assert dom_type, "Domain mutation container type not found"

  # Introspect BlogDomain mutation container for merge_posts arg type
  q_dom_type = (
    f'{{ __type(name: "{dom_type}") {{ fields {{ name args {{ name type '
    f'{{ kind name ofType {{ kind name ofType {{ kind name ofType {{ kind name }} }} }} }} }} }} }} }}'
  )
  res2 = await schema.execute(q_dom_type)
  assert res2.errors is None, res2.errors
  dom_fields = res2.data["__type"]["fields"]
  merge_posts = next((f for f in dom_fields if f["name"] == "merge_posts"), None)
  assert merge_posts is not None
  payload_arg = next((a for a in merge_posts.get("args", []) if a["name"] == "payload"), None)
  assert payload_arg is not None

  # Unwrap type wrappers to get base input object name
  t = payload_arg["type"]
  while t and t.get("ofType") is not None:
    t = t["ofType"]
  base_name = t.get("name")
  assert base_name == "PostQLInput"

  # Ensure read-only fields are not present on the input type
  q_in = '{ __type(name: "PostQLInput") { inputFields { name } } }'
  res3 = await schema.execute(q_in)
  assert res3.errors is None, res3.errors
  in_fields = {f["name"] for f in res3.data["__type"]["inputFields"]}
  assert "created_at" not in in_fields
  assert "content_length" not in in_fields
