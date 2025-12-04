import pytest

from tests.schema import schema as berry_strawberry_schema


@pytest.mark.asyncio
async def test_blog_domain_regular_strawberry_field(db_session, populated_db):
    q = """
    query {
      blogDomain {
        helloDomain
      }
    }
    """
    res = await berry_strawberry_schema.execute(q, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    assert res.data["blogDomain"]["helloDomain"] == "hello from blogDomain"


@pytest.mark.asyncio
async def test_blog_domain_annotated_lazy_field_exposed(db_session, populated_db):
    # Ensure field exists and returns null (None) but schema compiles
    q = """
    query {
      blogDomain {
  samplePostAnnotated(id: 1, title: "x") { id }
      }
    }
    """
    res = await berry_strawberry_schema.execute(q, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    # Value can be None; just ensure key present (null in GraphQL)
    assert "samplePostAnnotated" in res.data["blogDomain"], res.data


@pytest.mark.asyncio
async def test_user_domain_strawberry_field_users(db_session, populated_db):
    q = """
    query {
      userDomain {
        users_strawberry {
          id
          name
          posts {
            id
            title
          }
        }
      }
    }
    """
    res = await berry_strawberry_schema.execute(q, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    data = res.data["userDomain"]["users_strawberry"]
    assert len(data) > 0
    # Check if posts are populated
    has_posts = False
    for u in data:
        if u["posts"]:
            has_posts = True
            break
    assert has_posts
