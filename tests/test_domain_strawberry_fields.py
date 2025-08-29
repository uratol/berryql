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
        samplePostAnnotated { id }
      }
    }
    """
    res = await berry_strawberry_schema.execute(q, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    # Value can be None; just ensure key present (null in GraphQL)
    assert "samplePostAnnotated" in res.data["blogDomain"], res.data
