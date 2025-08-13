from tests.fixtures import *  # noqa: F401,F403
from tests.schema import schema

@pytest.mark.asyncio
async def test_post_comment_text_len_custom_field(db_session, populated_db):
    # Query first 2 posts including custom aggregated length field
    query = """
    query {
      posts(limit: 2) { id title comment_text_len }
    }
    """
    res = await schema.execute(query, context_value={"db_session": db_session})
    print(res.data)
    assert not res.errors
