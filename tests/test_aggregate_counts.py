import pytest
from tests.fixtures import *  # noqa: F401,F403
from tests.schema import schema


@pytest.mark.asyncio
async def test_post_comments_count_aggregate(db_session, populated_db):
    query = """
    query {
      posts(limit: 3) { id post_comments_agg }
    }
    """
    res = await schema.execute(query, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    data = res.data
    assert len(data['posts']) == 3
    counts = [p['post_comments_agg'] for p in data['posts']]
    # Ensure counts are integers and at least one non-zero
    assert all(isinstance(c, int) for c in counts)
    assert any(c > 0 for c in counts)
