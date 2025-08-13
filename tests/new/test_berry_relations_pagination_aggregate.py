import pytest
from tests.common.fixtures import *  # noqa: F401,F403
from tests.new.schema import schema


@pytest.mark.asyncio
async def test_single_object_relations_and_count(db_session, populated_db):
    from sqlalchemy import event
    engine = db_session.get_bind()
    query_counter = {'count': 0}
    def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):  # noqa: D401
        query_counter['count'] += 1
    event.listen(engine, "before_cursor_execute", before_cursor_execute)
    try:
        query = """
        query {
          posts(limit: 2) {
            id
            author { id name }
            post_comments(limit: 2) { id }
            post_comments_agg
          }
          users(limit: 1) { id name posts(limit: 2) { id title } post_agg post_agg_obj { count } }
        }
        """
        res = await schema.execute(query, context_value={"db_session": db_session})
        assert res.errors is None, res.errors
        data = res.data
        assert "posts" in data
        assert len(data["posts"]) == 2
        first_post = data["posts"][0]
        assert "author" in first_post and isinstance(first_post["author"], dict)
        assert "post_comments_agg" in first_post
        assert "users" in data
        assert len(data["users"]) == 1
        assert "post_agg" in data["users"][0]
        assert "post_agg_obj" in data["users"][0]
        agg_obj = data["users"][0]["post_agg_obj"]
        if agg_obj is not None:
            assert set(agg_obj.keys()) == {"count"}
        # Assert single SQL query executed (all relations batched)
        assert query_counter['count'] == 1, f"Expected 1 SQL query, got {query_counter['count']}"
    finally:
        event.remove(engine, "before_cursor_execute", before_cursor_execute)
