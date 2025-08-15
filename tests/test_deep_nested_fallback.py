import pytest
from sqlalchemy import event
from tests.schema import schema

# This test now asserts that a deeply nested, pushdown-friendly query executes in a single SQL SELECT
# on Postgres by leveraging JSON aggregation pushdown (users -> posts -> post_comments).


@pytest.mark.asyncio
async def test_deep_nested_pushdown_one_select(engine, db_session, populated_db):
    # Count SELECT statements executed by SQLAlchemy during the GraphQL query
    calls = {"selects": 0}

    def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        try:
            if str(statement).lstrip().upper().startswith("SELECT"):
                calls["selects"] += 1
        except Exception:
            pass

    event.listen(engine.sync_engine, "before_cursor_execute", before_cursor_execute)
    try:
        query = {
            'query': (
                '{ users { id name posts { post_comments { id likes { id } admin_likes { id user_id } } } } } '
            )
        }
        res = await schema.execute(query['query'], context_value={'db_session': db_session})
        assert res.errors is None, f"GraphQL errors: {res.errors}"
        # Assert real data came back: collect all likes and check count and known liker names
        data = res.data or {}
        assert 'users' in data and isinstance(data['users'], list)
        # Check returned user names include all seeded users
        user_names = {u.get('name') for u in data['users']}
        assert {'Alice Johnson', 'Bob Smith', 'Charlie Brown', 'Dave NoPosts'}.issubset(user_names)
        # Flatten likes across users -> posts -> post_comments
        all_likes = []
        all_admin_likes = []
        for u in data['users']:
            for p in (u.get('posts') or []):
                for c in (p.get('post_comments') or []):
                    for lk in (c.get('likes') or []):
                        all_likes.append(lk)
                    for alk in (c.get('admin_likes') or []):
                        all_admin_likes.append(alk)
        # From fixtures, there are exactly 4 likes across all comments
        assert len(all_likes) == 4, f"Expected 4 likes, got {len(all_likes)}"
        # Since pushdown returns only like ids at this depth, validate concrete IDs
        like_ids = sorted([lk.get('id') for lk in all_likes if lk is not None])
        assert like_ids == [1, 2, 3, 4], f"Unexpected like ids: {like_ids}"
        # Admin likes should be a subset: only likes by admin user (id=1) -> ids [1, 3]
        admin_like_ids = sorted([lk.get('id') for lk in all_admin_likes if lk is not None])
        assert admin_like_ids == [1, 3], f"Unexpected admin_like ids: {admin_like_ids}"
        assert all((lk.get('user_id') == 1) for lk in all_admin_likes), f"Non-admin like found in admin_likes: {all_admin_likes}"
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", before_cursor_execute)

    # Expect a single SELECT due to pushdown of nested relations
    assert calls["selects"] == 1, f"Expected 1 SELECT, got {calls['selects']}"
