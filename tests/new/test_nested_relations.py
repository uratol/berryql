import pytest
from tests.common.fixtures import *  # noqa: F401,F403
from tests.new.schema import schema


@pytest.mark.asyncio
async def test_nested_relations_query_count(db_session, populated_db):
        from sqlalchemy import event
        engine = db_session.get_bind()
        query_counter = {"count": 0}

        def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):  # noqa: D401
                if not statement.lstrip().upper().startswith("PRAGMA") and not statement.lstrip().upper().startswith("CREATE TABLE") and "INSERT INTO" not in statement:
                        query_counter["count"] += 1

        event.listen(engine, "before_cursor_execute", before_cursor_execute)
        query = """
        query Nested {
            users(limit: 1) {
                id
                name
                posts(limit: 2) {
                    id
                    title
                    post_comments(limit: 2) { id }
                }
            }
            posts(limit: 2) {
                id
                title
                post_comments(limit: 2) { id }
            }
        }
        """
        try:
                res = await schema.execute(query, context_value={"db_session": db_session})
                assert res.errors is None, res.errors
                data = res.data
                assert "users" in data
                assert len(data["users"]) == 1
                u = data["users"][0]
                assert "posts" in u and isinstance(u["posts"], list)
                for p in u["posts"]:
                        assert "post_comments" in p
                        assert isinstance(p["post_comments"], list)
                assert "posts" in data
                for p in data["posts"]:
                        assert "post_comments" in p
                try:
                        dialect_name = db_session.get_bind().dialect.name.lower()
                except Exception:
                        dialect_name = 'sqlite'
                if dialect_name.startswith('mssql'):
                        pytest.skip("Relation pushdown disabled for MSSQL adapter")
                assert query_counter['count'] == 2, f"Expected 2 SQL queries, got {query_counter['count']}"
        finally:
                event.remove(engine, "before_cursor_execute", before_cursor_execute)
