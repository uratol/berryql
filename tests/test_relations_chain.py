import pytest
from tests.schema import schema as berry_schema


@pytest.mark.asyncio
async def test_relations_chain_posts_author_post_comments_author_email_shape(db_session, populated_db):
    # multiple (posts) -> single (author) -> multiple (post_comments) -> single (author.email)
    # This mirrors the user's query exactly; we validate shape and presence of fields.
    query = """
    query {
      posts {
        author {
          post_comments {
            author {
              email
            }
          }
        }
      }
    }
    """

    res = await berry_schema.execute(query, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    assert res.data is not None and "posts" in res.data

    posts = res.data["posts"]
    assert isinstance(posts, list)
  # Basic structural checks and ensure nested authors are present even without author_id selected
    for p in posts:
        assert isinstance(p, dict)
        assert "author" in p
        a = p["author"]
        # author can be null; if present, it must have post_comments list
        if a is not None:
            assert "post_comments" in a and isinstance(a["post_comments"], list)
            for c in a["post_comments"]:
                assert isinstance(c, dict)
        # author must be present even without explicit author_id projection
        assert c.get("author") is not None
        assert isinstance(c["author"].get("email"), (str, type(None)))


@pytest.mark.asyncio
async def test_relations_chain_with_fk_projection_enables_nested_author(db_session, populated_db):
    # Including author_id ensures nested author resolution works without pushdown.
    query = """
    query {
      posts(limit: 3) {
        author {
          post_comments(limit: 3) {
            author_id
            author { email }
          }
        }
      }
    }
    """
    res = await berry_schema.execute(query, context_value={"db_session": db_session})
    assert res.errors is None, res.errors
    posts = res.data["posts"]
    known_emails = {"alice@example.com", "bob@example.com", "charlie@example.com", "dave@example.com"}
    # Now we expect non-null authors for comments since author_id was selected
    for p in posts:
        a = p.get("author")
        if a is None:
            continue
        for c in a.get("post_comments", []):
            if c.get("author_id") is None:
                continue
            au = c.get("author")
            assert au is not None
            assert au.get("email") in known_emails
