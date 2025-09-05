import pytest


@pytest.mark.asyncio
async def test_domain_type_scope_filters_views_via_context(db_session, populated_db):
    # Create a post with two views (for different users), then query via domain with
    # context-only type-level scope to filter views by user_id.
    from tests.schema import schema as berry_schema

    # Create a unique post with two views: user_id 1 and 2
    title = "DomainScopeProbe"
    create_mut = (
        """
        mutation Upsert($payload: [PostQLInput!]!) {
          merge_posts(payload: $payload) { id title views { id user_id entity_type } }
        }
        """
    )
    variables = {
        "payload": [
            {
                "title": title,
                "content": "Body",
                "author_id": 1,
                # Create two views with different user_ids; entity_type is defaulted by relation scope
                "views": [
                    {"user_id": 1},
                    {"user_id": 2},
                ],
            }
        ]
    }
    res1 = await berry_schema.execute(create_mut, variable_values=variables, context_value={"db_session": db_session})
    assert res1.errors is None, res1.errors
    created = res1.data["merge_posts"]
    assert created["title"] == title
    assert isinstance(created.get("views"), list)
    # Both views should be present before filtering
    created_uids = sorted(int(v["user_id"]) for v in created["views"])
    assert created_uids == [1, 2]

    # Now query posts under blogDomain with type-level scope on ViewQL driven by context
    # The ViewQL.type_scope reads only_view_user_id from context and filters by user_id
    q = (
        """
        query Q($title: String!) {
          blogDomain {
            posts(title_ilike: $title) {
              id
              title
              views { id user_id }
            }
          }
        }
        """
    )

    # Filter for user_id == 1
    res2 = await berry_schema.execute(
        q,
        variable_values={"title": title},
        context_value={"db_session": db_session, "only_view_user_id": 1},
    )
    assert res2.errors is None, res2.errors
    posts = res2.data["blogDomain"]["posts"]
    assert isinstance(posts, list) and posts, "expected at least one post"
    # Find our probe post
    probe = next((p for p in posts if p["title"] == title), None)
    assert probe is not None, "probe post not found"
    views = probe.get("views") or []
    # Type-level scope should filter down to only user_id == 1
    assert views and all(int(v["user_id"]) == 1 for v in views)

    # And for user_id == 2
    res3 = await berry_schema.execute(
        q,
        variable_values={"title": title},
        context_value={"db_session": db_session, "only_view_user_id": 2},
    )
    assert res3.errors is None, res3.errors
    posts2 = res3.data["blogDomain"]["posts"]
    probe2 = next((p for p in posts2 if p["title"] == title), None)
    assert probe2 is not None, "probe post not found (2)"
    views2 = probe2.get("views") or []
    assert views2 and all(int(v["user_id"]) == 2 for v in views2)
