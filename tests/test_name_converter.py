import pytest


try:
    # Prefer importing the actual NameConverter so we inherit required methods (from_type, etc.)
    from strawberry.schema.name_converter import NameConverter  # type: ignore
except Exception:  # pragma: no cover
    NameConverter = object  # type: ignore


class XPrefixCamelConverter(NameConverter):
    """Custom Strawberry name converter used in config.name_converter.

    - Leaves simple lower-case names like 'users', 'posts', 'author' unchanged.
    - For snake_case names (contain '_'), returns 'x' + CamelCase, e.g. 'post_comments' -> 'xPostComments'.
    """

    auto_camel_case = False  # hint; Berry reads this optionally

    def apply_naming_config(self, name: str) -> str:  # Strawberry calls this for field names
        if not isinstance(name, str):  # defensive
            return name
        if '_' not in name:
            return name
        parts = [p for p in name.split('_') if p]
        camel = parts[0].lower() + ''.join(p.capitalize() for p in parts[1:]) if parts else name
        return 'x' + camel[0].upper() + camel[1:] if camel else name


@pytest.mark.asyncio
async def test_custom_name_converter_for_relations(db_session, populated_db):
    from tests.schema import berry_schema
    try:
        from strawberry.schema.config import StrawberryConfig
    except Exception:
        pytest.skip("StrawberryConfig not available in this Strawberry version")

    schema = berry_schema.to_strawberry(
        strawberry_config=StrawberryConfig(name_converter=XPrefixCamelConverter())
    )

    # Root fields ('users', 'posts') are unchanged by our converter. Only snake_case relations get prefixed.
    query = '''
    query {
      users(limit: 1) {
        id
        posts(limit: 1) {
          id
          xPostComments { id }
        }
      }
    }
    '''

    res = await schema.execute(query, context_value={'db_session': db_session})
    assert res.errors is None, res.errors
    assert res.data and 'users' in res.data
    users = res.data['users']
    assert isinstance(users, list) and len(users) >= 0
    if users:
        posts = users[0].get('posts') or []
        assert isinstance(posts, list)
        if posts:
            assert 'xPostComments' in posts[0]
            assert isinstance(posts[0]['xPostComments'], list)
