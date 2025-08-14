import pytest


@pytest.mark.asyncio
async def test_camel_case_users_posts_post_comments(db_session, populated_db):
		# Build a Strawberry schema with auto_camel_case enabled
		from tests.schema import berry_schema
		try:
			from strawberry.schema.config import StrawberryConfig
		except Exception:
			pytest.skip("StrawberryConfig not available in this Strawberry version")
		schema = berry_schema.to_strawberry(strawberry_config=StrawberryConfig(auto_camel_case=True))

		query = '''
		query {
			users {
				id
				posts {
					id
					postComments { id }
				}
			}
		}
		'''

		res = await schema.execute(query, context_value={'db_session': db_session})
		assert res.errors is None, res.errors
		assert res.data is not None and 'users' in res.data
		users = res.data['users']
		assert isinstance(users, list)
		# Validate nested structure exists with camelCase field names
		saw_nested = False
		for u in users:
				posts = u.get('posts') or []
				assert isinstance(posts, list)
				for p in posts:
						assert 'postComments' in p
						assert isinstance(p['postComments'], list)
						saw_nested = True
						break
				if saw_nested:
						break
		assert saw_nested, 'Expected at least one user with posts.postComments'

