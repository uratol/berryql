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


@pytest.mark.asyncio
async def test_camel_case_root_mutations(db_session, populated_db):
	# Build schema with camelCase enabled
	from tests.schema import berry_schema
	try:
		from strawberry.schema.config import StrawberryConfig
	except Exception:
		pytest.skip("StrawberryConfig not available in this Strawberry version")
	schema = berry_schema.to_strawberry(strawberry_config=StrawberryConfig(auto_camel_case=True))

	# Root BerryQL mutation name should be camelCased: mergePosts, mergePost
	mutation = (
		"mutation($p: [PostQLInput!]!) { "
		"mergePosts(payload: $p) { id title authorId } }"
	)
	vars = {
		"p": [
			{"title": "CamelRoot", "content": "Body", "authorId": 1}
		]
	}
	res = await schema.execute(mutation, variable_values=vars, context_value={"db_session": db_session})
	assert res.errors is None, res.errors
	assert res.data and "mergePosts" in res.data
	post = res.data["mergePosts"]
	assert isinstance(post, dict) and {"id", "title", "authorId"} <= set(post)

	# Single-payload variant should be mergePost
	mutation_single = (
		"mutation($p: PostQLInput!) { "
		"mergePost(payload: $p) { id title authorId } }"
	)
	vars_single = {"p": {"title": "CamelSingle", "content": "Body", "authorId": 1}}
	res2 = await schema.execute(mutation_single, variable_values=vars_single, context_value={"db_session": db_session})
	assert res2.errors is None, res2.errors
	assert res2.data and "mergePost" in res2.data
	post2 = res2.data["mergePost"]
	assert isinstance(post2, dict) and {"id", "title", "authorId"} <= set(post2)


@pytest.mark.asyncio
async def test_camel_case_domain_mutations(db_session, populated_db):
	from tests.schema import berry_schema
	try:
		from strawberry.schema.config import StrawberryConfig
	except Exception:
		pytest.skip("StrawberryConfig not available in this Strawberry version")
	schema = berry_schema.to_strawberry(strawberry_config=StrawberryConfig(auto_camel_case=True))

	# Domain BerryQL mutation should be camelCased under blogDomain
	mutation = (
		"mutation($p: [PostQLInput!]!) { "
		"blogDomain { mergePosts(payload: $p) { id title authorId } } }"
	)
	vars = {"p": [{"title": "FromDomain", "content": "Body", "authorId": 1}]}
	res = await schema.execute(mutation, variable_values=vars, context_value={"db_session": db_session})
	assert res.errors is None, res.errors
	assert res.data and "blogDomain" in res.data
	data = res.data["blogDomain"]["mergePosts"]
	assert isinstance(data, dict) and {"id", "title", "authorId"} <= set(data)

	# Domain strawberry.mutation method should also be camelCased: createPostMut
	mutation2 = (
		"mutation { blogDomain { "
		"createPostMut(title: \"S\", content: \"B\", authorId: 1) { id title authorId } "
		"} }"
	)
	res2 = await schema.execute(mutation2, context_value={"db_session": db_session})
	assert res2.errors is None, res2.errors
	post = res2.data["blogDomain"]["createPostMut"]
	assert isinstance(post, dict) and {"id", "title", "authorId"} <= set(post)


@pytest.mark.asyncio
async def test_camel_case_domain_berryql_fields(db_session, populated_db):
	from tests.schema import berry_schema
	try:
		from strawberry.schema.config import StrawberryConfig
	except Exception:
		pytest.skip("StrawberryConfig not available in this Strawberry version")
	schema = berry_schema.to_strawberry(strawberry_config=StrawberryConfig(auto_camel_case=True))

	query = '''
	query {
		userDomain {
			users {
				id
				createdAt
				isAdmin
				nameUpper
				postAgg
				postAggObj { count }
				postsRecent { id }
				postsHaveComments { id }
			}
			userById(id: 1) { id nameUpper }
		}
	}
	'''
	res = await schema.execute(query, context_value={"db_session": db_session})
	assert res.errors is None, res.errors
	ud = res.data.get("userDomain")
	assert isinstance(ud, dict)
	users = ud.get("users")
	assert isinstance(users, list) and len(users) >= 1
	first = users[0]
	for key in ["id", "createdAt", "isAdmin", "nameUpper", "postAgg", "postAggObj", "postsRecent", "postsHaveComments"]:
		assert key in first
	# postAggObj may resolve to null depending on implementation; if present, it should have 'count'
	if first.get("postAggObj") is not None:
		assert isinstance(first["postAggObj"], dict) and "count" in first["postAggObj"]
	assert isinstance(first["postsRecent"], list)
	assert isinstance(first["postsHaveComments"], list)
	ubi = ud.get("userById")
	assert isinstance(ubi, dict) and {"id", "nameUpper"} <= set(ubi)


@pytest.mark.asyncio
async def test_camel_case_domain_strawberry_fields(db_session, populated_db):
	from tests.schema import berry_schema
	try:
		from strawberry.schema.config import StrawberryConfig
	except Exception:
		pytest.skip("StrawberryConfig not available in this Strawberry version")
	schema = berry_schema.to_strawberry(strawberry_config=StrawberryConfig(auto_camel_case=True))

	query = '''
	query {
		blogDomain {
			helloDomain
			samplePostAnnotated { id }
		}
	}
	'''
	res = await schema.execute(query, context_value={"db_session": db_session})
	assert res.errors is None, res.errors
	bd = res.data.get("blogDomain")
	assert isinstance(bd, dict)
	assert bd.get("helloDomain") == "hello from blogDomain"
	# samplePostAnnotated returns null by design
	assert "samplePostAnnotated" in bd

