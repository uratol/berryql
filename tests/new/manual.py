import pytest
from schema import schema as berry_schema

q = """
query { users(name_ilike: \"Alice\") { id posts(limit:1, offset:0) { id } } }
"""

import asyncio

async def main():
	res = await berry_schema.execute(q, context_value={'db_session': db_session})
	print(res)

asyncio.run(main())
