import sys, asyncio
sys.path.append(r'd:\Work\berryql')
from tests.schema import schema

async def main():
    q = "query { posts { id status __typename } }"
    res = await schema.execute(q, context_value={})
    print("errors:", res.errors)
    rows = res.data['posts']
    for r in rows:
        s = r['status']
        print(type(s), s, getattr(s, 'name', None))

if __name__ == '__main__':
    asyncio.run(main())
