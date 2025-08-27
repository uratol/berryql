import asyncio, sys
sys.path.append(r'd:\Work\berryql')
from tests.schema import schema

INTROSPECTION = {
    "query": "query { __type(name: \"PostQL\") { name fields { name type { kind name ofType { kind name } } } } }",
}

async def main():
    res = await schema.execute(INTROSPECTION["query"], context_value={})
    if res.errors:
        print("ERR:", res.errors)
        return
    t = res.data["__type"]
    for f in t["fields"]:
        if f["name"] == "status":
            typ = f["type"]
            kind = typ.get("kind")
            name = typ.get("name")
            ofType = typ.get("ofType")
            print("status type:", kind, name, ofType)
            break
    else:
        print("status field not found")
    # Also inspect input type for PostQLInput
    q2 = """
    query { __type(name: \"PostQLInput\") { name inputFields { name type { kind name ofType { kind name } } } } }
    """
    res2 = await schema.execute(q2, context_value={})
    if res2.errors:
        print("ERR2:", res2.errors)
    else:
        for f in res2.data["__type"]["inputFields"]:
            if f["name"] == "status":
                print("input status type:", f["type"])
                break

if __name__ == "__main__":
    asyncio.run(main())
