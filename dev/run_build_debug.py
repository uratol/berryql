import logging
import sys

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

try:
    from tests.schema import berry_schema
except Exception as e:
    print("import schema failed:", repr(e))
    sys.exit(1)

try:
    s = berry_schema.to_strawberry()
    print("built schema ok")
    print(s)
except Exception as e:
    print("build failed:", repr(e))
    raise
