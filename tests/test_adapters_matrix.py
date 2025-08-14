import os
import pytest

from berryql.adapters import get_adapter, SQLiteAdapter, PostgresAdapter, MSSQLAdapter


@pytest.mark.parametrize("dialect,cls", [
    ("sqlite", SQLiteAdapter),
    ("postgresql", PostgresAdapter),
    ("mssql", MSSQLAdapter),
])
def test_get_adapter_matrix(dialect, cls):
    a = get_adapter(dialect)
    assert isinstance(a, cls)


@pytest.mark.skipif(os.getenv('BERRYQL_TEST_DATABASE_URL', '').startswith('postgresql') is False, reason="requires Postgres URL")
def test_postgres_adapter_funcs():
    a = get_adapter('postgresql')
    # Ensure functions exist and return a SQLAlchemy expression-like object
    assert hasattr(a, 'json_object')
    assert a.json_object('a', 'b') is not None
    assert a.json_array_agg('x') is not None


@pytest.mark.skip(reason="MSSQL FOR JSON path tested indirectly in integration; adapter methods are placeholders")
def test_mssql_adapter_skip():
    a = get_adapter('mssql')
    assert a.supports_relation_pushdown()
