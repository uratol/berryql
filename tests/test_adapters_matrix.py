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


