import os
import pytest

from berryql.adapters import get_adapter, SQLiteAdapter, PostgresAdapter, MSSQLAdapter
from berryql.sql.builders import RelationSQLBuilders
from tests.models import Post, User
from tests.schema import berry_schema


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


def test_mssql_single_relation_json_uses_join_shape():
    adapter = get_adapter('mssql')
    builders = RelationSQLBuilders(berry_schema)

    expr = builders.build_single_relation_object(
        adapter=adapter,
        parent_model_cls=Post,
        child_model_cls=User,
        rel_name='author',
        projected_columns=['id', 'name'],
        parent_fk_col_name='author_id',
        json_object_fn=None,
        json_array_coalesce_fn=None,
        to_where_dict=lambda value, strict=True: value,
        expr_from_where_dict=lambda model, value, strict=True: value,
        info=None,
        rel_where=None,
        rel_default_where=None,
        type_default_where=None,
        filter_args={},
        arg_specs={},
        nested_cfg=None,
        json_array_agg_fn=None,
    )

    sql = str(expr)
    assert 'FROM [posts] AS [berryql_parent_posts_author_id]' in sql
    assert 'JOIN [users] ON [users].[id] = [berryql_parent_posts_author_id].[author_id]' in sql


def test_mssql_join_order_expr_uses_left_join_alias():
    adapter = get_adapter('mssql')
    builders = RelationSQLBuilders(berry_schema)
    join_clauses = []
    join_cache = {}

    expr = builders._resolve_mssql_join_order_expr(
        adapter,
        Post,
        'reviewer.id',
        join_cache=join_cache,
        join_clauses=join_clauses,
    )

    assert expr == '[berryql_ord_1_reviewer].[id]'
    assert join_clauses == [
        'LEFT JOIN [users] AS [berryql_ord_1_reviewer] ON [berryql_ord_1_reviewer].[id] = [posts].[reviewer_id]'
    ]


