import pytest
import strawberry
from tests.fixtures import *  # noqa: F401,F403

from berryql import BerrySchema, BerryType, field, relation
from tests.models import Post, User


berry_schema = BerrySchema()


@berry_schema.type(model=User)
class UserOrderingQL(BerryType):
    id = field()
    name = field()


@berry_schema.type(model=Post)
class PostOrderingQL(BerryType):
    id = field()
    title = field()
    author = relation('UserOrderingQL', single=True)


@berry_schema.type(model=Post)
class PostOrderingCustomTitleQL(BerryType):
    id = field()
    author = relation('UserOrderingQL', single=True)

    # This mirrors a real model column name through a regular Strawberry field.
    # Ordering by "title" should still resolve to Post.title even though it is not
    # declared as a Berry scalar via field().
    @strawberry.field
    def title(self) -> str | None:
        try:
            model = getattr(self, '_model', None)
            value = getattr(model, 'title', None) if model is not None else None
            if value is None:
                data = object.__getattribute__(self, '__dict__')
                value = data.get('title') if isinstance(data, dict) else None
            return None if value is None else str(value)
        except Exception:
            return None


@berry_schema.query()
class Query:
    posts = relation('PostOrderingQL', order_by='id', order_dir='asc')
    posts_custom_title = relation('PostOrderingCustomTitleQL', order_by='title', order_dir='asc')
    posts_by_author_default = relation(
        'PostOrderingQL',
        order_multi=["author.name:desc", "id:asc"],
    )


schema = berry_schema.to_strawberry()


def _expected_posts_by_author_desc(populated_db):
    users = populated_db['users']
    posts = populated_db['posts']
    return [
        (posts[4].id, users[2].name),
        (posts[2].id, users[1].name),
        (posts[3].id, users[1].name),
        (posts[0].id, users[0].name),
        (posts[1].id, users[0].name),
    ]


@pytest.mark.asyncio
async def test_root_order_multi_can_sort_by_single_relation_path(db_session, populated_db):
    query = """
    query {
      posts(order_multi: ["author.name:desc", "id:asc"]) {
        id
        author { name }
      }
    }
    """
    res = await schema.execute(query, context_value={'db_session': db_session})
    assert res.errors is None, res.errors

    ordered = [(post['id'], post['author']['name']) for post in res.data['posts']]
    assert ordered == _expected_posts_by_author_desc(populated_db)


@pytest.mark.asyncio
async def test_root_default_order_multi_can_sort_by_single_relation_path(db_session, populated_db):
    query = """
    query {
      posts_by_author_default {
        id
        author { name }
      }
    }
    """
    res = await schema.execute(query, context_value={'db_session': db_session})
    assert res.errors is None, res.errors

    ordered = [(post['id'], post['author']['name']) for post in res.data['posts_by_author_default']]
    assert ordered == _expected_posts_by_author_desc(populated_db)


@pytest.mark.asyncio
async def test_root_order_by_custom_strawberry_field_backed_by_model_column(db_session, populated_db):
    query = """
    query {
      posts_custom_title {
        id
        title
      }
    }
    """
    res = await schema.execute(query, context_value={'db_session': db_session})
    assert res.errors is None, res.errors

    ordered = [(post['id'], post['title']) for post in res.data['posts_custom_title']]
    assert ordered == [
        (post.id, post.title)
        for post in sorted(populated_db['posts'], key=lambda post: (post.title, post.id))
    ]