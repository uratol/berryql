from tests.schema import UserQL, PostQL, PostCommentQL, schema as berry_strawberry_schema

# Basic shape tests mapping to original tests/schema.py GraphQL types

def test_userql_fields_shape():
    f = set(UserQL.__berry_fields__.keys())
    assert {'id','name','email','is_admin','created_at','posts','post_comments','post_agg','post_agg_obj','new_posts','other_users','bloggers'} <= f


def test_postql_fields_shape():
    f = set(PostQL.__berry_fields__.keys())
    assert {'id','title','content','author_id','created_at','author','post_comments','post_comments_agg'} <= f


def test_postcommentql_fields_shape():
    f = set(PostCommentQL.__berry_fields__.keys())
    assert {'id','content','rate','post_id','author_id','created_at','post','author'} <= f


def test_strawberry_schema_compiles():
    assert berry_strawberry_schema.get_type_by_name('Query') is not None
