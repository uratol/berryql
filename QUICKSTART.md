# BerryQL Quick Start Guide

This guide will help you get started with BerryQL quickly.

## Installation

```bash
pip install berryql
```

## Basic Setup

1. **Define your SQLAlchemy models**:

```python
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import DeclarativeBase, relationship

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    email = Column(String(255))
    
    posts = relationship("Post", back_populates="author")

class Post(Base):
    __tablename__ = 'posts'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(200))
    content = Column(String(5000))
    author_id = Column(Integer, ForeignKey('users.id'))
    
    author = relationship("User", back_populates="posts")
```

2. **Define your Strawberry GraphQL types**:

```python
import strawberry
from typing import List, Optional
from berryql import get_resolved_field_data

@strawberry.type
class PostType:
    id: int
    title: str
    content: Optional[str]
    author_id: int

@strawberry.type  
class UserType:
    id: int
    name: str
    email: str
    
    @strawberry.field
    async def posts(self, info: strawberry.Info) -> List[PostType]:
        return get_resolved_field_data(self, info, 'posts')
```

3. **Create BerryQL resolvers**:

```python
from berryql import BerryQLFactory, GraphQLQueryParams

factory = BerryQLFactory()

user_resolver = factory.create_berryql_resolver(
    strawberry_type=UserType,
    model_class=User
)

post_resolver = factory.create_berryql_resolver(
    strawberry_type=PostType, 
    model_class=Post
)
```

4. **Use in your GraphQL schema**:

```python
@strawberry.type
class Query:
    @strawberry.field
    async def users(
        self,
        info: strawberry.Info,
        db: AsyncSession
    ) -> List[UserType]:
        return await user_resolver(db=db, info=info)
    
    @strawberry.field  
    async def posts(
        self,
        info: strawberry.Info,
        db: AsyncSession,
        limit: Optional[int] = None
    ) -> List[PostType]:
        params = GraphQLQueryParams(limit=limit)
        return await post_resolver(db=db, info=info, params=params)

schema = strawberry.Schema(query=Query)
```

## Advanced Features

### Filtering

```python
from berryql.input_types import StringComparisonInput

@strawberry.input
class UserWhereInput:
    name: Optional[StringComparisonInput] = None

@strawberry.field
async def users(
    self,
    info: strawberry.Info, 
    db: AsyncSession,
    where: Optional[UserWhereInput] = None
) -> List[UserType]:
    from berryql import convert_where_input
    
    where_dict = convert_where_input(where.__dict__ if where else {})
    params = GraphQLQueryParams(where=where_dict)
    return await user_resolver(db=db, info=info, params=params)
```

### Custom Fields

```python
from berryql import custom_field
from sqlalchemy import func

@strawberry.type
class UserType:
    id: int
    name: str
    
    @strawberry.field
    @custom_field
    def post_count(self) -> int:
        pass

def build_post_count_query(model_class, requested_fields):
    return func.count(Post.id).filter(Post.author_id == model_class.id).label('post_count')

user_resolver = factory.create_berryql_resolver(
    strawberry_type=UserType,
    model_class=User,
    custom_fields={
        UserType: {
            'post_count': build_post_count_query
        }
    }
)
```

## Benefits

- **No N+1 Problems**: BerryQL automatically optimizes queries using lateral joins
- **Dynamic Field Selection**: Only requested fields are queried from the database  
- **Type Safety**: Full type hints and GraphQL input types
- **Easy Integration**: Works seamlessly with existing Strawberry and SQLAlchemy code
- **Performance**: Intelligent query building and relationship resolution

## Next Steps

- Check out the [examples/](../examples/) directory for complete working examples
- Read the full [README.md](../README.md) for comprehensive documentation
- See [CONTRIBUTING.md](../CONTRIBUTING.md) for development guidelines
