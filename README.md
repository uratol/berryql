# BerryQL

A powerful GraphQL query optimization library for Strawberry GraphQL and SQLAlchemy that eliminates N+1 problems and provides advanced query building capabilities.

## Features

- **Unified Query Building**: Single approach using lateral joins + json_agg for all query levels
- **Automatic Field Mapping**: Maps Strawberry GraphQL types to SQLAlchemy models automatically
- **Dynamic Field Filtering**: Only requested fields are queried from the database
- **Recursive Configuration**: Support for custom configurations at any nesting level
- **Advanced Query Parameters**: Support for where/order_by/offset/limit parameters at any level
- **Complete N+1 Elimination**: Uses optimized lateral joins to prevent N+1 query problems
- **Type Safety**: Full type hints and proper GraphQL input types
- **Fragment Support**: Handles GraphQL fragments, inline fragments, and aliases
- **Custom Fields**: Support for computed fields with custom query builders

## Installation

```bash
pip install berryql
```

## Quick Start

### Basic Usage

```python
import strawberry
from sqlalchemy.ext.asyncio import AsyncSession
from berryql import BerryQLFactory, GraphQLQueryParams

# Define your Strawberry GraphQL type
@strawberry.type
class UserType:
    id: int
    name: str
    email: str
    posts: List["PostType"]

# Create a resolver factory
factory = BerryQLFactory()

# Create an optimized resolver
user_resolver = factory.create_berryql_resolver(
    strawberry_type=UserType,
    model_class=User  # Your SQLAlchemy model
)

# Use in your GraphQL schema
@strawberry.type
class Query:
    @strawberry.field
    async def users(
        self, 
        info: strawberry.Info,
        db: AsyncSession,
        params: Optional[GraphQLQueryParams] = None
    ) -> List[UserType]:
        return await user_resolver(db=db, info=info, params=params)
```

### Advanced Features

#### Custom Fields

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
        """Custom field with automatic query optimization"""
        pass

# Define the custom field query builder
def build_post_count_query(model_class, requested_fields):
    return func.count(model_class.posts).label('post_count')

# Create resolver with custom field configuration
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

#### Input Types for Filtering

```python
from berryql.input_types import StringComparisonInput, IntComparisonInput

@strawberry.input
class UserWhereInput:
    name: Optional[StringComparisonInput] = None
    age: Optional[IntComparisonInput] = None

# Use in your resolver
@strawberry.field
async def users(
    self,
    info: strawberry.Info,
    db: AsyncSession,
    where: Optional[UserWhereInput] = None
) -> List[UserType]:
    # Convert input to berryql format
    where_dict = convert_where_input(where.__dict__ if where else {})
    params = GraphQLQueryParams(where=where_dict)
    return await user_resolver(db=db, info=info, params=params)
```

#### Relationship Data Access

```python
from berryql import get_resolved_field_data

@strawberry.type
class UserType:
    id: int
    name: str
    
    @strawberry.field
    async def posts(self, info: strawberry.Info) -> List[PostType]:
        # Automatically retrieves pre-resolved relationship data
        return get_resolved_field_data(self, info, 'posts')
    
    @strawberry.field
    async def recent_posts(self, info: strawberry.Info) -> List[PostType]:
        # Access aliased fields
        return get_resolved_field_data(self, info, 'posts')
```

## Core Components

### BerryQLFactory

The main factory class that creates optimized GraphQL resolvers.

```python
factory = BerryQLFactory()

resolver = factory.create_berryql_resolver(
    strawberry_type=YourType,
    model_class=YourModel,
    custom_fields=None,  # Optional custom field definitions
    custom_where=None,   # Optional custom where conditions
    custom_order=None    # Optional default ordering
)
```

### GraphQLQueryParams

Parameter object for GraphQL queries supporting filtering, ordering, and pagination.

```python
params = GraphQLQueryParams(
    where={'name': {'like': '%john%'}},
    order_by=[{'field': 'created_at', 'direction': 'desc'}],
    offset=0,
    limit=10
)
```

### Input Types

Type-safe GraphQL input types for filtering:

- `StringComparisonInput`: String field comparisons (eq, ne, like, ilike, in, etc.)
- `IntComparisonInput`: Integer field comparisons
- `FloatComparisonInput`: Float field comparisons  
- `DateTimeComparisonInput`: DateTime field comparisons
- `UUIDComparisonInput`: UUID field comparisons
- `BoolComparisonInput`: Boolean field comparisons

## Advanced Usage

### Custom Query Decorators

```python
from berryql import berryql_field

@strawberry.type
class UserType:
    @strawberry.field
    @berryql_field  # Automatically handles resolved data
    async def posts(self, info: strawberry.Info) -> List[PostType]:
        pass  # Implementation handled by decorator
```

### Mixin Approach

```python
from berryql import ResolvedDataMixin

@strawberry.type
class UserType(ResolvedDataMixin):
    @strawberry.field
    async def posts(self, info: strawberry.Info) -> List[PostType]:
        return self.get_resolved_field_data(info, 'posts')
```

## Requirements

- Python 3.8+
- strawberry-graphql >= 0.200.0
- SQLAlchemy >= 2.0.0
- graphql-core >= 3.2.0

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and changes.
