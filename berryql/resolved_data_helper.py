"""
Helper utilities for working with resolved data in GraphQL types.
"""

from typing import List, TypeVar, Any, Optional, Callable, Union, Dict, Type, get_origin, get_args, get_type_hints
import strawberry
from functools import wraps
from sqlalchemy.sql import ColumnElement

T = TypeVar('T')


def get_resolved_field_data(instance: Any, info: strawberry.Info, relationship_name: str) -> List[T]:
    """
    Helper function to extract resolved data for a specific relationship field.
    
    This encapsulates the common pattern used in GraphQL type methods to retrieve
    pre-resolved relationship data from the _resolved attribute.
    
    Args:
        instance: The GraphQL type instance (e.g., ProjectType instance)
        info: GraphQL info object containing field context
        relationship_name: The name of the relationship field (e.g., 'tasks', 'locations')
        
    Returns:
        List of resolved objects for the requested field/alias
        
    Example:
        @strawberry.field
        async def tasks(self, info: strawberry.Info, ...) -> List[TaskType]:
            return get_resolved_field_data(self, info, 'tasks')
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Get the unified resolved data dictionary
    resolved_data = getattr(instance, '_resolved', {})
    
    # Get the actual field name from the GraphQL selection (this handles aliases)
    actual_field_name = info.field_name  # This is 'tasks' for both 'tasks' and 'a: tasks'
    
    # Check if this is an alias by looking at the GraphQL selection
    display_name = actual_field_name  # Default to the field name
    if hasattr(info, 'path') and info.path and hasattr(info.path, 'key'):
        # For aliases, info.path.key contains the alias name
        display_name = info.path.key
    
    # In the nested structure, data is organized by resolved field name, then by display name
    # For aliases like 'a: tasks', we look in resolved_data['tasks']['a']
    result = None
    
    # Check if we have nested resolved data for this field/alias combination
    field_data = resolved_data.get(actual_field_name, {})
    if isinstance(field_data, dict) and display_name in field_data:
        result = field_data[display_name]
    elif isinstance(field_data, list):
        # Direct list result (no aliases)
        result = field_data
    else:
        # Fallback: try to get data directly by relationship name
        result = resolved_data.get(relationship_name, [])
    
    # Ensure we always return a list, never None
    return result if result is not None else []


class ResolvedDataMixin:
    """Mixin class that provides convenient methods for accessing resolved data."""
    
    def get_resolved_field_data(self, info: strawberry.Info, relationship_name: str) -> List[Any]:
        """Instance method version of get_resolved_field_data."""
        return get_resolved_field_data(self, info, relationship_name)


def berryql_field(
    func_or_converter: Optional[Callable] = None, 
    *, 
    converter: Optional[Callable] = None,
    model_class: Optional[Type] = None,
    custom_fields: Optional[Dict[Type, Dict[str, callable]]] = None,
    custom_where: Optional[Dict[Type, Union[Dict[str, Any], callable]]] = None,
    custom_order: Optional[Dict[Type, List[str]]] = None
):
    """
    Enhanced decorator for BerryQL GraphQL field methods.
    
    This decorator automatically detects whether a field method is:
    1. A relationship field (with pass-only body) - provides auto-resolved data access
    2. A custom field (returning SQLAlchemy expression) - integrates with BerryQL's custom field system
    3. A resolver field (with resolver parameters) - creates and uses BerryQL resolver automatically
    
    For relationship fields, it automatically retrieves resolved data from the _resolved attribute.
    For custom fields, it extracts and registers the SQLAlchemy expression for use in queries.
    For resolver fields, it creates a BerryQL resolver with the provided configurations.
    
    Args:
        func_or_converter: Either the function to decorate (when used as @berryql_field)
                          or a converter function (when used as @berryql_field(converter=...))
        converter: Optional converter function to transform raw data
        model_class: SQLAlchemy model class for resolver creation
        custom_fields: Dict mapping {strawberry_type: {field_name: query_builder}}
        custom_where: Dict mapping {strawberry_type: where_conditions_or_function}
        custom_order: Dict mapping {strawberry_type: default_order_list}
        
    Example (resolver field - replaces direct factory.create_berryql_resolver usage):
        @strawberry.field
        @berryql.field(
            model_class=User,
            custom_fields={UserType: {'post_count': build_post_count_query}},
            custom_where={UserType: {'status': 'active'}},
            custom_order={UserType: ['created_at desc']}
        )
        async def users(
            self, 
            info: strawberry.Info, 
            db: AsyncSession,
            params: Optional[GraphQLQueryParams] = None
        ) -> List[UserType]:
            pass  # Implementation handled by decorator using BerryQL resolver
    """
    
    def create_decorator(actual_converter=None, actual_model_class=None, 
                        actual_custom_fields=None, actual_custom_where=None, 
                        actual_custom_order=None):
        """Create the actual decorator function."""
        
        def decorator(func):
            # Now we have the actual function to decorate
            import inspect
            from functools import wraps
            
            # Get the return type annotation for automatic conversion
            try:
                type_hints = get_type_hints(func)
                return_type = type_hints.get('return')
            except (NameError, AttributeError):
                return_type = None
            
            # Check if this is a resolver field (has model_class parameter)
            if actual_model_class is not None:
                # This is a resolver field - create and use BerryQL resolver
                from .factory import BerryQLFactory
                
                # Extract strawberry type from return type annotation
                strawberry_type = None
                if return_type:
                    # Handle List[SomeType] - extract the inner type
                    origin = get_origin(return_type)
                    if origin is list:
                        args = get_args(return_type)
                        if args and len(args) > 0:
                            inner_type = args[0]
                            # Validate that the inner type is a proper Strawberry type
                            if hasattr(inner_type, '__strawberry_definition__') or hasattr(inner_type, '__strawberry_type__'):
                                strawberry_type = inner_type
                            else:
                                import logging
                                logger = logging.getLogger(__name__)
                                logger.error(f"Invalid inner type in List annotation for {func.__name__}: {inner_type}")
                                logger.error(f"Return type was: {return_type}")
                                raise ValueError(f"Return type List[{inner_type}] is not a valid Strawberry type for {func.__name__}")
                    else:
                        # Check if it's a valid Strawberry type
                        if hasattr(return_type, '__strawberry_definition__') or hasattr(return_type, '__strawberry_type__'):
                            strawberry_type = return_type
                        else:
                            import logging
                            logger = logging.getLogger(__name__)
                            logger.error(f"Invalid return type annotation for {func.__name__}: {return_type}")
                            logger.error(f"Expected a Strawberry type or List[StrawberryType]")
                            raise ValueError(f"Return type {return_type} is not a valid Strawberry type for {func.__name__}")
                
                if strawberry_type is None:
                    raise ValueError(f"Could not determine strawberry type from return annotation of {func.__name__}. "
                                   f"Make sure the return type is annotated as List[YourStrawberryType] or YourStrawberryType")
                
                # Create the BerryQL resolver using the factory
                factory = BerryQLFactory()
                resolver = factory.create_berryql_resolver(
                    strawberry_type=strawberry_type,
                    model_class=actual_model_class,
                    custom_fields=actual_custom_fields,
                    custom_where=actual_custom_where,
                    custom_order=actual_custom_order
                )
                
                # Extract the original function's signature for the resolver wrapper
                import inspect as insp
                sig = insp.signature(func)
                
                # Create a new signature removing only internal parameters (db, params)
                new_params = []
                for param_name, param in sig.parameters.items():
                    if param_name in ['db', 'database', 'session', 'params']:
                        # Skip internal parameters - they'll be injected via context or built internally
                        continue
                    new_params.append(param)
                
                # Create new signature without internal parameters
                new_sig = sig.replace(parameters=new_params)
                
                # Create the new function with the modified signature
                async def new_resolver_func(self, info: strawberry.Info, **graphql_kwargs):
                    # Build the complete arguments for the original function
                    original_kwargs = {}
                    
                    # Add all GraphQL arguments
                    original_kwargs.update(graphql_kwargs)
                    
                    # Add info
                    original_kwargs['info'] = info
                    
                    # Add database session from context
                    db_value = None
                    if hasattr(info.context, 'get'):
                        db_value = (info.context.get('db_session') or 
                                  info.context.get('db') or 
                                  info.context.get('database') or
                                  info.context.get('session'))
                    elif hasattr(info.context, 'db_session'):
                        db_value = info.context.db_session
                    elif hasattr(info.context, 'db'):
                        db_value = info.context.db
                    
                    if db_value:
                        original_kwargs['db'] = db_value
                    
                    # Prepare resolver parameters - pass all arguments to the resolver
                    resolver_params = {
                        'db': db_value
                    }
                    
                    # Build GraphQLQueryParams from the GraphQL arguments
                    from .factory import GraphQLQueryParams
                    
                    # Build where conditions from filter parameters
                    where_conditions = {}
                    other_params = {}
                    
                    for param_name, value in original_kwargs.items():
                        if param_name not in ['info', 'db', 'database', 'session'] and value is not None:
                            # Handle filter parameters (like name_filter -> name LIKE %value%)
                            if param_name.endswith('_filter'):
                                field_name = param_name[:-7]  # Remove '_filter' suffix
                                where_conditions[field_name] = {'like': f'%{value}%'}
                            else:
                                other_params[param_name] = value
                    
                    # Build GraphQLQueryParams with where conditions and other parameters
                    query_params = GraphQLQueryParams(
                        where=where_conditions if where_conditions else None,
                        limit=other_params.get('limit'),
                        offset=other_params.get('offset'),
                        order_by=other_params.get('order_by')
                    )
                    
                    # Set the params in resolver_params
                    resolver_params['params'] = query_params
                    
                    # Also pass any other parameters that aren't handled specially
                    for param_name, value in other_params.items():
                        if param_name not in ['limit', 'offset', 'order_by']:
                            resolver_params[param_name] = value
                    
                    # Call the resolver with extracted parameters
                    return await resolver(info=info, **resolver_params)
                
                # Copy all essential metadata from the original function
                new_resolver_func.__name__ = func.__name__
                new_resolver_func.__qualname__ = func.__qualname__
                new_resolver_func.__module__ = func.__module__
                new_resolver_func.__doc__ = func.__doc__
                
                # Most importantly, copy the return type annotation exactly
                # This is crucial for Strawberry to understand the correct type
                new_resolver_func.__annotations__ = func.__annotations__.copy()
                
                # Remove any db-related and params parameters from annotations
                for param_name in ['db', 'database', 'session', 'params']:
                    if param_name in new_resolver_func.__annotations__:
                        del new_resolver_func.__annotations__[param_name]
                
                # Also preserve the modified signature
                new_resolver_func.__signature__ = new_sig
                
                # Clear any callable-related attributes that might confuse Strawberry
                if hasattr(new_resolver_func, '__wrapped__'):
                    delattr(new_resolver_func, '__wrapped__')
                
                return new_resolver_func
            
            else:
                # This is a relationship field - use resolved data
                # Determine the relationship name from the function name
                relationship_name = func.__name__
                
                @wraps(func)
                async def resolved_data_func(self, info: strawberry.Info, *args, **kwargs):
                    """Return pre-resolved relationship data."""
                    return get_resolved_field_data(self, info, relationship_name)
                
                return resolved_data_func
        
        return decorator
    
    # Handle case where decorator is used without parentheses: @berryql.field
    if func_or_converter is not None and callable(func_or_converter) and converter is None and model_class is None:
        # Direct decoration: @berryql_field
        return create_decorator()(func_or_converter)
    else:
        # Parametrized decoration: @berryql_field(...) 
        return create_decorator(
            actual_converter=func_or_converter or converter,
            actual_model_class=model_class,
            actual_custom_fields=custom_fields,
            actual_custom_where=custom_where,
            actual_custom_order=custom_order
        )


def custom_field(*args, **kwargs):
    """Decorator for custom field methods that return SQLAlchemy expressions."""
    # For backward compatibility, delegate to berryql_field
    return berryql_field(*args, **kwargs)


# Create a berryql namespace object
class BerryQLNamespace:
    """Namespace object for BerryQL decorators."""
    
    @property
    def field(self):
        """Property that returns the berryql_field decorator."""
        def decorator_caller(*args, **kwargs):
            # Handle both @berryql.field and @berryql.field()
            if len(args) == 1 and callable(args[0]) and not kwargs:
                # Direct decoration: @berryql.field
                return berryql_field(args[0])
            else:
                # Called with arguments: @berryql.field(...) 
                return berryql_field(*args, **kwargs)
        return decorator_caller

berryql = BerryQLNamespace()
