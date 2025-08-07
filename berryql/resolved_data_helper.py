"""
Helper utilities for working with resolved data in GraphQL types.
"""

from typing import List, TypeVar, Any, Optional, Callable, Union, Dict, Type, get_origin, get_args, get_type_hints
import strawberry
from functools import wraps
from sqlalchemy.sql import ColumnElement

T = TypeVar('T')

# Custom string class that has isoformat method for GraphQL datetime serialization
class DateTimeString(str):
    """String subclass that has isoformat method for GraphQL datetime serialization."""
    def isoformat(self):
        """Return the string as-is for GraphQL datetime serialization."""
        return str(self)


def convert_json_to_strawberry_instances(json_data: List[Dict], strawberry_type: Type) -> List[Any]:
    """
    Efficiently convert JSON data to Strawberry type instances.
    
    Args:
        json_data: List of dictionaries from JSON
        strawberry_type: The Strawberry GraphQL type to convert to
        
    Returns:
        List of Strawberry type instances
    """
    if not json_data:
        return []
    
    # Cache field type analysis for performance
    if not hasattr(convert_json_to_strawberry_instances, '_field_cache'):
        convert_json_to_strawberry_instances._field_cache = {}
    
    type_key = strawberry_type.__name__
    if type_key not in convert_json_to_strawberry_instances._field_cache:
        # Analyze field types once per strawberry type
        strawberry_fields = getattr(strawberry_type, '__annotations__', {})
        datetime_fields = set()
        
        for field_name, field_type in strawberry_fields.items():
            # Check for datetime fields by various indicators
            is_datetime = False
            
            # Direct datetime type
            if isinstance(field_type, type) and field_type.__name__ == 'datetime':
                is_datetime = True
            # Datetime class name check
            elif hasattr(field_type, '__name__') and 'datetime' in field_type.__name__.lower():
                is_datetime = True
            # Check for Optional[datetime] or Union[datetime, None] types
            elif hasattr(field_type, '__origin__'):
                from typing import Union, get_args
                if field_type.__origin__ is Union:
                    args = get_args(field_type)
                    for arg in args:
                        if (isinstance(arg, type) and arg.__name__ == 'datetime') or \
                           (hasattr(arg, '__name__') and 'datetime' in arg.__name__.lower()):
                            is_datetime = True
                            break
            # Field name pattern matching (covers min_created_at, max_created_at, etc.)
            elif any(pattern in field_name for pattern in ['created_at', 'updated_at', 'deleted_at', 'published_at', 'timestamp']):
                is_datetime = True
            
            if is_datetime:
                datetime_fields.add(field_name)
        
        convert_json_to_strawberry_instances._field_cache[type_key] = datetime_fields
    
    datetime_fields = convert_json_to_strawberry_instances._field_cache[type_key]
    
    # Convert data efficiently
    strawberry_instances = []
    if datetime_fields:
        # Only do field conversion if there are datetime fields
        for item_data in json_data:
            if isinstance(item_data, dict):
                # Separate constructor fields from relationship/method fields
                constructor_data = {}
                resolved_data = {}
                
                # Get type hints to understand which fields are constructor parameters
                type_hints = getattr(strawberry_type, '__annotations__', {})
                
                for field_name, field_value in item_data.items():
                    # Check if this field is a method (relationship field)
                    is_method = hasattr(strawberry_type, field_name) and callable(getattr(strawberry_type, field_name))
                    
                    # Check if this field exists in type hints (constructor parameter)
                    is_constructor_param = field_name in type_hints
                    
                    # For strawberry fields with @strawberry.field decorator:
                    # - They appear in both type_hints AND as callable methods
                    # - They should be treated as resolved data, not constructor parameters
                    if is_method:
                        # This is a relationship/method field - put in resolved data
                        if '_resolved' not in resolved_data:
                            resolved_data['_resolved'] = {}
                        
                        # For nested relationship data, we need to convert lists of dictionaries
                        # to proper Strawberry instances recursively
                        if isinstance(field_value, list) and field_value and isinstance(field_value[0], dict):
                            # This is a nested relationship field - we need to determine the target type
                            # and convert the list of dictionaries to Strawberry instances
                            
                            # Try to get the target type from the field annotation
                            target_type = None
                            if hasattr(strawberry_type, '__annotations__') and field_name in strawberry_type.__annotations__:
                                field_annotation = strawberry_type.__annotations__[field_name]
                                # Handle StrawberryAnnotation objects (from @strawberry.field decorated methods)
                                if hasattr(field_annotation, 'annotation'):
                                    field_annotation = field_annotation.annotation
                                    
                                # Handle List[TargetType] annotations
                                from typing import get_origin, get_args
                                if get_origin(field_annotation) is list:
                                    args = get_args(field_annotation)
                                    if args and len(args) > 0:
                                        target_type = args[0]
                            
                            # If we found a target type, convert the nested data recursively
                            if target_type and hasattr(target_type, '__strawberry_definition__'):
                                # Recursively convert nested relationship data
                                converted_instances = convert_json_to_strawberry_instances(field_value, target_type)
                                resolved_data['_resolved'][field_name] = converted_instances
                            else:
                                # Fallback - store the raw data
                                resolved_data['_resolved'][field_name] = field_value
                        else:
                            # Not a nested relationship or empty - store as-is
                            resolved_data['_resolved'][field_name] = field_value
                    elif is_constructor_param:
                        # This is a constructor parameter
                        if field_name in datetime_fields and isinstance(field_value, str):
                            constructor_data[field_name] = DateTimeString(field_value)
                        else:
                            constructor_data[field_name] = field_value
                    # If it's neither, skip it (shouldn't happen in well-formed data)
                
                # Create the instance with constructor data only
                instance = strawberry_type(**constructor_data)
                
                # Set resolved data if any
                if resolved_data:
                    for attr_name, attr_value in resolved_data.items():
                        setattr(instance, attr_name, attr_value)
                
                strawberry_instances.append(instance)
    else:
        # No datetime fields, but still need to separate constructor from resolved data
        for item_data in json_data:
            if isinstance(item_data, dict):
                # Separate constructor fields from relationship/method fields
                constructor_data = {}
                resolved_data = {}
                
                # Get type hints to understand which fields are constructor parameters
                type_hints = getattr(strawberry_type, '__annotations__', {})
                
                for field_name, field_value in item_data.items():
                    # Check if this field is a method (relationship field)
                    is_method = hasattr(strawberry_type, field_name) and callable(getattr(strawberry_type, field_name))
                    
                    # Check if this field exists in type hints (constructor parameter)
                    is_constructor_param = field_name in type_hints
                    
                    # For strawberry fields with @strawberry.field decorator:
                    # - They appear in both type_hints AND as callable methods
                    # - They should be treated as resolved data, not constructor parameters
                    if is_method:
                        # This is a relationship/method field - put in resolved data
                        if '_resolved' not in resolved_data:
                            resolved_data['_resolved'] = {}
                        
                        # For nested relationship data, we need to convert lists of dictionaries
                        # to proper Strawberry instances recursively
                        if isinstance(field_value, list) and field_value and isinstance(field_value[0], dict):
                            # This is a nested relationship field - we need to determine the target type
                            # and convert the list of dictionaries to Strawberry instances
                            
                            # Try to get the target type from the field annotation
                            target_type = None
                            if hasattr(strawberry_type, '__annotations__') and field_name in strawberry_type.__annotations__:
                                field_annotation = strawberry_type.__annotations__[field_name]
                                # Handle StrawberryAnnotation objects (from @strawberry.field decorated methods)
                                if hasattr(field_annotation, 'annotation'):
                                    field_annotation = field_annotation.annotation
                                    
                                # Handle List[TargetType] annotations
                                from typing import get_origin, get_args
                                if get_origin(field_annotation) is list:
                                    args = get_args(field_annotation)
                                    if args and len(args) > 0:
                                        target_type = args[0]
                            
                            # If we found a target type, convert the nested data recursively
                            if target_type and hasattr(target_type, '__strawberry_definition__'):
                                # Recursively convert nested relationship data
                                converted_instances = convert_json_to_strawberry_instances(field_value, target_type)
                                resolved_data['_resolved'][field_name] = converted_instances
                            else:
                                # Fallback - store the raw data
                                resolved_data['_resolved'][field_name] = field_value
                        else:
                            # Not a nested relationship or empty - store as-is
                            resolved_data['_resolved'][field_name] = field_value
                    elif is_constructor_param:
                        # This is a constructor parameter
                        constructor_data[field_name] = field_value
                    # If it's neither, skip it (shouldn't happen in well-formed data)
                
                # Create the instance with constructor data only
                instance = strawberry_type(**constructor_data)
                
                # Set resolved data if any
                if resolved_data:
                    for attr_name, attr_value in resolved_data.items():
                        setattr(instance, attr_name, attr_value)
                
                strawberry_instances.append(instance)
    
    return strawberry_instances


def find_relationship_name_by_type(parent_model_class: Type, target_strawberry_type: Type) -> Optional[str]:
    """
    Find the SQLAlchemy relationship name by matching the target type.
    
    Args:
        parent_model_class: The SQLAlchemy model class (e.g., User)
        target_strawberry_type: The Strawberry GraphQL type (e.g., PostType)
    
    Returns:
        The relationship name (e.g., 'posts') or None if not found
    """
    import logging
    logger = logging.getLogger(__name__)
    
    if not parent_model_class:
        return None
    
    try:
        # Map common Strawberry type names to SQLAlchemy model names
        # This assumes a naming convention like PostType -> Post, UserType -> User, etc.
        strawberry_type_name = getattr(target_strawberry_type, '__name__', str(target_strawberry_type))
        
        # Remove 'Type' suffix if present
        if strawberry_type_name.endswith('Type'):
            model_name = strawberry_type_name[:-4]  # Remove 'Type'
        else:
            model_name = strawberry_type_name
        
        # Look through all relationships on the parent model
        from sqlalchemy import inspect
        mapper = inspect(parent_model_class)
        
        for relationship_name, relationship_property in mapper.relationships.items():
            # Get the target model class from the relationship
            target_model_class = relationship_property.mapper.class_
            target_model_name = target_model_class.__name__
            
            # Check if this relationship points to our target model
            if target_model_name == model_name:
                logger.debug(f"Found relationship '{relationship_name}' from {parent_model_class.__name__} to {target_model_name}")
                return relationship_name
        
        logger.warning(f"No relationship found from {parent_model_class.__name__} to model matching {strawberry_type_name}")
        return None
        
    except Exception as e:
        logger.error(f"Error finding relationship name: {e}")
        return None


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
    custom_fields: Optional[Dict[str, callable]] = None,
    custom_where: Optional[Union[Dict[str, Any], callable]] = None,
    custom_order: Optional[List[str]] = None,
    **parameter_mappings  # Any named parameter will map to where conditions
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
        custom_fields: Dict mapping {field_name: query_builder}
        custom_where: where_conditions_or_function (simplified, no strawberry type key needed)
        custom_order: default_order_list (simplified, no strawberry type key needed)
        **parameter_mappings: Named parameters that map GraphQL parameters to where conditions
                            Format: parameter_name={'field': {'operator': 'value'}} or callable
        
    Example (resolver field with parameter mappings):
        @strawberry.field
        @berryql.field(
            model_class=User,
            name_filter={'name': {'like': lambda value: f'%{value}%'}},
            status_filter={'status': {'eq': lambda value: value}},
            custom_order={UserType: ['created_at desc']}
        )
        async def users(
            self, 
            info: strawberry.Info, 
            db: AsyncSession,
            name_filter: Optional[str] = None,
            status_filter: Optional[str] = None,
            limit: Optional[int] = None,
            offset: Optional[int] = None
        ) -> List[UserType]:
            pass  # Implementation handled by decorator
    """
    
    def create_decorator(actual_converter=None, actual_model_class=None, 
                        actual_custom_fields=None, actual_custom_where=None, 
                        actual_custom_order=None, actual_parameter_mappings=None):
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
                return_single = False  # Determine if we should return a single object vs list
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
                                return_single = False  # List return type
                            else:
                                import logging
                                logger = logging.getLogger(__name__)
                                logger.error(f"Invalid inner type in List annotation for {func.__name__}: {inner_type}")
                                logger.error(f"Return type was: {return_type}")
                                raise ValueError(f"Return type List[{inner_type}] is not a valid Strawberry type for {func.__name__}")
                    elif origin is Union:
                        # Handle Optional[SomeType] which is Union[SomeType, type(None)]
                        args = get_args(return_type)
                        if args and len(args) == 2 and type(None) in args:
                            # This is Optional[SomeType]
                            inner_type = args[0] if args[1] is type(None) else args[1]
                            if hasattr(inner_type, '__strawberry_definition__') or hasattr(inner_type, '__strawberry_type__'):
                                strawberry_type = inner_type
                                return_single = True  # Single object return type (can be None)
                            else:
                                import logging
                                logger = logging.getLogger(__name__)
                                logger.error(f"Invalid inner type in Optional annotation for {func.__name__}: {inner_type}")
                                logger.error(f"Return type was: {return_type}")
                                raise ValueError(f"Return type Optional[{inner_type}] is not a valid Strawberry type for {func.__name__}")
                        else:
                            import logging
                            logger = logging.getLogger(__name__)
                            logger.error(f"Unsupported Union type for {func.__name__}: {return_type}")
                            raise ValueError(f"Union types other than Optional are not supported for {func.__name__}")
                    else:
                        # Check if it's a valid Strawberry type
                        if hasattr(return_type, '__strawberry_definition__') or hasattr(return_type, '__strawberry_type__'):
                            strawberry_type = return_type
                            return_single = True  # Single object return type
                        else:
                            import logging
                            logger = logging.getLogger(__name__)
                            logger.error(f"Invalid return type annotation for {func.__name__}: {return_type}")
                            logger.error(f"Expected a Strawberry type, List[StrawberryType], or Optional[StrawberryType]")
                            raise ValueError(f"Return type {return_type} is not a valid Strawberry type for {func.__name__}")
                
                if strawberry_type is None:
                    raise ValueError(f"Could not determine strawberry type from return annotation of {func.__name__}. "
                                   f"Make sure the return type is annotated as List[YourStrawberryType], YourStrawberryType, or Optional[YourStrawberryType]")
                
                # Determine if this is a relationship field by checking if we're in a type class
                # and if the return type suggests a relationship to another model
                is_relationship_field = False
                parent_model_class = None
                relationship_name = None
                
                # Get the class that contains this method (the parent GraphQL type)
                if hasattr(func, '__qualname__'):
                    parent_class_name = func.__qualname__.split('.')[0] if '.' in func.__qualname__ else None
                    if parent_class_name:
                        # Try to find the corresponding SQLAlchemy model for the parent type
                        try:
                            # Look for the model class in the same way we determine strawberry types
                            import importlib
                            import sys
                            
                            # Get the calling frame to find the module where types are defined
                            calling_frame = sys._getframe(1)
                            calling_module = calling_frame.f_globals.get('__name__', '')
                            
                            if calling_module:
                                try:
                                    module = importlib.import_module(calling_module)
                                    # Try to find parent model by naming convention
                                    potential_model_names = [
                                        parent_class_name.replace('Type', ''),  # UserType -> User
                                        parent_class_name.replace('GraphQL', ''), # UserGraphQL -> User  
                                        parent_class_name # Exact match
                                    ]
                                    
                                    for model_name in potential_model_names:
                                        if hasattr(module, model_name):
                                            potential_parent_model = getattr(module, model_name)
                                            # Check if it's likely a SQLAlchemy model
                                            if (hasattr(potential_parent_model, '__tablename__') or 
                                                hasattr(potential_parent_model, '__table__')):
                                                parent_model_class = potential_parent_model
                                                break
                                except ImportError:
                                    pass
                                
                            # If we have a parent model, try to find the relationship
                            if parent_model_class and actual_model_class:
                                relationship_name = find_relationship_name_by_type(parent_model_class, actual_model_class)
                                if relationship_name:
                                    is_relationship_field = True
                        except Exception:
                            # If we can't determine the parent model, treat as non-relationship field
                            pass
                
                if is_relationship_field and relationship_name:
                    # For relationship fields, create a special resolver that executes the same query logic
                    # as the standard BerryQL resolver but applies it to load relationship data
                    def create_relationship_resolver():
                        async def relationship_resolver(self, info: strawberry.Info, relationship_name=None, **graphql_kwargs):
                            # Create a factory instance with our custom configurations
                            factory = BerryQLFactory()
                            
                            # Apply custom configurations to the factory - simplified structure
                            if actual_custom_where:
                                # Store as direct mapping to the strawberry type
                                factory._custom_where_config[strawberry_type] = actual_custom_where
                            if actual_custom_order:
                                # Store as direct mapping to the strawberry type  
                                factory._custom_order_config[strawberry_type] = actual_custom_order
                            if actual_custom_fields:
                                # Store custom fields for the strawberry type
                                factory.custom_field_manager.register_custom_fields(strawberry_type, actual_custom_fields)
                            
                            # Get the database session from context
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
                            
                            if not db_value:
                                raise ValueError("Database session not found in GraphQL context")
                            
                            # Build GraphQLQueryParams from the GraphQL arguments - same as for root fields
                            from .factory import GraphQLQueryParams
                            
                            # Build where conditions using explicit parameter mappings (same logic as root fields)
                            where_conditions = {}
                            other_params = {}
                            
                            for param_name, value in graphql_kwargs.items():
                                if value is not None:
                                    # Check if this parameter has an explicit mapping
                                    if actual_parameter_mappings and param_name in actual_parameter_mappings:
                                        mapping = actual_parameter_mappings[param_name]
                                        
                                        if callable(mapping):
                                            # If mapping is a callable, call it with the value
                                            where_clause = mapping(value)
                                            if isinstance(where_clause, dict):
                                                where_conditions.update(where_clause)
                                        elif isinstance(mapping, dict):
                                            # If mapping is a dict, process it
                                            for field_name, condition in mapping.items():
                                                if isinstance(condition, dict):
                                                    # Direct field condition like {'like': '%value%'}
                                                    if any(callable(v) for v in condition.values()):
                                                        # Contains callables - process them
                                                        processed_condition = {}
                                                        for op, op_value in condition.items():
                                                            if callable(op_value):
                                                                processed_condition[op] = op_value(value)
                                                            else:
                                                                processed_condition[op] = op_value
                                                        where_conditions[field_name] = processed_condition
                                                    else:
                                                        # Static condition
                                                        where_conditions[field_name] = condition
                                                elif callable(condition):
                                                    # Field condition is a callable
                                                    where_conditions[field_name] = condition(value)
                                                else:
                                                    # Simple field condition
                                                    where_conditions[field_name] = {'eq': condition}
                                    else:
                                        # No explicit mapping - treat as other parameter
                                        other_params[param_name] = value
                            
                            # Build GraphQLQueryParams with where conditions and other parameters
                            query_params = GraphQLQueryParams(
                                where=where_conditions if where_conditions else None,
                                limit=other_params.get('limit'),
                                offset=other_params.get('offset'),
                                order_by=other_params.get('order_by')
                            )
                            
                            # Now execute the same unified query logic but for the relationship
                            # We need to constrain the query to this specific parent instance
                            
                            # Get the parent's primary key value
                            parent_id = getattr(self, 'id', None)
                            if parent_id is None:
                                raise ValueError(f"Parent instance does not have 'id' attribute for relationship {relationship_name}")
                            
                            # Build a constrained query that only fetches related items for this parent
                            # Add the relationship constraint to the where conditions
                            relationship_where = where_conditions.copy() if where_conditions else {}
                            
                            # Get the foreign key field name for the relationship
                            from sqlalchemy import inspect
                            parent_mapper = inspect(parent_model_class)
                            if relationship_name in parent_mapper.relationships:
                                relationship_prop = parent_mapper.relationships[relationship_name]
                                # Get the foreign key column on the target model that points back to parent
                                foreign_key_columns = [col for col in relationship_prop.remote_side or relationship_prop.local_columns]
                                if relationship_prop.direction.name == 'ONETOMANY':
                                    # For one-to-many, the foreign key is on the target model
                                    target_fk_attr = None
                                    for local_col in relationship_prop.local_columns:
                                        for remote_col in relationship_prop.remote_side:
                                            if local_col.table == parent_mapper.local_table:
                                                # Find the attribute name for the foreign key on target model
                                                target_mapper = inspect(actual_model_class)
                                                for attr_name, attr in target_mapper.attrs.items():
                                                    if hasattr(attr, 'columns') and remote_col in attr.columns:
                                                        target_fk_attr = attr_name
                                                        break
                                    
                                    if target_fk_attr:
                                        relationship_where[target_fk_attr] = {'eq': parent_id}
                            
                            # Update query params with relationship constraint
                            query_params_with_relationship = GraphQLQueryParams(
                                where=relationship_where,
                                limit=query_params.limit,
                                offset=query_params.offset,
                                order_by=query_params.order_by
                            )
                            
                            # Execute the unified query for the target model with relationship constraints
                            result = await factory._execute_unified_query(
                                strawberry_type=strawberry_type,
                                model_class=actual_model_class,
                                db=db_value,
                                info=info,
                                params=query_params_with_relationship,
                                is_root=False  # This is a relationship query, not root
                            )
                            
                            # The result from _execute_unified_query for nested queries is raw JSON
                            # We need to parse it into proper strawberry type instances
                            if isinstance(result, str):
                                # Parse JSON string result
                                import json
                                try:
                                    json_data = json.loads(result) if result and result != '[]' else []
                                except (json.JSONDecodeError, TypeError):
                                    json_data = []
                            elif isinstance(result, list):
                                json_data = result
                            else:
                                json_data = []
                            
                            # Convert JSON data to strawberry type instances
                            converted_instances = convert_json_to_strawberry_instances(json_data, strawberry_type)
                            
                            # Return single object or list based on return type
                            if return_single:
                                return converted_instances[0] if converted_instances else None
                            else:
                                return converted_instances
                        
                        return relationship_resolver
                    
                    resolver = create_relationship_resolver()
                else:
                    # For non-relationship fields, use the standard factory resolver
                    factory = BerryQLFactory()
                    resolver = factory.create_berryql_resolver(
                        strawberry_type=strawberry_type,
                        model_class=actual_model_class,
                        custom_fields=actual_custom_fields,
                        custom_where=actual_custom_where,
                        custom_order=actual_custom_order,
                        return_single=return_single
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
                    # First, try to call the original function to see if it has custom logic
                    try:
                        # Build the complete arguments for the original function
                        original_kwargs = {}
                        
                        # Add all GraphQL arguments
                        original_kwargs.update(graphql_kwargs)
                        
                        # Add info
                        original_kwargs['info'] = info
                        
                        # Add database session from context if the original function expects it
                        if 'db' in sig.parameters or 'database' in sig.parameters or 'session' in sig.parameters:
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
                                if 'db' in sig.parameters:
                                    original_kwargs['db'] = db_value
                                elif 'database' in sig.parameters:
                                    original_kwargs['database'] = db_value
                                elif 'session' in sig.parameters:
                                    original_kwargs['session'] = db_value
                        
                        # Filter kwargs to only include parameters the original function expects
                        filtered_kwargs = {}
                        for param_name, value in original_kwargs.items():
                            if param_name in sig.parameters:
                                filtered_kwargs[param_name] = value
                        
                        # Call the original function
                        custom_result = await func(self, **filtered_kwargs)
                        
                        # If the original function returned something other than None, use it
                        if custom_result is not None:
                            return custom_result
                        
                    except Exception as e:
                        # If the original function raises an exception, let it propagate
                        # This allows custom validation, authorization, etc.
                        raise e
                    
                    # If we reach here, the original function returned None or passed
                    # Fall back to BerryQL resolver
                    
                    # Build the complete arguments for the BerryQL resolver
                    berryql_kwargs = {}
                    
                    # Add all GraphQL arguments
                    berryql_kwargs.update(graphql_kwargs)
                    
                    # Add info
                    berryql_kwargs['info'] = info
                    
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
                        berryql_kwargs['db'] = db_value
                    
                    # Prepare resolver parameters - pass all arguments to the resolver
                    resolver_params = {
                        'db': db_value
                    }
                    
                    # Build GraphQLQueryParams from the GraphQL arguments
                    from .factory import GraphQLQueryParams
                    
                    # Build where conditions using explicit parameter mappings
                    where_conditions = {}
                    other_params = {}
                    
                    for param_name, value in berryql_kwargs.items():
                        if param_name not in ['info', 'db', 'database', 'session'] and value is not None:
                            # Check if this parameter has an explicit mapping
                            if actual_parameter_mappings and param_name in actual_parameter_mappings:
                                mapping = actual_parameter_mappings[param_name]
                                
                                if callable(mapping):
                                    # If mapping is a callable, call it with the value
                                    where_clause = mapping(value)
                                    if isinstance(where_clause, dict):
                                        where_conditions.update(where_clause)
                                elif isinstance(mapping, dict):
                                    # If mapping is a dict, process it
                                    for field_name, condition in mapping.items():
                                        if isinstance(condition, dict):
                                            # Direct field condition like {'like': '%value%'}
                                            if any(callable(v) for v in condition.values()):
                                                # Contains callables - process them
                                                processed_condition = {}
                                                for op, op_value in condition.items():
                                                    if callable(op_value):
                                                        processed_condition[op] = op_value(value)
                                                    else:
                                                        processed_condition[op] = op_value
                                                where_conditions[field_name] = processed_condition
                                            else:
                                                # Static condition
                                                where_conditions[field_name] = condition
                                        elif callable(condition):
                                            # Field condition is a callable
                                            where_conditions[field_name] = condition(value)
                                        else:
                                            # Simple field condition
                                            where_conditions[field_name] = {'eq': condition}
                            else:
                                # No explicit mapping - treat as other parameter
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
                    if is_relationship_field and relationship_name:
                        # For relationship fields, pass self to the resolver
                        return await resolver(self, info, relationship_name)
                    else:
                        # For root fields, don't pass self
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
                # Determine the relationship name from the return type instead of function name
                relationship_name = func.__name__  # Fallback to function name
                
                # Try to determine relationship name from return type
                if return_type:
                    # Extract the target strawberry type from List[TargetType] or TargetType
                    target_strawberry_type = None
                    origin = get_origin(return_type)
                    if origin is list:
                        args = get_args(return_type)
                        if args and len(args) > 0:
                            target_strawberry_type = args[0]
                    else:
                        target_strawberry_type = return_type
                    
                    # If we have a target type, try to map it to a relationship name
                    if target_strawberry_type and hasattr(target_strawberry_type, '__name__'):
                        type_name = target_strawberry_type.__name__
                        
                        # Simple mapping: PostType -> posts, CommentType -> comments, etc.
                        if type_name.endswith('Type'):
                            base_name = type_name[:-4].lower()  # PostType -> post
                            # Try plural form first (most common)
                            potential_relationship_name = base_name + 's'  # post -> posts
                            
                            import logging
                            logger = logging.getLogger(__name__)
                            logger.debug(f"Inferring relationship name '{potential_relationship_name}' for field '{func.__name__}' "
                                       f"based on return type {type_name}")
                            
                            relationship_name = potential_relationship_name
                
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
            actual_custom_order=custom_order,
            actual_parameter_mappings=parameter_mappings
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
