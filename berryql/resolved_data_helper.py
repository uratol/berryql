"""
Helper utilities for working with resolved data in GraphQL types.

This module provides utilities to encapsulate the resolved data pattern,
eliminating code duplication across GraphQL type methods.

The resolved data pattern involves:
1. Getting pre-resolved relationship data from the `_resolved` attribute
2. Using the GraphQL field name (which could be an alias) to lookup the data
3. Returning the appropriate subset of resolved data

This module provides three main approaches:

1. Helper Function Approach (Recommended for existing code):
   ```python
   @strawberry.field
   async def tasks(self, info: strawberry.Info, ...) -> List[TaskType]:
       return get_resolved_field_data(self, info, 'tasks')
   ```

2. Decorator Approach (Clean for new code):
   ```python
   @strawberry.field
   @berryql_field
   async def tasks(self, info: strawberry.Info, ...) -> List[TaskType]:
       pass  # Implementation handled by decorator
   ```

3. Mixin Approach (For types that need resolved data methods):
   ```python
   class ProjectType(ResolvedDataMixin):
       @strawberry.field
       async def tasks(self, info: strawberry.Info, ...) -> List[TaskType]:
           return self.get_resolved_field_data(info, 'tasks')
   ```
"""

from typing import List, TypeVar, Any, Optional, Callable, Union
import strawberry
from functools import wraps
from sqlalchemy.sql import ColumnElement

T = TypeVar('T')


def get_resolved_field_data(instance: Any, info: strawberry.Info, relationship_name: str) -> List[T]:
    """
    Helper function to extract resolved data for a specific relationship field.
    
    This encapsulates the common pattern used in GraphQL type methods to retrieve
    pre-resolved relationship data from lateral joins.
    
    Args:
        instance: The GraphQL type instance (usually 'self' in a field method)
        info: Strawberry GraphQL info object containing field information
        relationship_name: The name of the relationship (e.g., 'tasks', 'locations', 'characters')
        
    Returns:
        List of resolved objects for the requested field
        
    Example:
        @strawberry.field
        async def tasks(self, info: strawberry.Info, ...) -> List[TaskType]:
            return get_resolved_field_data(self, info, 'tasks')
    """
    # Get the unified resolved data dictionary
    resolved_data = getattr(instance, '_resolved', {})
    
    # The field name in the GraphQL query (which could be an alias)
    field_name = info.field_name
    
    # Debug logging (remove in production)
    # import logging
    # logger = logging.getLogger("app.graphql.resolved_data_helper")
    # logger.info(f"get_resolved_field_data called: relationship_name={relationship_name}, field_name={field_name}")
    # logger.info(f"_resolved data keys: {list(resolved_data.keys())}")
    # logger.info(f"Full _resolved data: {resolved_data}")
    
    # In the nested structure, data is organized by resolved field name, then by display name
    relationship_data = resolved_data.get(relationship_name, {})
    result = relationship_data.get(field_name, [])
    
    # Debug logging (remove in production)
    # logger.info(f"relationship_data: {relationship_data}")
    # logger.info(f"final result: {result}")
    
    # Ensure we always return a list, never None
    return result if result is not None else []


class ResolvedDataMixin:
    """
    Mixin class that provides convenient methods for accessing resolved data.
    
    This can be used to add resolved data functionality to GraphQL types.
    """
    
    def get_resolved_field_data(self, info: strawberry.Info, relationship_name: str) -> List[Any]:
        """
        Instance method version of get_resolved_field_data.
        
        Args:
            info: Strawberry GraphQL info object
            relationship_name: The name of the relationship
            
        Returns:
            List of resolved objects for the requested field
        """
        return get_resolved_field_data(self, info, relationship_name)


def berryql_field(func_or_converter: Optional[Callable] = None, *, converter: Optional[Callable] = None):
    """
    Unified decorator that handles both relationship and custom fields automatically.
    
    This decorator inspects the method implementation to determine the behavior:
    - If method body is just 'pass': treats it as a relationship field, gets data from _resolved dict
    - If method returns SQLAlchemy expression: treats it as a custom field, stores query builder for SQL generation
    
    For custom fields, the decorator automatically converts raw data based on the return type annotation:
    - Dict data → Strawberry type instances (automatically creates instances)
    - Primitive types → direct conversion (int, str, float, bool)
    - Optional/Union types → handles None values gracefully
    
    Args:
        func_or_converter: The method to decorate (when used as @berryql_field) or the converter function (when used with parameters)
        converter: Optional function to convert raw database result to expected type (overrides automatic conversion)
        
    Returns:
        The decorated method that returns resolved data or custom field data
        
    Example (relationship field):
        @strawberry.field
        @berryql.field
        async def tasks(self, info: strawberry.Info, ...) -> List[TaskType]:
            pass  # Relationship data from _resolved
            
    Example (custom field with automatic conversion):
        @strawberry.field  
        @berryql.field
        def tasks_agg(self, info: strawberry.Info) -> Optional[TasksAggType]:
            return build_tasks_agg  # Query builder function - TasksAggType created automatically
            
    Example (custom field with manual converter):
        @strawberry.field  
        @berryql_field(converter=custom_converter)
        def custom_field(self, info: strawberry.Info) -> CustomType:
            return build_custom  # Manual converter overrides automatic conversion
    """
    # Handle case where decorator is used without parentheses: @berryql.field
    if func_or_converter is None:
        # This is when used as @berryql.field (no arguments)
        def decorator_without_args(func: Callable) -> Callable:
            return berryql_field(func, converter=converter)
        return decorator_without_args
    
    # Handle case where decorator is used with a function directly: @berryql_field
    if callable(func_or_converter) and converter is None:
        func = func_or_converter
    else:
        # Handle case with explicit converter: @berryql_field(converter=...)
        def decorator_with_converter(func: Callable) -> Callable:
            return berryql_field(func, converter=func_or_converter or converter)
        return decorator_with_converter
    
    # Now we have the actual function to decorate
    import inspect
    from functools import wraps
    from typing import get_type_hints, get_origin, get_args, Union
    
    # Get the return type annotation for automatic conversion
    try:
        type_hints = get_type_hints(func)
        return_type = type_hints.get('return')
    except (NameError, AttributeError):
        return_type = None
    
    def auto_convert_data(raw_data, target_type):
        """Automatically convert raw data to the target type based on type annotations."""
        if raw_data is None:
            return None
            
        if target_type is None:
            return raw_data
        
        # Handle Union types (like Optional[T] which is Union[T, None])
        origin = get_origin(target_type)
        if origin is Union:
            args = get_args(target_type)
            # For Optional[T], find the non-None type
            non_none_types = [arg for arg in args if arg is not type(None)]
            if len(non_none_types) == 1:
                target_type = non_none_types[0]
            elif raw_data is None:
                return None
        
        # Handle primitive types
        if target_type in (int, float, str, bool):
            try:
                return target_type(raw_data)
            except (ValueError, TypeError):
                return raw_data
        
        # Handle Strawberry types (classes with __strawberry_definition__)
        if hasattr(target_type, '__strawberry_definition__') and isinstance(raw_data, dict):
            try:
                # Create an instance of the Strawberry type from the dict data
                return target_type(**raw_data)
            except (TypeError, ValueError) as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Could not auto-convert {raw_data} to {target_type.__name__}: {e}")
                return raw_data
        
        # For other types, return as-is
        return raw_data
    
    # Analyze the function to determine if it's a relationship or custom field
    try:
        source_lines = inspect.getsourcelines(func)[0]
        # Find where the function body starts (after the colon)
        body_start_idx = -1
        for i, line in enumerate(source_lines):
            # Look for the line with : that ends the function signature
            if ':' in line and not line.strip().startswith('#'):
                body_start_idx = i + 1
                break
        
        if body_start_idx == -1:
            # Fallback: look for any line containing def or async def
            for i, line in enumerate(source_lines):
                if ('def ' in line or 'async def' in line) and not line.strip().startswith('#'):
                    body_start_idx = i + 1
                    break
        
        # Get the body lines
        if body_start_idx >= 0 and body_start_idx < len(source_lines):
            body_lines = source_lines[body_start_idx:]
        else:
            body_lines = []
        
        # Check if the function body is effectively just 'pass'
        body_code = ''.join(body_lines).strip()
        
        # Find the actual body after the function signature
        actual_body_lines = []
        for line in body_lines:
            stripped = line.strip()
            # Skip empty lines
            if not stripped:
                continue
            # Skip function signature continuation (lines with parameters)
            if ('Optional[' in stripped or 
                stripped.endswith(',') or 
                stripped.startswith(('offset:', 'limit:', 'where:', 'order_by:', 'self,', 'info:')) or
                ') ->' in stripped):
                continue
            # This is actual function body
            actual_body_lines.append(line)
        
        actual_body_code = ''.join(actual_body_lines).strip()
        
        # Remove docstrings ("""...""" or '''...''')
        import re
        actual_body_code = re.sub(r'""".*?"""', '', actual_body_code, flags=re.DOTALL)
        actual_body_code = re.sub(r"'''.*?'''", '', actual_body_code, flags=re.DOTALL)
        actual_body_code = actual_body_code.strip()
        
        # Check if what remains is just pass and optional comments
        remaining_lines = [line.strip() for line in actual_body_code.split('\n') if line.strip()]
        pass_line_found = False
        non_comment_non_pass_found = False
        
        for line in remaining_lines:
            if line == 'pass' or line.startswith('pass ') or line.startswith('pass#'):
                pass_line_found = True
            elif not line.startswith('#'):
                non_comment_non_pass_found = True
        
        is_pass_only = pass_line_found and not non_comment_non_pass_found
        
    except (OSError, TypeError) as e:
        # If we can't inspect the source, assume it's not a pass-only function
        is_pass_only = False
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Failed to inspect source for {func.__name__}: {e}")
        logger.info(f"  Assuming is_pass_only = False")
    
    if is_pass_only:
        # This is a relationship field - use the original generic_field behavior
        @wraps(func)
        def wrapper(self, info: strawberry.Info, *args, **kwargs):
            # Use the function name as the relationship name
            relationship_name = func.__name__
            result = get_resolved_field_data(self, info, relationship_name)
            
            # Ensure we always return a list, never None
            final_result = result if result is not None else []
            return final_result
        return wrapper
    else:
        # This is a custom field - the method should return a SQLAlchemy expression or a query builder function
        def query_builder(model_class, requested_fields):
            # Call the original function to get either a SQLAlchemy expression or a query builder function
            try:
                # Create a temporary instance just to call the method
                result = func(None, None)
                
                # Check if the result is a function (query builder) or a SQLAlchemy expression
                if callable(result):
                    # It's a function, call it with the proper parameters
                    return result(model_class, requested_fields)
                else:
                    # It's already a SQLAlchemy expression
                    return result
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error extracting SQLAlchemy expression from {func.__name__}: {e}")
                logger.error(f"Make sure the method returns either:")
                logger.error(f"  1. A SQLAlchemy expression directly, or")  
                logger.error(f"  2. A function that takes (model_class, requested_fields) and returns a SQLAlchemy expression")
                raise
        
        # Store the query builder on the function for the generic resolver to find
        func._custom_query_builder = query_builder
        func._custom_converter = converter
        func._is_custom_field = True
        
        @wraps(func)
        def wrapper(self, info: strawberry.Info, *args, **kwargs):
            # Look for the custom field data in the resolved data
            field_name = func.__name__
            raw_value = None
            
            # For custom fields, we should NOT check hasattr first because that will find the method
            # Instead, check _resolved data structure first
            if hasattr(self, '_resolved') and self._resolved:
                # First try direct access
                raw_value = self._resolved.get(field_name)
                # If we got a nested structure like {'fieldName': {'fieldName': actualData}}, extract the actual data
                if isinstance(raw_value, dict) and field_name in raw_value:
                    raw_value = raw_value[field_name]
                # If not found and it's a nested structure (for compatibility), try the nested format
                elif raw_value is None:
                    nested_data = self._resolved.get(field_name, {})
                    if isinstance(nested_data, dict):
                        raw_value = nested_data.get(field_name)

            # Only check if the custom field data is available directly on the instance as a fallback
            if raw_value is None and hasattr(self, field_name):
                attr_value = getattr(self, field_name)
                # Make sure it's not the method itself
                if not callable(attr_value):
                    raw_value = attr_value
                
            # Apply converter if provided and value exists
            if raw_value is not None:
                if converter:
                    # Use manual converter if provided
                    try:
                        return converter(raw_value)
                    except Exception as e:
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.error(f"Error converting custom field {field_name} with manual converter: {e}")
                        return None
                else:
                    # Use automatic conversion based on return type annotation
                    try:
                        return auto_convert_data(raw_value, return_type)
                    except Exception as e:
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.warning(f"Error auto-converting custom field {field_name}: {e}")
                        return raw_value
                    
            return raw_value
        return wrapper


def custom_field(query_builder: Callable, converter: Optional[Callable] = None):
    """
    Decorator for custom fields that use SQLAlchemy expressions.
    
    This decorator is used to mark methods that should have their data computed
    by a SQLAlchemy expression and included in the main query.
    
    Args:
        query_builder: A function that takes (model_class, requested_fields) and returns a SQLAlchemy expression
        converter: Optional function to convert the raw database result to the expected type
        
    Example:
        def build_tasks_count(model_class, requested_fields):
            return func.count(Task.id).where(Task.project_id == model_class.id)
            
        @strawberry.field
        @custom_field(build_tasks_count)
        def task_count(self, info: strawberry.Info) -> int:
            pass
    """
    def decorator(func: Callable) -> Callable:
        # Store the query builder on the function for the generic resolver to find
        func._custom_query_builder = query_builder
        func._custom_converter = converter
        func._is_custom_field = True
        
        @wraps(func)
        def wrapper(self, info: strawberry.Info, *args, **kwargs):
            # Look for the custom field data in the resolved data
            field_name = func.__name__
            raw_value = None
            
            # Check if the custom field data is available directly on the instance
            if hasattr(self, field_name):
                raw_value = getattr(self, field_name)
                
            # Check in _resolved data structure if not found directly
            if raw_value is None and hasattr(self, '_resolved') and self._resolved:
                raw_value = self._resolved.get(field_name)
                
            # Apply converter if provided and value exists
            if raw_value is not None and converter:
                try:
                    return converter(raw_value)
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(f"Error converting custom field {field_name}: {e}")
                    return None
                    
            return raw_value
        return wrapper
    return decorator


def create_resolved_field_method(relationship_name: str):
    """
    Factory function to create a resolved field method.
    
    This can be used to dynamically create field methods that return resolved data.
    
    Args:
        relationship_name: The name of the relationship to resolve
        
    Returns:
        An async method that returns resolved data
        
    Example:
        tasks_method = create_resolved_field_method('tasks')
    """
    async def resolved_method(
        self,
        info: strawberry.Info,
        offset: Optional[int] = 0,
        limit: Optional[int] = None,
        where: Optional[Any] = None,
        order_by: Optional[str] = None
    ) -> List[Any]:
        """Generated method for resolved field data."""
        return get_resolved_field_data(self, info, relationship_name)
    
    return resolved_method


# Example usage with decorator approach:
#
# @strawberry.field
# @generic_field
# async def tasks(
#     self,
#     info: strawberry.Info,
#     offset: Optional[int] = 0,
#     limit: Optional[int] = None,
#     where: Optional[str] = None,
#     order_by: Optional[str] = None
# ) -> List[TaskType]:
#     """Get tasks with resolved data."""
#     pass  # Implementation is handled by the decorator
#
# # Or using the helper function directly:
# @strawberry.field
# async def tasks(
#     self,
#     info: strawberry.Info,
#     offset: Optional[int] = 0,
#     limit: Optional[int] = None,
#     where: Optional[str] = None,
#     order_by: Optional[str] = None
# ) -> List[TaskType]:
#     """Get tasks with resolved data."""
#     return get_resolved_field_data(self, info, 'tasks')


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
