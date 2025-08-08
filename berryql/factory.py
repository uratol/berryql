"""
BerryQL Factory for GraphQL queries.

Features:
- Modular design with separate classes for different responsibilities
- Improved error handling and validation
- Better code organization and readability
- Comprehensive type hints and documentation
"""

import logging
import json
import re
import dataclasses
import asyncio
from typing import Optional, List, Dict, Any, Type, Set, Union, get_type_hints, cast, TypeVar, Callable, Awaitable
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text, asc, desc, inspect
from sqlalchemy.sql import ColumnElement
from datetime import datetime
from .query_analyzer import query_analyzer
from .database_adapters import get_database_adapter, DatabaseAdapter
from .resolved_data_helper import convert_json_to_strawberry_instances

logger = logging.getLogger(__name__)

# TypeVar for strawberry type annotation
T = TypeVar('T')


class InvalidFieldError(ValueError):
    """Raised when an invalid field is used in where conditions or ordering."""
    pass


class GraphQLQueryParams:
    """Parameters for GraphQL queries supporting filtering, ordering, and pagination."""
    
    def __init__(
        self,
        where: Optional[Union[Dict[str, Any], str]] = None,
        order_by: Optional[Union[List[Dict[str, str]], str]] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None
    ):
        self.where = self._parse_where_parameter(where)
        self.order_by = self._parse_order_by_parameter(order_by)
        self.offset = offset
        self.limit = limit
    
    def _parse_where_parameter(self, where: Optional[Union[Dict[str, Any], str]]) -> Dict[str, Any]:
        """Parse where parameter - can be dict or JSON string."""
        if isinstance(where, str):
            if not where.strip():
                return {}
            try:
                return json.loads(where.strip())
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse where condition '{where}': {e}")
                return {}
        return where or {}
    
    def _parse_order_by_parameter(self, order_by: Optional[Union[List[Dict[str, str]], str]]) -> List[Dict[str, str]]:
        """Parse order_by parameter - can be list of dicts or JSON string."""
        if order_by is None:
            return []
        
        if isinstance(order_by, list):
            return order_by
        
        if isinstance(order_by, str):
            if not order_by.strip():
                return []
            try:
                parsed = json.loads(order_by.strip())
                if isinstance(parsed, list):
                    return parsed
                elif isinstance(parsed, dict):
                    return [parsed]
                else:
                    logger.warning(f"Unexpected order_by format after JSON parsing: {parsed}")
                    return []
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse order_by condition '{order_by}': {e}")
                return []
        
        return []


class FieldMapper:
    """Handles field name mapping between GraphQL and database representations."""
    
    @staticmethod
    def camel_to_snake(name: str) -> str:
        """Convert camelCase GraphQL field names to snake_case database column names."""
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    
    @staticmethod
    def map_graphql_field_to_db_column(field_name: str, strawberry_type: Type = None) -> str:
        """Map GraphQL field name to database column name."""
        if strawberry_type:
            type_hints = get_type_hints(strawberry_type)
            if field_name in type_hints:
                return field_name
            
            snake_case_field = FieldMapper.camel_to_snake(field_name)
            if snake_case_field in type_hints:
                return snake_case_field
        
        return FieldMapper.camel_to_snake(field_name)


class TypeAnalyzer:
    """Analyzes Strawberry types to extract field and relationship information."""
    
    @staticmethod
    def get_required_fields(strawberry_type: Type) -> Set[str]:
        """Get required fields from Strawberry type annotations."""
        required_fields = set()
        
        try:
            if hasattr(strawberry_type, '__strawberry_definition__'):
                for field in strawberry_type.__strawberry_definition__.fields:
                    field_name = field.python_name
                    
                    is_optional = hasattr(field.type, '__class__') and field.type.__class__.__name__ == 'StrawberryOptional'
                    has_no_default = isinstance(field.default, dataclasses._MISSING_TYPE)
                    has_no_default_factory = isinstance(field.default_factory, dataclasses._MISSING_TYPE)
                    
                    if not is_optional and has_no_default and has_no_default_factory and field.init:
                        required_fields.add(field_name)
                            
        except Exception as e:
            logger.warning(f"Could not determine required fields for {strawberry_type.__name__}: {e}")
            required_fields = {'id'}
        
        return required_fields
    
    @staticmethod
    def get_relationship_inner_type(strawberry_type: Type, field_name: str) -> Optional[Type]:
        """Get the inner type for relationship fields by analyzing their return type."""
        type_hints = get_type_hints(strawberry_type)
        
        if field_name in type_hints:
            field_type = type_hints[field_name]
            return TypeAnalyzer._extract_inner_type(field_type)
        
        if hasattr(strawberry_type, field_name):
            field_attr = getattr(strawberry_type, field_name)
            if callable(field_attr) and hasattr(field_attr, '__annotations__'):
                return_type = field_attr.__annotations__.get('return')
                if return_type:
                    return TypeAnalyzer._extract_inner_type(return_type)
        
        return None
    
    @staticmethod
    def _extract_inner_type(field_type):
        """Extract the inner type from List[InnerType] or Optional[List[InnerType]]."""
        if not field_type:
            return None
        
        # Handle Strawberry annotation objects
        if hasattr(field_type, '__class__') and 'strawberry' in str(field_type.__class__):
            if hasattr(field_type, 'annotation'):
                actual_type = field_type.annotation
                return TypeAnalyzer._extract_inner_type(actual_type)
        
        origin = getattr(field_type, '__origin__', None)
        
        # Handle Optional[List[InnerType]]
        if origin is Union:
            args = getattr(field_type, '__args__', ())
            if len(args) == 2 and type(None) in args:
                non_none_type = args[0] if args[1] is type(None) else args[1]
                if getattr(non_none_type, '__origin__', None) is list:
                    list_args = getattr(non_none_type, '__args__', ())
                    return list_args[0] if list_args else None
        
        # Handle List[InnerType]
        elif origin is list:
            args = getattr(field_type, '__args__', ())
            return args[0] if args else None
        
        return None
    
    @staticmethod
    def is_relationship_field(field_type) -> bool:
        """Check if a field type represents a relationship (List type)."""
        origin = getattr(field_type, '__origin__', None)
        
        # Handle Optional[List[...]]
        if origin is Union:
            args = getattr(field_type, '__args__', ())
            if len(args) == 2 and type(None) in args:
                non_none_type = args[0] if args[1] is type(None) else args[1]
                return getattr(non_none_type, '__origin__', None) is list
        
        # Handle List[...]
        return origin is list


class CustomFieldManager:
    """Manages custom field configurations and processing."""
    
    def __init__(self):
        self._custom_fields_config = {}
    
    def register_custom_fields(self, strawberry_type: Type, custom_fields: Dict[str, callable]):
        """Register custom fields for a strawberry type."""
        self._custom_fields_config[strawberry_type] = custom_fields
    
    def is_custom_field(self, strawberry_type: Type, field_name: str) -> bool:
        """Check if a field is a custom field for the given strawberry type."""
        custom_fields = self._custom_fields_config.get(strawberry_type, {})
        
        # Check exact field name
        if field_name in custom_fields:
            return True
        
        # Check snake_case version
        snake_case_field = FieldMapper.camel_to_snake(field_name)
        if snake_case_field in custom_fields:
            return True
        
        # Check for @custom_field decorator
        try:
            if hasattr(strawberry_type, field_name):
                attr = getattr(strawberry_type, field_name)
                if callable(attr) and hasattr(attr, '_is_custom_field') and hasattr(attr, '_custom_query_builder'):
                    return True
        except (AttributeError, TypeError):
            pass
        
        return False
    
    def build_custom_field_columns(
        self,
        strawberry_type: Type,
        requested_fields: Set[str],
        model_class: Type
    ) -> List[ColumnElement]:
        """Build custom field columns using configuration and decorators."""
        custom_columns = []
        custom_fields = self._custom_fields_config.get(strawberry_type, {})
        
        # Scan for methods with @custom_field decorator
        for attr_name in dir(strawberry_type):
            if attr_name.startswith('_'):
                continue
                
            try:
                attr = getattr(strawberry_type, attr_name)
                if callable(attr) and hasattr(attr, '_is_custom_field') and hasattr(attr, '_custom_query_builder'):
                    if attr_name in requested_fields:
                        custom_fields[attr_name] = attr._custom_query_builder
            except (AttributeError, TypeError):
                continue
        
        # Build columns for all custom fields
        for field_name in requested_fields:
            query_builder = None
            custom_field_name = None
            
            if field_name in custom_fields:
                query_builder = custom_fields[field_name]
                custom_field_name = field_name
            else:
                snake_case_field = FieldMapper.camel_to_snake(field_name)
                if snake_case_field in custom_fields:
                    query_builder = custom_fields[snake_case_field]
                    custom_field_name = snake_case_field
            
            if query_builder:
                try:
                    custom_column = query_builder(model_class, requested_fields)
                    if hasattr(custom_column, 'label'):
                        custom_column = custom_column.label(custom_field_name)
                    custom_columns.append(custom_column)
                except Exception as e:
                    logger.error(f"Error building custom field {field_name} for {strawberry_type.__name__}: {e}")
        
        return custom_columns


class QueryConditionProcessor:
    """Processes where conditions and ordering for queries."""
    
    @staticmethod
    def apply_where_conditions(
        query, 
        model_class: Type, 
        where_conditions: Union[Dict[str, Any], str, None]
    ):
        """Apply where conditions to the query with validation."""
        if not where_conditions:
            return query
        
        # Handle string where conditions (JSON)
        if isinstance(where_conditions, str):
            if not where_conditions.strip():
                return query
            try:
                where_conditions = json.loads(where_conditions.strip())
                if not isinstance(where_conditions, dict):
                    raise InvalidFieldError(f"Where conditions must be a JSON object, got: {type(where_conditions).__name__}")
            except json.JSONDecodeError as e:
                raise InvalidFieldError(f"Invalid JSON format in where conditions: {where_conditions}. Error: {e}")
        
        if not isinstance(where_conditions, dict):
            raise InvalidFieldError(f"Where conditions must be a dict or JSON string, got: {type(where_conditions).__name__}")
        
        # Get valid model columns for validation
        inspector = inspect(model_class)
        valid_columns = {col.name for col in inspector.columns}
        
        # Define supported operators
        supported_operators = {'eq', 'ne', 'gt', 'gte', 'lt', 'lte', 'in', 'like', 'ilike'}
        
        for field_name, value in where_conditions.items():
            # Convert camelCase GraphQL field names to snake_case database column names
            db_field_name = FieldMapper.camel_to_snake(field_name)
            
            # Validate field exists on model
            if db_field_name in valid_columns:
                column_name = db_field_name
            elif field_name in valid_columns:
                column_name = field_name
            else:
                raise InvalidFieldError(
                    f"Where field '{field_name}' (converted: '{db_field_name}') not found in model {model_class.__name__}. "
                    f"Valid columns: {sorted(valid_columns)}"
                )
            
            logger.info(f"Processing where condition: {field_name} = {value}, using column: {column_name}")
            column = getattr(model_class, column_name)
                
            # Handle different comparison types
            if isinstance(value, dict):
                for op, op_value in value.items():
                    if op not in supported_operators:
                        raise InvalidFieldError(f"Unsupported operator '{op}'. Supported: {supported_operators}")
                    
                    converted_value = QueryConditionProcessor._convert_value_for_column(column, op_value)
                    query = QueryConditionProcessor._apply_operator(query, column, op, converted_value)
            else:
                if value is None:
                    continue
                    
                converted_value = QueryConditionProcessor._convert_value_for_column(column, value)
                query = query.where(column == converted_value)
        
        return query
    
    @staticmethod
    def _apply_operator(query, column, operator: str, value):
        """Apply a specific operator to a column."""
        if operator == 'eq':
            return query.where(column == value)
        elif operator == 'ne':
            return query.where(column != value)
        elif operator == 'gt':
            return query.where(column > value)
        elif operator == 'gte':
            return query.where(column >= value)
        elif operator == 'lt':
            return query.where(column < value)
        elif operator == 'lte':
            return query.where(column <= value)
        elif operator == 'in':
            return query.where(column.in_(value))
        elif operator == 'like':
            return query.where(column.like(value))
        elif operator == 'ilike':
            return query.where(column.ilike(value))
        else:
            raise InvalidFieldError(f"Unsupported operator: {operator}")
    
    @staticmethod
    def _convert_value_for_column(column, value):
        """Convert a value to the appropriate type for the database column."""
        if value is None:
            return value
            
        column_type = str(column.type).lower()
        
        if any(dt_type in column_type for dt_type in ['timestamp', 'datetime', 'date']):
            if isinstance(value, str):
                try:
                    return datetime.fromisoformat(value.replace('Z', '+00:00'))
                except ValueError:
                    logger.warning(f"Could not parse datetime value: {value}")
                    return value
        
        return value
    
    @staticmethod
    def apply_ordering(query, subquery, order_fields: List[Union[str, Dict[str, str]]]):
        """Apply ordering to a query using subquery columns."""
        if not order_fields:
            return query
        
        order_clauses = []
        for order_field in order_fields:
            if isinstance(order_field, str):
                # Skip empty strings to avoid index errors
                if not order_field.strip():
                    continue
                parts = order_field.strip().split()
                if not parts:  # Additional safety check
                    continue
                field_name = parts[0]
                direction = 'desc' if len(parts) > 1 and parts[1].lower() == 'desc' else 'asc'
                
                if hasattr(subquery.c, field_name):
                    column = getattr(subquery.c, field_name)
                    order_clauses.append(desc(column) if direction == 'desc' else asc(column))
                        
            elif isinstance(order_field, dict):
                field_name = order_field.get('field')
                direction = order_field.get('direction', 'asc').lower()
                
                if field_name and hasattr(subquery.c, field_name):
                    column = getattr(subquery.c, field_name)
                    order_clauses.append(desc(column) if direction == 'desc' else asc(column))
        
        if order_clauses:
            query = query.order_by(*order_clauses)
        
        return query
    
    @staticmethod
    def build_order_clauses_for_model(model_class: Type, order_fields: List[Union[str, Dict[str, str]]]):
        """Build order clauses using model columns directly."""
        if not order_fields:
            return []
        
        order_clauses = []
        for order_field in order_fields:
            if isinstance(order_field, str):
                # Skip empty strings to avoid index errors
                if not order_field.strip():
                    continue
                parts = order_field.strip().split()
                if not parts:  # Additional safety check
                    continue
                field_name = parts[0]
                direction = 'desc' if len(parts) > 1 and parts[1].lower() == 'desc' else 'asc'
                
                if hasattr(model_class, field_name):
                    column = getattr(model_class, field_name)
                    order_clauses.append(desc(column) if direction == 'desc' else asc(column))
                        
            elif isinstance(order_field, dict):
                field_name = order_field.get('field')
                direction = order_field.get('direction', 'asc').lower()
                
                if field_name and hasattr(model_class, field_name):
                    column = getattr(model_class, field_name)
                    order_clauses.append(desc(column) if direction == 'desc' else asc(column))
        
        return order_clauses


class InstanceCreator:
    """Handles creation of Strawberry instances from database data."""
    
    @staticmethod
    def create_strawberry_instance(strawberry_type: Type, data: Dict[str, Any]) -> Optional[Any]:
        """Create a strawberry instance from a dictionary of data."""
        if not data:
            return None
            
        try:
            type_hints = get_type_hints(strawberry_type)
            filtered_data = {}
            private_data = {}
            
            for field_name, value in data.items():
                if field_name.startswith('_'):
                    private_data[field_name] = value
                    continue
                
                actual_field_name = field_name
                if field_name not in type_hints:
                    snake_case_field = FieldMapper.camel_to_snake(field_name)
                    if snake_case_field in type_hints:
                        actual_field_name = snake_case_field
                    else:
                        logger.debug(f"Field {field_name} not found in {strawberry_type.__name__} type hints")
                        continue
                
                # Check if field is a method
                is_method = (hasattr(strawberry_type, field_name) and callable(getattr(strawberry_type, field_name))) or \
                           (hasattr(strawberry_type, actual_field_name) and callable(getattr(strawberry_type, actual_field_name)))
                
                if is_method:
                    if '_resolved' not in private_data:
                        private_data['_resolved'] = {}
                    private_data['_resolved'][field_name] = {field_name: value}
                    continue
                
                field_type = type_hints[actual_field_name]
                
                # Handle Optional types
                actual_field_type = field_type
                if hasattr(field_type, '__origin__') and field_type.__origin__ is Union:
                    args = getattr(field_type, '__args__', ())
                    for arg in args:
                        if arg is not type(None):
                            actual_field_type = arg
                            break
                
                # Handle relationship fields (List types with Strawberry types)  
                if TypeAnalyzer.is_relationship_field(field_type) and isinstance(value, list):
                    if hasattr(field_type, '__args__') and field_type.__args__:
                        inner_type = field_type.__args__[0]
                        converted_items = []
                        for item in value:
                            if isinstance(item, dict):
                                converted_item = InstanceCreator.create_strawberry_instance(inner_type, item)
                                if converted_item:
                                    converted_items.append(converted_item)
                            else:
                                converted_items.append(item)
                        filtered_data[actual_field_name] = converted_items
                    else:
                        filtered_data[actual_field_name] = value
                
                # Handle single Strawberry type instances
                elif isinstance(value, dict) and hasattr(actual_field_type, '__strawberry_definition__'):
                    converted_value = InstanceCreator.create_strawberry_instance(actual_field_type, value)
                    filtered_data[actual_field_name] = converted_value
                
                else:
                    filtered_data[actual_field_name] = value
            
            # Create the instance
            instance = strawberry_type(**filtered_data)
            
            # Set private fields directly on the instance
            for private_field, private_value in private_data.items():
                setattr(instance, private_field, private_value)
                    
            return instance
            
        except Exception as e:
            logger.error(f"Failed to create {strawberry_type.__name__} instance from data {data}: {e}")
            return None


class RelationshipProcessor:
    """Handles relationship data processing and parsing."""
    
    def __init__(self, custom_field_manager: CustomFieldManager):
        self.custom_field_manager = custom_field_manager
    
    def process_relationship_fields(
        self, 
        relationship_fields: Dict[str, Any], 
        alias_mapping: Dict[str, str],
        strawberry_type: Type
    ) -> Dict[str, Any]:
        """Process and resolve relationship field configurations."""
        resolved_relationships = {}
        
        for display_name, field_info in relationship_fields.items():
            actual_field_name = alias_mapping.get(display_name, display_name)
            
            # Create a unique key based on display name + field name + parameters
            field_arguments = field_info.get('field_arguments', {}) if isinstance(field_info, dict) else {}
            field_args_key = self._make_hashable(field_arguments) if field_arguments else ()
            unique_key = (display_name, actual_field_name, field_args_key)
            
            # Store the processed field info
            field_info_with_metadata = field_info.copy() if isinstance(field_info, dict) else {'fields': field_info}
            field_info_with_metadata['_resolved_field_name'] = actual_field_name
            field_info_with_metadata['_display_name'] = display_name
            field_info_with_metadata['_field_arguments'] = field_arguments
            
            resolved_relationships[unique_key] = field_info_with_metadata
        
        return resolved_relationships
    
    def get_relationship_strawberry_type(
        self,
        strawberry_type: Type,
        field_name: str,
        rel_config: Dict[str, Any]
    ) -> Optional[Type]:
        """Get the strawberry type for a relationship field."""
        if isinstance(rel_config, dict) and 'type' in rel_config:
            return rel_config['type']
        
        return TypeAnalyzer.get_relationship_inner_type(strawberry_type, field_name)
    
    def parse_json_relationship_data(self, json_data, rel_config) -> List[Any]:
        """Parse JSON relationship data into strawberry instances recursively."""
        if not json_data:
            return []
        
        if not isinstance(json_data, (list, tuple)):
            logger.warning(f"Expected list/tuple for relationship data, got {type(json_data)}: {json_data}")
            return []
        
        related_type = rel_config.get('type')
        if not related_type:
            logger.warning(f"No strawberry type found in rel_config for relationship parsing")
            return json_data
        
        return convert_json_to_strawberry_instances(json_data, related_type)
    
    @staticmethod
    def _make_hashable(obj):
        """Convert nested dictionaries and lists to hashable tuples for use as dictionary keys."""
        if isinstance(obj, dict):
            return tuple(sorted((k, RelationshipProcessor._make_hashable(v)) for k, v in obj.items()))
        elif isinstance(obj, list):
            return tuple(RelationshipProcessor._make_hashable(item) for item in obj)
        elif isinstance(obj, set):
            return tuple(sorted(RelationshipProcessor._make_hashable(item) for item in obj))
        else:
            return obj


class QueryBuilder:
    """Builds SQL queries for BerryQL operations."""
    
    def __init__(self, custom_field_manager: CustomFieldManager):
        self.custom_field_manager = custom_field_manager
        self._last_field_order = {}
    
    def build_columns_for_fields(
        self,
        model_class: Type,
        fields: Set[str],
        strawberry_type: Type = None
    ) -> List[ColumnElement]:
        """Build column list for requested fields."""
        inspector = inspect(model_class)
        available_columns = {col.name: getattr(model_class, col.name) for col in inspector.columns}
        
        columns = []
        sorted_fields = sorted(fields)
        for field_name in sorted_fields:
            db_column_name = FieldMapper.map_graphql_field_to_db_column(field_name, strawberry_type)
            
            if db_column_name in available_columns:
                columns.append(available_columns[db_column_name])
            elif field_name in available_columns:
                columns.append(available_columns[field_name])
        
        return columns
    
    async def build_entity_subquery(
        self,
        strawberry_type: Type,
        model_class: Type,
        requested_fields: Set[str],
        custom_where: Any,
        params: GraphQLQueryParams,
        parent_id_column = None,
        info = None
    ):
        """Build the base entity subquery with all scalar fields."""
        essential_fields = {'id'}
        
        if parent_id_column and hasattr(model_class, parent_id_column):
            essential_fields.add(parent_id_column)
        
        fields_to_include = essential_fields.union(requested_fields)
        
        # Separate regular fields from custom fields
        regular_fields = set()
        custom_fields_requested = set()
        
        for field_name in fields_to_include:
            if self.custom_field_manager.is_custom_field(strawberry_type, field_name):
                custom_fields_requested.add(field_name)
            else:
                regular_fields.add(field_name)
        
        # Build regular columns
        columns = self.build_columns_for_fields(model_class, regular_fields, strawberry_type)
        
        # Add custom field columns
        custom_columns = self.custom_field_manager.build_custom_field_columns(
            strawberry_type, custom_fields_requested, model_class
        )
        
        all_columns = columns + custom_columns
        
        # Store the field order for later extraction
        self._last_field_order = {
            'regular_fields': sorted(regular_fields),
            'custom_fields': sorted(custom_fields_requested)
        }
        
        # Build the subquery
        subquery = select(*all_columns)
        
        # Apply where conditions
        if custom_where:
            
            if callable(custom_where):
                
                # Check if the callable is async
                if asyncio.iscoroutinefunction(custom_where):
                    resolved_where = await custom_where(info)
                else:
                    resolved_where = custom_where(info)
                
                # Double-check if we still have a coroutine (shouldn't happen but let's be safe)
                if asyncio.iscoroutine(resolved_where):
                    logger.warning(f"custom_where returned a coroutine despite being awaited: {custom_where}")
                    resolved_where = await resolved_where
            else:
                resolved_where = custom_where
                
            subquery = QueryConditionProcessor.apply_where_conditions(subquery, model_class, resolved_where)
        if params.where:
            subquery = QueryConditionProcessor.apply_where_conditions(subquery, model_class, params.where)
        
        return subquery.subquery()


class BerryQLFactory:
    """
    Refactored BerryQL Factory that creates optimized resolvers for any Strawberry type.
    
    This version separates concerns into focused components for better maintainability.
    """
    
    def __init__(self):
        self._model_cache = {}
        self._field_cache = {}
        self._relationship_cache = {}
        self._custom_where_config = {}
        self._custom_order_config = {}
        self._db_adapter: Optional[DatabaseAdapter] = None
        
        # Initialize component managers
        self.custom_field_manager = CustomFieldManager()
        self.relationship_processor = RelationshipProcessor(self.custom_field_manager)
        self.query_builder = QueryBuilder(self.custom_field_manager)

    def _get_db_adapter(self, db: AsyncSession) -> DatabaseAdapter:
        """Get or initialize the database adapter for the current session."""
        if self._db_adapter is None:
            self._db_adapter = get_database_adapter(db.bind)
        return self._db_adapter

    def create_berryql_resolver(
        self,
        strawberry_type: Type[T],
        model_class: Type,
        custom_fields: Optional[Dict[str, callable]] = None,
        custom_where: Optional[Union[Dict[str, Any], callable]] = None,
        custom_order: Optional[List[str]] = None,
        return_single: bool = False
    ) -> Callable[..., Awaitable[Union[List[T], Optional[T]]]]:
        """
        Create a unified BerryQL resolver with global configurations.
        
        Args:
            strawberry_type: The root Strawberry GraphQL type class
            model_class: The corresponding root SQLAlchemy model class  
            custom_fields: Dict mapping {field_name: query_builder}
            custom_where: where_conditions_or_function (simplified, no strawberry type key needed)
            custom_order: default_order_list (simplified, no strawberry type key needed)
            return_single: If True, return a single object (or None) instead of a list
            
        Returns:
            Async resolver function that returns List[strawberry_type] or Optional[strawberry_type]
        """
        # Store configurations directly for the strawberry type
        if custom_fields:
            self.custom_field_manager.register_custom_fields(strawberry_type, custom_fields)
        if custom_where:
            self._custom_where_config[strawberry_type] = custom_where
        if custom_order:
            self._custom_order_config[strawberry_type] = custom_order
            
        async def resolver(
            db: AsyncSession,
            info=None,
            params: Optional[GraphQLQueryParams] = None,
            **kwargs
        ) -> Union[List[T], Optional[T]]:
            result = await self._execute_unified_query(
                strawberry_type=strawberry_type,
                model_class=model_class,
                db=db,
                info=info,
                params=params or GraphQLQueryParams(),
                is_root=True,
                **kwargs
            )
            
            if return_single:
                # For single object queries, return the first result or None
                return cast(Optional[T], result[0] if result and len(result) > 0 else None)
            else:
                # For list queries, return the full list
                return cast(List[T], result)
        
        return resolver

    @staticmethod
    def get_relationship_data(instance: Any, info, relationship_name: str) -> List[Any]:
        """
        Static helper method to retrieve relationship data with proper alias handling.
        
        Args:
            instance: The GraphQL type instance
            info: GraphQL info object containing field context
            relationship_name: The name of the relationship field
            
        Returns:
            List of relationship objects for the current field/alias
        """
        resolved_data = getattr(instance, '_resolved', {})
        actual_field_name = info.field_name
        
        display_name = actual_field_name
        if hasattr(info, 'path') and info.path and hasattr(info.path, 'key'):
            display_name = info.path.key
        
        relationship_data = resolved_data.get(relationship_name, {})
        result = relationship_data.get(display_name, [])
        
        return result

    async def _execute_unified_query(
        self,
        strawberry_type: Type,
        model_class: Type,
        db: AsyncSession,
        info,
        params: GraphQLQueryParams,
        is_root: bool = False,
        parent_id_column = None,
        **kwargs
    ) -> List[Any]:
        """
        Unified query execution that works for both root and nested queries using lateral joins.
        """
        # Analyze GraphQL query for field selection
        query_analysis = query_analyzer.analyze_query_fields(info, strawberry_type) if info else {
            'scalar_fields': set(),
            'relationship_fields': {}
        }
        
        # Get alias mapping to resolve display names to actual field names
        alias_mapping = query_analyzer.get_aliased_field_mapping(query_analysis) if info else {}
        
        # Resolve aliases to get actual field names for database queries
        display_fields = query_analysis['scalar_fields']
        requested_fields = set()
        for display_name in display_fields:
            if display_name.startswith('__'):
                continue
            actual_field_name = alias_mapping.get(display_name, display_name)
            requested_fields.add(actual_field_name)
        
        # Always include required fields even if not requested in GraphQL query
        required_fields = TypeAnalyzer.get_required_fields(strawberry_type)
        requested_fields.update(required_fields)
        
        # Process relationships
        resolved_relationships = self.relationship_processor.process_relationship_fields(
            query_analysis['relationship_fields'], alias_mapping, strawberry_type
        )
        
        # Build the unified query using lateral joins
        query_result = await self._build_and_execute_lateral_query(
            strawberry_type=strawberry_type,
            model_class=model_class,
            db=db,
            requested_fields=requested_fields,
            relationships=resolved_relationships,
            params=params,
            is_root=is_root,
            parent_id_column=parent_id_column,
            info=info
        )
        
        return query_result

    async def _build_and_execute_lateral_query(
        self,
        strawberry_type: Type,
        model_class: Type,
        db: AsyncSession,
        requested_fields: Set[str],
        relationships: Dict[str, Any],
        params: GraphQLQueryParams,
        is_root: bool,
        parent_id_column = None,
        info = None
    ) -> List[Any]:
        """
        Build and execute a unified lateral query that works for both root and nested entities.
        """
        # Apply custom configurations for this strawberry type
        custom_where = self._get_custom_config(strawberry_type, self._custom_where_config)
        custom_order = self._get_custom_config(strawberry_type, self._custom_order_config, default=["id"])
        
        # Build entity subquery with all required fields
        entity_subquery = await self.query_builder.build_entity_subquery(
            strawberry_type, model_class, requested_fields, custom_where, params, parent_id_column, info
        )
        
        # Build relationship aggregations
        relationship_aggregations = []
        if relationships:
            relationship_aggregations = await self._build_relationship_aggregations(
                strawberry_type, model_class, relationships, db, info
            )
        
        # Combine entity data with relationship data using lateral joins
        if is_root:
            final_query = self._build_root_lateral_query(
                entity_subquery, relationship_aggregations, custom_order, params
            )
        else:
            final_query = self._build_nested_lateral_query(
                entity_subquery, relationship_aggregations, custom_order, params, 
                self._get_db_adapter(db)
            )
        
        # Execute query
        logger.info(f"Executing unified lateral query for {strawberry_type.__name__}")
        result = await db.execute(final_query)
        
        if is_root:
            rows = result.all()
            logger.info(f"Root query found {len(rows)} rows")
            active_relationships = relationships if relationship_aggregations else {}
            return self._parse_root_results(rows, strawberry_type, requested_fields, active_relationships)
        else:
            row = result.first()
            return row[0] if row and row[0] else []

    def _get_custom_config(self, strawberry_type: Type, config_dict: Dict[Type, Any], default=None):
        """Get custom configuration for a strawberry type."""
        return config_dict.get(strawberry_type, default)

    async def _build_relationship_aggregations(
        self,
        strawberry_type: Type,
        model_class: Type,
        relationships: Dict[str, Any],
        db: AsyncSession,
        info
    ) -> List[ColumnElement]:
        """Build json_agg aggregations for all relationships."""
        aggregations = []
        
        for rel_key, rel_config in relationships.items():
            display_name, actual_field_name, _ = rel_key if isinstance(rel_key, tuple) else (rel_key, rel_key, ())
            
            if not hasattr(model_class, actual_field_name):
                logger.warning(f"Model {model_class.__name__} missing relationship: {actual_field_name}")
                continue
            
            # Get relationship info
            relationship_prop = getattr(model_class, actual_field_name).property
            related_model = relationship_prop.mapper.class_
            
            # Get the nested strawberry type
            related_strawberry_type = self.relationship_processor.get_relationship_strawberry_type(
                strawberry_type, actual_field_name, rel_config
            )
            
            if not related_strawberry_type:
                logger.warning(f"Could not determine strawberry type for {actual_field_name}")
                continue
            
            # Store the strawberry type in rel_config for later use during JSON parsing
            rel_config['type'] = related_strawberry_type
            
            # Build nested query parameters
            field_arguments = rel_config.get('_field_arguments', {})
            
            nested_params = GraphQLQueryParams(
                where=self._parse_where_from_string(field_arguments.get('where')),
                order_by=self._parse_order_by_from_string(field_arguments.get('order_by')),
                offset=field_arguments.get('offset'),
                limit=field_arguments.get('limit')
            )
            
            # Get nested field info
            nested_fields = rel_config.get('nested_fields', {}) if isinstance(rel_config, dict) else {}
            nested_scalar_fields = nested_fields.get('scalar_fields', set())
            nested_relationships = nested_fields.get('relationship_fields', {})
            
            # Build relationship subquery as SQL
            relationship_subquery = await self._build_relationship_sql_subquery(
                related_strawberry_type,
                related_model,
                nested_scalar_fields,
                nested_relationships,
                nested_params,
                relationship_prop,
                db,
                info,
                "anon_1",  # Root level uses anon_1
                actual_field_name,  # Pass the field name for alias generation
                1  # Root relationships start at nesting level 1
            )
            
            if relationship_subquery is not None:
                # The subquery already contains json_agg, so use it directly
                json_agg_column = relationship_subquery.label(f"_rel_{display_name}")
                aggregations.append(json_agg_column)
        
        return aggregations

    async def _build_relationship_sql_subquery(
        self,
        strawberry_type: Type,
        model_class: Type,
        scalar_fields: Set[str],
        relationships: Dict[str, Any],
        params: GraphQLQueryParams,
        parent_relationship_prop,
        db: AsyncSession,
        info,
        parent_alias: str = "anon_1",
        current_field_name: str = None,
        nesting_level: int = 1
    ):
        """Build relationship subquery as SQL for use in json_agg."""
        try:
            # Create a specific alias for this subquery based on field name and nesting level
            if current_field_name:
                current_alias = f"{current_field_name}_lvl{nesting_level}"
            else:
                current_alias = f"anon_{nesting_level + 1}"
            
            logger.info(f"Building subquery for {strawberry_type.__name__} with alias '{current_alias}', parent_alias='{parent_alias}'")
            
            # Always include required fields even if not explicitly requested in GraphQL query
            required_fields = TypeAnalyzer.get_required_fields(strawberry_type)
            all_scalar_fields = scalar_fields | required_fields
            
            # Build the base query for the related model
            related_columns = self.query_builder.build_columns_for_fields(
                model_class, all_scalar_fields, strawberry_type
            )
            
            if not related_columns:
                return None
            
            # Create base query
            query = select(*related_columns).select_from(model_class)
            
            # Apply where conditions
            if params.where:
                query = QueryConditionProcessor.apply_where_conditions(query, model_class, params.where)
            
            # Add foreign key relationship constraint 
            if parent_relationship_prop:
                # Get the correct foreign key column
                foreign_key_column = self._get_foreign_key_column(parent_relationship_prop, model_class)
                if foreign_key_column is not None:
                    # Use the correct parent alias for the nesting level
                    parent_id_ref = text(f"{parent_alias}.id")
                    query = query.where(foreign_key_column == parent_id_ref)
                    logger.info(f"Added foreign key constraint: {foreign_key_column} = {parent_alias}.id")
            
            # Apply ordering
            if params.order_by:
                order_clauses = QueryConditionProcessor.build_order_clauses_for_model(model_class, params.order_by)
                if order_clauses:
                    query = query.order_by(*order_clauses)
            
            # Apply pagination
            if params.offset:
                query = query.offset(params.offset)
            if params.limit is not None:
                query = query.limit(params.limit)
            
            # Apply json_agg and return as scalar subquery
            filtered_subquery = query.subquery(name=current_alias)
            
            # Build JSON object for each row using subquery columns
            json_columns = []
            for col in related_columns:
                # Reference the column from the subquery, not the original table
                subquery_col = getattr(filtered_subquery.c, col.name)
                json_columns.extend([text(f"'{col.name}'"), subquery_col])
            
            # Handle nested relationships recursively by adding them to the JSON object
            if relationships:
                logger.info(f"Processing {len(relationships)} nested relationships for {strawberry_type.__name__}")
                for rel_key, rel_config in relationships.items():
                    display_name, actual_field_name, _ = rel_key if isinstance(rel_key, tuple) else (rel_key, rel_key, ())
                    logger.info(f"Processing nested relationship: {display_name} -> {actual_field_name}")
                    
                    if not hasattr(model_class, actual_field_name):
                        logger.warning(f"Model {model_class.__name__} missing relationship: {actual_field_name}")
                        continue
                    
                    # Get relationship info
                    relationship_prop = getattr(model_class, actual_field_name).property
                    nested_related_model = relationship_prop.mapper.class_
                    logger.info(f"Nested relationship model: {nested_related_model.__name__}")
                    
                    # Get the nested strawberry type
                    nested_related_strawberry_type = self.relationship_processor.get_relationship_strawberry_type(
                        strawberry_type, actual_field_name, rel_config
                    )
                    
                    if not nested_related_strawberry_type:
                        logger.warning(f"Could not determine strawberry type for nested {actual_field_name}")
                        continue
                    
                    logger.info(f"Nested strawberry type: {nested_related_strawberry_type.__name__}")
                    
                    # Build nested query parameters
                    field_arguments = rel_config.get('_field_arguments', {})
                    
                    nested_params = GraphQLQueryParams(
                        where=self._parse_where_from_string(field_arguments.get('where')),
                        order_by=self._parse_order_by_from_string(field_arguments.get('order_by')),
                        offset=field_arguments.get('offset'),
                        limit=field_arguments.get('limit')
                    )
                    
                    logger.info(f"Nested params for {actual_field_name}: where={nested_params.where}, order_by={nested_params.order_by}")
                    
                    # Get nested field info
                    nested_fields = rel_config.get('nested_fields', {}) if isinstance(rel_config, dict) else {}
                    nested_scalar_fields = nested_fields.get('scalar_fields', set())
                    nested_nested_relationships = nested_fields.get('relationship_fields', {})
                    
                    logger.info(f"Nested fields: scalar={nested_scalar_fields}, relationships={list(nested_nested_relationships.keys()) if nested_nested_relationships else []}")
                    
                    # Recursively build the nested relationship subquery
                    # Pass the current subquery alias as parent and increment nesting level
                    nested_subquery = await self._build_relationship_sql_subquery(
                        nested_related_strawberry_type,
                        nested_related_model,
                        nested_scalar_fields,
                        nested_nested_relationships,
                        nested_params,
                        relationship_prop,
                        db,
                        info,
                        current_alias,  # Current subquery becomes the parent for nested
                        actual_field_name,  # Pass the field name for alias generation
                        nesting_level + 1  # Increment nesting level
                    )
                    
                    if nested_subquery is not None:
                        logger.info(f"Successfully built nested subquery for {display_name}")
                        # Add the nested relationship to the JSON object
                        json_columns.extend([text(f"'{display_name}'"), nested_subquery])
                    else:
                        logger.warning(f"Failed to build nested subquery for {display_name}")
            else:
                logger.info(f"No nested relationships to process for {strawberry_type.__name__}")
            
            # Get database adapter for cross-database compatibility
            db_adapter = self._get_db_adapter(db)
            json_object = db_adapter.json_build_object(*json_columns)
            
            json_agg_query = select(
                db_adapter.coalesce_json(
                    db_adapter.json_agg(json_object),
                    db_adapter.json_empty_array()
                )
            ).select_from(filtered_subquery)
            
            return json_agg_query.scalar_subquery()
            
        except Exception as e:
            logger.error(f"Error building relationship SQL subquery: {e}")
            return None

    def _get_foreign_key_column(self, relationship_prop, model_class):
        """Get the foreign key column for a relationship."""
        try:
            # Get the foreign key from the relationship using local_remote_pairs
            foreign_keys = relationship_prop.local_remote_pairs
            if foreign_keys:
                local_col, remote_col = foreign_keys[0]
                
                logger.info(f"Analyzing relationship {relationship_prop.key}:")
                # We want the column that's in the relationship target table (child table)
                if local_col.table.name == model_class.__tablename__:
                    foreign_key_column = local_col
                    logger.info(f"  Returning foreign key: {model_class.__tablename__}.{foreign_key_column.name}")
                    return foreign_key_column
                elif remote_col.table.name == model_class.__tablename__:
                    foreign_key_column = remote_col  
                    logger.info(f"  Returning foreign key: {model_class.__tablename__}.{foreign_key_column.name}")
                    return foreign_key_column
                else:
                    logger.warning(f"Could not determine correct foreign key column for relationship {relationship_prop.key}")
                    return local_col  # fallback
                    
        except Exception as e:
            logger.warning(f"Could not determine foreign key for relationship: {e}")
        
        return None

    def _parse_where_from_string(self, where_input) -> Optional[Dict[str, Any]]:
        """Parse where condition from JSON string or typed GraphQL input."""
        if where_input is None:
            return None
        
        # If it's already a dictionary, return it directly
        if isinstance(where_input, dict):
            return where_input
        
        # If it's a typed GraphQL input object, convert it
        if hasattr(where_input, '__dict__'):
            return where_input.__dict__
        
        # Try to parse as JSON string
        if isinstance(where_input, str):
            try:
                return json.loads(where_input)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse where input as JSON: {where_input}")
                return None
        
        return None

    def _parse_order_by_from_string(self, order_by_str: Optional[str]) -> Optional[List[Dict[str, str]]]:
        """Parse order by from JSON string."""
        if order_by_str is None:
            return None
        
        if isinstance(order_by_str, list):
            return order_by_str
        
        if isinstance(order_by_str, str):
            try:
                parsed = json.loads(order_by_str)
                if isinstance(parsed, list):
                    return parsed
                elif isinstance(parsed, dict):
                    return [parsed]
                else:
                    return self._parse_simple_order_by(order_by_str)
            except json.JSONDecodeError:
                return self._parse_simple_order_by(order_by_str)
        
        return None

    def _parse_simple_order_by(self, order_by_str: str) -> List[Dict[str, str]]:
        """Parse simple order by string like 'name desc' or 'id'."""
        if not order_by_str:
            return []
        
        parts = order_by_str.strip().split()
        if len(parts) == 1:
            return [{'field': parts[0], 'direction': 'asc'}]
        elif len(parts) == 2:
            return [{'field': parts[0], 'direction': parts[1].lower()}]
        else:
            # Multiple fields separated by commas
            fields = []
            for field_spec in order_by_str.split(','):
                field_spec = field_spec.strip()
                parts = field_spec.split()
                if len(parts) == 1:
                    fields.append({'field': parts[0], 'direction': 'asc'})
                elif len(parts) == 2:
                    fields.append({'field': parts[0], 'direction': parts[1].lower()})
            return fields

    def _build_root_lateral_query(
        self,
        entity_subquery,
        relationship_aggregations: List[ColumnElement],
        custom_order: List[str],
        params: GraphQLQueryParams
    ):
        """Build the final query for root entities."""
        query = select(entity_subquery)
        
        for agg in relationship_aggregations:
            query = query.add_columns(agg)
        
        if params.order_by:
            query = QueryConditionProcessor.apply_ordering(query, entity_subquery, params.order_by)
        elif custom_order:
            query = QueryConditionProcessor.apply_ordering(query, entity_subquery, custom_order)
        
        if params.offset:
            query = query.offset(params.offset)
        if params.limit is not None:
            query = query.limit(params.limit)
        
        return query

    def _build_nested_lateral_query(
        self,
        entity_subquery,
        relationship_aggregations: List[ColumnElement],
        custom_order: List[str],
        params: GraphQLQueryParams,
        db_adapter: DatabaseAdapter
    ):
        """Build the final query for nested entities using json_agg."""
        entity_columns = [c for c in entity_subquery.c]
        
        json_items = []
        for col in entity_columns:
            json_items.append(text(f"'{col.name}'"))
            json_items.append(col)
        
        entity_json = db_adapter.json_build_object(*json_items)
        
        final_agg = db_adapter.coalesce_json(
            db_adapter.json_agg(entity_json),
            db_adapter.json_empty_array()
        )
        
        query = select(final_agg).select_from(entity_subquery)
        
        if params.order_by:
            order_clauses = QueryConditionProcessor.build_order_clauses_for_model(entity_subquery, params.order_by)
            if order_clauses:
                ordered_agg = db_adapter.coalesce_json(
                    db_adapter.json_agg(entity_json.order_by(*order_clauses)),
                    db_adapter.json_empty_array()
                )
                query = select(ordered_agg).select_from(entity_subquery)
        
        return query

    def _parse_root_results(
        self,
        rows,
        strawberry_type: Type,
        requested_fields: Set[str],
        relationships: Dict[str, Any]
    ) -> List[Any]:
        """Parse results from root lateral query into Strawberry instances."""
        instances = []
        
        for row in rows:
            total_entity_fields = self._get_total_entity_field_count(strawberry_type, requested_fields)
            entity_data = self._extract_entity_data_from_row(row, requested_fields)
            
            if relationships:
                relationship_data = self._extract_relationship_data_from_row(row, relationships, total_entity_fields)
                entity_data.update(relationship_data)
            
            instance = InstanceCreator.create_strawberry_instance(strawberry_type, entity_data)
            instances.append(instance)
        
        return instances

    def _get_total_entity_field_count(self, strawberry_type: Type, requested_fields: Set[str]) -> int:
        """Get the total number of entity fields in the SQL result."""
        if hasattr(self.query_builder, '_last_field_order'):
            field_order = self.query_builder._last_field_order
            regular_fields_count = len(field_order['regular_fields'])
            custom_fields_count = len(field_order['custom_fields'])
            return regular_fields_count + custom_fields_count
        
        return len(requested_fields)

    def _extract_entity_data_from_row(self, row, requested_fields: Set[str]) -> Dict[str, Any]:
        """Extract entity data from a row, including custom fields."""
        entity_data = {}
        
        logger.debug(f"Extracting entity data from row with {len(row)} columns")
        logger.debug(f"Row data: {list(row)}")
        logger.debug(f"Requested fields: {requested_fields}")
        
        if hasattr(self.query_builder, '_last_field_order'):
            field_order = self.query_builder._last_field_order
            regular_fields = field_order['regular_fields']
            custom_fields = field_order['custom_fields']
            logger.debug(f"Using stored field order - regular: {regular_fields}, custom: {custom_fields}")
        else:
            sorted_fields = sorted(requested_fields)
            regular_fields = sorted_fields
            custom_fields = []
            logger.debug(f"Using fallback field order - regular: {regular_fields}")
        
        col_index = 0
        
        # Extract regular fields first
        for field_name in regular_fields:
            if col_index < len(row):
                entity_data[field_name] = row[col_index]
                logger.debug(f"Set regular field {field_name} = {row[col_index]} (index {col_index})")
                col_index += 1
            else:
                logger.warning(f"Column index {col_index} out of range for field {field_name}, row has {len(row)} columns")
        
        # Extract custom fields after regular fields
        for field_name in custom_fields:
            if col_index < len(row):
                entity_data[field_name] = row[col_index]
                logger.debug(f"Set custom field {field_name} = {row[col_index]} (index {col_index})")
                col_index += 1
            else:
                logger.warning(f"Column index {col_index} out of range for custom field {field_name}, row has {len(row)} columns")
        
        return entity_data

    def _extract_relationship_data_from_row(
        self, 
        row, 
        relationships: Dict[str, Any],
        entity_field_count: int
    ) -> Dict[str, Any]:
        """Extract relationship data from lateral join columns."""
        relationship_data = {}
        rel_start_idx = entity_field_count
        
        for i, (rel_key, rel_config) in enumerate(relationships.items()):
            display_name = rel_key[0] if isinstance(rel_key, tuple) else rel_key
            col_index = rel_start_idx + i
            
            json_data = row[col_index] if col_index < len(row) else []
            
            if json_data:
                if isinstance(json_data, str):
                    try:
                        json_data = json.loads(json_data)
                    except (json.JSONDecodeError, TypeError):
                        logger.warning(f"Failed to parse JSON string: {json_data}")
                        json_data = []
                
                parsed_relationships = self.relationship_processor.parse_json_relationship_data(json_data, rel_config)
                
                actual_field_name = rel_config.get('_resolved_field_name', display_name)
                if '_resolved' not in relationship_data:
                    relationship_data['_resolved'] = {}
                if actual_field_name not in relationship_data['_resolved']:
                    relationship_data['_resolved'][actual_field_name] = {}
                
                relationship_data['_resolved'][actual_field_name][display_name] = parsed_relationships
        
        return relationship_data
