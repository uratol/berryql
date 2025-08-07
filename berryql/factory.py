"""
Unified BerryQL Factory for GraphQL queries.

This module provides a completely refactored BerryQL factory that eliminates
all code duplication between root and relationship queries by using a unified lateral
join + json_agg approach for everything.

Features:
- Single create_berryql_resolver method with global configurations
- Unified query building logic for root and nested entities
- Recursive application of custom configurations at any level
- Complete elimination of N+1 problems and code duplication
"""

import logging
import json
import re
import dataclasses
import inflection
from typing import Optional, List, Dict, Any, Type, Set, Union, get_type_hints, cast, TypeVar, Callable, Awaitable
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text, asc, desc, inspect
from sqlalchemy.sql import ColumnElement
from datetime import datetime
from .query_analyzer import query_analyzer
from .database_adapters import get_database_adapter, DatabaseAdapter

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
        order_by: Optional[List[Dict[str, str]]] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None
    ):
        # Handle where parameter - can be dict or JSON string
        if isinstance(where, str):
            if not where.strip():
                self.where = {}
            else:
                try:
                    self.where = json.loads(where.strip())
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse where condition '{where}': {e}")
                    self.where = {}
        else:
            self.where = where or {}
        
        self.order_by = order_by or []
        self.offset = offset
        self.limit = limit


class BerryQLFactory:
    """
    Unified BerryQL Factory that creates optimized resolvers for any Strawberry type.
    
    Features:
    - Single query building approach using lateral joins + json_agg for all levels
    - Automatic field mapping from Strawberry types to SQLAlchemy models  
    - Dynamic field filtering - only requested fields are queried
    - Recursive configuration support at any nesting level
    - Support for where/order_by/offset/limit parameters at any level
    - Complete elimination of N+1 problems and code duplication
    """
    
    def __init__(self):
        self._model_cache = {}
        self._field_cache = {}
        self._relationship_cache = {}
        # Global configurations keyed by strawberry type
        self._custom_fields_config = {}  # Type -> {field_name: query_builder}
        self._custom_where_config = {}   # Type -> where_function_or_dict  
        self._custom_order_config = {}   # Type -> default_order_list
        # Database adapter for cross-database compatibility
        self._db_adapter: Optional[DatabaseAdapter] = None

    def _camel_to_snake(self, name: str) -> str:
        """Convert camelCase GraphQL field names to snake_case database column names."""
        # Handle cases like projectId -> project_id
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def _get_db_adapter(self, db: AsyncSession) -> DatabaseAdapter:
        """Get or initialize the database adapter for the current session."""
        if self._db_adapter is None:
            self._db_adapter = get_database_adapter(db.bind)
        return self._db_adapter

    def create_berryql_resolver(
        self,
        strawberry_type: Type[T],
        model_class: Type,
        custom_fields: Optional[Dict[Type, Dict[str, callable]]] = None,
        custom_where: Optional[Dict[Type, Union[Dict[str, Any], callable]]] = None,
        custom_order: Optional[Dict[Type, List[str]]] = None
    ) -> Callable[..., Awaitable[List[T]]]:
        """
        Create a unified BerryQL resolver with global configurations.
        
        Args:
            strawberry_type: The root Strawberry GraphQL type class
            model_class: The corresponding root SQLAlchemy model class  
            custom_fields: Dict mapping {strawberry_type: {field_name: query_builder}}
            custom_where: Dict mapping {strawberry_type: where_conditions_or_function}
            custom_order: Dict mapping {strawberry_type: default_order_list}
            
        Returns:
            Async resolver function that returns List[strawberry_type]
        """
        # Store global configurations
        if custom_fields:
            self._custom_fields_config.update(custom_fields)
        if custom_where:
            self._custom_where_config.update(custom_where)
        if custom_order:
            self._custom_order_config.update(custom_order)
            
        async def resolver(
            db: AsyncSession,
            info=None,
            params: Optional[GraphQLQueryParams] = None,
            **kwargs
        ) -> List[T]:
            result = await self._execute_unified_query(
                strawberry_type=strawberry_type,
                model_class=model_class,
                db=db,
                info=info,
                params=params or GraphQLQueryParams(),
                is_root=True,
                **kwargs
            )
            return cast(List[T], result)
        
        return resolver

    @staticmethod
    def _make_hashable(obj):
        """Convert nested dictionaries and lists to hashable tuples for use as dictionary keys."""
        if isinstance(obj, dict):
            return tuple(sorted((k, BerryQLFactory._make_hashable(v)) for k, v in obj.items()))
        elif isinstance(obj, list):
            return tuple(BerryQLFactory._make_hashable(item) for item in obj)
        elif isinstance(obj, set):
            return tuple(sorted(BerryQLFactory._make_hashable(item) for item in obj))
        else:
            return obj

    @staticmethod
    def get_relationship_data(instance: Any, info, relationship_name: str) -> List[Any]:
        """
        Static helper method to retrieve relationship data with proper alias handling.
        
        This method can be called from any GraphQL field method to retrieve relationship data
        that was resolved by the generic resolver with proper alias support.
        
        Args:
            instance: The GraphQL type instance (e.g., ProjectType instance)
            info: GraphQL info object containing field context
            relationship_name: The name of the relationship field (e.g., 'locations', 'characters')
            
        Returns:
            List of relationship objects for the current field/alias
        """
        logger = logging.getLogger(__name__)
        
        # Get the unified resolved data dictionary
        resolved_data = getattr(instance, '_resolved', {})
        logger.info(f"get_relationship_data called for {relationship_name}")
        logger.info(f"instance has _resolved: {resolved_data}")
        
        # Get the actual field name from the GraphQL selection (this handles aliases)
        # For alias 'a: locations', info.field_name='locations' but we need the alias 'a'
        actual_field_name = info.field_name  # This is 'locations' for both 'locations' and 'a: locations'
        
        # Check if this is an alias by looking at the GraphQL selection
        display_name = actual_field_name  # Default to the field name
        if hasattr(info, 'path') and info.path and hasattr(info.path, 'key'):
            # For aliases, info.path.key contains the alias name
            display_name = info.path.key
        
        logger.info(f"actual_field_name: {actual_field_name}, display_name: {display_name}")
        
        # In the nested structure, data is organized by resolved field name, then by display name
        # For aliases like 'a: locations', we look in resolved_data['locations']['a']
        # For non-aliases like 'locations', we look in resolved_data['locations']['locations']
        relationship_data = resolved_data.get(relationship_name, {})
        result = relationship_data.get(display_name, [])
        
        logger.info(f"relationship_data keys: {list(relationship_data.keys()) if isinstance(relationship_data, dict) else 'not dict'}")
        logger.info(f"returning: {len(result) if isinstance(result, list) else 'not list'} items")
        
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
        This eliminates duplication between root table and relationship queries.
        """
        # Analyze GraphQL query for field selection
        query_analysis = query_analyzer.analyze_query_fields(info, strawberry_type) if info else {
            'scalar_fields': set(),
            'relationship_fields': {}
        }
        
        logger.debug(f"Query analysis for {strawberry_type.__name__}: {query_analysis}")
        
        # Get alias mapping to resolve display names to actual field names
        alias_mapping = query_analyzer.get_aliased_field_mapping(query_analysis) if info else {}
        
        # Resolve aliases to get actual field names for database queries
        display_fields = query_analysis['scalar_fields']
        requested_fields = set()
        for display_name in display_fields:
            # Skip GraphQL meta fields that shouldn't be queried from database
            if display_name.startswith('__'):
                continue
            actual_field_name = alias_mapping.get(display_name, display_name)
            requested_fields.add(actual_field_name)
        
        # Always include required fields even if not requested in GraphQL query
        required_fields = self._get_required_fields(strawberry_type)
        requested_fields.update(required_fields)
        
        # Process relationships
        resolved_relationships = self._process_relationship_fields(
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

    def _process_relationship_fields(
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
        Uses json_agg approach for everything to eliminate code duplication.
        """
        # Apply custom configurations for this strawberry type
        custom_where = self._get_custom_config(strawberry_type, self._custom_where_config)
        custom_order = self._get_custom_config(strawberry_type, self._custom_order_config, default=["id"])
        
        # Build entity subquery with all required fields
        entity_subquery = self._build_entity_subquery(
            strawberry_type, model_class, requested_fields, custom_where, params, parent_id_column
        )
        
        # Build relationship aggregations
        relationship_aggregations = []
        if relationships:
            relationship_aggregations = await self._build_relationship_aggregations(
                strawberry_type, model_class, relationships, db, info
            )
        
        # Combine entity data with relationship data using lateral joins
        if is_root:
            # For root queries, return individual rows but use same lateral join logic
            final_query = self._build_root_lateral_query(
                entity_subquery, relationship_aggregations, custom_order, params
            )
        else:
            # For nested queries, return json_agg
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
            # Only pass relationships if they are actually being processed
            active_relationships = relationships if relationship_aggregations else {}
            return self._parse_root_results(rows, strawberry_type, requested_fields, active_relationships)
        else:
            # For nested queries, return the json result directly
            row = result.first()
            return row[0] if row and row[0] else []

    def _get_custom_config(self, strawberry_type: Type, config_dict: Dict[Type, Any], default=None):
        """Get custom configuration for a strawberry type."""
        config = config_dict.get(strawberry_type, default)
        if callable(config):
            return config()
        return config

    def _build_entity_subquery(
        self,
        strawberry_type: Type,
        model_class: Type,
        requested_fields: Set[str],
        custom_where: Any,
        params: GraphQLQueryParams,
        parent_id_column = None
    ):
        """Build the base entity subquery with all scalar fields."""
        # Get essential fields for SQL optimization
        # Only include truly essential fields like 'id' for relationships
        # Not all required fields from Strawberry type definition
        essential_fields = {'id'}
        
        # Add any parent relationship keys that might be needed
        if parent_id_column and hasattr(model_class, parent_id_column):
            essential_fields.add(parent_id_column)
        
        fields_to_include = essential_fields.union(requested_fields)
        
        # Separate regular fields from custom fields
        regular_fields = set()
        custom_fields_requested = set()
        
        for field_name in fields_to_include:
            if self._is_custom_field_for_type(strawberry_type, field_name):
                custom_fields_requested.add(field_name)
            else:
                regular_fields.add(field_name)
        
        # Build regular columns
        columns = self._build_columns_for_fields(model_class, regular_fields, strawberry_type)
        
        # Add custom field columns
        custom_columns = self._build_custom_field_columns_for_type(
            strawberry_type, custom_fields_requested, model_class
        )
        
        all_columns = columns + custom_columns
        
        # Store the field order for later extraction (this is the key fix)
        self._last_field_order = {
            'regular_fields': sorted(regular_fields),
            'custom_fields': sorted(custom_fields_requested)
        }
        
        # Build the subquery
        subquery = select(*all_columns)
        
        # Apply where conditions
        if custom_where:
            subquery = self._apply_where_conditions(subquery, model_class, custom_where)
        if params.where:
            subquery = self._apply_where_conditions(subquery, model_class, params.where)
        
        # Add parent relationship filtering for nested queries
        if parent_id_column is not None:
            # This will be used to join with parent entities
            pass
        
        return subquery.subquery()

    def _is_custom_field_for_type(self, strawberry_type: Type, field_name: str) -> bool:
        """Check if a field is a custom field for the given strawberry type."""
        # Check explicit custom field configuration
        custom_fields = self._custom_fields_config.get(strawberry_type, {})
        if field_name in custom_fields:
            return True
        
        # Check for @custom_field decorator
        if hasattr(strawberry_type, field_name):
            attr = getattr(strawberry_type, field_name)
            if callable(attr) and hasattr(attr, '_is_custom_field'):
                return True
        
        return False

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
            related_strawberry_type = self._get_relationship_strawberry_type(
                strawberry_type, actual_field_name, rel_config
            )
            
            if not related_strawberry_type:
                logger.warning(f"Could not determine strawberry type for {actual_field_name}")
                continue
            
            # Store the strawberry type in rel_config for later use during JSON parsing
            rel_config['type'] = related_strawberry_type
            
            # Build nested query parameters
            field_arguments = rel_config.get('_field_arguments', {})
            
            # Get default values from the method signature if arguments are not provided
            order_by_arg = field_arguments.get('order_by') or field_arguments.get('orderBy')
            if order_by_arg is None:
                # Try to get default value from the method signature
                default_order_by = self._get_method_default_value(strawberry_type, actual_field_name, 'order_by')
                order_by_arg = default_order_by
                logger.info(f"Using default order_by for {actual_field_name}: {default_order_by}")
            else:
                logger.info(f"Using provided order_by for {actual_field_name}: {order_by_arg}")
            
            nested_params = GraphQLQueryParams(
                where=self._parse_where_from_string(field_arguments.get('where')),
                order_by=self._parse_order_by_from_string(order_by_arg),
                offset=field_arguments.get('offset'),
                limit=field_arguments.get('limit')
            )
            
            # Get nested field info
            nested_fields = rel_config.get('nested_fields', {}) if isinstance(rel_config, dict) else {}
            nested_scalar_fields = nested_fields.get('scalar_fields', set())
            nested_relationships = nested_fields.get('relationship_fields', {})
            
            # Build relationship subquery as SQL, not as executed results
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
        """Build relationship subquery as SQL (not executed) for use in json_agg."""
        try:
            # Create a specific alias for this subquery based on field name and nesting level
            if current_field_name:
                current_alias = f"{current_field_name}_lvl{nesting_level}"
            else:
                current_alias = f"anon_{nesting_level + 1}"
            
            logger.info(f"Building subquery for {strawberry_type.__name__} with alias '{current_alias}', parent_alias='{parent_alias}'")
            
            # Always include required fields even if not explicitly requested in GraphQL query
            required_fields = self._get_required_fields(strawberry_type)
            all_scalar_fields = scalar_fields | required_fields
            
            # Build the base query for the related model
            related_columns = self._build_columns_for_fields(
                model_class, all_scalar_fields, strawberry_type
            )
            
            if not related_columns:
                return None
            
            # Create base query
            query = select(*related_columns).select_from(model_class)
            
            # Apply where conditions
            if params.where:
                query = self._apply_where_conditions(query, model_class, params.where)
            
            # Add foreign key relationship constraint 
            if parent_relationship_prop:
                # Get the correct foreign key column using the proven logic from old implementation
                foreign_key_column = self._get_foreign_key_column(parent_relationship_prop, model_class)
                if foreign_key_column is not None:
                    # Use the correct parent alias for the nesting level
                    parent_id_ref = text(f"{parent_alias}.id")
                    query = query.where(foreign_key_column == parent_id_ref)
                    logger.info(f"Added foreign key constraint: {foreign_key_column} = {parent_alias}.id")
            
            # Apply ordering - use model columns directly, not subquery columns
            if params.order_by:
                order_clauses = []
                for order_field in params.order_by:
                    if isinstance(order_field, str):
                        parts = order_field.strip().split()
                        field_name = parts[0]
                        direction = 'desc' if len(parts) > 1 and parts[1].lower() == 'desc' else 'asc'
                        
                        if hasattr(model_class, field_name):
                            column = getattr(model_class, field_name)
                            if direction == 'desc':
                                order_clauses.append(desc(column))
                            else:
                                order_clauses.append(asc(column))
                                
                    elif isinstance(order_field, dict):
                        field_name = order_field.get('field')
                        direction = order_field.get('direction', 'asc').lower()
                        
                        if field_name and hasattr(model_class, field_name):
                            column = getattr(model_class, field_name)
                            if direction == 'desc':
                                order_clauses.append(desc(column))
                            else:
                                order_clauses.append(asc(column))
                
                if order_clauses:
                    query = query.order_by(*order_clauses)
            
            # Apply pagination
            if params.offset:
                query = query.offset(params.offset)
            if params.limit is not None:
                query = query.limit(params.limit)
            
            # Apply json_agg inside the subquery and return as scalar subquery
            # Create a clean subquery from the filtered query with our specific alias
            filtered_subquery = query.with_only_columns(*related_columns).subquery(name=current_alias)
            
            # Build JSON object for each row using subquery columns
            json_columns = []
            for col in related_columns:
                # Reference the column from the subquery, not the original table
                subquery_col = getattr(filtered_subquery.c, col.name)
                json_columns.extend([text(f"'{col.name}'"), subquery_col])
            
            # Handle nested relationships recursively by adding them to the JSON object
            if relationships:
                logger.info(f"Processing {len(relationships)} nested relationships for {strawberry_type.__name__}")
                # Build nested relationship subqueries and add them to the JSON object
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
                    nested_related_strawberry_type = self._get_relationship_strawberry_type(
                        strawberry_type, actual_field_name, rel_config
                    )
                    
                    if not nested_related_strawberry_type:
                        logger.warning(f"Could not determine strawberry type for nested {actual_field_name}")
                        continue
                    
                    logger.info(f"Nested strawberry type: {nested_related_strawberry_type.__name__}")
                    
                    # Build nested query parameters
                    field_arguments = rel_config.get('_field_arguments', {})
                    
                    # Get default values from the method signature if arguments are not provided
                    order_by_arg = field_arguments.get('order_by') or field_arguments.get('orderBy')
                    if order_by_arg is None:
                        # Try to get default value from the method signature
                        default_order_by = self._get_method_default_value(strawberry_type, actual_field_name, 'order_by')
                        order_by_arg = default_order_by
                        logger.info(f"Using default order_by for {actual_field_name}: {default_order_by}")
                    else:
                        logger.info(f"Using provided order_by for {actual_field_name}: {order_by_arg}")
                    
                    nested_params = GraphQLQueryParams(
                        where=self._parse_where_from_string(field_arguments.get('where')),
                        order_by=self._parse_order_by_from_string(order_by_arg),
                        offset=field_arguments.get('offset'),
                        limit=field_arguments.get('limit')
                    )
                    
                    logger.info(f"Nested params for {actual_field_name}: where={nested_params.where}, order_by={nested_params.order_by}, offset={nested_params.offset}, limit={nested_params.limit}")
                    
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
            ).select_from(filtered_subquery)  # Only select from the subquery, not the original table
            
            return json_agg_query.scalar_subquery()
            
        except Exception as e:
            logger.error(f"Error building relationship SQL subquery: {e}")
            return None

    async def _build_relationship_subquery(
        self,
        strawberry_type: Type,
        model_class: Type,
        scalar_fields: Set[str],
        relationships: Dict[str, Any],
        params: GraphQLQueryParams,
        parent_relationship_prop,
        db: AsyncSession,
        info
    ):
        """Recursively build relationship subquery using the same unified approach."""
        # Convert relationship property to foreign key column name
        parent_column_name = None
        if parent_relationship_prop and hasattr(parent_relationship_prop, 'back_populates'):
            # For back-referenced relationships, get the foreign key column
            if hasattr(parent_relationship_prop, 'local_columns') and parent_relationship_prop.local_columns:
                parent_column_name = list(parent_relationship_prop.local_columns)[0].name
        
        # This recursively calls the same logic, applying custom configs at each level
        nested_results = await self._execute_unified_query(
            strawberry_type=strawberry_type,
            model_class=model_class,
            db=db,
            info=info,
            params=params,
            is_root=False,
            parent_id_column=parent_column_name
        )
        
        return nested_results

    def _build_root_lateral_query(
        self,
        entity_subquery,
        relationship_aggregations: List[ColumnElement],
        custom_order: List[str],
        params: GraphQLQueryParams
    ):
        """Build the final query for root entities."""
        # Select from entity subquery
        query = select(entity_subquery)
        
        # Add relationship aggregations as lateral joins
        for agg in relationship_aggregations:
            query = query.add_columns(agg)
        
        # No GROUP BY needed since we're using scalar subqueries with json_agg
        
        # Apply ordering
        if params.order_by:
            query = self._apply_ordering_to_query(query, entity_subquery, params.order_by)
        elif custom_order:
            query = self._apply_ordering_to_query(query, entity_subquery, custom_order)
        
        # Apply pagination
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
        """Build the final query for nested entities using json_agg - SIMPLIFIED VERSION."""
        # Get entity columns
        entity_columns = [c for c in entity_subquery.c]
        
        # For now, let's ignore relationships to fix the core issue
        # TODO: Add relationships back once basic query works
        
        # Create simple json object from entity columns
        json_items = []
        for col in entity_columns:
            json_items.append(text(f"'{col.name}'"))
            json_items.append(col)
        
        entity_json = db_adapter.json_build_object(*json_items)
        
        # Wrap in json_agg for multiple results - only select the aggregate, not individual columns
        final_agg = db_adapter.coalesce_json(
            db_adapter.json_agg(entity_json),
            db_adapter.json_empty_array()
        )
        
        query = select(final_agg).select_from(entity_subquery)
        
        # Apply ordering within the aggregation
        if params.order_by:
            order_clauses = self._build_order_clauses(entity_subquery, params.order_by)
            if order_clauses:
                # Re-build with ordering
                ordered_agg = db_adapter.coalesce_json(
                    db_adapter.json_agg(entity_json.order_by(*order_clauses)),
                    db_adapter.json_empty_array()
                )
                query = select(ordered_agg).select_from(entity_subquery)
        
        return query

    def _build_columns_for_fields(
        self,
        model_class: Type,
        fields: Set[str],
        strawberry_type: Type = None
    ) -> List[ColumnElement]:
        """Build column list for requested fields."""
        inspector = inspect(model_class)
        available_columns = {col.name: getattr(model_class, col.name) for col in inspector.columns}
        
        columns = []
        # Sort fields for consistent ordering
        sorted_fields = sorted(fields)
        for field_name in sorted_fields:
            # Map GraphQL field name to database column name
            db_column_name = self._map_graphql_field_to_db_column(field_name, strawberry_type)
            
            if db_column_name in available_columns:
                columns.append(available_columns[db_column_name])
            elif field_name in available_columns:
                columns.append(available_columns[field_name])
        
        return columns

    def _build_custom_field_columns_for_type(
        self,
        strawberry_type: Type,
        requested_fields: Set[str],
        model_class: Type
    ) -> List[ColumnElement]:
        """Build custom field columns using the global configuration and @custom_field decorators."""
        custom_columns = []
        
        # Get custom fields config for this type (from explicit configuration)
        custom_fields = self._custom_fields_config.get(strawberry_type, {})
        
        # Also scan the strawberry type for methods with @custom_field decorator
        for attr_name in dir(strawberry_type):
            if attr_name.startswith('_'):
                continue
                
            try:
                attr = getattr(strawberry_type, attr_name)
                if callable(attr) and hasattr(attr, '_is_custom_field') and hasattr(attr, '_custom_query_builder'):
                    # This is a method with @custom_field decorator
                    if attr_name in requested_fields:
                        custom_fields[attr_name] = attr._custom_query_builder
            except (AttributeError, TypeError):
                continue
        
        # Build columns for all custom fields
        for field_name in requested_fields:
            if field_name in custom_fields:
                query_builder = custom_fields[field_name]
                
                try:
                    custom_column = query_builder(model_class, requested_fields)
                    if hasattr(custom_column, 'label'):
                        custom_column = custom_column.label(field_name)
                    custom_columns.append(custom_column)
                except Exception as e:
                    logger.error(f"Error building custom field {field_name} for {strawberry_type.__name__}: {e}")
        
        return custom_columns

    def _get_relationship_strawberry_type(
        self,
        strawberry_type: Type,
        field_name: str,
        rel_config: Dict[str, Any]
    ) -> Optional[Type]:
        """Get the strawberry type for a relationship field."""
        # First try to get it from the relation config
        if isinstance(rel_config, dict) and 'type' in rel_config:
            return rel_config['type']
        
        # Try to infer from the strawberry type annotations
        return self._get_relationship_inner_type(strawberry_type, field_name)

    def _get_relationship_inner_type(self, strawberry_type: Type, field_name: str) -> Optional[Type]:
        """Get the inner type for relationship fields by analyzing their return type."""
        type_hints = get_type_hints(strawberry_type)
        
        if field_name in type_hints:
            field_type = type_hints[field_name]
            return self._get_inner_type(field_type)
        
        # Check if it's a strawberry field method
        if hasattr(strawberry_type, field_name):
            field_attr = getattr(strawberry_type, field_name)
            if callable(field_attr) and hasattr(field_attr, '__annotations__'):
                return_type = field_attr.__annotations__.get('return')
                if return_type:
                    return self._get_inner_type(return_type)
        
        return None

    def _get_inner_type(self, field_type):
        """Extract the inner type from List[InnerType] or Optional[List[InnerType]]."""
        if not field_type:
            return None
        
        # Handle Strawberry annotation objects
        if hasattr(field_type, '__class__') and 'strawberry' in str(field_type.__class__):
            # For StrawberryAnnotation, check the annotation attribute
            if hasattr(field_type, 'annotation'):
                actual_type = field_type.annotation
                return self._get_inner_type(actual_type)  # Recursive call with actual type
        
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
            # Calculate total entity field count (regular + custom fields)
            total_entity_fields = self._get_total_entity_field_count(strawberry_type, requested_fields)
            
            # Extract entity data (first columns are entity data)
            entity_data = self._extract_entity_data_from_row(row, requested_fields)
            
            # Extract relationship data (remaining columns are json aggregations)
            if relationships:
                relationship_data = self._extract_relationship_data_from_row(row, relationships, total_entity_fields)
                entity_data.update(relationship_data)
            
            # Create strawberry instance
            instance = self._create_strawberry_instance(strawberry_type, entity_data)
            instances.append(instance)
        
        return instances

    def _get_total_entity_field_count(self, strawberry_type: Type, requested_fields: Set[str]) -> int:
        """Get the total number of entity fields in the SQL result (regular + custom fields)."""
        # Count regular fields
        regular_field_count = len(requested_fields)
        
        # Count custom fields that were actually requested
        custom_field_count = 0
        for field_name in requested_fields:
            if self._is_custom_field_for_type(strawberry_type, field_name):
                custom_field_count += 1
        
        # Use the stored field order if available (more reliable)
        if hasattr(self, '_last_field_order'):
            field_order = self._last_field_order
            regular_fields_count = len(field_order['regular_fields'])
            custom_fields_count = len(field_order['custom_fields'])
            return regular_fields_count + custom_fields_count
        
        # Fallback calculation
        return regular_field_count

    def _extract_entity_data_from_row(self, row, requested_fields: Set[str]) -> Dict[str, Any]:
        """Extract entity data from a row, including custom fields."""
        entity_data = {}
        
        # Use the field order that was stored during subquery building
        if hasattr(self, '_last_field_order'):
            field_order = self._last_field_order
            regular_fields = field_order['regular_fields']
            custom_fields = field_order['custom_fields']
        else:
            # Fallback to the old method if field order is not available
            sorted_fields = sorted(requested_fields)
            regular_fields = sorted_fields
            custom_fields = []
        
        col_index = 0
        
        # Extract regular fields first
        for field_name in regular_fields:
            if col_index < len(row):
                entity_data[field_name] = row[col_index]
                col_index += 1
        
        # Extract custom fields after regular fields
        for field_name in custom_fields:
            if col_index < len(row):
                entity_data[field_name] = row[col_index]
                col_index += 1
        
        return entity_data

    def _extract_relationship_data_from_row(
        self, 
        row, 
        relationships: Dict[str, Any],
        entity_field_count: int
    ) -> Dict[str, Any]:
        """Extract relationship data from lateral join columns."""
        relationship_data = {}
        
        # Relationship data starts after entity columns
        rel_start_idx = entity_field_count
        
        for i, (rel_key, rel_config) in enumerate(relationships.items()):
            display_name = rel_key[0] if isinstance(rel_key, tuple) else rel_key
            col_index = rel_start_idx + i
            
            # Get json data from the row
            json_data = row[col_index] if col_index < len(row) else []
            
            # Parse the json data into strawberry instances
            if json_data:
                # Parse JSON string if needed (for SQLite and other databases that return JSON as strings)
                if isinstance(json_data, str):
                    try:
                        import json
                        json_data = json.loads(json_data)
                    except (json.JSONDecodeError, TypeError):
                        logger.warning(f"Failed to parse JSON string: {json_data}")
                        json_data = []
                
                parsed_relationships = self._parse_json_relationship_data(json_data, rel_config)
                
                # Store data in the nested structure expected by get_relationship_data
                actual_field_name = rel_config.get('_resolved_field_name', display_name)
                if '_resolved' not in relationship_data:
                    relationship_data['_resolved'] = {}
                if actual_field_name not in relationship_data['_resolved']:
                    relationship_data['_resolved'][actual_field_name] = {}
                
                relationship_data['_resolved'][actual_field_name][display_name] = parsed_relationships
        
        return relationship_data

    def _parse_json_relationship_data(self, json_data, rel_config) -> List[Any]:
        """Parse JSON relationship data into strawberry instances recursively."""
        if not json_data:
            return []
        
        # Ensure json_data is iterable (list)
        if not isinstance(json_data, (list, tuple)):
            logger.warning(f"Expected list/tuple for relationship data, got {type(json_data)}: {json_data}")
            return []
        
        # Get the related strawberry type
        related_type = rel_config.get('type')
        if not related_type:
            logger.warning(f"No strawberry type found in rel_config for relationship parsing")
            return json_data  # Return raw data if no type info
        
        # Convert json objects to strawberry instances
        instances = []
        for item_data in json_data:
            if isinstance(item_data, dict):
                # Recursively parse any nested relationship data within this item
                parsed_item_data = self._parse_nested_json_relationships(item_data, related_type)
                instance = self._create_strawberry_instance(related_type, parsed_item_data)
                instances.append(instance)
        
        return instances

    def _parse_nested_json_relationships(self, item_data: Dict[str, Any], strawberry_type: Type) -> Dict[str, Any]:
        """Recursively parse nested JSON relationships within an item."""
        parsed_data = item_data.copy()
        
        # Get the strawberry type's relationship fields
        type_hints = get_type_hints(strawberry_type)
        
        for field_name, field_value in item_data.items():
            # Check if this field represents a relationship (List type)
            if field_name in type_hints:
                field_type = type_hints[field_name]
                if self._is_relationship_field(field_type) and isinstance(field_value, (list, tuple)):
                    # Get the inner type for this relationship
                    inner_type = self._get_inner_type(field_type)
                    if inner_type and field_value:
                        # Create a mock rel_config with the type information
                        nested_rel_config = {'type': inner_type}
                        # Recursively parse this relationship data
                        parsed_relationships = self._parse_json_relationship_data(field_value, nested_rel_config)
                        parsed_data[field_name] = parsed_relationships
        
        return parsed_data

    # Utility methods
    def _get_required_fields(self, strawberry_type: Type) -> Set[str]:
        """Get required fields from Strawberry type annotations."""
        required_fields = set()
        
        try:
            # Get the strawberry type's fields
            if hasattr(strawberry_type, '__strawberry_definition__'):
                for field in strawberry_type.__strawberry_definition__.fields:
                    field_name = field.python_name
                    
                    # Check if field is required:
                    # - Not optional (not StrawberryOptional type)
                    # - No explicit default value (default is an instance of _MISSING_TYPE)
                    # - Init=True (participates in constructor)
                    
                    is_optional = hasattr(field.type, '__class__') and field.type.__class__.__name__ == 'StrawberryOptional'
                    has_no_default = isinstance(field.default, dataclasses._MISSING_TYPE)
                    has_no_default_factory = isinstance(field.default_factory, dataclasses._MISSING_TYPE)
                    
                    # Required fields: not optional, no explicit default, and participates in init
                    if not is_optional and has_no_default and has_no_default_factory and field.init:
                        required_fields.add(field_name)
                            
        except Exception as e:
            logger.warning(f"Could not determine required fields for {strawberry_type.__name__}: {e}")
            # Fallback to common required fields
            required_fields = {'id'}
        
        return required_fields

    def _map_graphql_field_to_db_column(self, field_name: str, strawberry_type: Type = None) -> str:
        """Map GraphQL field name to database column name."""
        if strawberry_type:
            # Check if the field exists directly in the type hints (snake_case)
            type_hints = get_type_hints(strawberry_type)
            if field_name in type_hints:
                return field_name
            
            # Convert camelCase to snake_case and check if it exists
            snake_case_field = self._camel_to_snake(field_name)
            if snake_case_field in type_hints:
                return snake_case_field
        
        # Fallback: try converting camelCase to snake_case
        return self._camel_to_snake(field_name)


    def _apply_where_conditions(self, query, model_class: Type, where_conditions: Union[Dict[str, Any], str, None]):
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
            db_field_name = self._camel_to_snake(field_name)
            
            # Validate field exists on model (try both original and converted names)
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
                    # Skip operators with None values (not provided by client)
                    if op_value is None:
                        continue
                        
                    logger.info(f"Processing operator '{op}' with value '{op_value}' (type: {type(op_value)}) for field '{field_name}'")
                    
                    # Validate operator is supported
                    if op not in supported_operators:
                        raise InvalidFieldError(f"Unsupported operator '{op}'. Supported: {sorted(supported_operators)}")
                    
                    # Special validation for True/False values - only allow equality/inequality
                    if op_value in (True, False) and op not in ('eq', 'ne'):
                        raise InvalidFieldError(f"Only equality operators ('eq', 'ne') can be used with True/False values, not '{op}'")
                    
                    # Convert the value to appropriate type for the column
                    converted_value = self._convert_value_for_column(column, op_value)
                    
                    if op == 'eq':
                        query = query.where(column == converted_value)
                    elif op == 'ne':
                        query = query.where(column != converted_value)
                    elif op == 'gt':
                        query = query.where(column > converted_value)
                    elif op == 'gte':
                        query = query.where(column >= converted_value)
                    elif op == 'lt':
                        query = query.where(column < converted_value)
                    elif op == 'lte':
                        query = query.where(column <= converted_value)
                    elif op == 'in':
                        query = query.where(column.in_(converted_value))
                    elif op == 'like':
                        query = query.where(column.like(converted_value))
                    elif op == 'ilike':
                        query = query.where(column.ilike(converted_value))
            else:
                # Simple equality - skip None values (not provided by client)
                if value is None:
                    continue
                    
                # Convert the value
                converted_value = self._convert_value_for_column(column, value)
                query = query.where(column == converted_value)
        
        return query

    def _convert_value_for_column(self, column, value):
        """Convert a value to the appropriate type for the database column."""
        if value is None:
            return value
            
        # Get the column type
        column_type = str(column.type).lower()
        
        # Handle datetime/timestamp/date columns
        if any(dt_type in column_type for dt_type in ['timestamp', 'datetime', 'date']):
            if isinstance(value, str):
                try:
                    # Handle different date/datetime formats
                    if 'T' in value or value.endswith('Z') or '+' in value[-6:] or value.endswith('+00:00'):
                        # ISO format with time (datetime)
                        normalized_value = value.replace('Z', '+00:00')
                        return datetime.fromisoformat(normalized_value)
                    elif '-' in value and len(value) == 10:
                        # Plain date format YYYY-MM-DD
                        if 'date' in column_type and 'datetime' not in column_type and 'timestamp' not in column_type:
                            from datetime import date
                            return datetime.strptime(value, '%Y-%m-%d').date()
                        else:
                            return datetime.strptime(value, '%Y-%m-%d')
                    else:
                        return datetime.fromisoformat(value)
                        
                except (ValueError, IndexError) as e:
                    logger.warning(
                        f"Could not parse date/datetime string '{value}' for column {column.name} "
                        f"of type {column_type}: {e}. Letting SQLAlchemy handle conversion."
                    )
                    return value
        
        return value

    def _apply_ordering_to_query(self, query, subquery, order_fields: List[Union[str, Dict[str, str]]]):
        """Apply ordering to a query using subquery columns."""
        if not order_fields:
            return query
        
        order_clauses = []
        for order_field in order_fields:
            if isinstance(order_field, str):
                parts = order_field.strip().split()
                if not parts:  # Skip empty strings
                    continue
                field_name = parts[0]
                direction = 'desc' if len(parts) > 1 and parts[1].lower() == 'desc' else 'asc'
                
                if hasattr(subquery.c, field_name):
                    column = getattr(subquery.c, field_name)
                    if direction == 'desc':
                        order_clauses.append(desc(column))
                    else:
                        order_clauses.append(asc(column))
                        
            elif isinstance(order_field, dict):
                field_name = order_field.get('field')
                direction = order_field.get('direction', 'asc').lower()
                
                if field_name and hasattr(subquery.c, field_name):
                    column = getattr(subquery.c, field_name)
                    if direction == 'desc':
                        order_clauses.append(desc(column))
                    else:
                        order_clauses.append(asc(column))
        
        if order_clauses:
            query = query.order_by(*order_clauses)
        
        return query

    def _build_order_clauses(self, subquery, order_fields: List[Union[str, Dict[str, str]]]):
        """Build order clauses for use within aggregations."""
        if not order_fields:
            return []
        
        order_clauses = []
        for order_field in order_fields:
            if isinstance(order_field, str):
                parts = order_field.strip().split()
                if not parts:  # Skip empty strings
                    continue
                field_name = parts[0]
                direction = 'desc' if len(parts) > 1 and parts[1].lower() == 'desc' else 'asc'
                
                if hasattr(subquery.c, field_name):
                    column = getattr(subquery.c, field_name)
                    if direction == 'desc':
                        order_clauses.append(desc(column))
                    else:
                        order_clauses.append(asc(column))
                        
            elif isinstance(order_field, dict):
                field_name = order_field.get('field')
                direction = order_field.get('direction', 'asc').lower()
                
                if field_name and hasattr(subquery.c, field_name):
                    column = getattr(subquery.c, field_name)
                    if direction == 'desc':
                        order_clauses.append(desc(column))
                    else:
                        order_clauses.append(asc(column))
        
        return order_clauses

    def _parse_where_from_string(self, where_input) -> Optional[Dict[str, Any]]:
        """Parse where condition from JSON string or typed GraphQL input."""
        if where_input is None:
            return None
        
        # If it's already a dictionary, return it directly
        if isinstance(where_input, dict):
            return where_input
        
        # If it's a typed GraphQL input object, convert it
        if hasattr(where_input, '__dict__'):
            # Import the converter to handle typed inputs
            try:
                from app.graphql.berryql.input_converter import convert_where_input
                
                # Convert the typed input to dictionary format
                where_dict = {}
                for field_name, field_value in where_input.__dict__.items():
                    if field_value is not None:
                        where_dict[field_name] = field_value
                
                # Convert to the format expected by the SQL generator
                return convert_where_input(where_dict)
            except ImportError:
                # Fallback to treating it as a regular object
                where_dict = {}
                for field_name, field_value in where_input.__dict__.items():
                    if field_value is not None:
                        where_dict[field_name] = field_value
                return where_dict
        
        # If it's a string, parse as JSON (original behavior)
        if isinstance(where_input, str):
            if not where_input.strip():
                return None
            try:
                parsed = json.loads(where_input.strip())
                if not isinstance(parsed, dict):
                    raise InvalidFieldError(f"Where clause must be a JSON object, got: {type(parsed).__name__}")
                return parsed
            except (json.JSONDecodeError, TypeError) as e:
                raise InvalidFieldError(f"Failed to parse where condition '{where_input}': {e}")
        
        # Unknown type
        raise InvalidFieldError(f"Where clause must be a string, dict, or typed input object, got: {type(where_input).__name__}")

    def _parse_order_by_from_string(self, order_by_str: Optional[str]) -> Optional[List[Dict[str, str]]]:
        """Parse order_by from JSON string or simple field name."""
        if not order_by_str:
            return None
        
        # First, try to parse as JSON
        try:
            parsed = json.loads(order_by_str)
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict):
                return [parsed]
            else:
                raise InvalidFieldError(f"Order by must be a list or object, got: {type(parsed).__name__}")
        except (json.JSONDecodeError, TypeError):
            # If JSON parsing fails, try to parse as simple field name or '{field} asc/desc'
            try:
                return self._parse_simple_order_by(order_by_str)
            except Exception as e:
                raise InvalidFieldError(f"Failed to parse order_by condition '{order_by_str}': {e}")

    def _parse_simple_order_by(self, order_by_str: str) -> List[Dict[str, str]]:
        """Parse simple order_by format: 'field' or 'field asc' or 'field desc' or comma-separated fields."""
        order_by_str = order_by_str.strip()
        
        if not order_by_str:
            raise ValueError("Empty order_by string")
        
        # Split by comma to handle multiple fields
        field_specs = [spec.strip() for spec in order_by_str.split(',')]
        result = []
        
        for field_spec in field_specs:
            if not field_spec:
                continue  # Skip empty specs from trailing commas
                
            # Split by whitespace to check for direction
            parts = field_spec.split()
            
            if len(parts) == 1:
                field_name = parts[0]
                direction = 'asc'  # Default direction
            elif len(parts) == 2:
                field_name, direction = parts
                direction = direction.lower()
                if direction not in ['asc', 'desc']:
                    raise ValueError(f"Invalid direction '{direction}'. Must be 'asc' or 'desc'")
            else:
                raise ValueError(f"Invalid order_by format for field '{field_spec}'. Expected 'field' or 'field direction'")
            
            result.append({"field": field_name, "direction": direction})
        
        if not result:
            raise ValueError("No valid field specifications found in order_by string")
            
        return result

    def _get_method_default_value(self, strawberry_type: Type, method_name: str, param_name: str):
        """Get default value for a parameter from a Strawberry field method signature."""
        try:
            logger.debug(f"Looking for default value: {strawberry_type.__name__}.{method_name}.{param_name}")
            if hasattr(strawberry_type, method_name):
                method = getattr(strawberry_type, method_name)
                if callable(method):
                    import inspect
                    sig = inspect.signature(method)
                    logger.debug(f"Method signature: {sig}")
                    if param_name in sig.parameters:
                        param = sig.parameters[param_name]
                        if param.default is not inspect.Parameter.empty:
                            logger.debug(f"Found default value: {param.default}")
                            return param.default
                        else:
                            logger.debug(f"Parameter {param_name} has no default value")
                    else:
                        logger.debug(f"Parameter {param_name} not found in method signature")
                else:
                    logger.debug(f"{method_name} is not callable")
            else:
                logger.debug(f"Method {method_name} not found on {strawberry_type.__name__}")
        except Exception as e:
            logger.debug(f"Error getting default value for {strawberry_type.__name__}.{method_name}.{param_name}: {e}")
        return None

    def _is_relationship_field(self, field_type) -> bool:
        """Check if a field type represents a relationship (List type)."""
        if not field_type:
            return False
        
        # Handle Strawberry annotation objects
        if hasattr(field_type, '__class__') and 'strawberry' in str(field_type.__class__):
            # For StrawberryAnnotation, check the annotation attribute
            if hasattr(field_type, 'annotation'):
                actual_type = field_type.annotation
                return self._is_relationship_field(actual_type)  # Recursive call with actual type
        
        # Handle Optional[List[...]] and List[...] types
        origin = getattr(field_type, '__origin__', None)
        if origin is Union:  # Optional type
            args = getattr(field_type, '__args__', ())
            if len(args) == 2 and type(None) in args:
                non_none_type = args[0] if args[1] is type(None) else args[1]
                return getattr(non_none_type, '__origin__', None) is list
        
        return origin is list

    def _create_strawberry_instance(self, strawberry_type: Type, data: Dict[str, Any]):
        """Create a Strawberry instance from dictionary data with proper type conversion."""
        if not data:
            return None
        
        try:
            # Get type hints for proper field mapping
            type_hints = get_type_hints(strawberry_type)
            filtered_data = {}
            private_data = {}
            
            for field_name, value in data.items():
                # Handle the special _resolved dictionary 
                if field_name == '_resolved':
                    private_data[field_name] = value
                    continue
                
                # Handle private fields (those starting with _resolved_)
                if field_name.startswith('_resolved_'):
                    private_data[field_name] = value
                    continue
                
                # Skip if field doesn't exist in type hints - move to _resolved instead
                if field_name not in type_hints:
                    # Check if this is relationship data that should go in _resolved
                    if isinstance(value, (list, dict)):
                        if '_resolved' not in private_data:
                            private_data['_resolved'] = {}
                        private_data['_resolved'][field_name] = {field_name: value}
                    continue
                
                # Check if field is a method (not a constructor parameter)
                # Methods decorated with @strawberry.field should not be passed to constructor
                if hasattr(strawberry_type, field_name) and callable(getattr(strawberry_type, field_name)):
                    # This is a method, move to _resolved
                    # For custom fields that use @generic_field, we need to store the raw data directly
                    if '_resolved' not in private_data:
                        private_data['_resolved'] = {}
                    # Store the data with the field name as key, and the field name as nested key for compatibility
                    private_data['_resolved'][field_name] = {field_name: value}
                    continue
                    
                field_type = type_hints[field_name]
                
                # Handle Optional types - extract the actual type
                actual_field_type = field_type
                if hasattr(field_type, '__origin__') and field_type.__origin__ is Union:
                    args = getattr(field_type, '__args__', ())
                    for arg in args:
                        if arg is not type(None):
                            actual_field_type = arg
                            break
                
                # Handle relationship fields (List types with Strawberry types)  
                if self._is_relationship_field(field_type) and isinstance(value, list):
                    # Get the inner type from List[SomeType]
                    if hasattr(field_type, '__args__') and field_type.__args__:
                        inner_type = field_type.__args__[0]
                        # Convert each dict to its Strawberry type
                        converted_items = []
                        for item in value:
                            if isinstance(item, dict):
                                converted_item = self._create_strawberry_instance(inner_type, item)
                                if converted_item:
                                    converted_items.append(converted_item)
                            else:
                                converted_items.append(item)
                        filtered_data[field_name] = converted_items
                    else:
                        filtered_data[field_name] = value
                # Handle single Strawberry type instances
                elif isinstance(value, dict) and hasattr(actual_field_type, '__strawberry_definition__'):
                    converted_value = self._create_strawberry_instance(actual_field_type, value)
                    filtered_data[field_name] = converted_value
                else:
                    filtered_data[field_name] = value
            
            # Create the instance
            instance = strawberry_type(**filtered_data)
            
            # Set private fields directly on the instance
            for private_field, private_value in private_data.items():
                if private_field == '_resolved':
                    setattr(instance, private_field, private_value)
                    continue
                    
                if isinstance(private_value, list) and private_value:
                    # Handle resolved relationship fields
                    converted_items = []
                    
                    # Try to determine the type dynamically
                    if private_field.startswith('_resolved_'):
                        type_name_base = private_field[10:]  # Remove '_resolved_' prefix
                        singular_base = inflection.singularize(type_name_base)
                        type_name = ''.join(word.capitalize() for word in singular_base.split('_')) + 'Type'
                        
                        try:
                            # Try to import the type
                            import importlib
                            types_module = importlib.import_module('app.graphql.types')
                            target_type = getattr(types_module, type_name, None)
                            
                            if target_type is None:
                                from .. import types
                                target_type = getattr(types, type_name, None)
                                
                            if target_type:
                                for item in private_value:
                                    if isinstance(item, dict):
                                        converted_item = self._create_strawberry_instance(target_type, item)
                                        if converted_item:
                                            converted_items.append(converted_item)
                                    else:
                                        converted_items.append(item)
                                setattr(instance, private_field, converted_items)
                            else:
                                setattr(instance, private_field, private_value)
                                
                        except (ImportError, AttributeError):
                            setattr(instance, private_field, private_value)
                    else:
                        setattr(instance, private_field, private_value)
                else:
                    setattr(instance, private_field, private_value)
                    
            return instance
            
        except Exception as e:
            logger.error(f"Failed to create {strawberry_type.__name__} instance from data {data}: {e}")
            return None

    def _get_foreign_key_column(self, relationship_prop, parent_model):
        """Get the foreign key column for a relationship using the proven logic from old implementation."""
        from sqlalchemy.sql import ColumnElement
        try:
            # Get the foreign key from the relationship using local_remote_pairs
            foreign_keys = relationship_prop.local_remote_pairs
            if foreign_keys:
                local_col, remote_col = foreign_keys[0]
                
                logger.info(f"Analyzing relationship {relationship_prop.key}:")
                # We want the column that's in the relationship target table (child table)
                # This will be the foreign key that references the parent
                if local_col.table.name == relationship_prop.mapper.class_.__tablename__:
                    foreign_key_column = local_col
                    logger.info(f"  Returning foreign key: {relationship_prop.mapper.class_.__tablename__}.{foreign_key_column.name}")
                    return foreign_key_column
                elif remote_col.table.name == relationship_prop.mapper.class_.__tablename__:
                    foreign_key_column = remote_col  
                    logger.info(f"  Returning foreign key: {relationship_prop.mapper.class_.__tablename__}.{foreign_key_column.name}")
                    return foreign_key_column
                else:
                    logger.warning(f"Could not determine correct foreign key column for relationship {relationship_prop.key}")
                    return local_col  # fallback
                    
        except Exception as e:
            logger.warning(f"Could not determine foreign key for relationship: {e}")
        
        return None
