"""
GraphQL Query Field Analysis

This module provides enhanced analysis of GraphQL query fields, including
proper handling of fragments, inline fragments, aliases, and multiple field nodes.
"""

import logging
from typing import Dict, Set, List, Any, Type, get_type_hints, Union, Optional
from graphql import GraphQLResolveInfo

logger = logging.getLogger(__name__)
 
class QueryFieldAnalyzer:
    """Analyzes GraphQL query fields using Strawberry's selected_fields API."""

    def analyze_query_fields(
        self,
        info: GraphQLResolveInfo,
        strawberry_type: Type,
        depth_limit: int = 10,
    ) -> Dict[str, Any]:
        if depth_limit <= 0:
            return {
                'scalar_fields': set(),
                'relationship_fields': {},
                'aliases': {},
                'fragments': []
            }

        all_selections: List[Any] = []
        if hasattr(info, 'selected_fields') and info.selected_fields:
            if len(info.selected_fields) > 0:
                current_field = info.selected_fields[0]
                if hasattr(current_field, 'selections') and current_field.selections:
                    all_selections = current_field.selections
                else:
                    return {
                        'scalar_fields': set(),
                        'relationship_fields': {},
                        'aliases': {},
                        'fragments': []
                    }
            else:
                return {
                    'scalar_fields': set(),
                    'relationship_fields': {},
                    'aliases': {},
                    'fragments': []
                }
        else:
            return {
                'scalar_fields': set(),
                'relationship_fields': {},
                'aliases': {},
                'fragments': []
            }

        analysis_result = self._analyze_selected_fields(
            all_selections,
            strawberry_type,
            info,
            depth_limit - 1,
        )
        return analysis_result

    def _analyze_selected_fields(
        self,
        selected_fields: List[Any],
        strawberry_type: Type,
        info: GraphQLResolveInfo,
        depth_limit: int,
    ) -> Dict[str, Any]:
        scalar_fields: Set[str] = set()
        relationship_fields: Dict[str, Any] = {}
        aliases: Dict[str, str] = {}
        fragments: List[Any] = []
        for selected_field in selected_fields:
            field_result = self._analyze_selected_field(selected_field, strawberry_type, info, depth_limit)
            scalar_fields.update(field_result['scalar_fields'])
            relationship_fields.update(field_result['relationship_fields'])
            aliases.update(field_result['aliases'])
        return {
            'scalar_fields': scalar_fields,
            'relationship_fields': relationship_fields,
            'aliases': aliases,
            'fragments': fragments,
        }

    def _analyze_selected_field(
        self,
        selected_field: Any,
        strawberry_type: Type,
        info: GraphQLResolveInfo,
        depth_limit: int,
    ) -> Dict[str, Any]:
        field_name = selected_field.name
        alias = getattr(selected_field, 'alias', None)
        field_arguments = getattr(selected_field, 'arguments', {}) or {}

        scalar_fields: Set[str] = set()
        relationship_fields: Dict[str, Any] = {}
        aliases: Dict[str, str] = {}
        display_name = alias or field_name
        if alias:
            aliases[alias] = field_name

        if self._is_relationship_field_by_name(field_name, strawberry_type):
            inner_type = self._get_relationship_inner_type(field_name, strawberry_type)
            if inner_type:
                if selected_field.selections and depth_limit > 0:
                    sub_analysis = self._analyze_selected_fields(
                        selected_field.selections, inner_type, info, depth_limit - 1
                    )
                    relationship_fields[display_name] = {
                        'type': inner_type,
                        'nested_fields': sub_analysis,
                        'field_arguments': field_arguments,
                    }
                else:
                    relationship_fields[display_name] = {
                        'type': inner_type,
                        'field_arguments': field_arguments,
                    }
            else:
                scalar_fields.add(display_name)
                if field_arguments:
                    self._validate_field_arguments(field_arguments, field_name)
        else:
            if not field_name.startswith('__'):
                scalar_fields.add(display_name)
                if field_arguments:
                    self._validate_field_arguments(field_arguments, field_name)

        return {
            'scalar_fields': scalar_fields,
            'relationship_fields': relationship_fields,
            'aliases': aliases,
        }

    def _validate_field_arguments(self, field_arguments: Dict[str, Any], field_name: str):
        from .factory import InvalidFieldError
        if 'where' in field_arguments:
            where_value = field_arguments['where']
            if isinstance(where_value, str):
                import json
                try:
                    parsed = json.loads(where_value.strip()) if where_value.strip() else {}
                    if not isinstance(parsed, dict):
                        raise InvalidFieldError(
                            f"Where clause in field '{field_name}' must be a JSON object"
                        )
                except json.JSONDecodeError as e:
                    raise InvalidFieldError(
                        f"Invalid JSON in where clause for field '{field_name}': {e}"
                    )

    def _is_relationship_field_by_name(self, field_name: str, strawberry_type: Type) -> bool:
        try:
            type_hints = get_type_hints(strawberry_type)
            if field_name in type_hints:
                return self._is_relationship_field_type(type_hints[field_name])
            if hasattr(strawberry_type, '__strawberry_definition__'):
                definition = strawberry_type.__strawberry_definition__
                for f in getattr(definition, 'fields', []):
                    if f.python_name == field_name:
                        method = getattr(strawberry_type, field_name, None)
                        if method and hasattr(method, '__annotations__') and 'return' in method.__annotations__:
                            return self._is_relationship_field_type(method.__annotations__['return'])
            if hasattr(strawberry_type, field_name):
                attr = getattr(strawberry_type, field_name)
                if callable(attr) and hasattr(attr, '__annotations__') and 'return' in attr.__annotations__:
                    return self._is_relationship_field_type(attr.__annotations__['return'])
            return self._is_special_relationship_field(field_name)
        except Exception:
            return False

    def _is_relationship_field_type(self, field_type) -> bool:
        if not field_type:
            return False
        if hasattr(field_type, 'annotation'):
            return self._is_relationship_field_type(field_type.annotation)
        origin = getattr(field_type, '__origin__', None)
        if origin is list:
            return True
        if origin is Union:
            args = getattr(field_type, '__args__', ())
            if len(args) == 2 and type(None) in args:
                non_none = args[0] if args[1] is type(None) else args[1]
                return getattr(non_none, '__origin__', None) is list
        return False

    def _is_special_relationship_field(self, field_name: str) -> bool:
        import inflection
        singular = inflection.singularize(field_name)
        if singular != field_name:
            return True
        return any(field_name.endswith(s) for s in ['_list', '_items', '_collection', '_set'])

    def _get_relationship_inner_type(self, field_name: str, strawberry_type: Type) -> Optional[Type]:
        try:
            type_hints = get_type_hints(strawberry_type)
            if field_name in type_hints:
                return self._extract_inner_type(type_hints[field_name])
            if hasattr(strawberry_type, '__strawberry_definition__'):
                definition = strawberry_type.__strawberry_definition__
                for f in getattr(definition, 'fields', []):
                    if f.python_name == field_name:
                        method = getattr(strawberry_type, field_name, None)
                        if method and hasattr(method, '__annotations__') and 'return' in method.__annotations__:
                            return self._extract_inner_type(method.__annotations__['return'])
            if hasattr(strawberry_type, field_name):
                attr = getattr(strawberry_type, field_name)
                if callable(attr) and hasattr(attr, '__annotations__') and 'return' in attr.__annotations__:
                    return self._extract_inner_type(attr.__annotations__['return'])
        except Exception:
            return None
        return None

    def _extract_inner_type(self, field_type):
        if not field_type:
            return None
        if hasattr(field_type, 'annotation'):
            return self._extract_inner_type(field_type.annotation)
        origin = getattr(field_type, '__origin__', None)
        if origin is Union:
            args = getattr(field_type, '__args__', ())
            if len(args) == 2 and type(None) in args:
                non_none = args[0] if args[1] is type(None) else args[1]
                if getattr(non_none, '__origin__', None) is list:
                    inner = getattr(non_none, '__args__', ())
                    return inner[0] if inner else None
        if origin is list:
            args = getattr(field_type, '__args__', ())
            return args[0] if args else None
        return None

    def get_flat_field_names(self, analysis_result: Dict[str, Any]) -> Set[str]:
        all_fields = set(analysis_result['scalar_fields'])
        all_fields.update(analysis_result['relationship_fields'].keys())
        return all_fields

    def get_aliased_field_mapping(self, analysis_result: Dict[str, Any]) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        aliases = analysis_result['aliases']
        for field in analysis_result['scalar_fields']:
            actual = aliases.get(field, field)
            mapping[field] = actual
        for field in analysis_result['relationship_fields'].keys():
            actual = aliases.get(field, field)
            mapping[field] = actual
        return mapping
    
    def _is_relationship_field_by_name(self, field_name: str, strawberry_type: Type) -> bool:
        """Check if a field is a relationship field by analyzing the Strawberry type."""
        try:
            logger.debug(f"Checking if {field_name} is relationship field on {strawberry_type.__name__}")
            
            # Check type annotations first (for directly annotated fields)
            type_hints = get_type_hints(strawberry_type)
            logger.debug(f"Type hints for {strawberry_type.__name__}: {list(type_hints.keys())}")
            
            if field_name in type_hints:
                field_type = type_hints[field_name]
                is_rel = self._is_relationship_field_type(field_type)
                logger.debug(f"Field {field_name} found in type hints, is_relationship: {is_rel}")
                return is_rel
            
            # Check strawberry definition fields (for @strawberry.field methods)
            if hasattr(strawberry_type, '__strawberry_definition__'):
                definition = strawberry_type.__strawberry_definition__
                if hasattr(definition, 'fields'):
                    for strawberry_field in definition.fields:
                        if strawberry_field.python_name == field_name:
                            logger.debug(f"Field {field_name} found in strawberry definition")
                            # Get the field's return type from the method annotations
                            if hasattr(strawberry_type, field_name):
                                method = getattr(strawberry_type, field_name)
                                if hasattr(method, '__annotations__') and 'return' in method.__annotations__:
                                    return_type = method.__annotations__['return']
                                    is_rel = self._is_relationship_field_type(return_type)
                                    logger.debug(f"Field {field_name} return type: {return_type}, is_relationship: {is_rel}")
                                    return is_rel
            
            # Check strawberry field methods (legacy detection)
            if hasattr(strawberry_type, field_name):
                attr = getattr(strawberry_type, field_name)
                logger.debug(f"Field {field_name} found as attribute: {type(attr)}")
                
                if hasattr(attr, '__strawberry_metadata__') and callable(attr):
                    logger.debug(f"Field {field_name} has strawberry metadata and is callable")
                    if hasattr(attr, '__annotations__') and 'return' in attr.__annotations__:
                        return_type = attr.__annotations__['return']
                        is_rel = self._is_relationship_field_type(return_type)
                        logger.debug(f"Field {field_name} return type: {return_type}, is_relationship: {is_rel}")
                        return is_rel
            
            # Check for special relationship patterns
            is_special = self._is_special_relationship_field(field_name)
            logger.debug(f"Field {field_name} is_special_relationship: {is_special}")
            return is_special
            
        except Exception as e:
            logger.debug(f"Error checking if {field_name} is relationship field: {e}")
            return False
    
    def _is_relationship_field_type(self, field_type) -> bool:
        """Check if a type annotation represents a relationship field."""
        if not field_type:
            return False
        
        # Handle StrawberryAnnotation objects
        if hasattr(field_type, '__class__') and 'strawberry' in str(field_type.__class__).lower():
            # This is a strawberry annotation, extract the actual type
            if hasattr(field_type, 'annotation'):
                actual_type = field_type.annotation
                return self._is_relationship_field_type(actual_type)
        
        origin = getattr(field_type, '__origin__', None)
        
        # Check for List[SomeType] or Optional[List[SomeType]]
        if origin is list:
            return True
        elif origin is Union:
            # Handle Optional[List[SomeType]]
            args = getattr(field_type, '__args__', ())
            if len(args) == 2 and type(None) in args:
                non_none_type = args[0] if args[1] is type(None) else args[1]
                return getattr(non_none_type, '__origin__', None) is list
        
        return False
    
    def _is_special_relationship_field(self, field_name: str) -> bool:
        """Check if a field name indicates a special relationship field."""
        # This method can be used to identify common relationship patterns
        # but should not hardcode domain-specific field names
        # Instead, rely on type introspection and naming conventions
        
        # Check for common plural naming patterns that suggest relationships
        import inflection
        singular = inflection.singularize(field_name)
        is_plural = singular != field_name
        
        # A field is likely a relationship if it's plural or ends with common relationship suffixes
        relationship_suffixes = ['_list', '_items', '_collection', '_set']
        has_relationship_suffix = any(field_name.endswith(suffix) for suffix in relationship_suffixes)
        
        return is_plural or has_relationship_suffix
    
    def _get_relationship_inner_type(self, field_name: str, strawberry_type: Type) -> Optional[Type]:
        """Get the inner type of a relationship field."""
        try:
            # Check type annotations first
            type_hints = get_type_hints(strawberry_type)
            if field_name in type_hints:
                field_type = type_hints[field_name]
                inner_type = self._extract_inner_type(field_type)
                if inner_type:
                    return inner_type
            
            # Check strawberry definition fields (for @strawberry.field methods)
            if hasattr(strawberry_type, '__strawberry_definition__'):
                definition = strawberry_type.__strawberry_definition__
                if hasattr(definition, 'fields'):
                    for strawberry_field in definition.fields:
                        if strawberry_field.python_name == field_name:
                            # Get the field's return type from the method annotations
                            if hasattr(strawberry_type, field_name):
                                method = getattr(strawberry_type, field_name)
                                if hasattr(method, '__annotations__') and 'return' in method.__annotations__:
                                    return_type = method.__annotations__['return']
                                    return self._extract_inner_type(return_type)
            
            # Check strawberry field methods (legacy)
            if hasattr(strawberry_type, field_name):
                attr = getattr(strawberry_type, field_name)
                if hasattr(attr, '__strawberry_metadata__') and callable(attr):
                    if hasattr(attr, '__annotations__') and 'return' in attr.__annotations__:
                        return_type = attr.__annotations__['return']
                        return self._extract_inner_type(return_type)
            
            # Try dynamic type resolution as fallback
            return self._resolve_type_dynamically(field_name)
            
        except Exception as e:
            logger.debug(f"Error getting inner type for {field_name}: {e}")
            return None
    
    def _extract_inner_type(self, field_type):
        """Extract the inner type from List[InnerType] or Optional[List[InnerType]]."""
        if not field_type:
            return None
        
        # Handle StrawberryAnnotation objects
        if hasattr(field_type, '__class__') and 'strawberry' in str(field_type.__class__).lower():
            # This is a strawberry annotation, extract the actual type
            if hasattr(field_type, 'annotation'):
                actual_type = field_type.annotation
                return self._extract_inner_type(actual_type)
        
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
    
    def _resolve_type_dynamically(self, field_name: str) -> Optional[Type]:
        """Resolve Strawberry type dynamically based on field name."""
        try:
            import inflection
            
            # Convert field name to singular and capitalize for type name
            # e.g., 'items' -> 'Item', 'orders' -> 'Order'
            singular_name = inflection.singularize(field_name)
            type_name = singular_name.capitalize()
            
            # Try to import from the types module (optional)
            try:
                # This is a fallback for cases where types are defined in a separate module
                # Import path may vary based on project structure
                import importlib
                types_module = importlib.import_module('types')  # Generic types module
                if hasattr(types_module, type_name):
                    resolved_type = getattr(types_module, type_name)
                    logger.debug(f"Dynamically resolved {field_name} -> {type_name}")
                    return resolved_type
            except (ImportError, ModuleNotFoundError):
                logger.debug("Could not import types module for dynamic resolution")
            
            # If not found, log and return None
            logger.debug(f"Could not dynamically resolve type for field: {field_name}")
            return None
            
        except Exception as e:
            logger.debug(f"Error in dynamic type resolution for {field_name}: {e}")
            return None
    
    def get_flat_field_names(self, analysis_result: Dict[str, Any]) -> Set[str]:
        """Get a flat set of all field names from analysis result."""
        all_fields = set(analysis_result['scalar_fields'])
        all_fields.update(analysis_result['relationship_fields'].keys())
        return all_fields
    
    def get_aliased_field_mapping(self, analysis_result: Dict[str, Any]) -> Dict[str, str]:
        """Get mapping from display names (aliases or field names) to actual field names."""
        mapping = {}
        aliases = analysis_result['aliases']
        
        # Add scalar fields
        for field in analysis_result['scalar_fields']:
            # If this is an alias, map it to the real field name
            actual_field = None
            for alias, real_field in aliases.items():
                if alias == field:
                    actual_field = real_field
                    break
            mapping[field] = actual_field or field
        
        # Add relationship fields
        for field in analysis_result['relationship_fields'].keys():
            # If this is an alias, map it to the real field name
            actual_field = None
            for alias, real_field in aliases.items():
                if alias == field:
                    actual_field = real_field
                    break
            mapping[field] = actual_field or field
        
        return mapping


# Global instance for easy importing
query_analyzer = QueryFieldAnalyzer()
