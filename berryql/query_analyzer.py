"""
GraphQL Query Field Analysis

This module provides enhanced analysis of GraphQL query fields, including
proper handling of fragments, inline fragments, aliases, and multiple field nodes.
"""

import logging
from typing import Dict, Set, List, Any, Type, get_type_hints, Union, Optional
from graphql import GraphQLResolveInfo
from .naming import camel_to_snake, snake_to_camel

logger = logging.getLogger(__name__)
 
class QueryFieldAnalyzer:
    """Analyzes GraphQL query fields using Strawberry's selected_fields API.

    DRY refactor:
    - Added _empty_analysis_result() factory to avoid repeating the empty analysis structure.
    - Removed duplicate implementations of relationship detection & helper methods.
    - Unified alias mapping & flat field name helpers (include object relationships).
    - Consolidated inner type extraction logic.
    """

    @staticmethod
    def _empty_analysis_result() -> Dict[str, Any]:
        return {
            'scalar_fields': set(),
            'relationship_fields': {},
            'object_relationship_fields': {},
            'aliases': {},
            'fragments': []
        }

    def analyze_query_fields(
        self,
        info: GraphQLResolveInfo,
        strawberry_type: Type,
        depth_limit: int = 10,
    ) -> Dict[str, Any]:
        if depth_limit <= 0:
            return self._empty_analysis_result()

        all_selections: List[Any] = []
        if hasattr(info, 'selected_fields') and info.selected_fields:
            if len(info.selected_fields) > 0:
                current_field = info.selected_fields[0]
                if hasattr(current_field, 'selections') and current_field.selections:
                    all_selections = current_field.selections
                else:
                    return self._empty_analysis_result()
            else:
                return self._empty_analysis_result()
        else:
            return self._empty_analysis_result()

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
        object_relationship_fields: Dict[str, Any] = {}
        aliases: Dict[str, str] = {}
        fragments: List[Any] = []
        for selected_field in selected_fields:
            field_result = self._analyze_selected_field(selected_field, strawberry_type, info, depth_limit)
            scalar_fields.update(field_result['scalar_fields'])
            relationship_fields.update(field_result['relationship_fields'])
            # merge object (single) relationships
            if field_result.get('object_relationship_fields'):
                object_relationship_fields.update(field_result['object_relationship_fields'])
            aliases.update(field_result['aliases'])
        return {
            'scalar_fields': scalar_fields,
            'relationship_fields': relationship_fields,
            'object_relationship_fields': object_relationship_fields,
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
        object_relationship_fields: Dict[str, Any] = {}
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
            # Detect single-object (parent) relationship (non-list) returning a Strawberry type
            obj_rel_type = self._get_object_relationship_type(field_name, strawberry_type)
            if obj_rel_type is not None:
                if selected_field.selections and depth_limit > 0:
                    sub_analysis = self._analyze_selected_fields(
                        selected_field.selections, obj_rel_type, info, depth_limit - 1
                    )
                    object_relationship_fields[display_name] = {
                        'type': obj_rel_type,
                        'nested_fields': sub_analysis,
                        'field_arguments': field_arguments,
                    }
                else:
                    object_relationship_fields[display_name] = {
                        'type': obj_rel_type,
                        'field_arguments': field_arguments,
                        'nested_fields': {
                            'scalar_fields': set(),
                            'relationship_fields': {},
                            'object_relationship_fields': {},
                            'aliases': {},
                            'fragments': []
                        }
                    }
            else:
                if not field_name.startswith('__'):
                    scalar_fields.add(display_name)
                    if field_arguments:
                        self._validate_field_arguments(field_arguments, field_name)

        return {
            'scalar_fields': scalar_fields,
            'relationship_fields': relationship_fields,
            'object_relationship_fields': object_relationship_fields,
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

    # (Removed earlier duplicate relationship helper implementations in favor of consolidated versions below.)

    def _is_relationship_field_by_name(self, field_name: str, strawberry_type: Type) -> bool:
        """Check if a field is a relationship field by analyzing the Strawberry type."""
        try:
            logger.debug(f"Checking if {field_name} is relationship field on {strawberry_type.__name__}")
            # Unified retrieval of return annotation (handles snake/camel, type hints, methods)
            return_annotation = self._get_field_return_annotation(field_name, strawberry_type)
            if return_annotation is not None:
                is_rel = self._is_relationship_field_type(return_annotation)
                logger.debug(f"Relationship check via unified helper: {field_name} -> {return_annotation} (is_rel={is_rel})")
                if is_rel:
                    return True
            
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
            return_annotation = self._get_field_return_annotation(field_name, strawberry_type)
            if return_annotation is not None:
                inner_type = self._extract_inner_type(return_annotation)
                if inner_type:
                    return inner_type
            # Dynamic fallback (singularize etc.)
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
    
    # Final public helpers (DRY, include object relationships) -----------------
    def get_flat_field_names(self, analysis_result: Dict[str, Any]) -> Set[str]:
        all_fields = set(analysis_result['scalar_fields'])
        all_fields.update(analysis_result['relationship_fields'].keys())
        all_fields.update(analysis_result.get('object_relationship_fields', {}).keys())
        return all_fields

    def get_aliased_field_mapping(self, analysis_result: Dict[str, Any]) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        aliases = analysis_result['aliases']
        for field in self.get_flat_field_names(analysis_result):
            mapping[field] = aliases.get(field, field)
        return mapping

    # -------- New helper methods for object (parent) relationships --------
    def _get_object_relationship_type(self, field_name: str, strawberry_type: Type) -> Optional[Type]:
        """Return the Strawberry type for a single-object (non-list) relationship if present."""
        try:
            type_hints = get_type_hints(strawberry_type)
            if field_name in type_hints:
                field_type = type_hints[field_name]
                # unwrap Optional
                origin = getattr(field_type, '__origin__', None)
                if origin is Union:
                    args = getattr(field_type, '__args__', ())
                    non_none = [a for a in args if a is not type(None)]  # noqa: E721
                    field_type = non_none[0] if non_none else field_type
                # If still a list, not object
                if getattr(field_type, '__origin__', None) is list:
                    return None
                if hasattr(field_type, '__strawberry_definition__'):
                    return field_type
            # method-based
            if hasattr(strawberry_type, field_name):
                attr = getattr(strawberry_type, field_name)
                if callable(attr) and hasattr(attr, '__annotations__') and 'return' in attr.__annotations__:
                    rt = attr.__annotations__['return']
                    origin = getattr(rt, '__origin__', None)
                    if origin is list:
                        return None
                    if origin is Union:
                        args = getattr(rt, '__args__', ())
                        non_none = [a for a in args if a is not type(None)]  # noqa: E721
                        if non_none:
                            rt = non_none[0]
                            if getattr(rt, '__origin__', None) is list:
                                return None
                    if hasattr(rt, '__strawberry_definition__'):
                        return rt
        except Exception:
            return None
        return None

    # ---------- Internal helpers for flexible field name matching ----------
    def _field_name_matches(self, strawberry_field, field_name: str) -> bool:
        """Return True if the provided GraphQL field_name matches the Strawberry field.

        Supports snake_case (python) vs camelCase (GraphQL) naming differences.
        Matching rules:
        - Direct match against python_name
        - Direct match against field's GraphQL name (if present)
        - Case-insensitive match with underscores removed
        - snake_case <-> camelCase transformations
        """
        try:
            python_name = getattr(strawberry_field, 'python_name', None)
            gql_name = getattr(strawberry_field, 'name', python_name)
            if not field_name:
                return False

            def normalize(s: str) -> str:
                return s.replace('_', '').lower()

            def to_camel(s: str) -> str:
                return snake_to_camel(camel_to_snake(s))  # normalize then convert

            candidates = set()
            for c in (python_name, gql_name):
                if c:
                    candidates.add(c)
            # Add transformed variants
            more = set()
            for c in list(candidates):
                more.add(to_camel(c))
                more.add(camel_to_snake(c))
            candidates.update(more)

            norm_field = normalize(field_name)
            for c in candidates:
                if normalize(c) == norm_field:
                    return True
            return False
        except Exception:
            return False

    # ---------- Shared low-level retrieval helper (DRY) ----------
    def _get_field_return_annotation(self, field_name: str, strawberry_type: Type) -> Optional[Any]:
        """Return the annotated return type for a field if resolvable.

        Handles:
        - Direct dataclass-style attribute annotations (type hints)
        - @strawberry.field methods (using strawberry definition for reliable python_name)
        - Legacy callable fields with __strawberry_metadata__
        - snake_case / camelCase matching via _field_name_matches
        Returns None if not resolvable.
        """
        # 1. Direct type hints (attempt flexible matching across keys)
        type_hints = get_type_hints(strawberry_type)
        if field_name in type_hints:
            return type_hints[field_name]
        else:
            # Try matching keys with normalization (snake/camel)
            for hint_key in type_hints.keys():
                class _Tmp:  # lightweight object for matching
                    python_name = hint_key
                    name = hint_key
                if self._field_name_matches(_Tmp, field_name):
                    return type_hints[hint_key]

        # 2. Strawberry definition based methods
        if hasattr(strawberry_type, '__strawberry_definition__'):
            definition = strawberry_type.__strawberry_definition__
            fields = getattr(definition, 'fields', [])
            for strawberry_field in fields:
                if self._field_name_matches(strawberry_field, field_name):
                    python_name = getattr(strawberry_field, 'python_name', field_name)
                    if hasattr(strawberry_type, python_name):
                        method = getattr(strawberry_type, python_name)
                        annotations = getattr(method, '__annotations__', {})
                        if 'return' in annotations:
                            return annotations['return']

        # 3. Legacy direct attribute method (only if exact attribute exists)
        if hasattr(strawberry_type, field_name):
            attr = getattr(strawberry_type, field_name)
            if callable(attr):
                annotations = getattr(attr, '__annotations__', {})
                if 'return' in annotations:
                    return annotations['return']
        return None


# Global instance for easy importing
query_analyzer = QueryFieldAnalyzer()
