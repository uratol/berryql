"""
GraphQL Query Field Analysis

This module provides enhanced analysis of GraphQL query fields, including
proper handling of fragments, inline fragments, aliases, and multiple field nodes.
"""

import logging
from typing import Dict, Set, List, Any, Type, get_type_hints, Union, Optional
from graphql import GraphQLResolveInfo, FieldNode, InlineFragmentNode, FragmentSpreadNode
from graphql.language.ast import ObjectValueNode, ListValueNode, ValueNode

logger = logging.getLogger(__name__)


class QueryFieldAnalyzer:
    """Analyzes GraphQL query fields with support for fragments and aliases."""
    
    def _ast_value_to_python(self, value_node: ValueNode, info: GraphQLResolveInfo = None) -> Any:
        """Convert a GraphQL AST value node to a Python object."""
        if hasattr(value_node, 'value'):
            # Simple literal value (StringValue, IntValue, BooleanValue, etc.)
            return value_node.value
        elif isinstance(value_node, ObjectValueNode):
            # Object literal - convert to dictionary
            result = {}
            for field in value_node.fields:
                field_name = field.name.value
                field_value = self._ast_value_to_python(field.value, info)
                result[field_name] = field_value
            logger.debug(f"Converted ObjectValueNode to dict: {result}")
            return result
        elif isinstance(value_node, ListValueNode):
            # Array literal - convert to list
            return [self._ast_value_to_python(item, info) for item in value_node.values]
        elif hasattr(value_node, 'name'):
            # Variable reference - resolve from execution context
            variable_name = value_node.name.value
            if info and hasattr(info, 'variable_values') and variable_name in info.variable_values:
                logger.debug(f"Resolving variable ${variable_name} to: {info.variable_values[variable_name]}")
                return info.variable_values[variable_name]
            else:
                # Fallback to variable reference string if we can't resolve it
                logger.warning(f"Could not resolve variable ${variable_name}, treating as literal string")
                return f"${variable_name}"
        else:
            # Fallback - convert to string representation
            logger.warning(f"Unknown value node type: {type(value_node)} - falling back to string conversion")
            return str(value_node)
    
    def analyze_query_fields(
        self, 
        info: GraphQLResolveInfo, 
        strawberry_type: Type,
        depth_limit: int = 10  # Increased from 3 to support deeper nesting
    ) -> Dict[str, Any]:
        """
        Analyze all requested fields from GraphQL query including fragments and aliases.
        
        Args:
            info: GraphQL resolve info containing field nodes
            strawberry_type: The Strawberry GraphQL type being resolved
            depth_limit: Maximum depth to analyze (prevents infinite recursion)
            
        Returns:
            Dict containing:
            - 'scalar_fields': Set of scalar field names
            - 'relationship_fields': Dict mapping field names to their inner types
            - 'aliases': Dict mapping aliases to actual field names
            - 'fragments': List of fragment information
        """
        if depth_limit <= 0:
            logger.warning("Depth limit reached in query analysis")
            return {
                'scalar_fields': set(),
                'relationship_fields': {},
                'aliases': {},
                'fragments': []
            }
        
        logger.debug(f"Analyzing query fields for {strawberry_type.__name__}")
        
        # Process all field nodes (not just the first one)
        all_selections = []
        for field_node in info.field_nodes:
            if hasattr(field_node, 'selection_set') and field_node.selection_set:
                all_selections.extend(field_node.selection_set.selections)
        
        # Analyze all selections
        analysis_result = self._analyze_selections(
            all_selections, 
            strawberry_type, 
            info,
            depth_limit - 1
        )
        
        logger.debug(f"Analysis result for {strawberry_type.__name__}: "
                    f"scalar_fields={analysis_result['scalar_fields']}, "
                    f"relationship_fields={list(analysis_result['relationship_fields'].keys())}")
        
        return analysis_result
    
    def _analyze_selections(
        self, 
        selections: List[Any], 
        strawberry_type: Type,
        info: GraphQLResolveInfo,
        depth_limit: int
    ) -> Dict[str, Any]:
        """Analyze a list of GraphQL selections (fields, fragments, inline fragments)."""
        scalar_fields = set()
        relationship_fields = {}
        aliases = {}
        fragments = []
        
        for selection in selections:
            if isinstance(selection, FieldNode):
                # Handle regular field selection
                field_result = self._analyze_field_node(
                    selection, strawberry_type, info, depth_limit
                )
                scalar_fields.update(field_result['scalar_fields'])
                relationship_fields.update(field_result['relationship_fields'])
                aliases.update(field_result['aliases'])
                
            elif isinstance(selection, InlineFragmentNode):
                # Handle inline fragment (... on TypeName { fields })
                fragment_result = self._analyze_inline_fragment(
                    selection, strawberry_type, info, depth_limit
                )
                scalar_fields.update(fragment_result['scalar_fields'])
                relationship_fields.update(fragment_result['relationship_fields'])
                aliases.update(fragment_result['aliases'])
                fragments.append({
                    'type': 'inline',
                    'type_condition': selection.type_condition.name.value if selection.type_condition else None,
                    'fields': fragment_result['scalar_fields'].union(set(fragment_result['relationship_fields'].keys()))
                })
                
            elif isinstance(selection, FragmentSpreadNode):
                # Handle fragment spread (...FragmentName)
                fragment_result = self._analyze_fragment_spread(
                    selection, strawberry_type, info, depth_limit
                )
                scalar_fields.update(fragment_result['scalar_fields'])
                relationship_fields.update(fragment_result['relationship_fields'])
                aliases.update(fragment_result['aliases'])
                fragments.append({
                    'type': 'spread',
                    'name': selection.name.value,
                    'fields': fragment_result['scalar_fields'].union(set(fragment_result['relationship_fields'].keys()))
                })
        
        return {
            'scalar_fields': scalar_fields,
            'relationship_fields': relationship_fields,
            'aliases': aliases,
            'fragments': fragments
        }
    
    def _analyze_field_node(
        self, 
        field_node: FieldNode, 
        strawberry_type: Type,
        info: GraphQLResolveInfo,
        depth_limit: int
    ) -> Dict[str, Any]:
        """Analyze a single field node."""
        field_name = field_node.name.value
        alias = field_node.alias.value if field_node.alias else None
        
        scalar_fields = set()
        relationship_fields = {}
        aliases = {}
        
        # Extract field arguments
        field_arguments = {}
        if hasattr(field_node, 'arguments') and field_node.arguments:
            for arg in field_node.arguments:
                arg_name = arg.name.value
                # Convert GraphQL AST value to Python object
                field_arguments[arg_name] = self._ast_value_to_python(arg.value, info)
        
        # Handle alias
        if alias:
            aliases[alias] = field_name
            display_name = alias
        else:
            display_name = field_name
        
        # Check if this is a relationship field
        if self._is_relationship_field_by_name(field_name, strawberry_type):
            logger.debug(f"Field {field_name} detected as relationship field")
            # Get the inner type for the relationship
            inner_type = self._get_relationship_inner_type(field_name, strawberry_type)
            if inner_type:
                relationship_fields[display_name] = inner_type
                
                # If the field has sub-selections, analyze them recursively
                if hasattr(field_node, 'selection_set') and field_node.selection_set and depth_limit > 0:
                    logger.debug(f"Analyzing sub-selections for {display_name}, depth_limit: {depth_limit}")
                    sub_analysis = self._analyze_selections(
                        field_node.selection_set.selections,
                        inner_type,
                        info,
                        depth_limit - 1
                    )
                    logger.debug(f"Sub-analysis for {display_name}: scalar_fields={sub_analysis['scalar_fields']}, relationship_fields={list(sub_analysis['relationship_fields'].keys())}")
                    # Store nested field info in the relationship
                    relationship_fields[display_name] = {
                        'type': inner_type,
                        'nested_fields': sub_analysis,
                        'field_arguments': field_arguments
                    }
                else:
                    logger.debug(f"No sub-selections for {display_name} or depth limit reached")
                    relationship_fields[display_name] = {
                        'type': inner_type,
                        'field_arguments': field_arguments
                    }
            else:
                # Fallback: treat as scalar if we can't determine the type
                logger.debug(f"Field {field_name} relationship detected but no inner type found, treating as scalar")
                scalar_fields.add(display_name)
                
                # Validate field arguments even for fallback scalar fields
                if field_arguments:
                    self._validate_field_arguments(field_arguments, field_name)
        else:
            # Regular scalar field - but skip GraphQL meta fields
            if field_name.startswith('__'):
                # All GraphQL meta fields start with __ (like __typename, __schema, __type)
                # These are handled by GraphQL execution engine, not database
                logger.debug(f"Skipping GraphQL meta field: {field_name}")
            else:
                logger.debug(f"Field {field_name} detected as scalar field")
                scalar_fields.add(display_name)
                
                # Even scalar fields can have arguments that need validation
                # For example, scalar field computed fields with parameters
                if field_arguments:
                    # Validate field arguments for scalar fields too
                    self._validate_field_arguments(field_arguments, field_name)
        
        return {
            'scalar_fields': scalar_fields,
            'relationship_fields': relationship_fields,
            'aliases': aliases
        }
    
    def _validate_field_arguments(self, field_arguments: Dict[str, Any], field_name: str):
        """Validate field arguments and throw InvalidFieldError for invalid JSON."""
        # Import here to avoid circular imports
        from .factory import InvalidFieldError
        
        # Check if there's a 'where' argument that needs JSON validation
        if 'where' in field_arguments:
            where_value = field_arguments['where']
            if isinstance(where_value, str):
                try:
                    import json
                    parsed = json.loads(where_value.strip()) if where_value.strip() else {}
                    if not isinstance(parsed, dict):
                        raise InvalidFieldError(f"Where clause in field '{field_name}' must be a JSON object, got: {type(parsed).__name__}")
                except json.JSONDecodeError as e:
                    raise InvalidFieldError(f"Invalid JSON in where clause for field '{field_name}': {where_value}. Error: {e}")
        
        # Add other argument validations as needed
    
    def _analyze_inline_fragment(
        self, 
        inline_fragment: InlineFragmentNode, 
        strawberry_type: Type,
        info: GraphQLResolveInfo,
        depth_limit: int
    ) -> Dict[str, Any]:
        """Analyze an inline fragment."""
        # For inline fragments, we need to check the type condition
        target_type = strawberry_type  # Default to current type
        
        if inline_fragment.type_condition:
            type_name = inline_fragment.type_condition.name.value
            # Try to resolve the target type - for now, use the current type
            # In a more sophisticated implementation, you'd resolve the actual type
            logger.debug(f"Inline fragment for type: {type_name}")
        
        # Analyze the fragment's selections
        if inline_fragment.selection_set:
            return self._analyze_selections(
                inline_fragment.selection_set.selections,
                target_type,
                info,
                depth_limit
            )
        
        return {
            'scalar_fields': set(),
            'relationship_fields': {},
            'aliases': {},
            'fragments': []
        }
    
    def _analyze_fragment_spread(
        self, 
        fragment_spread: FragmentSpreadNode, 
        strawberry_type: Type,
        info: GraphQLResolveInfo,
        depth_limit: int
    ) -> Dict[str, Any]:
        """Analyze a fragment spread."""
        fragment_name = fragment_spread.name.value
        
        # Get the fragment definition from the document
        fragment_def = None
        if info.fragments and fragment_name in info.fragments:
            fragment_def = info.fragments[fragment_name]
        
        if fragment_def and fragment_def.selection_set:
            # Determine target type from fragment
            target_type = strawberry_type  # Default fallback
            if fragment_def.type_condition:
                type_name = fragment_def.type_condition.name.value
                logger.debug(f"Fragment spread '{fragment_name}' for type: {type_name}")
                # In a more sophisticated implementation, resolve the actual type
            
            # Analyze the fragment's selections
            return self._analyze_selections(
                fragment_def.selection_set.selections,
                target_type,
                info,
                depth_limit
            )
        
        logger.warning(f"Fragment '{fragment_name}' not found or has no selections")
        return {
            'scalar_fields': set(),
            'relationship_fields': {},
            'aliases': {},
            'fragments': []
        }
    
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
            
            # Try to import from the types module
            try:
                from app.graphql import types
                if hasattr(types, type_name):
                    resolved_type = getattr(types, type_name)
                    logger.debug(f"Dynamically resolved {field_name} -> {type_name}")
                    return resolved_type
            except ImportError:
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
