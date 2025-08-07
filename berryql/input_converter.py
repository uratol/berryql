"""
Input converter utilities for GraphQL input types.

This module provides utilities to convert Strawberry GraphQL input types
to the dictionary format expected by the generic resolver factory.
"""

from typing import Dict, Any, List, Optional, Union
from .input_types import (
    ComparisonValue
)


def convert_comparison_input(comparison_input: ComparisonValue) -> Dict[str, Any]:
    """
    Convert a comparison input to the dictionary format expected by the generic resolver.
    
    Args:
        comparison_input: GraphQL comparison input or direct value
        
    Returns:
        Dictionary with operator keys and values
    """
    if comparison_input is None:
        return {}
    
    # Handle direct values (simple equality)
    if not hasattr(comparison_input, '__dict__'):
        return comparison_input  # Return the direct value for simple equality
    
    # Handle comparison input objects
    result = {}
    
    # Get all non-None fields from the input
    for field_name, value in comparison_input.__dict__.items():
        if value is not None:
            # Handle the special 'in_' field (maps to 'in' operator)
            if field_name == 'in_':
                result['in'] = value
            else:
                result[field_name] = value
    
    return result


def convert_where_input(where_fields: Dict[str, ComparisonValue]) -> Dict[str, Any]:
    """
    Convert GraphQL where input fields to the format expected by the generic resolver.
    
    Args:
        where_fields: Dictionary mapping field names to comparison inputs
        
    Returns:
        Dictionary in the format expected by _apply_where_conditions
    """
    if not where_fields:
        return {}
    
    result = {}
    for field_name, comparison_input in where_fields.items():
        converted = convert_comparison_input(comparison_input)
        if converted:  # Only add non-empty conversions
            result[field_name] = converted
    
    return result


def convert_order_by_input(order_by_list: Optional[List[Union[str, Any]]]) -> List[Dict[str, str]]:
    """
    Convert GraphQL order by input to the format expected by the generic resolver.
    
    Args:
        order_by_list: List of strings (e.g., "field", "field asc", "field desc") or objects with 'field' and 'direction' attributes
        
    Returns:
        List of dictionaries with 'field' and 'direction' keys
    """
    if not order_by_list:
        return []
    
    result = []
    for order_input in order_by_list:
        if isinstance(order_input, str):
            # Handle string format: "field" or "field direction" or comma-separated "field1, field2 desc"
            if ',' in order_input:
                # Handle comma-separated fields
                field_specs = [spec.strip() for spec in order_input.split(',')]
                for field_spec in field_specs:
                    if not field_spec:
                        continue  # Skip empty specs from trailing commas
                        
                    parts = field_spec.split()
                    if len(parts) == 1:
                        result.append({
                            'field': parts[0],
                            'direction': 'asc'
                        })
                    elif len(parts) == 2:
                        field_name, direction = parts
                        direction = direction.lower()
                        if direction not in ['asc', 'desc']:
                            raise ValueError(f"Invalid direction '{direction}'. Must be 'asc' or 'desc'")
                        result.append({
                            'field': field_name,
                            'direction': direction
                        })
                    else:
                        raise ValueError(f"Invalid order_by format for field '{field_spec}'. Expected 'field' or 'field direction'")
            else:
                # Handle single field: "field" or "field direction"
                parts = order_input.strip().split()
                if len(parts) == 1:
                    result.append({
                        'field': parts[0],
                        'direction': 'asc'
                    })
                elif len(parts) == 2:
                    field_name, direction = parts
                    direction = direction.lower()
                    if direction not in ['asc', 'desc']:
                        raise ValueError(f"Invalid direction '{direction}'. Must be 'asc' or 'desc'")
                    result.append({
                        'field': field_name,
                        'direction': direction
                    })
                else:
                    raise ValueError(f"Invalid order_by format: '{order_input}'. Expected 'field' or 'field direction'")
        else:
            # Handle object format (for backward compatibility)
            result.append({
                'field': order_input.field,
                'direction': order_input.direction or 'asc'
            })
    
    return result


def convert_field_based_order_by(order_by_input: Any) -> List[Dict[str, str]]:
    """
    Convert field-based order by input to the format expected by the generic resolver.
    Preserves the field order as defined in the GraphQL input.
    
    Args:
        order_by_input: Object with field names as attributes and directions as values
        
    Returns:
        List of dictionaries with 'field' and 'direction' keys in proper order
    """
    if not order_by_input:
        return []
    
    # Use the get_ordered_fields method to preserve field order
    if hasattr(order_by_input, 'get_ordered_fields'):
        ordered_fields = order_by_input.get_ordered_fields()
        result = []
        for field_name, direction in ordered_fields:
            result.append({
                'field': field_name,
                'direction': validate_order_direction(direction)
            })
        return result
    
    # Fallback to dict iteration (may not preserve order)
    result = []
    for field_name, direction in order_by_input.__dict__.items():
        if direction is not None:
            result.append({
                'field': field_name,
                'direction': validate_order_direction(direction)
            })
    
    return result


def validate_order_direction(direction: str) -> str:
    """
    Validate and normalize order direction.
    
    Args:
        direction: Direction string
        
    Returns:
        Normalized direction ('asc' or 'desc')
        
    Raises:
        ValueError: If direction is invalid
    """
    direction = direction.lower().strip()
    if direction not in ('asc', 'desc'):
        raise ValueError(f"Invalid order direction '{direction}'. Must be 'asc' or 'desc'.")
    return direction
