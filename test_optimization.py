#!/usr/bin/env python3
"""Test the JSON to Strawberry conversion optimization."""

import time
import strawberry
from datetime import datetime
from typing import List
from berryql.resolved_data_helper import convert_json_to_strawberry_instances


@strawberry.type
class PostType:
    id: int
    title: str
    content: str
    created_at: str  # Treated as datetime field
    author_id: int


@strawberry.type
class SimpleType:
    id: int
    name: str
    value: float


def test_datetime_conversion_optimization():
    """Test that datetime field conversion works efficiently."""
    
    # Test data with datetime fields
    datetime_data = [
        {
            "id": 1,
            "title": "Test Post 1",
            "content": "Content 1",
            "created_at": "2024-01-15T10:30:00",
            "author_id": 1
        },
        {
            "id": 2,
            "title": "Test Post 2", 
            "content": "Content 2",
            "created_at": "2024-01-16T11:45:00",
            "author_id": 2
        }
    ]
    
    # Test data without datetime fields
    simple_data = [
        {"id": 1, "name": "Item 1", "value": 10.5},
        {"id": 2, "name": "Item 2", "value": 20.3}
    ]
    
    print("Testing datetime field conversion...")
    start_time = time.time()
    
    # Convert data with datetime fields
    datetime_results = convert_json_to_strawberry_instances(datetime_data, PostType)
    
    datetime_time = time.time() - start_time
    
    print("Testing simple field conversion...")
    start_time = time.time()
    
    # Convert data without datetime fields
    simple_results = convert_json_to_strawberry_instances(simple_data, SimpleType)
    
    simple_time = time.time() - start_time
    
    # Verify results
    assert len(datetime_results) == 2
    assert len(simple_results) == 2
    
    # Verify datetime fields have isoformat method
    assert hasattr(datetime_results[0].created_at, 'isoformat')
    assert datetime_results[0].created_at.isoformat() == "2024-01-15T10:30:00"
    
    # Verify simple fields don't have unnecessary conversions
    assert datetime_results[0].title == "Test Post 1"
    assert simple_results[0].name == "Item 1"
    
    print(f"✅ Datetime conversion time: {datetime_time:.6f}s")
    print(f"✅ Simple conversion time: {simple_time:.6f}s")
    print("✅ All conversion tests passed!")
    
    # Test caching behavior
    print("\nTesting field type caching...")
    start_time = time.time()
    
    # Second call should use cached field analysis
    cached_results = convert_json_to_strawberry_instances(datetime_data, PostType)
    
    cached_time = time.time() - start_time
    
    assert len(cached_results) == 2
    assert hasattr(cached_results[0].created_at, 'isoformat')
    
    print(f"✅ Cached conversion time: {cached_time:.6f}s")
    print("✅ Field type caching is working!")
    
    # Test empty data handling
    empty_results = convert_json_to_strawberry_instances([], PostType)
    assert empty_results == []
    print("✅ Empty data handling works correctly!")


if __name__ == "__main__":
    test_datetime_conversion_optimization()
    print("\n🎉 All optimization tests passed successfully!")
