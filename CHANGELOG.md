# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of BerryQL library
- Unified BerryQL Factory for GraphQL query optimization
- Support for Strawberry GraphQL and SQLAlchemy integration
- Advanced query field analysis with fragment support
- Resolved data helper utilities
- Input type converters for GraphQL filtering
- Comprehensive input types for field comparison operations

### Features
- Complete elimination of N+1 query problems through lateral joins
- Dynamic field filtering - only requested fields are queried
- Recursive configuration support at any nesting level
- Support for where/order_by/offset/limit parameters at any level
- Custom field support with query builders
- Automatic field mapping from Strawberry types to SQLAlchemy models
- Type-safe GraphQL input types for filtering and ordering
- Fragment, inline fragment, and alias support
- Mixin and decorator approaches for resolved data access

## [0.1.0] - 2025-01-XX

### Added
- Initial release
