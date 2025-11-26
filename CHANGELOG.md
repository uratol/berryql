# Changelog

All notable changes to this project will be documented here.

## [0.1.4] - 2025-11-26
### Added
- Enhanced foreign key handling for single relations in BerrySchema.
- Propagate `parent_ctx` in nested mutations to ensure context availability in deep nesting.

### Changed
- Refactored callback helpers in `mutations.py` to inner scope for better closure handling.
- Simplified meta handling in FieldDescriptor and related functions.
- Improved UUID type imports in utils.

### Removed
- Deprecated inspection and testing scripts related to Strawberry configuration and GraphQL schema introspection.
- Removed `build.py` script.

## [0.1.2] - 2025-11-20

### Changed
- Minor changes

## [0.1.1] - 2025-11-20
- Initial public release on PyPI.
