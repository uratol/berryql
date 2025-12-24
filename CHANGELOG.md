# Changelog

All notable changes to this project will be documented here.

## [0.2.0] - 2025-12-24
### Added
- Implemented `on_delete` strategy for foreign key handling in merge resolver.

## [0.1.9] - 2025-12-21
### Added
- Added 'not_like', 'not_ilike', 'not_in', 'not_between' operators for query field arguments definition

## [0.1.8] - 2025-12-20
### Fixed
- Fixed concurrency issues in `custom` and `custom_object` field resolvers by introducing a session-level lock during builder execution.

## [0.1.7] - 2025-12-12
### Fixed
- Fixed evaluation of `builder`-based `custom` / `custom_object` fields on mutation responses when the builder relies on SQLAlchemy correlation (e.g. `.correlate(M)`), ensuring results are scoped to the parent instance.


## [0.1.6] - 2025-12-12
### Changed
- **BREAKING**: Mutations that accept a list payload (default behavior or `single=False`) now return a list of objects (`[Type]`) instead of a single object (`Type`). Previously, it returned the last modified object.

## [0.1.5] - 2025-12-04
### Fixed
- Fixed relation resolution when the parent object is a raw SQLAlchemy model instance (e.g. returned from a native Strawberry field) instead of a BerryQL type instance.
- Fixed `self.__berry_registry__` access in relation resolvers by capturing the schema instance in closure.

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
