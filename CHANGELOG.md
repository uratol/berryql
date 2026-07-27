# Changelog

All notable changes to this project will be documented here.

## [0.3.6] - 2026-07-27
### Fixed
- Fixed `ArgumentError: SQL expression for WHERE/HAVING role expected, got {...}` raised when a relation/type `scope` defined as a **callable returning a dict** (or JSON string) was resolved on the per-parent fallback path (e.g. polymorphic relations declared with `fk_column_name='entity_id'`, which cannot use LATERAL JSON pushdown). The fallback previously passed the callable's return value straight to `Select.where()` without converting dict/str forms to a SQL expression. A new shared `scope_to_sql_expr` helper now normalizes `None`/dict/JSON-str/callable/SQL-expression scope values consistently across the registry and SQL builders.

### Added
- Added `scope_to_sql_expr` helper in `berryql.core.utils` for uniform scope-value normalization.
- Added regression coverage (`tests/test_callable_scope_dict_fallback.py`) for callable scopes returning a dict on a polymorphic relation, plus dispatch unit tests for the helper.

## [0.3.5] - 2026-06-01
### Fixed
- Ensure a caller-supplied root/domain list `where` filter is ANDed with the relation/domain `scope` guard instead of replacing it, so scoped rows cannot leak when `where` is provided.

### Added
- Added regression coverage for scoped root and domain list queries with caller-provided `where` filters.

## [0.3.4] - 2026-05-27
### Fixed
- Allow `orderBy` to target regular Strawberry fields when they mirror a real model column name, including fields that are not declared as Berry scalars via `field()`.

## [0.3.3] - 2026-04-27
### Added
- Added default pagination configuration support to `BerrySchema`.

## [0.3.2] - 2026-04-23
### Changed
- Added memoization for column type mapping and order-segment normalization to reduce repeated schema work.
- Simplified relation meta retrieval in ordering and selection paths for better readability and maintainability.

## [0.3.1] - 2026-03-25
### Added
- Merge resolver child instances now receive parent context and relation name metadata.

## [0.3.0] - 2026-03-25
### Changed
- Refactored ordering logic in `BerrySchema` and `RootSQLBuilders` for clearer order resolution and maintenance.

## [0.2.9] - 2026-03-25
### Added
- Added normalization for where-clause SQL expression handling, including camelCase coverage in tests.

## [0.2.8] - 2026-03-25
### Added
- Enhanced relation SQL ordering support.
- Improved MSSQL adapter join handling for nested relations and order expressions.

## [0.2.7] - 2026-03-19
### Added
- Added JSON pushdown support for nested single relations.
- Enhanced `RelationSQLBuilders` SQL expression handling and added context-aware tests for custom scalar fields.

## [0.2.6] - 2026-03-19
### Added
- Merge resolver now coerces ISO date and datetime strings to native Python objects, including nested comment merge paths.

## [0.2.5] - 2026-03-18
### Added
- Added fragment resolution in AST processing, with tests covering fragment-based projections.

## [0.2.3] - 2026-02-02
### Added
- Allow `custom` and `custom_object` builders to receive Strawberry `Info` (context) when declared with `(model_cls, info)` signature.

## [0.2.2] - 2026-01-14
### Added
- Added support for `orderBy`, `orderDir`, and `orderMulti` variables in GraphQL queries.
- Added support for variables in `limit` and `offset` arguments.

## [0.2.1] - 2026-01-02
### Fixed
- Fixed `BigInteger` column type mapping to use `Float` instead of `Int` to avoid overflow errors for large values.

## [0.2.0] - 2025-12-24
### Added
- Added nullable reviewer foreign-key coverage and cascade delete tests.

## [0.1.9] - 2025-12-21
### Added
- Added 'not_like', 'not_ilike', 'not_in', 'not_between' operators for query field arguments definition
- Implemented `on_delete` strategy for foreign key handling in merge resolver.

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
