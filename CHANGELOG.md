# Changelog

All notable changes to this project will be documented here.

## [0.5.1] - 2026-08-15
### Fixed
- Restored support for SQLAlchemy SQL operands in where-dict scopes:
  `{'business_id': {'in': select(...).scalar_subquery()}}` — the documented
  multi-tenant scope form — was rejected by 0.5.0's operand-shape validation
  with `Where operator 'in' for '<column>' requires a non-empty list`,
  breaking every scoped query and mutation in consumer apps.
- `in`/`not_in` now pass SQL selectables/expressions to `column.in_()`
  unwrapped, so plain `Select` operands compile to `col IN (SELECT ...)`
  instead of raising SQLAlchemy's `IN expression list, SELECT construct, or
  bound parameter object expected`.
- `coerce_where_value` no longer type-coerces SQL constructs (previously a
  `Select` bound to a boolean column could be coerced to `True`).
- The MSSQL predicate path now compiles SQL-operand predicates through the
  SQLAlchemy dialect instead of silently dropping the condition, and
  `where_from_dict` renders `not_in` list conditions instead of skipping them.
- Trusted scopes are no longer subject to `max_in_items` caller cost limits,
  aligning enforcement with the documented `FilterLimits` contract
  (shape validation still applies; caller filters are unchanged).

## [0.5.0] - 2026-08-14
### Added
- Introduced capability-based `FieldPermissions` with independent `select`,
  `filter`, `order`, `create`, and `update` field masks, plus exported
  `OperationPermissions` controls for create, update, delete, and replace.
- Added internal `PolicyEngine`, immutable-IR `PredicateCompiler`, and
  `OrderingCompiler` components shared by root, relation, fallback, adapter,
  and mutation paths.
- Added immutable typed query/selection/relation plans with typed provenance,
  authorization copies, hydration metadata, and pushdown fallback reasons.
- Added declared-filter dependency metadata, schema-local operator registries,
  optional filter-cost limits, strict operator value-shape validation, and
  uniform sync/async filter execution.
- Added additive keyset pagination through the `after` argument and
  `encode_cursor`/`decode_cursor`, optional `nulls_first`/`nulls_last` ordering,
  and schema/type ordering caches.
- Split query-root and relation-resolver orchestration behind dedicated internal
  service modules and introduced typed configuration, predicate, ordering,
  authorization, and adapter error categories.

### Fixed
- Closed `_Replace` authorization bypasses: relation names are normalized and
  sanitized before hooks, denied relations cannot become delete-all commands,
  parent replace/update and child delete capabilities are enforced, and the
  same resolved relation scope is applied to ID discovery and deletion.
- Made unsupported MSSQL scopes fail closed instead of issuing a broader query.
- Added final select guards for ordinary nullable Strawberry scalar/object/list
  fields declared on `BerryType`.
- Made multi-order validation atomic and added primary-key tie-breakers to
  paginated ordering, with dialect-aware null ordering in Python fallbacks.
- Canonicalized and validated every explicit `FieldSet` name, including deny
  lists, before intersection; unknown names now fail closed with type/capability
  context.
- Applied type row scopes consistently to mutation create/update/delete/replace
  paths and removed payload values from authorization diagnostics.
- Replaced per-row dotted-order fallback loads with batched per-hop queries.

## [0.4.1] - 2026-08-03
### Fixed
- Made `_Insert: false` an explicit update-only mode. It now requires a
  primary key that resolves to an existing row instead of falling through to
  normal merge behavior and inserting an unknown ID. `_Insert` omitted keeps
  normal merge semantics; `_Insert: true` continues to force an explicit-PK
  insert.

## [0.4.0] - 2026-07-29
### Added
- Added deterministic sibling relation merging based on owning `BerryType`
  declaration order, with normalized local `declaration_index` metadata and a
  single ordering helper shared by precreation, `_Replace`, normal recursion,
  and cascade traversal.
- Added public `MergeOperationContext`, `MergeNodeContext`, and
  `DeferredValidation` APIs. All recursive nodes and list items share one
  operation, which records touched model primary keys and shared application
  state.
- Added deduplicated sync/async deferred validators that run after the complete
  graph and final flush but before commit.
- Added incremental schema-level `before_merge`, `before_commit`,
  `after_commit`, and `on_error` hooks through `BerrySchema.merge_hooks()`.

### Changed
- Centralized single and list merge transaction handling under the existing
  AsyncSession lock. Merge, type-hook, validator, and before-commit failures
  now explicitly roll back before `on_error`; an `on_error` failure cannot
  mask the original exception.
- Defined inherited declaration composition: bases retain relative order and
  compose left-to-right, subclass fields append, and overrides move to their
  subclass declaration position.
- Documented explicit-PK `_Insert` references from earlier-declared relations
  to later-declared sibling payloads.

## [0.3.9] - 2026-07-28
### Fixed
- Fixed `_Replace` mutation argument validation failing under `autoCamelCase` (Strawberry `StrawberryConfig(auto_camel_case=True)`). Relation names listed in `_Replace` are now resolved in **both** snake_case (python field name) and camelCase (GraphQL field name) forms. Previously `_Replace: ["postComments"]` raised `_Replace references unknown relation 'postComments'` because only the snake_case key (`post_comments`) was recognized; the string values inside `_Replace` are not remapped by Strawberry the way input field names are.

## [0.3.8] - 2026-07-28
### Added
- Added **nested relation re-parenting** support in merge mutations. When a child record nested under a parent's relation specifies a foreign key pointing to a DIFFERENT parent (e.g. `merge_posts(payload: [{ id: 10, post_comments: [{ id: 123, content: "bar", post_id: 11 }] }])`), the input value now wins: the child is edited AND moved onto the target parent identified by its own FK. Previously such a move was either silently ignored or blocked by the parent-ownership guard.
- Enforces the merge/relation security scope against BOTH the **origin parent** (where the child currently lives) and the **target parent** (where the child is being moved to) during re-parenting. A move across a scope boundary is rejected with `Mutation out of scope for re-parent`.

### Changed
- Relaxed the parent-ownership guard (`Mutation out of scope for update; child does not belong to parent`) for scope-validated re-parents, so a legitimate cross-parent move is no longer blocked once both parents pass the scope check.

## [0.3.7] - 2026-07-28
### Added
- Added the `_Replace` control flag on mutation input payloads for **replace-semantics** on nested list relations. When `_Replace: ["<relation>"]` is set, the mutation upserts the listed child items and deletes every other child belonging to that parent (matching the relation `scope`/FK) whose PK is not in the payload. The delete runs **before** the upsert loop to avoid unique-constraint collisions between new inserts and soon-to-be-deleted rows. Grandchildren are cascade-deleted first for MSSQL FK safety. Entries are validated (unknown relation names and single relations raise before any DB work).
- Added the `_Insert` control flag on mutation input payloads to force insertion of a new row even when a PK is provided. When `_Insert: true`, the update-lookup is skipped and a new row is inserted **with the provided PK value** (explicit-PK insert). If the PK already exists, the DB raises a duplicate-key error (caller's responsibility).

### Fixed
- Fixed MSSQL duplicate-PK failure in `tests/test_callable_scope_dict_fallback.py`: the test's separate `DeclarativeBase` tables are now dropped before `create_all` to ensure a clean slate on persistent backends.

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
