# 11. Cross-adapter verification, документація та release readiness

## Мета

Завершити roadmap повною матрицею invariants, міграційною документацією для єдиного breaking API та release checks.

## Test matrix

Покрити комбінації:

- root/single/list/nested relation;
- pushdown/fallback;
- SQLite/PostgreSQL/MSSQL adapter behavior;
- dict/string/expression/sync/async scopes;
- select/filter/order/create/update і operations capabilities;
- schema/type policy intersection;
- aliases, camelCase, fragments і variables;
- create/update/delete/replace/reparent;
- plain Strawberry, custom, custom_object та aggregate fields;
- concurrent sibling roots і reused context;
- invalid configuration та provider/compiler failures.

Додати property-based або exhaustive unit tests для `FieldSet`/capability intersections і predicate/order normalization, якщо dev dependencies це дозволяють без ускладнення package runtime.

## Документація

- Оновити README declaration examples на capability-based `FieldPermissions`.
- Додати migration guide `read → select/filter/order`, `write → create/update` та operation permissions. Це документація міграції, не runtime adapter.
- Описати row scope проти field capability та request/type policy проти per-row authorization.
- Описати fail-closed behavior і adapter limitations.
- Документувати denied output semantics: scalar/single/custom/aggregate `null`, list relation `[]`.
- Описати `_Insert`, `_Delete`, `_Replace` authorization semantics.
- Оновити changelog із єдиним дозволеним breaking change.

## Release checks

- Full pytest suite без нових unexpected skips.
- Formatter/linter/type checks, `compileall`, package build та `git diff --check`.
- GraphQL SDL comparison для підтвердження відсутності неузгоджених API changes.
- SQL statement assertions, що trusted scopes присутні у всіх relevant paths.
- Немає sensitive field values у warnings/audit logs.

## Acceptance criteria

- Усі задачі 01–10 завершені та їхні regression tests увійшли до загальної suite.
- Єдиний задокументований breaking Python API — `FieldPermissions`.
- Existing `scope`, `where`, filter, order, mutation і schema declaration APIs залишаються робочими.
- Documentation examples виконуються як tests або перевіряються окремими smoke tests.

## Залежності

- Задачі 01–10.
