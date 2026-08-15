# Authorization and query pipeline roadmap

## Глобальні правила

- Задачі виконуються за номером; задача може починатися лише після завершення всіх указаних залежностей.
- Єдина дозволена несумісна зміна публічного API — новий capability-based constructor і атрибути `FieldPermissions` у задачі 01.
- Для `relation()`, `scope()`, `mutation()`, `FilterSpec`, `order_by`, `order_multi`, `where`, decorators та сформованої GraphQL schema зберігається сумісність, якщо конкретна задача не додає нову опціональну можливість.
- Security paths працюють fail-closed: помилки policy/scope/authorization не можна перетворювати на ширшу вибірку або mutation.
- Кожна задача включає документацію, unit/regression tests і повний запуск тестів, релевантний підтримуваним адаптерам.
- Не змішувати security fixes із несуміжним форматуванням чи змінами моделей.

## Хвиля 1 — виконати першою

1. [01-capability-field-permissions.md](01-capability-field-permissions.md) — capability-based `FieldPermissions` без compatibility adapter.
2. [02-p0-authorization-gaps.md](02-p0-authorization-gaps.md) — закрити P0 bypass-и та додати regression tests.
3. [03-extract-policy-engine.md](03-extract-policy-engine.md) — винести policy resolution і enforcement.
4. [04-extract-predicate-compiler.md](04-extract-predicate-compiler.md) — уніфікувати `scope`/`where` compilation.
5. [05-extract-ordering-compiler.md](05-extract-ordering-compiler.md) — уніфікувати validation та compilation ordering.

## Хвиля 2 — решта покращень

6. [06-policy-and-scope-hardening.md](06-policy-and-scope-hardening.md)
7. [07-where-and-filter-hardening.md](07-where-and-filter-hardening.md)
8. [08-typed-query-plan.md](08-typed-query-plan.md)
9. [09-registry-modularization.md](09-registry-modularization.md)
10. [10-ordering-pagination-performance.md](10-ordering-pagination-performance.md)
11. [11-verification-docs-release.md](11-verification-docs-release.md)

## Definition of done для roadmap

- Усі authorization invariants перевіряються до SQL execution і повторно на resolver boundary, де це потрібно для fallback paths.
- Жоден adapter або fallback не ігнорує scope, permission чи invalid user control.
- Query і mutation paths використовують одні й ті самі policy/predicate/order abstractions.
- За винятком нового `FieldPermissions`, чинний Python і GraphQL API залишається сумісним.
