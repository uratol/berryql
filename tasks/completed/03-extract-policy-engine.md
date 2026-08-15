# 03. Винести PolicyEngine

## Мета

Забрати resolution, caching, validation та enforcement permissions із `BerrySchema`, query builders і mutation executor у єдиний внутрішній `PolicyEngine`.

## Відповідальність компонента

- Резолвити schema-level і type-level providers та обчислювати effective `FieldPermissions`.
- Валідувати canonical field names і capability invariants.
- Кешувати policy один раз на execution/type з підтримкою concurrent single-flight.
- Перевіряти select/filter/order capabilities.
- Санітизувати create/update nested payloads і operation controls до hooks.
- Перевіряти create/update/delete/replace operations.
- Надавати фінальні resolver guards для pushdown і fallback paths.
- Генерувати typed authorization/configuration errors; не генерувати SQL.

## Контракт кешу

- Provider є request/type policy та не залежить від конкретного row або resolver path.
- Контракт явно документований і перевіряється тестами.
- Не використовувати naked `id(...)` як єдину execution identity у довгоживучому context.
- Failed task видаляється з cache; успішний результат не протікає в наступний execution при reuse context.
- Cache не повинен безмежно накопичувати operations.

## Роботи

- Створити окремий internal module, наприклад `berryql/core/policy.py`.
- Перенести `FieldPermissionResolver` і permission helpers із `registry.py`.
- Замінити прямі `permissions.*.allows()` у query/mutation paths методами engine.
- Залишити мінімальний delegation facade на `BerrySchema` лише там, де він потрібен внутрішній сумісності.
- Не переносити predicate/order compilation у цей компонент.

## Acceptance criteria

- У query, relation fallback і mutation немає незалежних реалізацій capability checks.
- Provider викликається рівно один раз на execution/type навіть для sibling root fields і nested selection.
- Reused context не повертає policy попереднього execution.
- Provider exception fail-closed і не кешується як успішний результат.
- Public API не змінено.

## Тести

- Unit tests PolicyEngine без GraphQL execution.
- Concurrent resolution, exception retry і context reuse.
- Schema/type intersection для всіх capabilities/operations.
- Integration tests, що pushdown і fallback дають однаковий результат.

## Залежності

- Задачі 01–02.
