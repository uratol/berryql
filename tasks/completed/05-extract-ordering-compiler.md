# 05. Винести OrderingCompiler

## Мета

Зосередити parsing, permission validation, expression resolution та deterministic ordering в одному компоненті для root, relation pushdown і fallback execution.

## Публічна сумісність

- Зберегти `order_by`, `order_dir`, `order_multi=["field:direction"]` та relation/type defaults.
- Зберегти camelCase, scalar column aliases, callable/default SQL expressions і dotted single-relation paths.
- Нові typed helpers можуть бути лише additive.

## Роботи

- Створити `berryql/core/ordering.py` або еквівалент.
- Нормалізувати input у immutable `OrderTerm(path, direction, ...)`.
- Strictly validate кожен `order_multi` term, field path і direction; invalid term не можна мовчки пропустити.
- Перевіряти `order` capability через `PolicyEngine` для кожного segment path.
- Використовувати один expression resolver для root і nested relation builders.
- Для paginated query завжди додавати primary key як останній tie-breaker, якщо його ще немає.
- Узгодити SQL ordering і Python fallback ordering, включно з `None` values.
- Прибрати дубльоване partition/normalization з registry/builders.

## Acceptance criteria

- Invalid `order_by`, будь-який invalid `order_multi` term і invalid direction дають стабільну GraphQL error.
- Частково валідний `order_multi` не виконується частково.
- Однаковий ordering result у pushdown і fallback paths.
- Offset/limit pagination детермінована при однакових значеннях основного sort field.
- Denied order path завершується authorization error до SQL execution.
- Public API не змінено.

## Тести

- Direct, aliased, camelCase та dotted paths.
- Invalid first/middle/last multi-order term і direction.
- Duplicate PK term та automatic PK tie-breaker.
- Null values, asc/desc, root/nested/fallback parity.
- SQL assertion, що PK присутній у paginated ORDER BY.

## Залежності

- Задачі 01–04.
