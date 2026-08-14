# 07. Where та declared-filter hardening

## Мета

Зробити caller filters строгими, однаковими для root/nested paths і безпечними щодо hidden fields та query cost.

## Роботи

- Виправити `FilterSpec.clone_with()`, щоб зберігався `arg_type` та всі інші metadata.
- Додати additive `depends_on`/`fields` metadata для builder filters.
- `PolicyEngine` перевіряє `filter` capability для `column` і всіх declared dependencies до виклику transform/builder.
- Root і nested filter transforms/builders мають однакові sync/async semantics та однакове error handling.
- Не ковтати transform/builder/unknown operator/unknown column errors у relation builders.
- Валідувати arity/value shape для `between`, `in`, `not_in` та custom operators до SQLAlchemy construction.
- Додати configurable limits на кількість clauses, розмір `in`, JSON length і nesting depth із backward-compatible permissive defaults.
- Зробити operator registry schema-aware; чинний global `register_operator` залишити робочим як compatibility source/default.
- Зберегти поточний GraphQL `where` argument shape; typed predicate IR є внутрішнім. Будь-який новий typed public filter API — лише additive.

## Acceptance criteria

- Builder filter без dependency metadata не може неявно заявляти доступ до hidden field; documented compatibility behavior явний.
- Root і relation filter failures дають однаковий class/message structure.
- Жоден invalid filter не перетворюється на ширшу вибірку.
- Multi-op `FilterSpec(arg_type=...)` зберігає правильні GraphQL argument types.
- Existing filters і `where` queries продовжують працювати без змін declarations.

## Тести

- Custom builder за readable/hidden dependency.
- Sync/async transform і builder для root/nested.
- Invalid arity, empty/malformed values та custom operators.
- Query-cost limits і defaults.
- Schema-local operator не протікає в іншу schema.

## Залежності

- Задачі 01–06.
