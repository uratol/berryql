# 04. Винести PredicateCompiler

## Мета

Уніфікувати parsing, async resolution, validation і SQL compilation для caller `where`, relation/type/domain scopes та mutation scopes.

## Внутрішня модель

Запровадити immutable predicate IR, який розрізняє:

- відсутній predicate;
- conjunction із кількох fragments;
- column/operator/value predicate;
- trusted SQLAlchemy expression;
- provider, який ще треба resolve-нути.

Назви конкретних dataclasses можуть відрізнятися, але builders не повинні повторно парсити AST/JSON/callables.

## Публічна сумісність

Без змін підтримати чинні форми:

- `where` як JSON string/dict на внутрішніх resolver boundaries;
- `scope` як dict, JSON string, SQLAlchemy expression, callable або list/tuple;
- sync та async callable `(model_cls, info)`;
- camelCase/column aliases;
- `None` у чинних optional scope declarations як відсутність додаткового predicate.

Можна додати explicit `Scope.allow_all()`/`Scope.deny_all()` як additive API, але не видаляти чинні форми.

## Роботи

- Створити `berryql/core/predicates.py` або еквівалент.
- Єдиний раз resolve GraphQL variables і callable/awaitable values до compilation.
- Не використовувати `except TypeError` для повторного виклику callback з іншою кількістю аргументів; signature визначати до виклику або стандартизувати documented signature.
- Компілювати IR у SQLAlchemy expression або adapter-native representation.
- Adapter capability mismatch має бути явним і fail-closed.
- Замінити `scope_to_sql_expr`, `_apply_where_common`, `_mssql_where_from_value` та mutation-local scope resolution одним pipeline або thin compatibility wrappers над ним.

## Acceptance criteria

- Root/nested query, pushdown/fallback і mutation використовують той самий compiler.
- Predicate provider викликається не більше одного разу для одного planned use.
- Unknown column/operator, malformed JSON, provider exception і unsupported adapter не ігноруються.
- Caller `where` завжди AND-иться з trusted scopes.
- Public query/mutation API не змінено.

## Тести

- Matrix: dict/string/expression/sync callable/async callable/list fragments.
- Matrix: root, single relation, list relation, nested relation, mutation update/delete/replace.
- SQLite/PostgreSQL compilation та MSSQL adapter tests.
- Regression test: `TypeError` всередині scope виконує callback один раз і повертає original error.

## Залежності

- Задачі 01–03.
