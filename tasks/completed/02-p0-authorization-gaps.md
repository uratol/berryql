# 02. Закрити P0 authorization gaps

## Мета

Виправити відомі fail-open та permission-bypass paths до подальшого структурного рефакторингу.

## Обов'язкові fixes

### `_Replace` і denied relation

- Нормалізувати `_Replace` names до canonical Berry field names до mutation hooks.
- Target relation має пройти `update` capability parent type і `operations.replace`.
- Видалення child rows додатково вимагає `operations.delete` child type.
- Якщо relation недоступна, її nested payload і відповідний `_Replace` target повністю ігноруються; не можна видаляти rows.
- `_Replace` не повинен використовувати відсутність відфільтрованого relation payload як команду `delete all`.

### Scope під час `_Replace`

- Застосовувати resolved scope value, а не початковий callable.
- Підтримати sync/async callable, dict, JSON string, list/tuple fragments і SQLAlchemy expression.
- Не ковтати scope resolution/compilation errors.
- Однаковий scoped predicate має використовуватися для визначення IDs, cascade handling і фінального DELETE.
- Помилка повинна rollback-нути всю merge operation.

### MSSQL scope

- Direct SQLAlchemy expression або async callable не можна мовчки ігнорувати.
- Якщо adapter не може безпечно pushdown-нути predicate, перейти на коректний SQLAlchemy/fallback path або завершити operation явною unsupported/configuration error.
- Ніколи не виконувати ширший query без scope.

### Звичайні Strawberry fields на BerryType

- Final select guard має охоплювати nullable plain Strawberry scalar/object/list resolvers, а не лише Berry descriptors.
- Denied list повертає `[]`, інші nullable outputs — `None`.
- Зафіксувати GraphQL-spec поведінку для user-declared non-null fields; schema не переписувати під час request.
- `requested_other_root` не повинен автоматично обходити policy enforcement.

## Acceptance criteria

- Усі чотири gaps мають окремий failing-before/passing-after regression test.
- Authorization/scope exception ніколи не перетворюється на unscoped query або delete.
- Hooks не бачать denied relation payload чи denied `_Replace` target.
- SQLite tests проходять; MSSQL behavior покритий adapter/compiler unit tests навіть без live database.
- Публічні API, крім уже зміненого `FieldPermissions`, не змінені.

## Regression tests

- Denied `post_comments` + `_Replace=["post_comments"]` з relation payload і без нього.
- Child delete capability denied при дозволеному parent replace.
- Sync, async і throwing relation scope під час replace.
- Scope, що дозволяє лише частину children: unlisted out-of-scope rows не видаляються.
- MSSQL callable, що повертає dict/expression/awaitable.
- Nullable plain Strawberry field/list field denied через select capability.

## Залежності

- Задача 01.
