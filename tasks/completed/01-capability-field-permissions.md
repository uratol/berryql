# 01. Capability-based FieldPermissions

## Мета

Замінити `FieldPermissions(read=..., write=...)` capability-based моделлю. Compatibility adapter для `read`/`write` не створювати.

## Новий API

Базовий shape:

```python
FieldPermissions(
    select=FieldSet.all(),
    filter=FieldSet.all(),
    order=FieldSet.all(),
    create=FieldSet.all(),
    update=FieldSet.all(),
    operations=OperationPermissions(
        create=True,
        update=True,
        delete=True,
        replace=True,
    ),
)
```

Допускається інше точне ім'я для operation dataclass, але capabilities та їхня семантика мають залишитися явними.

## Семантика

- `select`: чи можна повертати значення field.
- `filter`: чи можна використовувати field у caller-controlled `where` і declared filter arguments.
- `order`: чи можна використовувати field у caller-controlled `order_by`/`order_multi`.
- `create`: які payload fields можна прийняти при створенні.
- `update`: які payload fields можна змінювати в існуючому row.
- `operations`: дозволи на самі create/update/delete/replace operations.
- Effective `filter` і `order` завжди перетинаються з `select`, щоб приховане поле не створювало side-channel.
- Schema-level і type-level policies перетинаються для кожної capability та operation.
- Denied scalar/single/custom/aggregate field повертає `None`; denied list relation повертає `[]`.
- Primary key може бути selector для update/delete незалежно від `update`, але не повинен неявно ставати writable value існуючого row.
- `_Insert=True` вимагає create operation; `_Insert=False` — update operation; звичайний merge перевіряє operation після визначення create/update path.
- `_Delete` і `_Replace` перевіряють відповідні operation capabilities.

## Роботи

- Переробити dataclasses, intersection і validation у `berryql/core/permissions.py`.
- Оновити schema/type provider resolution.
- Оновити query selection, controls validation і mutation sanitization на нові capability names.
- Оновити public exports, type hints, README, examples і changelog.
- Видалити підтримку `FieldPermissions.read`, `FieldPermissions.write` та старих constructor arguments.

## Acceptance criteria

- Використання `read=` або `write=` завершується чіткою `TypeError`; автоматичної міграції немає.
- Усі шість field/operation capabilities тестуються окремо та в schema/type intersection.
- `filter`/`order` не можуть ефективно дозволити field, заборонений у `select`.
- Create і update masks працюють рекурсивно для nested relations.
- Existing behavior `None`/`[]` для denied outputs збережено.
- Жодний інший публічний API не змінено.

## Тести

- Unit tests для всіх комбінацій `FieldSet.intersection()` і operation intersection.
- Query tests для select/filter/order capabilities, включно з aliases і camelCase.
- Mutation tests для create/update/delete/replace capabilities та primary-key selector.
- Provider failure і concurrent single-flight resolution.

## Залежності

Немає. Це перша задача roadmap.
