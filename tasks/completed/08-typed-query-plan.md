# 08. Typed та phased QueryPlan

## Мета

Замінити mutable relation-config dictionaries із службовими string flags на typed internal plan та чіткі фази analysis → authorization → compilation → hydration.

## Внутрішні типи

Запровадити dataclasses на кшталт:

- `QueryPlan`;
- `RelationPlan`;
- `SelectionPlan`;
- `AuthorizedQueryPlan`;
- `PushdownDecision`;
- typed filter/order/predicate references з попередніх задач.

Точні назви можуть відрізнятися. Public API не повинен залежати від цих класів.

## Роботи

- Прибрати mutation таких flags, як `_has_explicit_order_by`, `_has_explicit_order_multi`, `_field_permissions_applied`, `from_pushdown`, `skip_reason` із generic dicts.
- Зберігати provenance: caller argument, relation default, type default або trusted scope.
- Resolve GraphQL AST variables один раз під час analysis.
- Authorization phase повертає новий plan, не змінюючи original selection plan.
- Compilation працює лише з authorized plan.
- Hydration отримує explicit projection/relation metadata, а не здогадується за відсутніми keys.
- Fallback decision має typed reason і не може приховати authorization/configuration error.

## Acceptance criteria

- Builders не читають raw GraphQL AST і не викликають permission providers.
- Authorization завжди відбувається до SQL projection/relationship pushdown.
- Selection із лише denied fields все ще повертає правильну кількість rows з `None`/`[]` через мінімальну PK projection.
- Pushdown і fallback produce identical GraphQL data.
- Жодних змін GraphQL schema або public declarations.

## Тести

- Unit snapshot/structural tests для plan phases.
- Fragments, aliases, variables, repeated selections і nested arguments.
- Only-denied selection, custom fields, aggregates і mixed relations.
- Pushdown decision не ковтає compiler/policy errors.

## Залежності

- Задачі 01–07.
