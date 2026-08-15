# 06. Policy та scope hardening

## Мета

Після extraction закріпити однозначні policy/scope invariants без зміни чинних declarations.

## Роботи

- Валідувати всі explicit names у `FieldSet` проти canonical fields/type metadata під час policy resolution.
- Unknown name у `only` або `all_except` має давати configuration error із type/provider context; typo не може fail-open.
- Підтримати aliases і auto-camel-case лише на input boundary; effective policy зберігати в canonical Python names.
- Зафіксувати type-level scope semantics як row policy і застосувати до root/nested reads та create/update/delete/replace mutations.
- Relation/domain/type scopes завжди поєднувати через AND.
- Додати dependency metadata для derived/custom/aggregate fields, щоб policy міг перевірити можливі side-channels; існуючі declarations без metadata зберігають поточну поведінку.
- Завершити policy enforcement для plain Strawberry fields на registered BerryType.
- Документувати, що cached type policy не є per-row policy. Для per-row authorization використовувати row scopes або окремий additive object-policy mechanism.
- Додати structured audit context до denied operations без логування payload values або secrets.

## Acceptance criteria

- Typo в deny-list не відкриває field.
- Type scope дає однаковий row set для query та mutation authorization.
- Scope/provider errors fail-closed на всіх adapters і fallback paths.
- Derived field dependencies можна перевірити, не виконуючи builder.
- Reused context та sibling operations не ділять stale policy.
- Existing public declarations залишаються валідними.

## Тести

- Unknown canonical/alias/camelCase names для кожного `FieldSet` mode.
- Type scope для create, update, delete, replace і nested reparent.
- Derived/custom/aggregate dependency allow/deny cases.
- Nullable plain Strawberry scalar/object/list fields.
- Cache lifecycle і audit records без sensitive values.

## Залежності

- Задачі 01–05.
