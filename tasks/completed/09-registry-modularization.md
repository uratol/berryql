# 09. Модульний registry та typed errors

## Мета

Зменшити відповідальність `registry.py`, прибрати дублювання execution logic і broad exception swallowing із security/correctness paths.

## Роботи

- Винести query root assembly/resolver factory в окремий module.
- Винести relation resolver generation/fallback execution в окремий module.
- Залишити в `BerrySchema` registration, public decorators, schema assembly і delegation до internal services.
- Використовувати вже винесені `PolicyEngine`, `PredicateCompiler`, `OrderingCompiler` та typed QueryPlan.
- Запровадити typed internal exceptions: configuration, invalid predicate, invalid ordering, authorization denied, adapter unsupported.
- Замінити `except Exception: pass/continue` у policy/scope/filter/order/SQL projection paths вузькими handlers або propagated errors.
- Best-effort behavior залишати лише для diagnostics/optional optimizations; fallback має бути явним і observable.
- Прибрати дубльовані filter expansion, AST value resolution, scope application та ordering resolution.

## Acceptance criteria

- `registry.py` більше не містить SQL predicate/order implementation та mutation authorization logic.
- Security-relevant exceptions не ковтаються.
- Schema SDL і public Python declarations, крім `FieldPermissions`, сумісні з попереднім станом.
- Full tests проходять на SQLite; adapter unit tests проходять для PostgreSQL/MSSQL.
- Circular imports та runtime dynamic resolver generation не погіршують startup.

## Тести

- GraphQL SDL snapshot до/після refactor з дозволеним винятком лише для відображення нового `FieldPermissions`, якщо воно взагалі впливає на SDL.
- Error mapping tests до GraphQL errors.
- Import/schema-build smoke tests.
- Pushdown/fallback and mutation regression suite.

## Залежності

- Задачі 01–08.
