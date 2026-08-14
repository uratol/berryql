# 10. Ordering, pagination та performance

## Мета

Усунути потенційний N+1 і зробити складні ordering/pagination paths прогнозованими без видалення чинних можливостей.

## Роботи

- Замінити per-item `session.get()`/SELECT у Python dotted-relation ordering на batch/preload strategy або SQL ordering.
- Кешувати compiled order expressions і normalized allowed paths на schema/type рівні без request data.
- Не обчислювати автоматично весь список nested order paths для кожної помилки; використовувати lazy/capped diagnostics.
- Зберегти чинні nested single-relation paths; додати additive explicit allow-list/config для застосунків, які хочуть обмежити дорогі paths.
- Додати optional `nulls_first`/`nulls_last` у backward-compatible form, якщо всі adapters можуть дати визначену семантику.
- Додати additive cursor/keyset pagination поверх deterministic OrderingCompiler; існуючий limit/offset зберегти.
- Додати query-count і SQL-shape regression tests для root/nested ordering.
- Перевірити, що policy та scope resolution не створюють N+1 external ACL calls.

## Acceptance criteria

- Dotted fallback ordering не виконує O(rows × depth) SQL queries.
- Limit/offset завжди має stable total order.
- Existing order arguments і results залишаються сумісними, крім виправлення раніше недетермінованого tie order.
- Нові options є additive та документованими.
- Performance tests мають explicit query-count bounds.

## Тести

- Велика collection із repeated FK targets і багаторівневим dotted ordering.
- Pushdown disabled/enabled comparison.
- Stable pagination без duplicates/missing rows між сторінками.
- Null ordering per adapter.
- Permission provider call count.

## Залежності

- Задачі 01–09.
