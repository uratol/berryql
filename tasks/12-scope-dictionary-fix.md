# Task for BerryQL — Regression in 0.5.0: `PredicateCompiler` rejects SQLAlchemy subquery operands for `in` in scope where-dicts

> **Repo**: https://github.com/uratol/berryql
> **Type**: bug / breaking-change regression
> **Affected version**: 0.5.0 (introduced by the new `PredicateCompiler`, commit `b47a8a3` "Implement Policy Engine and Predicate Compiler", 2026-08-14)
> **Proposed target release**: 0.5.1 (backward-compatibility restore)
> **Severity**: critical — every BerryQL deployment that uses subquery-based row-level security scopes is broken.

---

## 1. Summary

After upgrading to BerryQL 0.5.0, **every query and mutation protected by a scope that returns a where-dict with a SQLAlchemy subquery operand** fails with:

```
PredicateError: Where operator 'in' for 'business_id' requires a non-empty list
```

GraphQL response (as observed in a production app on 0.5.0):

```json
{
  "errors": [
    {
      "message": "Where operator 'in' for 'business_id' requires a non-empty list",
      "extensions": { "code": "VALIDATION", "httpStatus": 400 }
    }
  ]
}
```

This is a regression of a **documented and previously working** feature. `berryql/mutations.py` (`_enforce_scope`, lines 393–398 in 0.5.0) still documents the contract:

```python
async def _enforce_scope(model_cls_local, instance_local, scope_raw, scope_ctx=None):
    """Enforce mutation scope using the same where builder as queries.

    Accepts dict/JSON string/callable/SQLA expression. Callable may return any of those.
    Supports ScalarSelect inside where dicts (e.g., {'id': {'in': select(...).scalar_subquery()}}).
    """
```

## 2. Root cause

### 2.1 New unconditional operand-shape validation

`berryql/core/predicates.py` (0.5.0):

- `parse()` — line 188 calls the validator for **every** operand, **regardless of `trusted`**:

  ```python
  for operator, operand in operator_map.items():
      self._validate_operand_shape(str(column_name), str(operator), operand)
      fragments.append(ColumnPredicate(str(column_name), str(operator), operand))
  ```

- `_validate_operand_shape()` — lines 199–218 rejects anything that is not a non-empty `list/tuple/set`:

  ```python
  elif operator in {"in", "not_in"}:
      if not isinstance(operand, (list, tuple, set)) or not operand:
          raise PredicateError(
              f"Where operator '{operator}' for '{column}' requires a non-empty list"
          )
      if self.limits.max_in_items is not None and len(operand) > self.limits.max_in_items:
          ...
  ```

A `ScalarSelect` (result of `select(...).scalar_subquery()`) is not a `list/tuple/set`, so the scope dict
`{"business_id": {"in": <scalar_subquery>}}` — the canonical multi-tenant RLS scope pattern — is rejected
**before compilation ever runs**.

Both code paths route through the same compiler, so this breaks queries *and* mutations:

- Query path: `berryql/registry.py` (~line 2248):
  `await self._predicate_compiler.apply(stmt, child_model_cls, eff_scope, info, strict=True, trusted=True)`
- Mutation path: `berryql/mutations.py` `_enforce_scope` (~line 411):
  `await schema._predicate_compiler.resolve(scope_raw, model_cls_local, info, strict=True, trusted=True)`

### 2.2 The compile path itself still supports subqueries (proof it's purely a validation regression)

`compile_sqlalchemy()` passes the operand through `coerce_where_value()` (which returns non-string SQL
constructs untouched) into the operator lambda. Building the `ColumnPredicate` manually (what `parse()`
produced in ≤0.4.x) compiles to exactly the expected SQL:

```
SELECT widgets.id FROM widgets
WHERE widgets.business_id IN ((SELECT members.business_id FROM members WHERE members.user_id = :user_id_1))
```

### 2.3 Secondary latent bug: plain `Select` operands are wrapped into a list by the operator lambda

`berryql/core/filters.py` lines 18–19:

```python
'in':     lambda col, v: col.in_(v if isinstance(v, (list, tuple, set)) else [v]),
'not_in': lambda col, v: ~col.in_(v if isinstance(v, (list, tuple, set)) else [v]),
```

- `ScalarSelect` wrapped as `[scalar_subquery]` — works (element of an IN list).
- Plain `Select` wrapped as `[select_stmt]` — **fails**:

  ```
  sqlalchemy.exc.ArgumentError: IN expression list, SELECT construct, or bound parameter object
  expected, got [<sqlalchemy.sql.selectable.Select ...>]
  ```

  SQLAlchemy requires `col.in_(select_stmt)` (unwrapped) for a plain `Select`.

So if validation is relaxed to allow `Select` operands, the operator lambdas must be adjusted in the same
release, otherwise the failure just moves from validation to compilation.

### 2.4 Docstring/behavior mismatch for cost limits

`FilterLimits` docstring says: *"Trusted scopes are validated for shape but are not subject to caller cost
limits."* But `_validate_operand_shape` enforces `max_in_items` for **trusted** scopes too. Not the cause of
this outage, but should be aligned while touching this code.

## 3. Reproduction

Script (run with the venv that has berryql 0.5.0; also saved at `tmp_berryql_repro.py` in the coretaxa workspace root):

```python
import asyncio
from types import SimpleNamespace
from sqlalchemy import Column, Integer, String, select
from sqlalchemy.orm import DeclarativeBase
from berryql.core.predicates import PredicateCompiler

class Base(DeclarativeBase): pass

class Member(Base):
    __tablename__ = "members"
    user_id = Column(String, primary_key=True)
    business_id = Column(String, primary_key=True)

class Widget(Base):
    __tablename__ = "widgets"
    id = Column(Integer, primary_key=True)
    business_id = Column(String)

class _Info:
    context = {}
    operation = object()
    variable_values = {}

compiler = PredicateCompiler(SimpleNamespace(_auto_camel_case=False))
subq = select(Member.business_id).where(Member.user_id == "u1").scalar_subquery()
scope = {"business_id": {"in": subq}}   # canonical multi-tenant scope shape

async def main():
    await compiler.resolve(scope, Widget, _Info(), strict=True, trusted=True)

asyncio.run(main())
```

Observed output on 0.5.0:

```
1) trusted scope resolve:   FAILED -> PredicateError: Where operator 'in' for 'business_id' requires a non-empty list
2) untrusted filter resolve: FAILED -> PredicateError: (same)
3) empty list operand:      rejected (intended behavior — keep)
4) manual ColumnPredicate with ScalarSelect compiles to: ... WHERE widgets.business_id IN ((SELECT ...))
5) plain Select operand:    compiles, but only if NOT wrapped into [v] by the operator lambda
```

## 4. Impact (real consumer)

Coretaxa backend uses the documented pattern as its **standard row-level-security scope** in
`app/backend/app/graphql/domains/`:

- `common_scopes.py` — `items_of_user_businesses_scope` / `items_of_user_businesses_write_scope`
- `people_domain.py` — `_business_owned_read_scope` / `_business_owned_write_scope` (Person/Worker/Occupation/Address domains)
- `business_domain.py`, `availability_domain.py`, and every domain importing `common_scopes`

```python
def _business_owned_read_scope(model_cls, info):
    uid = get_user_id_from_info(info)
    return {"business_id": {"in": business_membership_subquery(uid)}}
```

After the 0.5.0 upgrade, a plain `worker { workers(businessId: $id) }` query returns
`400 VALIDATION: Where operator 'in' for 'business_id' requires a non-empty list` — i.e. **all tenant-isolated
reads/writes are down**, which also means the app cannot even fall back to a less-privileged view.

## 5. Required fix

Restore the ≤0.4.x where-dict contract without weakening the new caller-side (untrusted) validation.
Caller filters arrive from GraphQL JSON and can never contain SQL constructs, so accepting SQL operands in
`_validate_operand_shape` does not open any injection surface.

### 5.1 `berryql/core/predicates.py` — `_validate_operand_shape`

Accept SQLAlchemy selectable/scalar-expression operands for `in` / `not_in` (and, for symmetry,
`between` / `not_between` where each side may be a SQL expression):

```python
from sqlalchemy.sql.selectable import ScalarSelect, Select
from sqlalchemy.sql.elements import ColumnElement  # covers scalar_subquery(), bindparam(), column expressions

def _is_sql_operand(operand: Any) -> bool:
    # Duck-typing fallback keeps compatibility with dialect-specific wrappers
    return isinstance(operand, (Select, ScalarSelect, ColumnElement)) or (
        hasattr(operand, "selectable") or hasattr(operand, "self_group")
    ) and not isinstance(operand, (list, tuple, set))
```

Validation logic:

```python
elif operator in {"in", "not_in"}:
    if _is_sql_operand(operand):
        pass  # trusted SQL construct: no length semantics, no max_in_items
    elif not isinstance(operand, (list, tuple, set)) or not operand:
        raise PredicateError(
            f"Where operator '{operator}' for '{column}' requires a non-empty list"
        )
    elif self.limits.max_in_items is not None and len(operand) > self.limits.max_in_items:
        raise PredicateError(...)
```

Also:
- `between`/`not_between`: accept a 2-sequence **whose items may be SQL expressions** (only the container
  must be a sequence of exactly two).
- Per the `FilterLimits` docstring, skip `max_in_items` when `trusted=True` (or update the docstring —
  pick one, but make code and docs agree).

### 5.2 `berryql/core/filters.py` — operator registry

Do not wrap SQL constructs into a list; `Select` must reach `col.in_()` unwrapped:

```python
def _in_value(v):
    if isinstance(v, (list, tuple, set)):
        return v
    if _is_sql_operand(v):
        return v                      # col.in_(select) / col.in_(scalar_subquery)
    return [v]

OPERATOR_REGISTRY = {
    ...
    'in':     lambda col, v: col.in_(_in_value(v)),
    'not_in': lambda col, v: ~col.in_(_in_value(v)),
    ...
}
```

(Alternatively normalize the operand once in `PredicateCompiler.compile_sqlalchemy` before invoking the
operator — either is fine, but keep `register_operator` custom operators working.)

### 5.3 Tests to add (regression guard)

In `tests/test_predicate_compiler.py` (and one end-to-end scope test):

1. Trusted scope dict `{"business_id": {"in": select(...).scalar_subquery()}}`:
   - `resolve(..., trusted=True)` succeeds;
   - `compile_sqlalchemy` produces `... IN (SELECT ...)`.
2. Plain `Select` operand for `in` / `not_in` compiles (no `[<Select>]` wrapping).
3. `between` with `[scalar_expr, scalar_expr]` compiles.
4. Untrusted caller `{"id": {"in": []}}` still raises `PredicateError` (empty-list validation preserved).
5. Untrusted caller `{"id": {"in": [ ... >max_in_items... ]}}` still raises when `max_in_items` configured.
6. End-to-end: a `relation(..., scope=lambda M, info: {"business_id": {"in": subq}})` returns only in-scope
   rows, and the corresponding `merge*` mutation with the same scope enforces/permits correctly
   (covers `_enforce_scope`'s documented contract).
7. MSSQL adapter path: decide and test explicitly whether `ColumnPredicate` with SQL operands goes through
   `compile_mssql` (currently it re-dispatches to `adapter.where_from_dict` with the raw value) — either
   compile via the SQLAlchemy compiler like `TrustedExpression` does, or raise the existing
   `UnsupportedPredicateError` deterministically.

### 5.4 Documentation

- README / docs "Scopes & row-level security" section: re-state the supported scope forms:
  dict / JSON string / callable / raw SQLAlchemy expression / dict containing `ScalarSelect`/`Select` operands.
- CHANGELOG: 0.5.0 entry should be annotated with the breaking change, 0.5.1 with the restoration.

## 6. Workaround for consumers (until 0.5.1 ships)

Return a raw SQLAlchemy expression from the scope instead of a where-dict — raw expressions are wrapped as
`TrustedExpression` and skip dict-shape validation entirely:

```python
def _business_owned_read_scope(model_cls, info):
    uid = get_user_id_from_info(info)
    return model_cls.business_id.in_(business_membership_subquery(uid))   # not a dict -> TrustedExpression
```

(Verified against installed 0.5.0: scopes returning `or_(...)` / `column.in_(subquery)` expressions keep
working — the regression only affects the dict form.)

## 7. Acceptance criteria

- [ ] The reproduction script in §3 resolves and compiles on the fixed version.
- [ ] Full BerryQL test suite passes, including the new tests from §5.3.
- [ ] Empty-list and cost-limit validation still reject untrusted caller filters (no security regression).
- [ ] Docstring in `mutations._enforce_scope` matches actual behavior (it already claims this support).
- [ ] Released as 0.5.1 and coretaxa's `GetBusinessWorkers` query (and all scoped domains) work again
      after upgrade.
