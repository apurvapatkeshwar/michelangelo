# primary-export-first

## What this rule enforces

A file's primary export should come first. Private helper functions and implementation details should follow, not precede, the thing they support — a reader opening the file should see the main export before the internals it depends on.

## Flagged patterns

```ts
// ❌ helper function declared above the export it supports
function formatRowLabel(row: Row) {
  return row.name.toUpperCase();
}

export function RowSummary(row: Row) {
  return formatRowLabel(row);
}
```

```tsx
// ❌ arrow-function helper above the component that uses it
const withStickySides = (Component: RowComponent) => (props: RowProps) => (
  <Sticky>
    <Component {...props} />
  </Sticky>
);

export const TableRow = withStickySides(BaseRow);
```

## Correct patterns

```ts
// ✓ export first, helper follows
export function RowSummary(row: Row) {
  return formatRowLabel(row);
}

function formatRowLabel(row: Row) {
  return row.name.toUpperCase();
}
```

Function hoisting means a `function` declaration used before its textual position still works at runtime, but reordering it after the primary export is what this rule enforces (and what the file should look like either way).

## What is not flagged

This rule only flags **function-like** declarations: a top-level `function` declaration, or a top-level `const`/`let` whose initializer is a function expression, arrow function, or class. Plain data — object literals, arrays, strings, numbers — sitting above the primary export is a different, more debatable style question and is intentionally out of scope:

```ts
// ✓ not flagged — a plain constant, not a helper function
const SUFFIX_DELIMITER = '::';

export function buildKey(id: string) {
  return `${id}${SUFFIX_DELIMITER}suffix`;
}
```

## Exemptions (auto-detected)

This rule only analyzes files where a single "primary export" can be identified unambiguously. It reports nothing (rather than guessing) for:

| Case                                                  | Why exempt                                                            |
| ------------------------------------------------------ | ----------------------------------------------------------------------- |
| Non-`.ts`/`.tsx` files, `.d.ts`                       | not applicable                                                         |
| `*.test.ts(x)`, files under `__tests__/`              | test files have their own structure conventions                       |
| `index.ts(x)`                                         | entry points are intentionally multi-export                           |
| `types.ts`                                            | type-only files                                                       |
| `*styled-components.ts(x)`                            | multi-component styled collections, no single "primary" export        |
| Zero or more than one top-level value export          | no single primary export to anchor on                                 |
| `export { a, b }` specifier-style or re-export         | resolving which declaration an alias refers to is out of scope         |
| `export default`                                      | already banned elsewhere in this config; skipped defensively           |
| Single export whose initializer is a class             | classes are out of scope for this rule                                |
| Single export whose initializer is a `CallExpression`  | e.g. `forwardRef(...)`/`memo(...)`-wrapped exports — unwrapping HOCs is a separate, more complex analysis |

Type-only top-level declarations (`type X = ...`, `interface X {}`) are never flagged and never count toward the "exactly one export" check — colocating a component's prop type above the component is a distinct, idiomatic convention.
