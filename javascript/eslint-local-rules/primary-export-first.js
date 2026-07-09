/**
 * @fileoverview Flags private helper functions declared before a file's primary export.
 *
 * Per "Code Ordering Within a File" in the ui-coding-principles skill: put the
 * primary export first. Private helpers and implementation details should
 * follow, not precede, the thing they support — readers should see the main
 * thing before the internals.
 *
 * SCOPE (intentionally narrow):
 *
 * This only analyzes files with EXACTLY ONE top-level, non-type, value-level
 * export, and only when that export is unambiguously a function declaration
 * or a `const` bound directly to a function/arrow expression. Everything else
 * is skipped without reporting anything, rather than guessing:
 *
 *   - Non ts/tsx files, `.d.ts` files
 *   - `*.test.ts(x)`, anything under `__tests__/`
 *   - `index.ts(x)`, `types.ts`
 *   - `*styled-components.ts(x)`
 *   - Files with zero or >1 top-level value exports
 *   - `export { a, b }` specifier-style exports (including re-exports) and
 *     `export default` — out of scope for this analysis
 *   - A single export whose init is a class or a CallExpression (e.g.
 *     `export const Foo = forwardRef(...)` or `memo(Foo)`) — unwrapping
 *     HOC-wrapped exports is left for a follow-up
 *
 * Within an in-scope file, only FUNCTION-like preceding declarations are
 * flagged: a top-level `function` declaration, or a top-level `const`/`let`
 * whose initializer is a function expression, arrow function, or class.
 * Plain object/array/literal-valued constants (e.g. a `SUFFIX_DELIMITER` or
 * `TERMINATED_STATES` sitting near the imports) are a different, more
 * debatable style question and are intentionally never flagged here.
 */

const isHelperDeclaration = (stmt) => {
  if (stmt.type === 'FunctionDeclaration') return true;
  if (stmt.type !== 'VariableDeclaration') return false;

  return stmt.declarations.some((declarator) => {
    const init = declarator.init;
    if (!init) return false;
    return (
      init.type === 'FunctionExpression' ||
      init.type === 'ArrowFunctionExpression' ||
      init.type === 'ClassExpression'
    );
  });
};

const getHelperName = (stmt) => {
  if (stmt.type === 'FunctionDeclaration') return stmt.id?.name ?? '(anonymous)';

  const declarator = stmt.declarations.find((d) => {
    const init = d.init;
    return (
      init?.type === 'FunctionExpression' ||
      init?.type === 'ArrowFunctionExpression' ||
      init?.type === 'ClassExpression'
    );
  });
  return declarator?.id?.type === 'Identifier' ? declarator.id.name : '(anonymous)';
};

/** @type {import('eslint').Rule.RuleModule} */
const rule = {
  meta: {
    type: 'suggestion',
    docs: {
      description:
        "Flag private helper functions declared before a file's single primary export",
      recommended: true,
      url: 'https://github.com/michelangelo-ai/michelangelo/blob/main/javascript/eslint-local-rules/primary-export-first.md',
    },
    messages: {
      helperBeforePrimary:
        "'{{helperName}}' is defined before the primary export '{{primaryName}}'. " +
        'Move private helpers after the thing they support (Code Ordering Within a File).',
    },
    schema: [],
  },

  create(context) {
    const filename = context.getPhysicalFilename?.() ?? context.filename;
    const basename = filename.split('/').pop() ?? '';

    // --- File-level skip rules: report nothing, don't even analyze. ---
    if (basename.endsWith('.d.ts')) return {};
    if (!/\.(tsx|ts)$/.test(basename)) return {};
    if (basename.endsWith('.test.ts') || basename.endsWith('.test.tsx')) return {};
    if (filename.includes('/__tests__/')) return {};
    if (basename === 'index.ts' || basename === 'index.tsx') return {};
    if (basename === 'types.ts') return {};
    if (basename.endsWith('styled-components.ts') || basename.endsWith('styled-components.tsx')) {
      return {};
    }

    return {
      'Program:exit'(program) {
        const body = program.body;

        // Bail if the file uses specifier-style exports or a default export
        // anywhere at top level — out of scope for this rule.
        const hasSpecifierOrDefaultExport = body.some((stmt) => {
          if (stmt.type === 'ExportDefaultDeclaration') return true;
          if (stmt.type === 'ExportNamedDeclaration' && !stmt.declaration) return true;
          return false;
        });
        if (hasSpecifierOrDefaultExport) return;

        // Collect top-level, non-type, value-level exports.
        const valueExports = [];
        for (const stmt of body) {
          if (stmt.type !== 'ExportNamedDeclaration' || !stmt.declaration) continue;
          if (stmt.exportKind === 'type') continue;

          const decl = stmt.declaration;
          if (decl.type === 'FunctionDeclaration' && decl.id) {
            valueExports.push({ node: stmt, name: decl.id.name, kind: 'function' });
          } else if (decl.type === 'ClassDeclaration' && decl.id) {
            valueExports.push({ node: stmt, name: decl.id.name, kind: 'class' });
          } else if (decl.type === 'VariableDeclaration') {
            for (const d of decl.declarations) {
              if (d.id?.type === 'Identifier') {
                valueExports.push({ node: stmt, name: d.id.name, kind: 'variable', init: d.init });
              }
            }
          }
          // TSTypeAliasDeclaration / TSInterfaceDeclaration / TSEnumDeclaration
          // intentionally fall through unmatched — types don't count.
        }

        // Skip: no exports, or more than one — can't identify a single primary.
        if (valueExports.length !== 1) return;

        const primary = valueExports[0];

        // Skip: primary is a class — not in scope per spec.
        if (primary.kind === 'class') return;

        if (primary.kind === 'variable') {
          const init = primary.init;
          if (!init) return; // ambient `export const x: Foo;` — nothing to anchor on
          if (init.type === 'CallExpression') return; // forwardRef(...)/memo(...)/HOC(...)
          if (init.type !== 'FunctionExpression' && init.type !== 'ArrowFunctionExpression') {
            return; // not a function-valued const — ambiguous
          }
        }

        const primaryIndex = body.indexOf(primary.node);

        // Flag only function-like top-level declarations before the primary:
        // a `function` declaration, or a `const`/`let` bound to a function,
        // arrow function, or class. Plain data constants are left alone.
        for (let i = 0; i < primaryIndex; i++) {
          const stmt = body[i];
          if (!isHelperDeclaration(stmt)) continue;

          context.report({
            node: stmt,
            messageId: 'helperBeforePrimary',
            data: { helperName: getHelperName(stmt), primaryName: primary.name },
          });
        }
      },
    };
  },
};

export default rule;
