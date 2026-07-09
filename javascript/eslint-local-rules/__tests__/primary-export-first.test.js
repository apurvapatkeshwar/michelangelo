import { RuleTester } from 'eslint';
import tseslint from 'typescript-eslint';

import rule from '../primary-export-first.js';

RuleTester.describe = describe;
RuleTester.it = it;

const tester = new RuleTester({
  languageOptions: {
    ecmaVersion: 2020,
    sourceType: 'module',
    parser: tseslint.parser,
  },
});

tester.run('primary-export-first', rule, {
  valid: [
    {
      name: 'primary export first, helper function follows',
      filename: 'row-summary.ts',
      code: `export function RowSummary() { return formatLabel(); } function formatLabel() { return ''; }`,
    },
    {
      name: 'primary export first, arrow helper follows',
      filename: 'row-summary.ts',
      code: `export const RowSummary = () => formatLabel(); const formatLabel = () => '';`,
    },
    {
      name: 'no helpers, just the export',
      filename: 'row-summary.ts',
      code: `export function RowSummary() { return null; }`,
    },
    {
      name: 'plain object constant before the export is not a helper function',
      filename: 'row-summary.ts',
      code: `const CONFIG = { max: 10 }; export function RowSummary() { return CONFIG.max; }`,
    },
    {
      name: 'plain string constant before the export is not a helper function',
      filename: 'row-summary.ts',
      code: `const SUFFIX_DELIMITER = '::'; export function buildKey() { return SUFFIX_DELIMITER; }`,
    },
    {
      name: 'plain array constant before the export is not a helper function',
      filename: 'row-summary.ts',
      code: `const TERMINATED_STATES = ['DONE', 'FAILED']; export function isTerminal(s) { return TERMINATED_STATES.includes(s); }`,
    },
    {
      name: 'multiple top-level exports — no single primary to anchor on',
      filename: 'multi.ts',
      code: `function helper() { return ''; } export function A() { return helper(); } export function B() { return helper(); }`,
    },
    {
      name: 'zero value exports — nothing to analyze',
      filename: 'constants.ts',
      code: `function helper() { return ''; } const X = helper();`,
    },
    {
      name: 'index.ts is always skipped',
      filename: 'index.ts',
      code: `function helper() { return ''; } export function Anything() { return helper(); }`,
    },
    {
      name: 'index.tsx is always skipped',
      filename: 'index.tsx',
      code: `function helper() { return null; } export function Anything() { return helper(); }`,
    },
    {
      name: 'types.ts is always skipped',
      filename: 'types.ts',
      code: `function helper() { return ''; } export function Anything() { return helper(); }`,
    },
    {
      name: 'styled-components.tsx is skipped',
      filename: 'styled-components.tsx',
      code: `function helper() { return null; } export const StyledRow = helper();`,
    },
    {
      name: 'styled-components.ts is skipped',
      filename: 'row-styled-components.ts',
      code: `function helper() { return null; } export const styles = helper();`,
    },
    {
      name: '.test.ts files are skipped',
      filename: 'row-summary.test.ts',
      code: `function helper() { return ''; } export function RowSummary() { return helper(); }`,
    },
    {
      name: 'files under __tests__/ are skipped',
      filename: 'some-dir/__tests__/row-summary.ts',
      code: `function helper() { return ''; } export function RowSummary() { return helper(); }`,
    },
    {
      name: '.d.ts files are skipped',
      filename: 'row-summary.d.ts',
      code: `function helper() { return ''; } export function RowSummary(): string;`,
    },
    {
      name: 'non-ts/tsx files are skipped',
      filename: 'row-summary.js.snap',
      code: `function helper() { return ''; } export function RowSummary() { return helper(); }`,
    },
    {
      name: 'forwardRef/HOC-wrapped export is skipped (CallExpression init)',
      filename: 'row.tsx',
      code: `function BaseRow() { return null; } export const Row = forwardRef(BaseRow);`,
    },
    {
      name: 'class primary export is skipped',
      filename: 'row-model.ts',
      code: `function helper() { return ''; } export class RowModel {}`,
    },
    {
      name: 'specifier-style export is skipped',
      filename: 're-export.ts',
      code: `function helper() { return ''; } const X = helper(); export { X };`,
    },
    {
      name: 'type-only declarations never count as the helper or the export',
      filename: 'row-summary.ts',
      code: `type Config = { max: number }; export function RowSummary(c: Config) { return c.max; }`,
    },
  ],

  invalid: [
    {
      name: 'function helper declared before the function export it supports',
      filename: 'row-summary.ts',
      code: `function formatLabel() { return ''; } export function RowSummary() { return formatLabel(); }`,
      errors: [{ messageId: 'helperBeforePrimary' }],
    },
    {
      name: 'arrow-function helper declared before the const export it supports',
      filename: 'row-summary.ts',
      code: `const formatLabel = () => ''; export const RowSummary = () => formatLabel();`,
      errors: [{ messageId: 'helperBeforePrimary' }],
    },
    {
      name: 'class expression helper declared before the export it supports',
      filename: 'row-model.ts',
      code: `const RowBase = class { toString() { return ''; } }; export const RowModel = () => new RowBase();`,
      errors: [{ messageId: 'helperBeforePrimary' }],
    },
    {
      name: 'multiple helper functions declared before the export — each flagged',
      filename: 'row-summary.ts',
      code: `function formatLabel() { return ''; } function formatValue() { return ''; } export function RowSummary() { return formatLabel() + formatValue(); }`,
      errors: [{ messageId: 'helperBeforePrimary' }, { messageId: 'helperBeforePrimary' }],
    },
    {
      name: 'plain constant is not flagged, but the function helper alongside it still is',
      filename: 'row-summary.ts',
      code: `const CONFIG = { max: 10 }; function formatLabel() { return ''; } export function RowSummary() { return formatLabel() + CONFIG.max; }`,
      errors: [{ messageId: 'helperBeforePrimary' }],
    },
  ],
});
