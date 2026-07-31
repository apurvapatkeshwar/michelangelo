import { MarkdownField } from './markdown-field';

import type { BaseFieldProps } from '#core/components/form/fields/types';
import type { FieldRendererProps } from '#core/components/form/types';

export function SchemaMarkdownField({ config, ...baseProps }: FieldRendererProps) {
  // cast: schema field bridges untyped config to typed MarkdownField props
  const props = baseProps as BaseFieldProps<string>;
  // cast: config bag narrowed to MarkdownField-specific props
  const { rows, maxLength } = config as { rows?: number; maxLength?: number };

  return <MarkdownField {...props} rows={rows} maxLength={maxLength} />;
}
