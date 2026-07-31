import { TextareaField } from './textarea-field';

import type { BaseFieldProps } from '#core/components/form/fields/types';
import type { FieldRendererProps } from '#core/components/form/types';

export function SchemaTextareaField({ config, ...baseProps }: FieldRendererProps) {
  // cast: schema field bridges untyped config to typed TextareaField props
  const props = baseProps as BaseFieldProps<string>;
  // cast: config bag narrowed to TextareaField-specific props
  const { rows, maxLength } = config as { rows?: number; maxLength?: number };

  return <TextareaField {...props} rows={rows} maxLength={maxLength} />;
}
