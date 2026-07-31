import { CheckboxField } from './checkbox-field';

import type { BaseFieldProps } from '#core/components/form/fields/types';
import type { FieldRendererProps } from '#core/components/form/types';
import type { CheckboxOption } from './types';

export function SchemaCheckboxField({ config, ...baseProps }: FieldRendererProps) {
  // cast: schema field bridges untyped config to typed CheckboxField props
  const props = baseProps as BaseFieldProps<string[]>;
  // cast: config bag narrowed to CheckboxField-specific props
  const options = (config.options ?? []) as CheckboxOption[];

  return <CheckboxField {...props} options={options} />;
}
