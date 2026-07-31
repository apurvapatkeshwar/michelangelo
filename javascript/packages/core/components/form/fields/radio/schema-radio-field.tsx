import { RadioField } from './radio-field';

import type { BaseFieldProps } from '#core/components/form/fields/types';
import type { FieldRendererProps } from '#core/components/form/types';
import type { RadioOption } from './types';

export function SchemaRadioField({ config, ...baseProps }: FieldRendererProps) {
  // cast: schema field bridges untyped config to typed RadioField props
  const props = baseProps as BaseFieldProps<string | boolean>;
  // cast: config bag narrowed to RadioField-specific props
  const { options: rawOptions, align } = config as {
    options?: RadioOption[];
    align?: 'horizontal' | 'vertical';
  };

  return <RadioField {...props} options={rawOptions ?? []} align={align} />;
}
