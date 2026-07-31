import { NumberField } from './number-field';

import type { BaseFieldProps } from '#core/components/form/fields/types';
import type { FieldRendererProps } from '#core/components/form/types';

export function SchemaNumberField({ config: _config, ...baseProps }: FieldRendererProps) {
  // cast: narrowing from unknown to number | undefined for NumberField
  return <NumberField {...(baseProps as BaseFieldProps<number | undefined>)} />;
}
