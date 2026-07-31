import { BooleanField } from './boolean-field';

import type { BaseFieldProps } from '#core/components/form/fields/types';
import type { FieldRendererProps } from '#core/components/form/types';

export function SchemaBooleanField({ config, ...baseProps }: FieldRendererProps) {
  // cast: schema field bridges untyped config to typed BooleanField props
  const props = baseProps as BaseFieldProps<boolean>;
  // cast: config bag narrowed to BooleanField-specific props
  const { checkboxLabel, toggle } = config as { checkboxLabel?: string; toggle?: boolean };

  return <BooleanField {...props} checkboxLabel={checkboxLabel} toggle={toggle} />;
}
