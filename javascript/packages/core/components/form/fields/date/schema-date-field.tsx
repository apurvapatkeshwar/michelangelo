import { DateField } from './date-field';

import type { BaseFieldProps } from '#core/components/form/fields/types';
import type { FieldRendererProps } from '#core/components/form/types';
import type { DateFormat } from './types';

export function SchemaDateField({ config, ...baseProps }: FieldRendererProps) {
  // cast: schema field bridges untyped config to typed DateField props
  const props = baseProps as BaseFieldProps<string, Date | null>;
  // cast: config bag narrowed to DateField-specific props
  const { dateFormat, noFutureDate } = config as {
    dateFormat?: DateFormat;
    noFutureDate?: boolean;
  };

  return <DateField {...props} dateFormat={dateFormat} noFutureDate={noFutureDate} />;
}
