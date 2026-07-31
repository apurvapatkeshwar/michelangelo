import { UrlField } from './url-field';

import type { BaseFieldProps } from '#core/components/form/fields/types';
import type { FieldRendererProps } from '#core/components/form/types';

export function SchemaUrlField({ config, ...baseProps }: FieldRendererProps) {
  // cast: schema field bridges untyped config to typed UrlField props
  const props = baseProps as BaseFieldProps<string>;
  // cast: config bag narrowed to UrlField-specific props
  const { urlName } = config as { urlName?: string };

  return <UrlField {...props} urlName={urlName} />;
}
