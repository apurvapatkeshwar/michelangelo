import { MapField } from './map-field';

import type { BaseFieldProps } from '#core/components/form/fields/types';
import type { FieldRendererProps } from '#core/components/form/types';
import type { MapFieldOwnProps } from './types';

export function SchemaMapField({ config, ...baseProps }: FieldRendererProps) {
  // cast: schema field bridges untyped config to typed MapField props
  const props = baseProps as BaseFieldProps<Record<string, string>>;
  // cast: config bag narrowed to MapField-specific props
  const mapConfig = config as Partial<MapFieldOwnProps>;

  return (
    <MapField
      {...props}
      singleValue={mapConfig.singleValue}
      creatable={mapConfig.creatable}
      deletable={mapConfig.deletable}
      emptyMessage={mapConfig.emptyMessage}
      keyConfig={mapConfig.keyConfig}
      valueConfig={mapConfig.valueConfig}
      size={mapConfig.size}
    />
  );
}
