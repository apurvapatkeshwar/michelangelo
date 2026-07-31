import { SelectField } from './select-field';

import type { BaseFieldProps } from '#core/components/form/fields/types';
import type { FieldRendererProps } from '#core/components/form/types';
import type { SelectOption } from './types';

export function SchemaSelectField({ config, ...baseProps }: FieldRendererProps) {
  const {
    options: rawOptions,
    clearable,
    searchable,
    creatable,
    isLoading,
    visibleOptionLimit,
    // cast: config bag narrowed to SelectField-specific props
  } = config as {
    options?: SelectOption[];
    clearable?: boolean;
    searchable?: boolean;
    creatable?: boolean;
    isLoading?: boolean;
    visibleOptionLimit?: number;
  };
  const options = rawOptions ?? [];

  if (config.multi) {
    // cast: multi=true narrows SelectField to the array variant
    const props = baseProps as BaseFieldProps<(string | number)[]>;
    return (
      <SelectField
        {...props}
        options={options}
        multi
        clearable={clearable}
        searchable={searchable}
        creatable={creatable}
        isLoading={isLoading}
        visibleOptionLimit={visibleOptionLimit}
      />
    );
  }

  // cast: multi=false/absent narrows SelectField to the single variant
  const props = baseProps as BaseFieldProps<string | number>;
  return (
    <SelectField
      {...props}
      options={options}
      clearable={clearable}
      searchable={searchable}
      creatable={creatable}
      isLoading={isLoading}
      visibleOptionLimit={visibleOptionLimit}
    />
  );
}
