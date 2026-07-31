import { useFieldRenderer } from '#core/components/form/hooks/use-field-renderer';
import { resolveValidation } from '#core/components/form/validation/resolve-validation';

import type { FieldConfig } from '#core/components/form/types';

export function SchemaField({
  fieldPath,
  config,
}: {
  fieldPath: string;
  config: FieldConfig | undefined;
}) {
  const {
    type,
    label,
    required,
    disabled,
    readOnly,
    placeholder,
    description,
    caption,
    validation,
    ...rest
    // cast: FieldConfig union always has these shared properties plus a string `type`
  } = (config ?? { type: '' }) as FieldConfig & { type: string };

  const Renderer = useFieldRenderer(type);
  if (!Renderer) return null;

  return (
    <Renderer
      name={fieldPath}
      label={label}
      required={required}
      disabled={disabled}
      readOnly={readOnly}
      placeholder={placeholder}
      description={description}
      caption={caption}
      validate={resolveValidation(validation)}
      config={rest}
    />
  );
}
