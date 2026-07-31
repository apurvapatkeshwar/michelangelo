import { Form } from '#core/components/form/form';
import { ResolvedFormContent } from '#core/components/form/resolved-form-content';

import type { FormConfigSchema, FormData } from '#core/components/form/types';

type ConfigDrivenFormProps = {
  config: FormConfigSchema;
  onSubmit: (values: FormData) => void | object | Promise<object>;
  initialValues?: Record<string, unknown>;
};

export function ConfigDrivenForm({ config, onSubmit, initialValues }: ConfigDrivenFormProps) {
  return (
    <Form onSubmit={onSubmit} initialValues={initialValues}>
      <ResolvedFormContent config={config} />
    </Form>
  );
}
