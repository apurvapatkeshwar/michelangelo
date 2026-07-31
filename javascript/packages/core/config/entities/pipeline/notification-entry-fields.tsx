import { CheckboxField } from '#core/components/form/fields/checkbox/checkbox-field';
import { RadioField } from '#core/components/form/fields/radio/radio-field';
import { StringField } from '#core/components/form/fields/string/string-field';
import { useField } from '#core/components/form/hooks/use-field';
import { useForm } from '#core/components/form/hooks/use-form';
import { combineValidators } from '#core/components/form/validation/combine-validators';
import { email, required } from '#core/components/form/validation/validators';

import type { CheckboxOption } from '#core/components/form/fields/checkbox/types';
import type { RadioOption } from '#core/components/form/fields/radio/types';
import type { NotificationType } from '#core/config/entities/shared/notification-types';

const NOTIFICATION_TYPE_OPTIONS: RadioOption[] = [
  { value: 'EMAIL', label: 'Email' },
  { value: 'SLACK', label: 'Slack' },
];

const EVENT_TYPE_OPTIONS: CheckboxOption[] = [
  { id: 'PIPELINE_RUN_STATE_SUCCEEDED', label: 'Succeeded' },
  { id: 'PIPELINE_RUN_STATE_FAILED', label: 'Failed' },
  { id: 'PIPELINE_RUN_STATE_KILLED', label: 'Killed' },
  { id: 'PIPELINE_RUN_STATE_SKIPPED', label: 'Skipped' },
];

const slackChannelValidator = combineValidators(required(), (value) => {
  if (!Array.isArray(value)) return undefined;
  // cast: Array.isArray narrows to any[]; the multi StringField stores string[]
  const entries = value as string[];
  const invalid = entries.find((v) => !v.startsWith('#'));
  return invalid ? `"${invalid}" must start with #.` : undefined;
});

export function NotificationEntryFields({ fieldPath }: { fieldPath: string }) {
  const { input } = useField<NotificationType>(`${fieldPath}.notification_type`);
  const { change } = useForm();

  const notificationType = input.value;

  const handleTypeChange = (newType: NotificationType) => {
    if (newType === 'EMAIL') {
      change(`${fieldPath}.slack_destinations`, []);
    } else {
      change(`${fieldPath}.emails`, []);
    }
  };

  return (
    <>
      <RadioField
        name={`${fieldPath}.notification_type`}
        label="Notification type"
        options={NOTIFICATION_TYPE_OPTIONS}
        required
        parse={(value) => {
          // cast: RadioField's parse receives string|boolean; narrow to NotificationType for routing
          handleTypeChange(value as NotificationType);
          return value;
        }}
      />

      {notificationType === 'EMAIL' && (
        <StringField
          name={`${fieldPath}.emails`}
          label="Email addresses"
          placeholder="Enter an email and press Enter"
          required
          multi
          validate={email()}
        />
      )}

      {notificationType === 'SLACK' && (
        <StringField
          name={`${fieldPath}.slack_destinations`}
          label="Slack channels"
          placeholder="#channel-name"
          required
          multi
          validate={slackChannelValidator}
        />
      )}

      <CheckboxField
        name={`${fieldPath}.event_types`}
        label="Notify on"
        options={EVENT_TYPE_OPTIONS}
        required
      />
    </>
  );
}
