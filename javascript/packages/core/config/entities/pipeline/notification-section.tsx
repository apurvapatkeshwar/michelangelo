import { ArrayFormGroup } from '#core/components/form/layout/array-form-group/array-form-group';
import { FormGroup } from '#core/components/form/layout/form-group/form-group';
import { NotificationEntryFields } from '#core/config/entities/pipeline/notification-entry-fields';

export function NotificationSection() {
  return (
    <FormGroup title="Notifications" collapsible>
      <ArrayFormGroup
        rootFieldPath="spec.notifications"
        groupLabel="Notification"
        addLabel="Add notification"
      >
        {(fieldPath) => <NotificationEntryFields fieldPath={fieldPath} />}
      </ArrayFormGroup>
    </FormGroup>
  );
}
