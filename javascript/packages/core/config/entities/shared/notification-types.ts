export type NotificationType = 'EMAIL' | 'SLACK';

export type NotificationEventType =
  | 'PIPELINE_RUN_STATE_SUCCEEDED'
  | 'PIPELINE_RUN_STATE_FAILED'
  | 'PIPELINE_RUN_STATE_KILLED'
  | 'PIPELINE_RUN_STATE_SKIPPED';

export type Notification = {
  notification_type: NotificationType;
  event_types: NotificationEventType[];
  resource_type: string;
  emails: string[];
  slack_destinations: string[];
};
