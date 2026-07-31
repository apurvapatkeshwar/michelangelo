import { useState } from 'react';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { CreatePipelineRunForm } from '#core/config/entities/pipeline/create-pipeline-run-form';
import { buildWrapper } from '#core/test/wrappers/build-wrapper';
import { getBaseProviderWrapper } from '#core/test/wrappers/get-base-provider-wrapper';
import { getErrorProviderWrapper } from '#core/test/wrappers/get-error-provider-wrapper';
import { getIconProviderWrapper } from '#core/test/wrappers/get-icon-provider-wrapper';
import { getInterpolationProviderWrapper } from '#core/test/wrappers/get-interpolation-provider-wrapper';
import { getRouterWrapper } from '#core/test/wrappers/get-router-wrapper';
import {
  createQueryMockRouter,
  getServiceProviderWrapper,
} from '#core/test/wrappers/get-service-provider-wrapper';

function FormWrapper() {
  const [mounted, setMounted] = useState(true);
  const data = {
    metadata: { name: 'test-pipeline', namespace: 'test-namespace' },
    spec: { owner: { name: 'test-owner' } },
  };
  if (!mounted) return null;
  return <CreatePipelineRunForm record={data} onClose={() => setMounted(false)} />;
}

describe('CreatePipelineRunForm notifications', () => {
  it('renders collapsed Notifications section', async () => {
    render(
      <FormWrapper />,
      buildWrapper([
        getBaseProviderWrapper(),
        getIconProviderWrapper(),
        getErrorProviderWrapper(),
        getInterpolationProviderWrapper(),
        getRouterWrapper({ location: '/ma-dev-test/train/pipelines' }),
        getServiceProviderWrapper({ request: createQueryMockRouter({ CreatePipelineRun: {} }) }),
      ])
    );

    await screen.findByRole('dialog', { name: 'Start new pipeline run' });
    expect(screen.getByText('Notifications')).toBeInTheDocument();
    expect(screen.queryByText('Add notification')).not.toBeInTheDocument();
  });

  it('expands Notifications section and shows Add button', async () => {
    const user = userEvent.setup();
    render(
      <FormWrapper />,
      buildWrapper([
        getBaseProviderWrapper(),
        getIconProviderWrapper(),
        getErrorProviderWrapper(),
        getInterpolationProviderWrapper(),
        getRouterWrapper({ location: '/ma-dev-test/train/pipelines' }),
        getServiceProviderWrapper({ request: createQueryMockRouter({ CreatePipelineRun: {} }) }),
      ])
    );

    await screen.findByRole('dialog', { name: 'Start new pipeline run' });
    await user.click(screen.getByText('Notifications'));

    expect(screen.getByText('Add notification')).toBeInTheDocument();
  });

  it('adds a notification entry with type, destination, and event fields', async () => {
    const user = userEvent.setup();
    render(
      <FormWrapper />,
      buildWrapper([
        getBaseProviderWrapper(),
        getIconProviderWrapper(),
        getErrorProviderWrapper(),
        getInterpolationProviderWrapper(),
        getRouterWrapper({ location: '/ma-dev-test/train/pipelines' }),
        getServiceProviderWrapper({ request: createQueryMockRouter({ CreatePipelineRun: {} }) }),
      ])
    );

    await screen.findByRole('dialog', { name: 'Start new pipeline run' });
    await user.click(screen.getByText('Notifications'));
    await user.click(screen.getByText('Add notification'));

    expect(screen.getByText('Notification 1')).toBeInTheDocument();
    expect(screen.getByText('Notification type')).toBeInTheDocument();
    expect(screen.getByText('Notify on')).toBeInTheDocument();
  });

  it('shows email field when Email type is selected', async () => {
    const user = userEvent.setup();
    render(
      <FormWrapper />,
      buildWrapper([
        getBaseProviderWrapper(),
        getIconProviderWrapper(),
        getErrorProviderWrapper(),
        getInterpolationProviderWrapper(),
        getRouterWrapper({ location: '/ma-dev-test/train/pipelines' }),
        getServiceProviderWrapper({ request: createQueryMockRouter({ CreatePipelineRun: {} }) }),
      ])
    );

    await screen.findByRole('dialog', { name: 'Start new pipeline run' });
    await user.click(screen.getByText('Notifications'));
    await user.click(screen.getByText('Add notification'));
    await user.click(screen.getByRole('radio', { name: 'Email' }));

    expect(screen.getByText('Email addresses')).toBeInTheDocument();
    expect(screen.queryByText('Slack channels')).not.toBeInTheDocument();
  });

  it('shows slack field when Slack type is selected', async () => {
    const user = userEvent.setup();
    render(
      <FormWrapper />,
      buildWrapper([
        getBaseProviderWrapper(),
        getIconProviderWrapper(),
        getErrorProviderWrapper(),
        getInterpolationProviderWrapper(),
        getRouterWrapper({ location: '/ma-dev-test/train/pipelines' }),
        getServiceProviderWrapper({ request: createQueryMockRouter({ CreatePipelineRun: {} }) }),
      ])
    );

    await screen.findByRole('dialog', { name: 'Start new pipeline run' });
    await user.click(screen.getByText('Notifications'));
    await user.click(screen.getByText('Add notification'));
    await user.click(screen.getByRole('radio', { name: 'Slack' }));

    expect(screen.getByText('Slack channels')).toBeInTheDocument();
    expect(screen.queryByText('Email addresses')).not.toBeInTheDocument();
  });

  it('submits email notification with resource_type hardcoded to PIPELINE_RUN', async () => {
    const user = userEvent.setup();
    const mockRequest = createQueryMockRouter({ CreatePipelineRun: {} });
    render(
      <FormWrapper />,
      buildWrapper([
        getBaseProviderWrapper(),
        getIconProviderWrapper(),
        getErrorProviderWrapper(),
        getInterpolationProviderWrapper(),
        getRouterWrapper({ location: '/ma-dev-test/train/pipelines' }),
        getServiceProviderWrapper({ request: mockRequest }),
      ])
    );

    const dialog = await screen.findByRole('dialog', { name: 'Start new pipeline run' });
    await user.click(screen.getByText('Notifications'));
    await user.click(screen.getByText('Add notification'));
    await user.click(screen.getByRole('radio', { name: 'Email' }));

    const emailInput = screen.getByPlaceholderText('Enter an email and press Enter');
    await user.type(emailInput, 'test@example.com{Enter}');

    await user.click(screen.getByRole('checkbox', { name: 'Failed' }));

    await user.click(within(dialog).getByRole('button', { name: 'Run' }));

    await waitFor(() => {
      expect(mockRequest).toHaveBeenCalledWith(
        'CreatePipelineRun',
        expect.objectContaining({
          spec: expect.objectContaining({
            notifications: [
              expect.objectContaining({
                notification_type: 'EMAIL',
                emails: ['test@example.com'],
                event_types: ['PIPELINE_RUN_STATE_FAILED'],
                resource_type: 'PIPELINE_RUN',
              }),
            ],
          }) as Record<string, unknown>,
        }),
        {}
      );
    });
  });

  it('submits slack notification with resource_type hardcoded to PIPELINE_RUN', async () => {
    const user = userEvent.setup();
    const mockRequest = createQueryMockRouter({ CreatePipelineRun: {} });
    render(
      <FormWrapper />,
      buildWrapper([
        getBaseProviderWrapper(),
        getIconProviderWrapper(),
        getErrorProviderWrapper(),
        getInterpolationProviderWrapper(),
        getRouterWrapper({ location: '/ma-dev-test/train/pipelines' }),
        getServiceProviderWrapper({ request: mockRequest }),
      ])
    );

    const dialog = await screen.findByRole('dialog', { name: 'Start new pipeline run' });
    await user.click(screen.getByText('Notifications'));
    await user.click(screen.getByText('Add notification'));
    await user.click(screen.getByRole('radio', { name: 'Slack' }));

    const slackInput = screen.getByPlaceholderText('#channel-name');
    await user.type(slackInput, '#alerts{Enter}');

    await user.click(screen.getByRole('checkbox', { name: 'Succeeded' }));
    await user.click(screen.getByRole('checkbox', { name: 'Failed' }));

    await user.click(within(dialog).getByRole('button', { name: 'Run' }));

    await waitFor(() => {
      expect(mockRequest).toHaveBeenCalledWith(
        'CreatePipelineRun',
        expect.objectContaining({
          spec: expect.objectContaining({
            notifications: [
              expect.objectContaining({
                notification_type: 'SLACK',
                slack_destinations: ['#alerts'],
                event_types: ['PIPELINE_RUN_STATE_SUCCEEDED', 'PIPELINE_RUN_STATE_FAILED'],
                resource_type: 'PIPELINE_RUN',
              }),
            ],
          }) as Record<string, unknown>,
        }),
        {}
      );
    });
  });

  it('removes a notification entry', async () => {
    const user = userEvent.setup();
    render(
      <FormWrapper />,
      buildWrapper([
        getBaseProviderWrapper(),
        getIconProviderWrapper(),
        getErrorProviderWrapper(),
        getInterpolationProviderWrapper(),
        getRouterWrapper({ location: '/ma-dev-test/train/pipelines' }),
        getServiceProviderWrapper({ request: createQueryMockRouter({ CreatePipelineRun: {} }) }),
      ])
    );

    await screen.findByRole('dialog', { name: 'Start new pipeline run' });
    await user.click(screen.getByText('Notifications'));
    await user.click(screen.getByText('Add notification'));

    expect(screen.getByText('Notification 1')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: /remove/i }));

    expect(screen.queryByText('Notification 1')).not.toBeInTheDocument();
  });

  it('submits without notifications when none are added', async () => {
    const user = userEvent.setup();
    const mockRequest = createQueryMockRouter({ CreatePipelineRun: {} });
    render(
      <FormWrapper />,
      buildWrapper([
        getBaseProviderWrapper(),
        getIconProviderWrapper(),
        getErrorProviderWrapper(),
        getInterpolationProviderWrapper(),
        getRouterWrapper({ location: '/ma-dev-test/train/pipelines' }),
        getServiceProviderWrapper({ request: mockRequest }),
      ])
    );

    const dialog = await screen.findByRole('dialog', { name: 'Start new pipeline run' });
    await user.click(within(dialog).getByRole('button', { name: 'Run' }));

    await waitFor(() => {
      expect(mockRequest).toHaveBeenCalledWith(
        'CreatePipelineRun',
        expect.objectContaining({
          spec: expect.objectContaining({
            pipeline: { name: 'test-pipeline', namespace: 'ma-dev-test' },
          }) as Record<string, unknown>,
        }),
        {}
      );

      expect(mockRequest).not.toHaveBeenCalledWith(
        'CreatePipelineRun',
        expect.objectContaining({
          spec: expect.objectContaining({
            notifications: expect.anything() as unknown,
          }) as Record<string, unknown>,
        }),
        {}
      );
    });
  });
});
