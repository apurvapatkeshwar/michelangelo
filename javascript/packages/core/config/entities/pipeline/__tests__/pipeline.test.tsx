import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { InterpolatableActionsPopover } from '#core/components/actions/interpolatable-actions-popover';
import { PIPELINE_ENTITY_CONFIG } from '#core/config/entities/pipeline/pipeline';
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
import { getSnackbarProviderWrapper } from '#core/test/wrappers/get-snackbar-provider-wrapper';

import type { ActionConfigSchema, Data } from '#core/components/actions/types';
import type { Pipeline } from '#core/config/entities/pipeline/types';

const RUN_ACTIONS = PIPELINE_ENTITY_CONFIG.actions as ActionConfigSchema<Data>[];

function buildPipeline(): Pipeline {
  return {
    metadata: { name: 'my-pipeline', namespace: 'test-ns' },
    spec: { owner: { name: 'test-owner' } },
  };
}

describe('PIPELINE_ENTITY_CONFIG: run action', () => {
  it('opens a confirm dialog naming the pipeline and fires CreatePipelineRun with a clean payload', async () => {
    const user = userEvent.setup();
    const record = buildPipeline();
    const mockRequest = createQueryMockRouter({
      CreatePipelineRun: { pipelineRun: { metadata: { name: 'run-created' } } },
    });

    render(
      <InterpolatableActionsPopover actions={RUN_ACTIONS} record={record as unknown as Data} />,
      buildWrapper([
        getBaseProviderWrapper(),
        getErrorProviderWrapper(),
        getIconProviderWrapper(),
        getInterpolationProviderWrapper(),
        getRouterWrapper({ location: '/ma-dev-test/train/pipelines' }),
        getServiceProviderWrapper({ request: mockRequest }),
        getSnackbarProviderWrapper(),
      ])
    );

    await user.click(screen.getByRole('button', { name: 'Actions' }));
    await user.click(await screen.findByRole('option', { name: 'Run' }));

    const dialog = await screen.findByRole('dialog', { name: 'Start new pipeline run' });
    expect(within(dialog).getByText(/my-pipeline/)).toBeInTheDocument();

    await user.click(within(dialog).getByRole('button', { name: 'Run' }));

    await waitFor(() => {
      expect(mockRequest).toHaveBeenCalledWith(
        'CreatePipelineRun',
        expect.objectContaining({
          metadata: expect.objectContaining({
            name: expect.stringMatching(/^run-\d{8}-\d{6}-.+$/) as string,
            namespace: 'ma-dev-test',
          }) as Record<string, unknown>,
          spec: expect.objectContaining({
            actor: { name: 'mastudio-user' },
            pipeline: { name: 'my-pipeline', namespace: 'ma-dev-test' },
          }) as Record<string, unknown>,
        })
      );
    });

    await waitFor(() => {
      expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    });
  });

  it('keeps dialog open and shows error when mutation fails', async () => {
    const user = userEvent.setup();
    const record = buildPipeline();
    const mockRequest = createQueryMockRouter({ CreatePipelineRun: new Error('Create failed') });

    render(
      <InterpolatableActionsPopover actions={RUN_ACTIONS} record={record as unknown as Data} />,
      buildWrapper([
        getBaseProviderWrapper(),
        getErrorProviderWrapper(),
        getIconProviderWrapper(),
        getInterpolationProviderWrapper(),
        getRouterWrapper({ location: '/ma-dev-test/train/pipelines' }),
        getServiceProviderWrapper({ request: mockRequest }),
        getSnackbarProviderWrapper(),
      ])
    );

    await user.click(screen.getByRole('button', { name: 'Actions' }));
    await user.click(await screen.findByRole('option', { name: 'Run' }));
    const dialog = await screen.findByRole('dialog', { name: 'Start new pipeline run' });
    await user.click(within(dialog).getByRole('button', { name: 'Run' }));

    await within(dialog).findByText(/Test error/);
    expect(screen.getByRole('dialog', { name: 'Start new pipeline run' })).toBeInTheDocument();
  });
});
