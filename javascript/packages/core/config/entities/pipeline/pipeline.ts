import { ActionHierarchy } from '#core/components/actions/types';
import { interpolate } from '#core/interpolation/interpolate';
import { generateSuffix } from '#core/utils/name-utils';
import { PIPELINE_DETAIL_CONFIG } from './detail';
import { PIPELINE_LIST_CONFIG } from './list';

import type { PhaseEntityConfig } from '#core/types/common/studio-types';
import type { Pipeline } from '#core/config/entities/pipeline/types';

export const PIPELINE_ENTITY_CONFIG: PhaseEntityConfig = {
  id: 'pipelines',
  name: 'Pipelines',
  service: 'pipeline',
  state: 'active',
  views: [PIPELINE_LIST_CONFIG, PIPELINE_DETAIL_CONFIG],
  actions: [
    {
      display: { label: 'Run', icon: 'playerPlay' },
      hierarchy: ActionHierarchy.PRIMARY,
      operation: {
        type: 'mutation',
        mutation: { mutationName: 'CreatePipelineRun' },
        middleware: {
          startEmpty: true,
          operations: [
            {
              destination: 'metadata.name',
              default: () => `run${generateSuffix({ withDate: true })}`,
            },
            { destination: 'metadata.namespace', default: ({ studio }) => studio.projectId },
            { destination: 'spec.actor.name', default: 'mastudio-user' },
            { destination: 'spec.pipeline.name', source: 'metadata.name' },
            { destination: 'spec.pipeline.namespace', default: ({ studio }) => studio.projectId },
          ],
        },
      },
      modal: {
        type: 'confirm',
        header: { title: 'Start new pipeline run' },
        body: interpolate(({ data }) => `Run pipeline **${(data as Pipeline).metadata.name}**?`),
        button: { label: 'Run' },
      },
    },
  ],
};
