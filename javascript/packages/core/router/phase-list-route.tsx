import { useNavigate } from 'react-router-dom-v5-compat';
import { useStyletron } from 'baseui';

import { ErrorView } from '#core/components/error-view/error-view';
import { CircleExclamationMark } from '#core/components/illustrations/circle-exclamation-mark/circle-exclamation-mark';
import { CircleExclamationMarkKind } from '#core/components/illustrations/circle-exclamation-mark/types';
import { PhaseEntityView } from '#core/components/views/phase-entity-view/phase-entity-view';
import { isListableEntity } from '#core/components/views/phase-entity-view/utils';
import { useStudioParams } from '#core/hooks/routing/use-studio-params/use-studio-params';
import { useStudioConfig } from '#core/providers/config-provider/use-studio-config';

export function PhaseListRoute() {
  const [, theme] = useStyletron();
  const { phase, projectId } = useStudioParams('list');
  const navigate = useNavigate();
  const { getPhase } = useStudioConfig();

  const phaseConfig = getPhase(phase);

  if (!phaseConfig) {
    return (
      <ErrorView
        title="Phase not found"
        description={`Phase "${phase}" configuration not found.`}
        illustration={
          <CircleExclamationMark
            kind={CircleExclamationMarkKind.ERROR}
            width={theme.sizing.scale1600}
            height={theme.sizing.scale1600}
          />
        }
        buttonConfig={{
          onClick: () => navigate(`/${projectId}`),
          content: 'Go home',
        }}
      />
    );
  }

  const listableEntities = phaseConfig.entities.filter(isListableEntity);

  if (listableEntities.length === 0) {
    return (
      <ErrorView
        title="Phase has no active entities"
        description={`Phase "${phase}" has no active entities with list views configured.`}
        illustration={
          <CircleExclamationMark
            kind={CircleExclamationMarkKind.ERROR}
            width={theme.sizing.scale1600}
            height={theme.sizing.scale1600}
          />
        }
        buttonConfig={{
          onClick: () => navigate(`/${projectId}`),
          content: 'Go home',
        }}
      />
    );
  }

  return <PhaseEntityView phaseConfig={phaseConfig} entities={listableEntities} />;
}
