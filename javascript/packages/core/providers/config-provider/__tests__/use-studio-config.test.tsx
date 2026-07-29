import { renderHook } from '@testing-library/react';

import { buildWrapper } from '#core/test/wrappers/build-wrapper';
import { getConfigProviderWrapper } from '#core/test/wrappers/get-config-provider-wrapper';
import { useStudioConfig } from '../use-studio-config';

import type { StudioConfig } from '#core/types/common/studio-types';

describe('useStudioConfig', () => {
  function buildConfig(): StudioConfig {
    return {
      categories: [
        {
          id: 'core-ml',
          name: 'Core ML',
          phases: [
            {
              id: 'train',
              icon: 'train',
              name: 'Train',
              state: 'active',
              entities: [
                {
                  id: 'pipelines',
                  name: 'Pipelines',
                  service: 'pipeline',
                  state: 'active',
                  views: [],
                },
                {
                  id: 'runs',
                  name: 'Runs',
                  service: 'pipelineRun',
                  state: 'active',
                  views: [],
                },
              ],
            },
            {
              id: 'deploy',
              icon: 'deploy',
              name: 'Deploy',
              state: 'active',
              entities: [
                {
                  id: 'targets',
                  name: 'Targets',
                  service: 'inferenceServer',
                  state: 'active',
                  views: [],
                },
              ],
            },
          ],
        },
      ],
    };
  }

  test('returns categories from config', () => {
    const config = buildConfig();
    const { result } = renderHook(
      () => useStudioConfig(),
      buildWrapper([getConfigProviderWrapper(config)])
    );

    expect(result.current.categories).toEqual(config.categories);
  });

  test('getPhase returns matching phase', () => {
    const config = buildConfig();
    const { result } = renderHook(
      () => useStudioConfig(),
      buildWrapper([getConfigProviderWrapper(config)])
    );

    expect(result.current.getPhase('train')).toEqual(config.categories[0].phases[0]);
  });

  test('getPhase returns undefined for unknown phase', () => {
    const { result } = renderHook(
      () => useStudioConfig(),
      buildWrapper([getConfigProviderWrapper(buildConfig())])
    );

    expect(result.current.getPhase('nonexistent')).toBeUndefined();
  });

  test('getEntity returns matching entity within phase', () => {
    const config = buildConfig();
    const { result } = renderHook(
      () => useStudioConfig(),
      buildWrapper([getConfigProviderWrapper(config)])
    );

    expect(result.current.getEntity('train', 'pipelines')).toEqual(
      config.categories[0].phases[0].entities[0]
    );
  });

  test('getEntity returns undefined for unknown entity in valid phase', () => {
    const { result } = renderHook(
      () => useStudioConfig(),
      buildWrapper([getConfigProviderWrapper(buildConfig())])
    );

    expect(result.current.getEntity('train', 'nonexistent')).toBeUndefined();
  });

  test('getEntity returns undefined for unknown phase', () => {
    const { result } = renderHook(
      () => useStudioConfig(),
      buildWrapper([getConfigProviderWrapper(buildConfig())])
    );

    expect(result.current.getEntity('nonexistent', 'pipelines')).toBeUndefined();
  });

  test('throws when used outside ConfigProvider', () => {
    expect(() => {
      const { result } = renderHook(() => useStudioConfig());
      result.current.getPhase('train');
    }).toThrow('must be used within a ConfigProvider');
  });
});
