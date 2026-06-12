import { cloneDeep, get, isNil, set, unset } from 'lodash';

import { applyScaffold } from './apply-scaffold';

import type { StudioParamsBase } from '#core/hooks/routing/use-studio-params/types';
import type { MiddlewareOptions, MiddlewareSchema } from './types';

export function applyMiddleware<T extends object>(
  record: T,
  schema: MiddlewareSchema,
  context?: StudioParamsBase,
  options?: MiddlewareOptions
): T {
  const base = schema.startEmpty ? ({} as T) : cloneDeep(record);
  const clone = applyScaffold(base, schema);

  if (!schema.operations) return clone;

  // When startEmpty, read sources from the original record so operations can
  // pull field values from it while building a fresh payload.
  const sourceObject = options?.sourceFromObject ?? (schema.startEmpty ? record : clone);

  for (const op of schema.operations) {
    if (op.subTypes && !op.subTypes.includes(get(clone, schema.subTypePath!) as string)) {
      continue;
    }

    if (op.transformation === 'unset') {
      unset(clone, op.destination);
      continue;
    }

    const sourceValue: unknown = op.source !== undefined ? get(sourceObject, op.source) : undefined;

    if (!isNil(sourceValue)) {
      const transformed =
        typeof op.transformation === 'function' ? op.transformation(sourceValue) : sourceValue;
      set(clone, op.destination, transformed);
    } else if ('default' in op) {
      const defaultVal =
        typeof op.default === 'function'
          ? (op.default as (args: { studio: StudioParamsBase }) => unknown)({ studio: context! })
          : op.default;
      set(clone, op.destination, defaultVal);
    }
  }

  return clone;
}
