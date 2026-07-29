import { createContext } from 'react';

import type { StudioConfigContextType } from './types';

export const ConfigContext = createContext<StudioConfigContextType>({
  categories: [],
  getPhase: () => {
    throw new Error('getPhase must be used within a ConfigProvider');
  },
  getEntity: () => {
    throw new Error('getEntity must be used within a ConfigProvider');
  },
});
