import { Block } from 'baseui/block';
import { HeadingXXLarge } from 'baseui/typography';

import { ConfigDrivenForm } from '#core/components/form/config-driven-form';
import { MainViewContainer } from '#core/components/views/main-view-container';

import type { FormConfig } from '#core/components/form/types';

const SAMPLE_FORM_CONFIG: FormConfig = {
  entities: {
    'spec.title': { type: 'string', label: 'Title', required: true, placeholder: 'Enter a title' },
    'spec.description': { type: 'textarea', label: 'Description', rows: 3 },
    'spec.tags': { type: 'string', label: 'Tags', multi: true, placeholder: 'Add a tag' },
    'spec.priority': {
      type: 'select',
      label: 'Priority',
      options: [
        { id: 'low', label: 'Low' },
        { id: 'medium', label: 'Medium' },
        { id: 'high', label: 'High' },
      ],
    },
    'spec.enabled': { type: 'boolean', label: 'Enabled', toggle: true },
    'spec.replicas': { type: 'number', label: 'Replicas', placeholder: '1' },
  },
  layout: [
    {
      type: 'group',
      label: 'General',
      items: [
        'spec.title',
        'spec.description',
        { type: 'row', span: [1, 1], items: ['spec.priority', 'spec.replicas'] },
      ],
    },
    {
      type: 'group',
      label: 'Settings',
      items: ['spec.enabled', 'spec.tags'],
    },
  ],
};

export function Sandbox() {
  return (
    <MainViewContainer>
      <HeadingXXLarge>Component Sandbox</HeadingXXLarge>
      <Block marginBottom="24px">Config-driven form proof-of-life.</Block>
      <Block width="600px">
        <ConfigDrivenForm config={SAMPLE_FORM_CONFIG} onSubmit={() => undefined} />
      </Block>
    </MainViewContainer>
  );
}
