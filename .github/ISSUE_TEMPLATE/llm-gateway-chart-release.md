---
name: LLM gateway chart release
about: Release the independently versioned LiteLLM gateway Helm chart
title: 'Release LLM gateway chart vX.Y.Z'
labels: ['release']
assignees: ''
---

## Automated checks

- [ ] The `llm-gateway-chart-release` environment exists and requires approval.
- [ ] A tag ruleset restricts creation of `michelangelo-llm-gateway-v*` tags.
- [ ] The version matches `helm/michelangelo-llm-gateway/Chart.yaml`.
- [ ] `appVersion`, the LiteLLM image tag, and the image digest were reviewed together.
- [ ] The LLM gateway chart validation workflow passed.
- [ ] The release tag uses `michelangelo-llm-gateway-vX.Y.Z`.

## Real-cluster evidence

- Staging run:
- Migration and rollback run:
- Provider, metrics, and shutdown run:
- Uninstall cleanup run:

## Publication

- [ ] The OCI artifact is publicly installable without registry credentials.
