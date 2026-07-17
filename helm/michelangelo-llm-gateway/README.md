# Michelangelo LLM Gateway Helm chart

This chart installs LiteLLM as Michelangelo's private LLM gateway data plane. It is installed, versioned, and released separately from the Michelangelo control-plane chart, allowing inference to scale and roll independently.

The chart owns the LiteLLM Deployment, Service, non-secret configuration, database migration Job, health probes, HPA, PodDisruptionBudget, topology spread, and optional Ingress, NetworkPolicy, ServiceMonitor, and Helm test.

Cloud SQL, Redis, Secret Manager, External Secrets Operator, private DNS, TLS, platform routing, dashboards, and alerts remain external.

This is a data-plane-only chart. It does not install the Michelangelo control plane, project-sync controller, Project or key reconciliation, caller credential delivery, an authentication adapter, or provider infrastructure.

In the LiteLLM integration rollout, Milestone 0 validates the upstream LiteLLM chart in staging. This reusable Michelangelo-owned chart is the Milestone 1 data-plane packaging deliverable; it is not Milestone 0 runtime evidence or control-plane automation.

## Prerequisites

- Kubernetes 1.27 or newer.
- Helm 3.12 or newer.
- An existing PostgreSQL database supported by the pinned LiteLLM release.
- Existing Kubernetes Secrets for the master key, salt key, database URL, and provider credentials.
- At least one provider model approved for use by your organization.

## Secret contract

The chart maps individual keys from existing Secrets into fixed runtime variables:

| Values | Container variable |
| --- | --- |
| `masterKey.existingSecret` / `masterKey.key` | `PROXY_MASTER_KEY` |
| `saltKey.existingSecret` / `saltKey.key` | `LITELLM_SALT_KEY` |
| `database.existingSecret` / `database.key` | `DATABASE_URL` |

The defaults read `masterkey`, `saltkey`, and `database-url` from `litellm-secrets`. Add provider credential Secrets to `environmentSecrets`; add non-secret provider configuration to `environmentConfigMaps`.

Provision Secrets through External Secrets Operator or your organization's secret-management workflow before installation. Do not put plaintext credentials in `config.data`, `extraEnv.value`, or committed values files: Helm stores supplied values in the release Secret. The chart creates no Secret resources.

## GCP values

Copy `examples/values-gcp.yaml` and replace each `REPLACE_WITH_...` placeholder with values approved for your deployment. The chart deliberately does not hardcode model generations.

The GCP profile reuses one externally managed `litellm-secrets` Secret. Configure the ExternalSecret target with these keys:

| Kubernetes Secret key | GCP Secret Manager source | Runtime use |
| --- | --- | --- |
| `masterkey` | `litellm/master-key` | LiteLLM master key |
| `saltkey` | `litellm/salt-key` | LiteLLM salt key |
| `database-url` | `litellm/db-dsn` | PostgreSQL DSN |
| `openai-api-key` | `litellm/openai-key` | `OPENAI_API_KEY` |

The profile creates the `litellm` Kubernetes ServiceAccount so an existing `litellm/litellm` Workload Identity binding can be reused. Bind it to a least-privilege Google service account for Vertex AI:

```yaml
serviceAccount:
  name: litellm
  annotations:
    iam.gke.io/gcp-service-account: litellm@my-project.iam.gserviceaccount.com
```

The GCP profile also creates a GKE internal LoadBalancer by setting `service.type=LoadBalancer` and `networking.gke.io/load-balancer-type=Internal`. Create the private Cloud DNS record outside this chart after GKE allocates the address. Never expose this Service through a public load balancer.

The default LiteLLM and Helm-test images are pinned to verified multi-architecture digests. Review and update each image tag and digest together.

## Validate and install

From the Michelangelo repository root:

```bash
helm lint --strict helm/michelangelo-llm-gateway \
  -f helm/michelangelo-llm-gateway/examples/values-gcp.yaml

helm template litellm helm/michelangelo-llm-gateway \
  --namespace litellm \
  -f helm/michelangelo-llm-gateway/examples/values-gcp.yaml

helm upgrade --install litellm helm/michelangelo-llm-gateway \
  --namespace litellm \
  --create-namespace \
  --wait \
  --timeout 15m \
  -f path/to/environment-values.yaml
```

Published releases use an independent OCI chart version. Repository administrators must protect the `llm-gateway-chart-release` environment with required reviewers and restrict the `michelangelo-llm-gateway-v*` tag pattern before the first release. The tag version must equal `Chart.yaml` and point to a commit on `main`. Install a version-pinned release with:

```bash
helm upgrade --install litellm \
  oci://ghcr.io/michelangelo-ai/michelangelo/charts/michelangelo-llm-gateway \
  --version 1.0.0 \
  --namespace litellm \
  --create-namespace \
  --wait \
  --timeout 15m \
  -f path/to/environment-values.yaml
```

Run the post-rollout health check with:

```bash
helm test litellm --namespace litellm
```

## Migrations

The default migration Job is a fail-closed Helm pre-install and pre-upgrade hook. It uses LiteLLM's v2 migration resolver by default. Its database Secret and `migrationJob.serviceAccountName` must exist before installation. Completed and failed Helm and Argo CD hooks delete themselves; capture controller and pod logs while the operation is running. For Argo CD, disable the Helm hook and enable the Argo CD hook. For another orchestrator, disable the Job, set `migrationJob.managedExternally=true`, and disable both hooks.

The proxy never owns schema changes. Chart-owned configuration requires `general_settings.disable_prisma_schema_update: true`; an external ConfigMap must set the same value.

Require backward-compatible expand/contract migrations and a current database recovery point before upgrades. Helm rollback does not roll back the database schema. Qualify the v2 resolver against every pinned LiteLLM upgrade before production.

## Rotation and external configuration

Secret and external ConfigMap values are read when a pod starts. After a secret rotation, change `rollout.runtimeRevision`. After changing an external LiteLLM ConfigMap, change `rollout.configRevision`. Either change performs a rolling restart. A cluster reloader can instead be configured through `podAnnotations`.

## Uninstall

Do not uninstall while a migration hook is running. Remove or redirect the external private DNS record, then uninstall the release:

```bash
helm uninstall litellm --namespace litellm --wait
```

Uninstall deletes the chart-managed Service and its GKE load balancer. It does not delete Cloud SQL, Secret Manager values, ExternalSecret resources, external Secrets or ConfigMaps, private DNS records, or provider infrastructure. It also does not reverse database migrations. Retain or restore those resources according to the environment rollback plan.

## Metrics

Chart-owned configuration enables the Prometheus callback. When `serviceMonitor.enabled=true`, provide `serviceMonitor.authorization.secretName` and `secretKey` for a dedicated LiteLLM scrape credential. An external LiteLLM ConfigMap must also enable the Prometheus callback. Restrict `/metrics` access with the environment NetworkPolicy.

## Production requirements

- Use either the GCP internal LoadBalancer profile or a `ClusterIP` Service behind platform-owned private ingress. Never create a public endpoint.
- Add layer-7 isolation because LiteLLM serves inference and Management API routes on the same port.
- Set `topologySpread.whenUnsatisfiable=DoNotSchedule` and provide capacity in at least two zones.
- Define caller, observability, database, provider, telemetry, and GKE metadata-server NetworkPolicy rules before enabling the policy. Protect the pre-install migration hook with an equivalent namespace or platform policy.
- Keep response caching disabled. If Redis is selected for cross-replica coordination, provide it externally and qualify failure behavior.
- Use a hardened image validated against your cluster's security policy, or document an explicit exception. The upstream image declares a root runtime user; the default chart drops capabilities, blocks privilege escalation, and applies the runtime-default seccomp profile but does not claim restricted-policy compliance.

When `networkPolicy.enabled=true` and this chart owns migrations, provision the migration policy before installing the release. It must select the rendered `app.kubernetes.io/name`, `app.kubernetes.io/instance`, and `app.kubernetes.io/component: migration` labels and allow only DNS and database egress. Then set `networkPolicy.migration.managedExternally=true` to acknowledge that prerequisite. The chart does not create this policy because Helm and Argo CD migration hooks run before ordinary release resources.

## Upgrade policy

The initial runtime contract is LiteLLM `v1.85.1`. For each upgrade:

1. Pin the chart app version, image tag, and digest together.
2. Review release notes and schema migrations.
3. Lint and render every environment profile.
4. Test migrations, readiness, streaming cancellation, budget enforcement, zonal disruption, and rollback in staging.
5. Confirm logs, traces, spend records, and callbacks exclude prompts, completions, authorization headers, and credentials.
