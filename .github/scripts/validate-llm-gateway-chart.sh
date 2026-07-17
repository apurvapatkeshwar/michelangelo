#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHART_DIR="$ROOT_DIR/helm/michelangelo-llm-gateway"

render() {
  helm template gateway "$CHART_DIR" \
    --namespace litellm \
    --kube-version 1.27.0 \
    "$@" \
    >/dev/null
}

lint_profile() {
  helm lint --strict "$CHART_DIR" "$@"
}

validate_profile() {
  lint_profile "$@"
  render "$@"
}

expect_failure() {
  local name="$1"
  local expected="$2"
  local normalized_output
  local output
  shift 2

  if output="$(helm template gateway "$CHART_DIR" "$@" 2>&1)"; then
    printf 'Expected validation failure: %s\n' "$name" >&2
    exit 1
  fi
  normalized_output="${output//\//.}"
  if [[ "$normalized_output" != *"$expected"* ]]; then
    printf 'Validation failed for the wrong reason: %s\n%s\n' "$name" "$output" >&2
    exit 1
  fi
}

assert_contains() {
  local output="$1"
  local expected="$2"

  if [[ "$output" != *"$expected"* ]]; then
    printf 'Expected rendered output to contain: %s\n' "$expected" >&2
    exit 1
  fi
}

validate_profile
validate_profile -f "$CHART_DIR/examples/values-gcp.yaml"
validate_profile \
  --set migrationJob.hooks.helm.enabled=false \
  --set migrationJob.hooks.argocd.enabled=true
validate_profile \
  --set config.create=false \
  --set config.existingConfigMap=litellm-config \
  --set migrationJob.enabled=false \
  --set migrationJob.managedExternally=true \
  --set migrationJob.hooks.helm.enabled=false
validate_profile \
  --set networkPolicy.enabled=true \
  --set networkPolicy.migration.managedExternally=true \
  --set serviceMonitor.enabled=true \
  --set serviceMonitor.authorization.secretName=litellm-metrics \
  --set ingress.enabled=true \
  --set 'ingress.hosts[0].host=litellm.example.com' \
  --set 'ingress.hosts[0].paths[0].path=/' \
  --set 'ingress.hosts[0].paths[0].pathType=Prefix'
validate_profile \
  --set-string 'extraEnv[0].name=my.env-name' \
  --set-string 'extraEnv[0].value=ok'

private_registry_values=(
  --set 'imagePullSecrets[0].name=registry-creds'
  --set image.repository=registry.example.com/litellm
  --set tests.image.repository=registry.example.com/curl
  --set tests.image.tag=8.12.1
)
lint_profile "${private_registry_values[@]}"
private_registry_output="$(helm template gateway "$CHART_DIR" \
  --namespace litellm \
  --kube-version 1.27.0 \
  "${private_registry_values[@]}")"
assert_contains "$private_registry_output" 'registry.example.com/litellm@sha256:'
assert_contains "$private_registry_output" 'registry.example.com/curl@sha256:'
image_pull_secret_count=0
remaining_output="$private_registry_output"
while [[ "$remaining_output" == *'- name: registry-creds'* ]]; do
  remaining_output="${remaining_output#*'- name: registry-creds'}"
  ((image_pull_secret_count += 1))
done
if [[ "$image_pull_secret_count" -ne 3 ]]; then
  printf 'Expected imagePullSecrets on Deployment, migration Job, and Helm test Pod\n' >&2
  exit 1
fi

test_digest="sha256:$(printf 'a%.0s' {1..64})"
test_output="$(helm template gateway "$CHART_DIR" \
  --show-only templates/tests/test-connection.yaml \
  --set probes.liveness.path=/live \
  --set probes.readiness.path=/ready \
  --set-string tests.image.digest="$test_digest")"
assert_contains "$test_output" "curlimages/curl@$test_digest"
assert_contains "$test_output" '/ready'
if [[ "$test_output" == *'/live'* ]]; then
  printf 'Helm test must use the readiness probe path\n' >&2
  exit 1
fi

expect_failure "config ownership" "config.existingConfigMap is required" \
  --set config.create=false
expect_failure "migration ownership" "exactly one of migrationJob.enabled" \
  --set migrationJob.managedExternally=true
expect_failure "migration hook ownership" "managed migrations require exactly one" \
  --set migrationJob.hooks.argocd.enabled=true
expect_failure "reserved migration hook annotation" "cannot override chart-owned hook annotation" \
  --set-string 'migrationJob.annotations.helm\.sh/hook=post-install'
expect_failure "migration NetworkPolicy prerequisite" "pre-existing externally managed NetworkPolicy" \
  --set networkPolicy.enabled=true
expect_failure "ServiceMonitor authentication" "authorization.secretName" \
  --set serviceMonitor.enabled=true
expect_failure "Ingress hosts" "ingress.hosts is required" \
  --set ingress.enabled=true
expect_failure "PDB availability" "set only one of podDisruptionBudget" \
  --set podDisruptionBudget.minAvailable=1
expect_failure "autoscaling bounds" "minReplicas cannot exceed" \
  --set autoscaling.minReplicas=11
expect_failure "plaintext sensitive environment variable" "must use valueFrom" \
  --set 'extraEnv[0].name=API_TOKEN' \
  --set 'extraEnv[0].value=secret'
expect_failure "image digest" "image.digest" \
  --set image.digest=sha256:bad
expect_failure "test image digest" "tests.image.digest" \
  --set-string tests.image.digest=sha256:bad
expect_failure "service account token type" "automountServiceAccountToken" \
  --set-string serviceAccount.automountServiceAccountToken=not-a-bool
expect_failure "image pull secret type" "imagePullSecrets" \
  --set-string 'imagePullSecrets[0]=registry-creds'
expect_failure "unknown value" "unknownGatewayValue" \
  --set unknownGatewayValue=true
expect_failure "existing Secret name" "masterKey.existingSecret" \
  --set masterKey.existingSecret=BAD_NAME
expect_failure "existing Secret key" "masterKey.key" \
  --set masterKey.key=BAD/KEY
expect_failure "existing Secret traversal key" "masterKey.key" \
  --set-string masterKey.key=.
expect_failure "environment variable traversal prefix" "extraEnv" \
  --set-string 'extraEnv[0].name=..BAD' \
  --set-string 'extraEnv[0].value=ok'
expect_failure "environment variable traversal name" "extraEnv" \
  --set-string 'extraEnv[0].name=.' \
  --set-string 'extraEnv[0].value=ok'
expect_failure "migration ServiceAccount name" "migrationJob.serviceAccountName" \
  --set migrationJob.serviceAccountName=BAD_NAME

if [[ "${SKIP_PACKAGE:-false}" != "true" ]]; then
  package_dir="$(mktemp -d)"
  trap 'rm -rf "$package_dir"' EXIT
  helm package "$CHART_DIR" --destination "$package_dir" >/dev/null
fi

printf 'Validated %s\n' "$CHART_DIR"
