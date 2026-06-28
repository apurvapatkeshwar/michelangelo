[![GitHub Release](https://img.shields.io/github/v/release/michelangelo-ai/michelangelo)](https://github.com/michelangelo-ai/michelangelo/releases)
[![License](https://img.shields.io/github/license/michelangelo-ai/michelangelo)](http://www.apache.org/licenses/LICENSE-2.0)
[![codecov](https://codecov.io/gh/michelangelo-ai/michelangelo/graph/badge.svg?token=HKJDT0I6CW)](https://codecov.io/gh/michelangelo-ai/michelangelo)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/11481/badge)](https://www.bestpractices.dev/projects/11481)

# Michelangelo-AI

Michelangelo-AI is an open-source ML platform built for **production scale**: multi-cluster model deployment, metric-driven rollbacks, and runtime-agnostic serving — all through a single declarative API.

> :warning: **Beta** — APIs and features may evolve. Breaking changes possible as we stabilize.

---

## Why Michelangelo

Most ML serving tools answer "how do I serve one model on one cluster?" Michelangelo answers the harder question: **how do I safely roll out a new model version across many clusters, catch regressions automatically, and recover without human intervention?**

| Capability | What it means |
|---|---|
| **Multi-cluster deployment** | One `Deployment` CR targets N clusters; the controller manages traffic shifting per cluster independently |
| **Metric health gates** | Define PromQL rules in your deployment spec — if P99 or error rate exceeds threshold, the controller rolls back automatically |
| **Runtime-agnostic** | Works with KServe, Triton, vLLM, or any Kubernetes-based serving runtime; swap runtimes without changing how deployments work |
| **Condition-based rollout engine** | Rollout advances stage by stage (placement → traffic shift → health check → complete) only when each gate passes |
| **Full ML lifecycle** | Pipelines, training, evaluation, and serving under one platform with a unified project/revision model |

---

## Features

- **ML Pipelines**: Define DAGs with `@task` and `@workflow` decorators; run on distributed compute with Temporal-backed orchestration.
- **Model Training**: Distributed training across nodes via Ray Train + PyTorch Lightning, with pluggable experiment tracking (Comet, MLflow). See the [MovieLens NCF example](python/examples/movielens/).
- **Model Serving**: Deploy to real-time or batch inference endpoints backed by any serving runtime.
- **Multi-cluster Deployment**: Roll out across clusters with configurable traffic increments, automatic canary advancement, and rollback.
- **Metric Health Gates**: PromQL-based rollback rules evaluated per reconcile cycle — no alert manager or external automation needed.
- **Monitoring**: Continuous model performance tracking tied to the deployment lifecycle.

---

## Quickstart

### 1. Install dependencies

```bash
git clone https://github.com/michelangelo-ai/michelangelo.git
cd michelangelo/python
poetry install
source .venv/bin/activate
```

### 2. Create the local sandbox

```bash
# Single-cluster sandbox
ma sandbox create

# Multi-cluster sandbox (adds a compute cluster for serving demos)
ma sandbox create --create-compute-cluster
```

### 3. Run a demo

```bash
# ML pipeline demo
ma sandbox demo pipeline

# InferenceServer + multi-cluster serving demo
ma sandbox demo inference

# KServe serving demo with metric-driven rollback (requires --create-compute-cluster)
ma sandbox demo kserve
```

---

## Demo: Multi-Cluster Model Serving with KServe

The KServe demo shows Michelangelo's three core strengths end-to-end.

### What gets set up

`ma sandbox demo kserve` automates the full stack on `michelangelo-compute-1`:

- cert-manager + KServe v0.13.1 (RawDeployment mode — no Knative or Istio required)
- Custom Triton `ClusterServingRuntime`
- `sklearn-iris` InferenceService serving predictions from GCS
- Michelangelo `InferenceServer` and `Deployment` CRs wiring it into the platform

### Strength 1 — Declarative multi-cluster deployment

A single `Deployment` CR rolls out the model across clusters, advancing traffic in configurable increments:

```yaml
apiVersion: michelangelo.api/v2
kind: Deployment
metadata:
  name: sklearn-iris-deployment
spec:
  inferenceServer:
    name: inference-server-multi   # targets compute-1 and compute-2
  strategy:
    rolling:
      incrementPercentage: 10      # advance 10% at a time
  desiredRevision:
    name: sklearn-iris-v2
```

The controller handles cluster-by-cluster traffic shifting automatically — no per-cluster kubectl required.

### Strength 2 — Metric health gates (automatic rollback)

Define rollback rules directly in the deployment spec using PromQL:

```yaml
spec:
  healthCheckConfig:
    prometheusUrl: "http://prometheus:9090"
    rules:
      - name: high-error-rate
        query: >-
          rate(kserve_inference_service_request_total{name="sklearn-iris",response_code!~"2.."}[2m])
          / rate(kserve_inference_service_request_total{name="sklearn-iris"}[2m])
        op: GT
        threshold: 0.05      # roll back if error rate > 5%

      - name: high-latency-p99
        query: >-
          histogram_quantile(0.99,
            rate(kserve_inference_service_request_latency_seconds_bucket{name="sklearn-iris"}[2m]))
        op: GT
        threshold: 1.0       # roll back if P99 > 1s
```

Apply the full example:

```bash
kubectl apply -f python/michelangelo/cli/sandbox/demo/kserve/deployment-with-healthcheck.yaml
```

**Force a rollback** (useful for live demos) by injecting an always-failing rule:

```bash
kubectl patch deployment sklearn-iris-deployment --type=merge -p '{
  "spec": {"healthCheckConfig": {"rules": [
    {"name":"force-rollback","query":"vector(1)","op":"GT","threshold":0}
  ]}}
}'
# → controller detects breach within one reconcile cycle and rolls back
```

**Restore and re-deploy:**

```bash
kubectl patch deployment sklearn-iris-deployment --type=merge -p '{
  "spec": {"healthCheckConfig": {"rules": []}}
}'
# → rollout resumes and advances automatically
```

### Strength 3 — Runtime-agnostic control plane

The `InferenceServer` CR targets clusters and manages routing; it doesn't care what serves the model. Switch from KServe to Triton by updating `servingSpec.version` — the deployment lifecycle is identical:

```yaml
spec:
  servingSpec:
    version: "nvcr.io/nvidia/tritonserver:23.04-py3"   # any image works
```

### Test the endpoint

```bash
kubectl --context k3d-michelangelo-compute-1 \
  port-forward svc/sklearn-iris-predictor -n default 8081:80

curl -X POST http://localhost:8081/v1/models/sklearn-iris:predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [[6.8, 2.8, 4.8, 1.4]]}'
# → {"predictions": [1]}
```

---

## Define your own pipeline

```python
import michelangelo.uniflow.core as uniflow

@uniflow.task()
def train(learning_rate: float = 0.01) -> str:
    # your training logic here
    return "model_path"

@uniflow.workflow()
def my_pipeline(learning_rate: float = 0.01):
    model = train(learning_rate=learning_rate)
```

For a full walkthrough, see the [Getting Started with ML Pipelines](https://michelangelo-ai.org/docs/user-guides/getting-started/getting-started) guide.

---

## Build and Test

See the [User Guides](https://michelangelo-ai.org/docs/user-guides/) for instructions on running tests and working with the development environment.

## Contributing

We welcome contributions! Please read our [Contributing Guidelines](https://github.com/michelangelo-ai/michelangelo/blob/main/CONTRIBUTING.md) to get started.

## License

[Apache 2.0](https://github.com/michelangelo-ai/michelangelo/blob/main/LICENSE)

## Acknowledgments

Thank you to the Michelangelo Open Source team for getting this project off the ground, and to all contributors.
