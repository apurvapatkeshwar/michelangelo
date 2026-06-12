# Comet ML

This guide explains how platform operators and ML engineers can connect [Comet ML](https://www.comet.com) to Michelangelo workloads for experiment tracking, model management, and offline evaluation. Comet ML is an experiment management platform that captures hyperparameters, metrics, code snapshots, artifacts, and system metrics per training run and makes them searchable and comparable across your team.

> **Note:** Comet ML is proprietary software and is not included in this repository. This guide demonstrates how an open-source ML platform like Michelangelo can be integrated with an Experiment Management system. Comet's EM platform can be procured separately — see [comet.com](https://www.comet.com) for licensing and trial options. The integration patterns shown here are illustrative; other Experiment Management platforms with compatible logging APIs could be substituted.

Michelangelo does not bundle a Comet ML server. This guide assumes you have an active Comet ML account (cloud or self-hosted) and an API key.

> **Before you begin:** Complete [Experiment Tracking Setup](../experiment-tracking.md) — the platform-level guide for network reachability, ConfigMap injection, and auth. This Comet guide builds on those foundations.

---

## How Comet ML Works with Michelangelo

```text
┌─────────────────────────────────────────────┐
│ Operator Responsibility                     │
│ ├─ Verify network egress to comet.com       │
│ │   (or to self-hosted COMET_URL_OVERRIDE)  │
│ └─ Ensure comet_ml is in the task image     │
└─────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────┐
│ User Responsibility (task code)             │
│ ├─ Import comet_ml first (before ML libs)   │
│ ├─ Set COMET_API_KEY at pipeline submission │
│ ├─ Call comet_ml.start() inside the task    │
│ └─ Log params, metrics, models, artifacts  │
└─────────────────────────────────────────────┘
```

Michelangelo does not intercept or wrap Comet ML calls. Users call the Comet SDK directly inside `@uniflow.task()` functions. The operator's job is to ensure the Comet ML endpoint is reachable from task pods and that the `comet_ml` package is available in the task's Docker image.

### Concept mapping (Michelangelo → Comet)

| Michelangelo | Comet | Why it matters |
|---|---|---|
| Pipeline run / training job | Experiment | Single, versioned record of what ran and what it produced |
| Training configuration | Parameters + metadata (e.g., tags) | Makes runs searchable and comparable across many experiments |
| Offline evaluation (global + sliced) | Metrics + assets | Captures outcomes and supporting artifacts (plots, reports) per run |
| Model registry entry | Logged model artifact (linked back to the experiment) | Traceability from a promoted model back to the training run |

---

## Prerequisites

- A Comet ML account. Sign up at [comet.com](https://www.comet.com) or contact your administrator for a self-hosted instance.
- A Comet ML API key. Generate one from **Account → API Keys** in the Comet ML UI.
- `comet_ml` added to your task's `requirements.txt` (users add this; not installed by default).
- Network egress from task pods to `https://www.comet.com` (or to your self-hosted endpoint). See [Step 1](#step-1-verify-network-reachability) below.

---

## Step 1: Verify Network Reachability

Task pods run inside the compute cluster namespace registered with Michelangelo. Confirm that pods in that namespace can reach the Comet ML endpoint before proceeding.

```bash
kubectl run comet-connectivity-test \
  --image=curlimages/curl \
  --namespace=<compute-namespace> \
  --restart=Never \
  --rm -it -- \
  curl -sv https://www.comet.com/api/isAlive
```

A `200 OK` response confirms reachability. If your organization runs a self-hosted Comet instance, replace the URL with your internal endpoint.

If you need to add an egress rule for task pods:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-comet-egress
  namespace: <compute-namespace>
spec:
  podSelector:
    matchLabels:
      <your-pod-selector-label>: <your-value>
  policyTypes:
    - Egress
  egress:
    # Allow DNS resolution
    - ports:
        - protocol: UDP
          port: 53
        - protocol: TCP
          port: 53
    # Allow HTTPS to Comet ML cloud
    - to:
        - ipBlock:
            cidr: 0.0.0.0/0
      ports:
        - protocol: TCP
          port: 443
```

For a self-hosted instance on a known IP, replace the `ipBlock` with the specific CIDR for your Comet server.

---

## Step 2: Configure Credentials

`COMET_API_KEY` is a user-space configuration — pass it at pipeline submission time, not in the Michelangelo system ConfigMap. The Comet SDK reads these environment variables automatically; no configuration calls are needed in task code.

### Cloud Comet ML

```bash
ma pipeline dev-run -f pipeline.yaml --env COMET_API_KEY=<your-api-key>
```

You can also set a default workspace and project to avoid hardcoding them in task code:

```bash
ma pipeline dev-run -f pipeline.yaml \
  --env COMET_API_KEY=<your-api-key> \
  --env COMET_WORKSPACE=<your-workspace> \
  --env COMET_PROJECT_NAME=<your-project>
```

### Self-hosted Comet ML

If your organization runs a self-hosted Comet instance, also set `COMET_URL_OVERRIDE` to point the SDK at your internal server:

```bash
ma pipeline dev-run -f pipeline.yaml \
  --env COMET_API_KEY=<your-api-key> \
  --env COMET_URL_OVERRIDE=<your-org-comet-endpoint>
```

:::warning
Avoid hardcoding API keys in source code or pipeline YAML files committed to version control. Pass them at runtime via `--env` or a secrets manager integrated with your CI/CD system.
:::

---

## What Users Do (Task Code)

Once the operator has confirmed network reachability (Step 1), users instrument their `@uniflow.task()` functions with the Comet SDK.

### Import order matters

`comet_ml` must be imported **before** any ML framework imports. The SDK uses import-time patching to detect your framework — importing it after PyTorch or TensorFlow may disable automatic logging.

```python
import comet_ml  # must be first import
import michelangelo.uniflow.core as uniflow
from michelangelo.uniflow.plugins.ray import RayTask
```

### What Comet captures

Every call to `comet_ml.start()` creates a new **experiment** — a single, versioned record of a training run. Comet captures the following automatically or via explicit SDK calls:

| Category | What gets recorded |
|---|---|
| **Parameters** | Hyperparameters and configuration values logged with `log_parameters()` |
| **Metrics** | Loss, accuracy, and any custom scalar values logged per step or epoch with `log_metrics()` |
| **Code** | A snapshot of the training code, plus optional git metadata (commit SHA/branch and local diffs/patches) |
| **System info** | CPU, GPU, and memory usage during the run |
| **Models** | Model files or folders logged with `log_model()` |
| **Tags** | Arbitrary labels for filtering and grouping experiments in the UI |

### Basic experiment logging

```python
import comet_ml  # must be first import
import michelangelo.uniflow.core as uniflow
from michelangelo.uniflow.plugins.ray import RayTask

@uniflow.task(config=RayTask(head_cpu=4, head_memory="8Gi"))
def train(train_data, config: dict):
    experiment = comet_ml.start(
        project_name="eta-estimation",
        experiment_config=comet_ml.ExperimentConfig(
            name="transformer-sf-market",
            tags=["transformer", "san-francisco"],
        ),
    )

    experiment.log_parameters(config)

    model = _run_training(train_data, config)

    experiment.log_metrics({
        "val_mae":  model.val_mae,
        "test_mae": model.test_mae,
    })

    experiment.log_model("eta-transformer", "./model_output/")

    experiment.end()
    return model
```

:::tip
Wrap your training body in `try/finally` to ensure `experiment.end()` is always called, even if the task raises an exception:

```python
try:
    model = _run_training(train_data, config)
    experiment.log_metrics({...})
    experiment.log_model(...)
finally:
    experiment.end()
```
:::

### Logging metrics per step

For training loops where you want to track metrics at each epoch:

```python
@uniflow.task(config=RayTask(head_cpu=4, head_memory="8Gi"))
def train(train_data, config: dict):
    experiment = comet_ml.start(project_name="eta-estimation")
    experiment.log_parameters(config)

    for epoch in range(config["epochs"]):
        train_loss = _train_epoch(epoch)
        val_mae    = _validate(epoch)

        experiment.log_metrics(
            {"train_loss": train_loss, "val_mae": val_mae},
            epoch=epoch,
        )

    experiment.end()
```

### Logging sliced metrics

For segmented evaluation (e.g., by market or product category), log both the global metric and per-slice metrics using a consistent naming convention. This makes it easy to filter and compare slices in the Comet ML UI:

```python
slice_metrics = {
    "mae/global":            overall_mae,
    "mae/market=sf":         sf_mae,
    "mae/market=nyc":        nyc_mae,
    "mae/category=grocery":  grocery_mae,
    "mae/category=rideshare": rideshare_mae,
}

experiment.log_metrics(slice_metrics)
```

### What to log for reproducibility

In addition to parameters and scalar metrics, most teams get better reproducibility and easier debugging by explicitly logging:

- **Dataset lineage:** dataset name + version / snapshot hash, sampling window, feature set version
- **Code & runtime fingerprint:** git commit SHA, container image tag, key package versions, and important environment flags
- **Evaluation specifics:** evaluation dataset version, thresholds, and sliced metrics (e.g., per-market / per-category) to catch regressions that global metrics hide
- **Richer evaluation artifacts:** confusion matrices, PR/ROC curves, calibration plots, and curated error examples (redacted when needed)

---

## Comet ML Model Registry vs Michelangelo Model Registry

Comet ML includes its own model registry. Michelangelo also has a built-in model registry backed by a `Model` Kubernetes custom resource. The two are independent and can be used simultaneously.

| | Comet ML Model Registry | Michelangelo Model Registry |
|---|---|---|
| Backed by | Comet ML experiment database (linked to training run) | Kubernetes `Model` CRD + S3 |
| Queried via | Comet ML UI / SDK | `kubectl get models` / `ma model get` |
| Integrates with serving | Comet serving | Michelangelo `InferenceServer` |
| Required for Michelangelo pipelines? | No | No |

**When to use Comet's registry:** If your organization uses Comet for model governance and lineage, continue using it. Registering a model in Comet links it back to the experiment that produced it, giving you a full audit trail from trained artifact to deployed model.

**When to use Michelangelo's registry:** If you want models to be deployable via Michelangelo's `InferenceServer` (Triton, vLLM, etc.), register them in Michelangelo's registry. You can do this in addition to logging to Comet.

**Using both:** Log experiments and register models to Comet for lineage and governance, and separately register the deployable artifact to Michelangelo for serving. Both calls can live in the same task function.

---

## TensorBoard Profiler Viewer

Comet includes support for custom Python panels that can render profiler traces and other rich artifacts directly inside the Comet UI. For Michelangelo training jobs, this means users can:

- **Trace individual run behavior:** view operator-level execution timelines, kernel launches, and memory activity captured during training
- **Monitor system metrics in context:** GPU utilization, memory bandwidth, and CPU activity displayed alongside training metrics
- **Compare across runs:** spot regressions in throughput or efficiency when you change model architecture, batch size, or hardware configuration

To use it, log profiler output during training (e.g., via PyTorch Profiler or TensorFlow Profiler), then add the [TensorBoard Profiler Viewer panel](https://github.com/comet-ml/comet-examples/tree/master/panels/TensorboardTorchProfilerViewer) to your Comet project. See the [Comet Python panels guide](https://www.comet.com/docs/v2/guides/comet-ui/experiment-management/visualizations/python-panel/) for setup instructions.

---

## Best Practices

- **Import `comet_ml` first.** It must be imported before any ML framework to enable automatic logging.
- **Always call `experiment.end()`.** Wrap your training body in `try/finally` to ensure it runs even if the task raises an exception.
- **Use `experiment_config` for consistent naming.** Setting a name and tags on every experiment makes the Comet ML UI much easier to navigate across large projects.
- **Log dataset provenance.** Use `experiment.log_parameters()` to record dataset versions alongside hyperparameters. This is what makes a run fully reproducible.
- **Log metrics at the step level.** Per-epoch metrics give you a complete picture of training dynamics, not just the final value.
- **Log artifacts for interpretability.** Attach confusion matrices, PR/ROC curves, calibration plots, and sample error cases so reviewers can understand *why* a run changed.
- **Avoid sensitive data in logs.** Prefer aggregate metrics and redacted examples; do not log raw PII or secrets as parameters, metrics, or artifacts.

---

## Troubleshooting

| Symptom | Likely cause | Resolution |
|---|---|---|
| `ConnectionError` or `requests.exceptions.ConnectionError` | Comet endpoint unreachable from pod | Re-run the connectivity test from Step 1; check NetworkPolicy and firewall rules |
| `CometRestApiException: Invalid API key` | `COMET_API_KEY` not set or incorrect | Verify the env var is passed at pipeline submission time |
| `ModuleNotFoundError: No module named 'comet_ml'` | `comet_ml` not in task's Docker image | Add `comet_ml` to `requirements.txt` or the project Dockerfile |
| Automatic framework logging not working | `comet_ml` imported after ML framework | Move `import comet_ml` to the top of the file, before PyTorch/TensorFlow imports |
| Experiment logged but artifacts missing | Artifact upload timeout or network issue | Check pod egress to Comet artifact storage; consider increasing task timeout |
| Self-hosted endpoint not reached | `COMET_URL_OVERRIDE` not set | Add `--env COMET_URL_OVERRIDE=<your-endpoint>` at pipeline submission time |

For additional FAQs, see the [Comet ML troubleshooting guide](https://www.comet.com/docs/v2/guides/experiment-management/troubleshooting-and-faq/).

---

## Next Steps

- [Experiment Tracking Setup](../experiment-tracking.md) — platform-level setup guide: network reachability, ConfigMap injection, and auth patterns for any tracking server
- [Model Registry](../components/model-registry.md) — Michelangelo's built-in model registry: storage configuration, RBAC, and serving integration
- [MLflow Integration](mlflow.md) — guide for connecting a self-hosted or Databricks-managed MLflow Tracking Server
- [Register a Compute Cluster](../setup/register-a-compute-cluster-to-michelangelo-control-plane.md) — how to add a Kubernetes cluster so Michelangelo can dispatch jobs to it
- [Comet ML Documentation](https://www.comet.com/docs/v2/) — official Comet docs for SDK reference, UI guide, and Python panels
