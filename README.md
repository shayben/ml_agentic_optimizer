# agentic-optimizer

[![CI](https://github.com/shayben/ml_agentic_optimizer/actions/workflows/ci.yml/badge.svg)](https://github.com/shayben/ml_agentic_optimizer/actions/workflows/ci.yml)

Give a **local agent** (your interactive **GitHub Copilot CLI** session) live, two-way access to one or more
**remote PyTorch training runs**. Everything except the training job runs on your machine — a local **MCP stdio
server** plus a local **broker** published to the node over a **Microsoft Dev Tunnel**, so there is **no
separately-hosted broker or third endpoint**. The agent can stream telemetry/metrics (including **MLflow**),
diagnose training, and interject mid-run: tune optimizer
hyperparameters, adjust throughput/hardware settings, queue data/model interrogations, flag suspicious samples,
and drive optional HPO.

**What the agent can do to a live run, without restarting it:**

- **Observe** — step/epoch, loss history, gradient norm, throughput, GPU memory & utilization, pause state,
  `last_error`, anomalies, scheduler state, a step-time profile, and MLflow run linkage.
- **Tune the optimizer** — `lr`, `weight_decay`, `momentum`, `grad_clip` (bounded by **guardrails**).
- **Steer hardware/throughput** — `batch_size`, `num_workers`, `grad_accum_steps`, `amp` (via your callback).
- **Control the schedule** — read scheduler state and reconfigure the LR scheduler mid-run.
- **Interrogate** — per-class loss or any custom model/data query you register, mid-run.
- **Fix data & robustness** — pull the highest per-sample losses and flag likely label noise to down-weight or
  filter, and toggle data augmentation mid-run (`set_augmentation`).
- **Train safely** — **checkpoint + rollback** a bad change, **guardrails** clamp unsafe hyperparameters, and
  **anomaly auto-pause** halts on NaN/Inf/grad-explosion.
- **Profile** — a built-in step-time breakdown (data-wait vs forward vs backward) for throughput RCA.
- **Manage the run** — graceful `stop_training` (free the GPU) and `extend_training` (raise `max_epochs`).
- **Search** — an optional **Optuna** advisor the agent layers root-cause reasoning on top of.

Runs on a raw loop, **PyTorch Lightning**, or **Hugging Face `Trainer`**, and scales to **DDP** (rank-0 telemetry
+ command broadcast). A one-line `attach(optimizer, model)` makes the *same* script run unchanged with or without
a broker.

```
 Local dev box  (agent + broker — nothing hosted separately)          Remote node (AML/GPU)
 ┌────────────────────────────────────────────────┐                   ┌───────────────────────┐
 │ GitHub Copilot CLI ─MCP stdio─► mcp_server      │     Dev Tunnel     │ PyTorch training loop  │
 │ broker (FastAPI, --tunnel)                      │  public HTTPS URL  │ + TrainingBridge       │
 │ telemetry · queues · runs · bearer-token auth   │◄══════════════════►│ + callbacks/handlers   │
 └────────────────────────────────────────────────┘  telemetry ↑       └───────────────────────┘
                                                      commands/results ↓
```

The Copilot CLI, MCP server, **and broker** all run on your **local** machine; only the training job is remote.
`agentic-optimizer-broker --tunnel` binds the broker to localhost and publishes it to the node through a
**Microsoft Dev Tunnel** (a public HTTPS URL) — no separately-hosted broker or third endpoint required. Prefer a
fixed, shared endpoint? You can still self-host the broker on any reachable host and point both sides at it. See
[`docs/architecture.md`](docs/architecture.md).

## Components

| Module | Role |
| --- | --- |
| `agentic_optimizer.controlplane` | FastAPI broker (`agentic-optimizer-broker`), in-memory or SQLite-backed, bearer-token protected; publishes itself to the node via Dev Tunnel (`--tunnel`). |
| `agentic_optimizer.bridge` | `TrainingBridge` — remote-side glue in the PyTorch loop (+ `attach`/`NoOpBridge` ergonomics). |
| `agentic_optimizer.mcp_server` | MCP **stdio** server exposing control tools to the local CLI. |
| `agentic_optimizer.contract` | Shared pydantic models for telemetry, commands, run IDs, and results. |
| `agentic_optimizer.safety` | Anomaly detection (NaN/Inf/grad-explosion) + hyperparameter guardrails. |
| `agentic_optimizer.profiling` | `StepProfiler` — per-section step-time breakdown for throughput RCA. |
| `agentic_optimizer.distributed` | DDP helpers: rank-0 telemetry, command broadcast/barrier. |
| `agentic_optimizer.integrations` | `LightningBridgeCallback` + `HFBridgeCallback` adapters. |
| `agentic_optimizer.tunnel` | Dev Tunnel wrapper that publishes the broker over a public HTTPS URL. |
| `agentic_optimizer.callback` / `driver` | Secondary legacy agent-on-node file-contract mode. |

## Install matrix

Base install is intentionally small: `pydantic` + `httpx`.

| Role | Install |
| --- | --- |
| Broker host | `pip install -e ".[broker]"` |
| Local CLI / MCP side | `pip install -e ".[mcp,hpo]"` |
| Training node | `pip install -e ".[torch,gpu,mlflow]"` |
| Lightning / HF node | add `".[lightning]"` or `".[hf]"` |
| Optional extras | `[hpo]` = Optuna, `[gpu]` = NVML GPU telemetry (`nvidia-ml-py`), `[mlflow]` = MLflow, `[lightning]` = PyTorch Lightning adapter, `[hf]` = Hugging Face Trainer adapter, `[all]`, `[dev]` |

The remote node does **not** need FastAPI, uvicorn, MCP, or the Copilot CLI.

## MCP tools the agent gets

Observe and control: `get_training_state`, `get_metrics`, `get_mlflow_info`, `list_knobs`, `list_runs`,
`select_run`, `set_hyperparameters`, `set_training_config`, `set_knob`, `invoke`, `interrogate`,
`invoke_and_wait`, `interrogate_and_wait`, `get_suspicious_samples`, `flag_samples`, `set_augmentation`,
`run_evaluation`, `pause_training`, `resume_training`, and `wait_for_result`.

Safe live control and lifecycle: `save_checkpoint`, `restore_checkpoint`, `list_checkpoints`, `set_guardrails`,
`get_profile`, `get_scheduler`, `set_scheduler`, `get_anomalies`, `stop_training`, and `extend_training`.

If installed with `[hpo]`, optional advisor tools are also available: `hpo_configure`, `hpo_suggest`,
`hpo_report`, `hpo_report_intermediate`, and `hpo_best`.

## Quick start

```bash
# 1) broker — runs locally and publishes itself via Dev Tunnel (needs the `devtunnel` CLI).
#    Prints a public HTTPS URL for the node to use as <broker-url> below. No 3rd endpoint to host.
pip install -e ".[broker]"
CONTROL_PLANE_TOKEN=<strong-token> agentic-optimizer-broker --tunnel
# Alternative — self-host a fixed, reachable broker instead of a tunnel:
#   CONTROL_PLANE_TOKEN=<strong-token> CONTROL_PLANE_HOST=0.0.0.0 agentic-optimizer-broker

# 2) remote training job (AML/GPU node), pointed at the tunnel URL
pip install -e ".[torch,gpu,mlflow]"
CONTROL_PLANE_URL=<broker-url> CONTROL_PLANE_TOKEN=<strong-token> \
  CONTROL_PLANE_RUN_ID=run-001 python examples/train_with_bridge.py --broker <broker-url>

# 3) local agent — Copilot CLI starts the MCP stdio server from agent/mcp-config.json
pip install -e ".[mcp,hpo]"
copilot --additional-mcp-config @agent/mcp-config.json
```

Set `CONTROL_PLANE_URL`, `CONTROL_PLANE_TOKEN`, and `CONTROL_PLANE_RUN_ID` (default `default`) in
`agent/mcp-config.json` or your local environment. Then ask Copilot things like: *“inspect GPU utilization and
throughput, tune batch size if safe, then check suspicious samples.”*

> On PowerShell, set env vars with `$env:CONTROL_PLANE_TOKEN = "<strong-token>"` rather than the `VAR=value`
> prefix shown above.

## Try the end-to-end demo (no GPU, no Copilot CLI)

```bash
pip install -e ".[dev,torch]"
python examples/live_demo.py
```

`live_demo.py` starts an in-process broker, launches a synthetic training run wired to the `TrainingBridge`, and
runs a scripted stand-in agent against it over **real HTTP** — proving live interjection without a GPU or the real
CLI. It changes the learning rate mid-run, interrogates per-class loss, pulls the worst per-sample losses and
flags them as label noise (then verifies they were genuinely corrupted), raises the batch size, and runs a short
Optuna HPO loop. It prints a pass/fail check matrix and exits non-zero on failure.

Or run the pieces separately:

```bash
python examples/run_broker.py --port 8765                                          # 1) broker
python examples/train_with_bridge.py --broker http://127.0.0.1:8765 --epochs 30    # 2) training + bridge
python examples/agent_sim.py --broker http://127.0.0.1:8765                         # 3) scripted agent
```

## Instrumenting your own loop

The lowest-friction path is `attach(optimizer, model)` plus `bridge.train_step(...)`. `attach` returns a live
bridge when `CONTROL_PLANE_URL` is set and an inert **`NoOpBridge`** otherwise — so the **same script** runs
unchanged (and still trains) with or without a broker, becoming agent-steerable only when one is present:

```python
from agentic_optimizer import attach

bridge = attach(optimizer, model)          # live if CONTROL_PLANE_URL is set, else a no-op stand-in
with bridge:                               # on_train_begin / on_train_end
    for epoch in range(epochs):
        if bridge.should_stop():           # agent can request a graceful stop
            break
        for x, y in loader:
            loss = loss_fn(model(x), y)
            bridge.train_step(loss, batch_size=len(x))   # backward + grad-clip + step + zero_grad + telemetry
        bridge.epoch_end(epoch, val_acc=acc)             # push metrics + apply queued agent commands
```

That is the whole integration — see [`examples/minimal_bridge.py`](examples/minimal_bridge.py). `bridge(loss,
batch_size=len(x))` is shorthand for `train_step`. For full control, construct the bridge directly and opt into the
safety, profiling, scheduler, and checkpoint surfaces:

```python
from agentic_optimizer.bridge import TrainingBridge
from agentic_optimizer.controlplane import ControlPlaneClient
from agentic_optimizer.telemetry import compute_grad_norm

client = ControlPlaneClient.from_url("https://<broker>", token="...")
bridge = TrainingBridge(
    optimizer,
    client,
    model=model,
    scheduler=scheduler,                       # exposed via get_scheduler / set_scheduler
    mlflow=True,
    poll_interval=1.0,                         # optional background poller for prompt command pickup
    auto_pause_on_anomaly=True,                # halt on NaN/Inf/grad-explosion
    guardrails={"bounds": {"lr": {"min": 1e-5, "max": 0.5}}, "max_rel_change": 10.0},
    checkpoint_dir="ckpts",                    # enables save_checkpoint / restore_checkpoint rollback
    on_scheduler_reconfig=lambda args: build_scheduler(optimizer, **args),
    on_training_config=rebuilt_loader_or_amp_callback,
    on_flagged_samples=downweight_or_filter_callback,
)

bridge.register("per_class_loss", lambda args, ctx: compute_per_class_loss(), safe_async=True)
bridge.register_knob("label_smoothing", set_label_smoothing, value=0.0)

bridge.on_train_begin()
for epoch in range(epochs):
    if bridge.should_stop():
        break
    for x, y in loader:
        with bridge.section("data"):           # step-time profiling sections (optional)
            ...
        with bridge.section("forward"):
            loss = loss_fn(model(x), y)
        with bridge.section("backward"):
            loss.backward()
        bridge.clip_gradients(model)
        optimizer.step(); optimizer.zero_grad()
        bridge.on_batch_end(loss.item(), batch_size=len(x), grad_norm=compute_grad_norm(model))
    bridge.scheduler_step()
    bridge.on_epoch_end(epoch, metrics={"val_acc": acc})
bridge.on_train_end()
```

`set_hyperparameters(lr, weight_decay, momentum, grad_clip)` changes optimizer settings (clamped by guardrails).
`set_training_config` can change `batch_size`, `num_workers`, `grad_accum_steps`, and `amp` through your callback.

### PyTorch Lightning and Hugging Face Trainer

Drop in a callback instead of editing the loop:

```python
# PyTorch Lightning
from agentic_optimizer.integrations import LightningBridgeCallback
trainer = pl.Trainer(callbacks=[LightningBridgeCallback.from_env()], ...)

# Hugging Face Trainer
from agentic_optimizer.integrations import HFBridgeCallback
trainer = Trainer(model=model, ..., callbacks=[HFBridgeCallback.from_env()])
```

Both wrap a `TrainingBridge`/`NoOpBridge` (via `from_env`) and stream telemetry + apply commands at the framework's
own step/epoch hooks. Install with `".[lightning]"` or `".[hf]"`.

## Live-control recipes

Patterns the agent (and `agent_sim.py`) follow — inspect first, change one thing, then verify it landed.

| Symptom | Read | Act | Verify |
| --- | --- | --- | --- |
| Loss diverging / grad-norm spikes | `get_training_state`, `get_anomalies` | `save_checkpoint` → `set_hyperparameters(lr=…/10, grad_clip=…)` | `get_metrics`; `restore_checkpoint` if worse |
| Plateau | `get_metrics` | modest LR / regularization change, or start an HPO loop | `get_metrics` |
| Low GPU util, memory to spare | `get_profile`, `gpu.util_pct`, throughput | `set_training_config(batch_size=…, num_workers=…, amp=true)` | `get_profile` / throughput in `get_metrics` |
| LR fighting the schedule | `get_scheduler` | `set_scheduler({...})` to reshape it mid-run | `get_scheduler`, `get_metrics` |
| Suspected label noise | `get_suspicious_samples`, `interrogate_and_wait("per_class_loss")` | `flag_samples([...])` → your `on_flagged_samples` down-weights/filters | re-read per-sample losses |
| Converged early / want more | `get_metrics` | `extend_training(max_epochs=…)` or `stop_training` to free the GPU | `get_training_state` |
| Need a principled search | — | `hpo_configure` → `hpo_suggest` → apply → `hpo_report` | `hpo_best` |

Mutations apply at the bridge's safe sync points; read-only interrogations registered `safe_async=True` can
return sooner. **Guardrails** clamp out-of-range hyperparameters, and **anomaly auto-pause** can halt the run on
NaN/Inf/grad-explosion. Use `wait_for_result(command_id)` to confirm a queued change was applied.

## What the agent sees

`get_training_state` returns the latest `Telemetry` for a run: `run_id`, `paused`, `last_error`, advertised
`knobs`, optional `mlflow` linkage, recorded `anomalies` and `checkpoints`, optional `distributed` (rank/world)
info, and a `state` snapshot with `step`/`epoch`/`max_epochs`, recent `loss_history`, `metrics`,
per-`param_groups` `lr`/`weight_decay`/`momentum`, `grad_norm`, `throughput_samples_per_s`, `gpu` (`device`,
`mem_used_mb`, `mem_total_mb`, `util_pct`), `per_sample_losses` (highest-loss samples for label-noise triage), a
`scheduler` snapshot (`get_scheduler`), and a step-time `profile` (`get_profile`). All schemas live in
`agentic_optimizer.contract`.

## Multiple runs on one broker

Every client sets `CONTROL_PLANE_RUN_ID` (default `default`); telemetry, command queues, knobs, and results are
namespaced by it, so many training jobs can share one broker. The agent calls `list_runs()` to discover runs and
`select_run(run_id)` to switch which run subsequent tools target.

## Auth & hosting

- **Broker auth** is a privileged bearer token (`CONTROL_PLANE_TOKEN`) that can mutate live training.
- Store the token in Key Vault/AML secrets or another secret manager and set it on the broker, node, and local CLI
  side. If you self-host a broker instead of tunnelling, put TLS (or an SSH tunnel) in front of any non-loopback
  bind.
- The broker uses constant-time token comparison, request-size limits (`CONTROL_PLANE_MAX_BODY_BYTES`), and
  **refuses to start** an unauthenticated control plane over a public `--tunnel` **or** a non-loopback bind unless
  `CONTROL_PLANE_INSECURE=1` (a loopback-only bind with no token is still allowed).
- Optional SQLite persistence: set `CONTROL_PLANE_PERSIST=<path.db>` on the broker.
- **Default — no third endpoint (Dev Tunnel):** run `agentic-optimizer-broker --tunnel` to bind the broker to
  localhost *and* publish a public HTTPS URL through Microsoft Dev Tunnels (requires the `devtunnel` CLI). The
  printed URL is what the remote node uses as `CONTROL_PLANE_URL` — no separately hosted broker required. Because a
  tunnel is public, the broker **requires** `CONTROL_PLANE_TOKEN` for `--tunnel` (override with
  `CONTROL_PLANE_INSECURE=1`, unsafe).

## Containers

See [`docker/`](docker/) for separate broker and training-node images. The old single “Copilot CLI on the node”
image is deprecated; the CLI now runs locally. [`examples/aml_job.yml`](examples/aml_job.yml) submits the bridge as
an Azure ML v2 command job pointed at a separately hosted broker.

## Development

```bash
pip install -e ".[dev]"      # dev tooling; note: intentionally does NOT pull torch
python -m pytest -q          # full suite
python -m pytest tests/test_integration.py::test_run_id_isolation   # a single test
python -m ruff check .       # lint (line length 100)
```

The test suite is **torch-free** so CI can run it without PyTorch; the few tests that need torch guard with
`pytest.importorskip("torch")`. CI (`.github/workflows/ci.yml`) runs ruff + pytest on Python 3.10 and 3.12.
Architecture and contributor notes: [`docs/architecture.md`](docs/architecture.md) and
[`.github/copilot-instructions.md`](.github/copilot-instructions.md).

### Project layout

```
src/agentic_optimizer/
  contract.py        shared pydantic models (single source of truth)
  controlplane.py    FastAPI broker + httpx ControlPlaneClient
  bridge.py          TrainingBridge + HandlerRegistry + attach/NoOpBridge (remote-loop glue)
  mcp_server.py      MCP stdio server: tools + standalone *_impl functions
  telemetry.py       grad-norm + GPU/NVML telemetry helpers
  safety.py          anomaly detection + hyperparameter guardrails
  profiling.py       StepProfiler step-time breakdown
  distributed.py     DDP rank-0 telemetry + command broadcast helpers
  tunnel.py          Dev Tunnel wrapper for the broker
  integrations/      Lightning + Hugging Face Trainer callbacks
  optuna_advisor.py  optional Optuna ask/tell advisor
  callback.py        secondary legacy file-contract mode
  driver.py          ↳ agent drivers for that legacy mode
examples/  run_broker · minimal_bridge · train_with_bridge · agent_sim · live_demo · cifar10_resnet · aml_job.yml
docs/ · agent/ · docker/ · auth/
```

## Examples

| File | What it is |
| --- | --- |
| [`examples/run_broker.py`](examples/run_broker.py) | Start a local broker for the demo (loopback); for a public tunnel use `agentic-optimizer-broker --tunnel`. |
| [`examples/minimal_bridge.py`](examples/minimal_bridge.py) | Smallest integration: `attach` + `train_step` in a vanilla loop (runs standalone as a no-op). |
| [`examples/train_with_bridge.py`](examples/train_with_bridge.py) | Remote-style PyTorch job exercising the full surface (scheduler, profiler, guardrails, checkpoints, label noise). |
| [`examples/agent_sim.py`](examples/agent_sim.py) | Scripted stand-in for the CLI agent; drives the same MCP tool impls. |
| [`examples/live_demo.py`](examples/live_demo.py) | One command: broker + training + agent, end to end over real HTTP. |
| [`examples/aml_job.yml`](examples/aml_job.yml) | Azure ML v2 command job that runs the bridge on a GPU node. |
| [`examples/cifar10_resnet.py`](examples/cifar10_resnet.py) | Secondary **legacy** file-contract demo (`state.json`/`control.json`). |

## Secondary mode: agent on the node (file contract)

`agentic_optimizer.callback.AgenticCallback` + `driver.CopilotOptimizerDriver` retain the earlier design where the
agent runs *on* the training node and exchanges `state.json` / `control.json` (see `cifar10_resnet.py`). Use it
only where no broker is reachable; the MCP control plane above is the primary path.
