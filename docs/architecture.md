# Architecture

## Why this shape

The agent (GitHub Copilot CLI) is **local** and interactive; training is **remote** and long-lived. Rather than
ship the agent onto the GPU node, a local MCP **stdio** server talks to a reachable broker, and the remote
`TrainingBridge` talks to the same broker. This keeps GitHub/Copilot auth and human interaction local while GPU
work stays remote.

```
 Local dev box                         Reachable control plane                    Remote node (AML/GPU)
 ┌────────────────────┐ MCP stdio       ┌────────────────────────┐ HTTP/REST       ┌──────────────────────┐
 │ GitHub Copilot CLI │◄───────────────►│ broker (FastAPI)       │◄───────────────►│ PyTorch training loop │
 │ + mcp-config       │ mcp_server      │ runs/telemetry/queues  │ telemetry ↑     │ + TrainingBridge      │
 └────────────────────┘                 │ bearer-token auth      │ commands  ↓     │ + HandlerRegistry     │
                                        └────────────────────────┘ results   ↑     └──────────────────────┘
```

## Three processes

1. **LOCAL: MCP server** (`python -m agentic_optimizer.mcp_server`) — launched by GitHub Copilot CLI as a
   **stdio** subprocess using `agent/mcp-config.json`. It is a thin broker client configured with
   `CONTROL_PLANE_URL`, `CONTROL_PLANE_TOKEN`, and `CONTROL_PLANE_RUN_ID` (default `default`). It is not network
   exposed and does not run on the training node.
2. **BROKER: control plane** (`agentic-optimizer-broker` / `agentic_optimizer.controlplane`) — the only
   network-reachable component. It is FastAPI + bearer-token auth, in-memory by default, optionally SQLite-backed
   via `CONTROL_PLANE_PERSIST=<path.db>`. Runs are namespaced by `run_id`; `GET /runs` lists them.
3. **NODE: training job** — PyTorch loop + `TrainingBridge`. The node needs only `pydantic`, `httpx`, `torch`, and
   optional telemetry packages (`pynvml`, `mlflow`), not FastAPI/uvicorn/MCP/Copilot CLI.

## Command lifecycle and reliability

```
agent (MCP tool)         broker                              bridge (training loop)
   │ enqueue command  ─────► pending queue scoped by run_id
   │ ◄── command_id
   │                       GET /commands/next  ◄────────────── poller or safe sync point
   │                       claim lease          ──────────────► run handler / mutate safely
   │                       POST result         ◄────────────── CommandResult{ok,data,error}
   │ wait_for_result ─────► return result
```

- Read-only interrogations can run immediately; mutations apply at safe sync points (epoch/batch boundaries).
- An optional bridge background poller (`poll_interval`) improves prompt command pickup without mutating mid-step.
- Broker command leases/redelivery prevent a crashed bridge from stranding commands.
- Structured logging and `last_error` in telemetry help diagnose handler or transport failures.
- `pause_training` / `resume_training` pause at the bridge with a max-pause safety timeout.

## Influence points

Built-ins include:

- `set_hyperparameters(lr, weight_decay, momentum, grad_clip)` for optimizer controls, clamped by guardrails.
- `set_training_config(batch_size, num_workers, grad_accum_steps, amp)` for throughput/hardware levers. The bridge
  invokes a user callback to rebuild the DataLoader, change gradient accumulation, or toggle AMP.
- `save_checkpoint` / `restore_checkpoint` / `list_checkpoints` for checkpoint-and-rollback of a bad change.
- `set_guardrails` to set per-knob min/max bounds and a max relative-change limit at runtime.
- `get_profile` for a step-time breakdown; `get_scheduler` / `set_scheduler` to observe/reshape the LR schedule.
- `get_anomalies` to read recorded NaN/Inf/grad-explosion events; `stop_training` / `extend_training` for lifecycle.
- `run_evaluation`, `pause_training`, `resume_training`, `get_training_state`, `get_metrics`, `get_mlflow_info`,
  and `wait_for_result`.

The `HandlerRegistry` exposes custom loop logic:

- `invoke` / `interrogate` queue custom actions and data/model queries.
- `invoke_and_wait` / `interrogate_and_wait` enqueue and collect results in one call.
- `set_knob` / `list_knobs` expose named custom knobs.
- `flag_samples(indices)` calls `on_flagged_samples` so user code can down-weight/filter likely noisy labels.
- `get_suspicious_samples(limit)` returns the highest per-sample losses reported by the bridge.

## Safe live control

Mutating a live run is risky, so the bridge de-risks the agent's actions:

- **Checkpoint + rollback** — `save_checkpoint` snapshots model/optimizer/scheduler/scaler/RNG (in-memory, plus
  `torch.save` when `checkpoint_dir` is set). `restore_checkpoint` rolls the live run back if a change made things
  worse. The bridge keeps the last `max_checkpoints` and evicts in insertion order.
- **Guardrails** — `_apply_guardrails` clamps requested `lr`/`weight_decay`/`momentum`/`grad_clip` to configured
  bounds and a max relative-change limit *before* applying, returning notes so the agent sees what was clamped.
- **Anomaly auto-pause** — an `AnomalyDetector` watches each `on_batch_end` for NaN/Inf loss/grad and grad
  explosion. Events are recorded (`get_anomalies`) and, when `auto_pause_on_anomaly=True`, pause the run. Non-finite
  values are never serialized into telemetry (they are recorded as `null`) so JSON push cannot silently fail.

## Profiling, scheduler, lifecycle, and distributed

- **Profiling** — `StepProfiler` records per-section timings via `with bridge.section("data"|"forward"|"backward")`
  and `mark_step()`; `get_profile` returns the average step time, per-section percentages, and throughput
  suggestions for root-cause analysis of slow training.
- **Scheduler** — pass `scheduler=` so the bridge advances it via `scheduler_step()` and reports `SchedulerState`
  (`get_scheduler`). `set_scheduler(config)` calls the user `on_scheduler_reconfig` hook to rebuild the schedule;
  without that hook the command fails cleanly.
- **Run lifecycle** — `stop_training` sets a flag the loop polls via `should_stop()` (free the GPU gracefully);
  `extend_training(max_epochs)` raises the budget for a promising run.
- **Distributed (DDP)** — when `torch.distributed` is initialized, only rank 0 pushes telemetry and drains the
  command queue; it broadcasts processed commands so every rank applies the same mutation at the same step.
  Read-only/local commands are excluded from broadcast. All distributed paths are gated on `dist.is_available()`,
  so single-process and CI behavior is unchanged.

## Ergonomics and framework adapters

- `attach(optimizer, model)` (alias of `TrainingBridge.from_env`) returns a live bridge when `CONTROL_PLANE_URL`
  is set and an inert **`NoOpBridge`** otherwise. The `NoOpBridge` still performs real optimization
  (`train_step` → backward/step/zero_grad, `scheduler_step`), so the *same* script trains unchanged off the control
  plane and becomes steerable when a broker is present.
- `bridge.train_step(loss, batch_size=n)` (or `bridge(loss, batch_size=n)`) does backward → grad-clip →
  `optimizer.step()` → `zero_grad()` → telemetry in one call; `bridge.epoch_end(...)` and `bridge.should_stop()`
  complete the loop. The bridge is also a context manager (`with attach(...) as bridge:`).
- `integrations.LightningBridgeCallback` and `integrations.HFBridgeCallback` wrap a bridge and stream telemetry +
  apply commands at the framework's own hooks; both import without the framework installed and build via
  `from_env()`.

## Telemetry, MLflow, and multi-run

Telemetry includes step/epoch, recent losses, optimizer param groups, gradient norm, throughput, GPU memory,
optional GPU utilization (`gpu.util_pct` via `pynvml`), `last_error`, pause state, metrics, and optional MLflow
run info. With `mlflow=True`, the bridge logs pushed metrics and surfaces run metadata to the agent.

Multiple runs can share one broker. Each client sets `CONTROL_PLANE_RUN_ID`; the agent can call `list_runs` and
`select_run(run_id)` to switch namespaces.

## Optional HPO advisor

When the local side is installed with `[hpo]`, the agent can combine Optuna TPE suggestions with root-cause
reasoning:

1. `hpo_configure(param_space, direction)`
2. `hpo_suggest()`
3. `set_hyperparameters(...)` and/or `set_training_config(...)`
4. observe objective metrics with `get_metrics`
5. `hpo_report(trial_id, value, step)` or `hpo_report_intermediate(...)`
6. repeat, then inspect `hpo_best()`

## Transport & security

Plain HTTP is easy to tunnel, but broker access is privileged: the bearer token can mutate a live training
process. Require TLS or an SSH tunnel for any non-loopback broker and store `CONTROL_PLANE_TOKEN` in Key Vault,
AML secrets, or an equivalent secret manager. The broker uses constant-time token comparison, request-size limits
(`CONTROL_PLANE_MAX_BODY_BYTES`), and refuses to bind a non-loopback host without a token unless
`CONTROL_PLANE_INSECURE=1`.

To avoid hosting a separate reachable endpoint, `agentic-optimizer-broker --tunnel` binds the broker to localhost
and publishes a public HTTPS URL through Microsoft Dev Tunnels (`agentic_optimizer.tunnel`, requires the
`devtunnel` CLI). The printed URL becomes the node's `CONTROL_PLANE_URL`; always pair it with a strong token since
the tunnel is internet-reachable.

The broker stores telemetry and small command/result payloads, not model weights. Custom interrogation handlers
decide what data/model details are exposed; keep results free of sensitive raw data.

## Secondary mode: agent-on-node file contract

`agentic_optimizer.callback.AgenticCallback` + `driver.CopilotOptimizerDriver` retain the earlier `state.json` /
`control.json` flow for environments with no reachable broker. The MCP control plane above is the primary path.
