# Copilot instructions for `agentic-optimizer`

A local **GitHub Copilot CLI** session drives one or more **remote PyTorch training runs** through a local
**MCP stdio server** and a reachable **HTTP broker**, to stream telemetry/MLflow and interject mid-run
(hyperparameters, throughput/hardware, interrogations, label-noise handling, checkpoint/rollback, guardrails,
profiler RCA, scheduler control, run lifecycle, DDP, optional HPO). It runs on a raw loop, **Lightning**, or
**HF `Trainer`**, and `attach(optimizer, model)` makes the *same* script run unchanged with or without a broker.

> This file is about working *on* this repository. `agent/AGENTS.md` is a different audience: it is the
> runtime playbook for the agent *operating a live run* via the MCP tools. Don't conflate the two.

## Build / test / lint

Windows + PowerShell dev box; Python ≥ 3.10 (CI runs 3.10 and 3.12).

```powershell
pip install -e ".[dev]"                 # dev tooling — NOTE: intentionally has NO torch
python -m pytest -q                      # full suite (keep it green)
python -m pytest tests/test_integration.py::test_run_id_isolation   # single test
python -m pytest tests/test_bridge_control.py -q                    # single file (live-control suite)
python -m ruff check .                   # lint (line-length 100, target py310)
python examples/minimal_bridge.py        # zero-config smoke (runs as NoOpBridge, no broker needed)
python examples/live_demo.py             # end-to-end smoke over real HTTP; exits 0 on success
```

CI (`.github/workflows/ci.yml`) installs `[dev]` then runs `ruff check .` and `pytest -q`. `[dev]` deliberately
omits `torch`, `pytorch-lightning`, and `transformers`, so torch/Lightning tests self-skip in CI.

## Architecture (three processes, one broker)

Read `docs/architecture.md` for the full picture. The shape:

- **LOCAL** — Copilot CLI launches `agentic_optimizer.mcp_server` as a **stdio** subprocess (config in
  `agent/mcp-config.json`). It is a thin broker client; it is *not* network-exposed and does *not* run on the node.
- **BROKER** — `agentic_optimizer.controlplane`, the only network-reachable component. FastAPI + bearer token,
  in-memory by default, optional SQLite (`CONTROL_PLANE_PERSIST`). Console script `agentic-optimizer-broker`.
- **NODE** — your PyTorch loop + `agentic_optimizer.bridge.TrainingBridge`, which pushes telemetry and applies
  agent commands at safe sync points. **Never-idle invariant:** the bridge must *never block or pause the training
  loop*. Telemetry pushes are fire-and-forget and command draining is non-blocking, so the agent always reads
  slightly stale state and its influence lands **with delay** at the next sync point. Do not add any command,
  handler, or wait that idles the loop waiting on the agent; `stop_training` (a polled flag) is the only allowed
  loop-halting control, and it terminates rather than idles.

`agentic_optimizer.contract` holds the shared pydantic models and is the **single source of truth** — change it
first, then the broker/bridge/MCP sides. Command lifecycle: agent enqueues via an MCP tool → broker per-`run_id`
queue → bridge claims with a lease (background poller or sync point) → runs handler → posts `CommandResult` →
`wait_for_result`. `callback.py`/`driver.py` are a **secondary** legacy file-contract mode (`state.json`/
`control.json`); the MCP path is primary.

The bridge composes four newer, file-disjoint helper modules: `safety.py` (anomaly detection + guardrails),
`profiling.py` (`StepProfiler` step-time breakdown), `distributed.py` (DDP rank/broadcast helpers), and
`tunnel.py` (publishes the broker over Dev Tunnels via `agentic-optimizer-broker --tunnel`). `integrations/`
adapts the bridge to Lightning/HF as callbacks. `attach()` / `NoOpBridge` (in `bridge.py`) are the low-friction
entry point — see the conventions below.

## Project-specific conventions (the non-obvious ones)

- **Keep the training node lightweight.** It must run on only `pydantic`, `httpx`, `torch`. `controlplane.py`
  imports FastAPI/uvicorn/starlette **lazily inside functions** (`create_app`, `from_app`) so that
  `import agentic_optimizer.bridge` never pulls FastAPI. Optional deps (`torch`, `optuna`, `mlflow`, `pynvml`)
  are likewise imported lazily at point of use. Don't hoist these to module top level.
- **Tests must run without torch.** CI's `[dev]` has no torch, so guard any torch use with
  `pytest.importorskip("torch")` (see `tests/test_telemetry.py`). `telemetry.gpu_telemetry()` returns `None`
  (never raises) when torch/NVML are absent.
- **Dependencies are role-based extras**, not one blob: `[broker]`, `[mcp]`, `[hpo]` (Optuna), `[gpu]`
  (`nvidia-ml-py`, imported as the `pynvml` module), `[torch]`, `[mlflow]`, `[lightning]`, `[hf]`, `[all]`, `[dev]`.
  Put new deps in the narrowest extra that needs them. `[all]` and `[dev]` deliberately exclude the heavy
  `torch`/`lightning`/`hf` sets.
- **MCP tools come in pairs.** Each tool is a standalone `*_impl(client, ...)` function (unit-testable in-process)
  wrapped by a thin `@mcp.tool()` in `build_server`. Impls are decorated with `_safe_impl`, so they return
  `{"error": ..., "available": False}` instead of raising. Tests and `examples/agent_sim.py` drive the `*_impl`
  functions directly — add new tools the same way.
- **Bridge command safety classes.** `_is_safe_async` decides execution: `flag_samples`/
  `stop_training`/`extend_training`/`set_guardrails` and any interrogation/action registered via
  `bridge.register(name, fn, safe_async=True)` run **immediately** in the background poller thread and **must be
  read-only / non-torch-mutating**; everything that mutates the loop or torch state (`set_hyperparameters`,
  `set_training_config`, `set_knob`, `set_augmentation`, `set_scheduler`, `save_checkpoint`, `restore_checkpoint`,
  `run_evaluation`) is **deferred** to a training-thread sync point (`drain_commands` at `on_epoch_end`). When you
  add a command, classify it here; anything touching tensors/optimizer/model must be deferred. A `safe_async`
  handler may update plain Python bookkeeping immediately, but if it must fire a caller-supplied callback that
  touches tensors (e.g. `flag_samples` updates `flagged_indices` inline yet queues `on_flagged_samples`), it
  enqueues the work and lets `drain_commands` run it on the training thread via `_flush_flag_callbacks` — never
  call such a callback inline on the poller thread. `_poll_once()` is split out for deterministic testing.
- **`attach()` returns a `NoOpBridge` off the control plane — and it must still TRAIN.** `attach(optimizer, model)`
  (== `TrainingBridge.from_env`) returns a live bridge when `CONTROL_PLANE_URL` is set and a `NoOpBridge`
  otherwise, so the *same* script runs unchanged. `NoOpBridge` is **not fully inert**: `train_step`/`__call__`
  still run `backward`→clip→`optimizer.step()`→`zero_grad()` and `scheduler_step` still steps the scheduler — only
  the control-plane I/O (telemetry/commands/checkpoints) is a no-op. If you add loop-driving behavior to
  `TrainingBridge.train_step`/`scheduler_step`, mirror it in `NoOpBridge` or models silently stop training
  off-plane.
- **`bridge.step` is an int, not a method.** `self.step` is the step counter (read in telemetry/MLflow). The
  one-call ergonomic step is `bridge.train_step(loss, batch_size=n)` (or `bridge(loss, batch_size=n)` via
  `__call__`). Never call `bridge.step(...)`.
- **Non-finite floats must be sanitized before telemetry.** `client.push_telemetry` JSON-encodes the payload and
  **raises** on `inf`/`nan` (and `push_telemetry` then swallows it, so the push is silently lost). Coerce
  non-finite `loss`/`grad_norm`/`metrics`/`per_sample_losses`/anomaly values to `None` (or drop them) before they
  enter `TrainingState` — guard with `math.isfinite`. This matters most during divergence, exactly when the agent
  needs to see the anomaly.
- **Distributed paths are gated on `dist.is_available()`.** The bridge references the module-global `dist`
  (`from . import distributed as dist`); every collective is guarded so single-process and CI behave identically.
  Under DDP, only rank 0 pushes telemetry and drains commands, then broadcasts the processed commands so all ranks
  apply the same mutation at the same step (`_NON_REPLICATED` is excluded). Keep the per-`drain_commands` collective
  count identical across ranks — an unmatched broadcast hangs the process group.
- **In-memory checkpoints must clone tensors.** `torch` `state_dict()` returns tensors that **alias** live
  params, so an in-memory snapshot must `.detach().clone()` (recursively for optimizer state) or
  `restore_checkpoint` becomes a silent no-op once `optimizer.step()` mutates them. The `checkpoint_dir`
  (`torch.save`) path is already safe. Re-saving an existing checkpoint id **pops-then-reinserts** it so it moves
  to newest — restore-latest (no id) and LRU eviction both key off `dict` insertion order.
- **Backward-compatible contract changes.** New fields/params default to preserve today's behavior
  (`run_id="default"`, new `on_batch_end(..., sample_indices=None, sample_losses=None)` kwargs optional). Existing
  loops and tests call hooks with old signatures.
- **`run_id` everywhere.** All client/broker/tool calls thread a `run_id` (default `"default"`); multiple runs
  share one broker (`list_runs` / `select_run`).
- **Bodyless 204s.** The broker returns bare `204` for "nothing here" (real uvicorn/h11 drops the connection if a
  204 carries a body); clients treat `204` as `None`. Don't add a body to a 204.
- **In-process tests use `ControlPlaneClient.from_app(app)`** (starlette `TestClient`), because
  `httpx.ASGITransport` is async-only. Prefer this over real sockets in tests.
- **`__init__.py` lazy-exports** `TrainingBridge`, `NoOpBridge`, `attach`, `HandlerRegistry`, `OptunaAdvisor`,
  `optuna_available`, `AgenticCallback` via `__getattr__` to keep the top-level import torch/fastapi-free. The new
  contract models (`SchedulerState`, `ProfileSummary`, `CheckpointInfo`, `AnomalyEvent`, `GuardrailConfig`, …) are
  eagerly re-exported (pure-pydantic, no heavy deps).

## Env vars

- Local + node clients: `CONTROL_PLANE_URL`, `CONTROL_PLANE_TOKEN`, `CONTROL_PLANE_RUN_ID`,
  `CONTROL_PLANE_TUNNEL_ACCESS_TOKEN` (Dev Tunnels *connect* token for a non-anonymous tunnel; forwarded as the
  `X-Tunnel-Authorization: tunnel <token>` header by `ControlPlaneClient.from_env`/`from_url`, independent of the
  bearer token).
- Broker: `CONTROL_PLANE_HOST`, `CONTROL_PLANE_PORT`, `CONTROL_PLANE_TOKEN`, `CONTROL_PLANE_PERSIST`,
  `CONTROL_PLANE_MAX_BODY_BYTES`, `CONTROL_PLANE_INSECURE=1`, `CONTROL_PLANE_TUNNEL_ID` (persistent named Dev
  Tunnel → stable public URL for `--tunnel`, so the node/MCP config stays static across restarts),
  `CONTROL_PLANE_TUNNEL_LOGIN` (non-interactive Dev Tunnels host login for node-hosted mode; anonymous access is
  client-only so a headless host must authenticate), `CONTROL_PLANE_TUNNEL_URL_FILE` (write the discovered
  public URL for cross-machine discovery), `CONTROL_PLANE_TUNNEL_ANONYMOUS=0` / `--no-tunnel-anonymous` (make the
  relay reject unauthenticated clients; then no bearer token is forced) and `CONTROL_PLANE_TUNNEL_TOKEN_FILE`
  (node-hosted: mint a connect token via `tunnel.issue_connect_token` and write it beside the URL file — needs a
  non-anonymous named tunnel; connect tokens expire ~24h). Broker startup (`controlplane._check_exposure`)
  **refuses** to serve an unauthenticated control plane over an *anonymous* public `--tunnel` **or** a non-loopback
  bind (anyone reachable could drive the run); set `CONTROL_PLANE_TOKEN`, host a non-anonymous tunnel, or
  `CONTROL_PLANE_INSECURE=1` to override (unsafe). A loopback bind with no token is allowed.

## Where things live

`src/agentic_optimizer/`: `contract.py`, `controlplane.py` (broker + httpx client + `--tunnel`), `bridge.py`
(`TrainingBridge` + `HandlerRegistry` + `attach`/`NoOpBridge`), `mcp_server.py` (FastMCP tools + impls),
`telemetry.py`, `safety.py` (anomaly + guardrails), `profiling.py` (`StepProfiler`), `distributed.py` (DDP),
`tunnel.py` (Dev Tunnel), `integrations/` (`lightning.py`, `hf.py` callbacks), `optuna_advisor.py`,
`callback.py`/`driver.py` (legacy). Runnable `examples/` (`run_broker.py`, `minimal_bridge.py` = 3-line `attach`
demo, `train_with_bridge.py`, `agent_sim.py`, `live_demo.py`; `cifar10_resnet.py` = the legacy
`AgenticCallback`/driver `state.json`/`control.json` demo, runs offline with `--fake-data`; `aml_job.yml` = an
Azure ML job spec [agent-hosted broker], `selfhost_and_train.py` + `aml_job_selfhost.yml` = node-hosted
"submit-then-attach" broker+tunnel on the training node). Docs in `README.md`, `docs/`, `agent/`, `docker/`,
`auth/`.
