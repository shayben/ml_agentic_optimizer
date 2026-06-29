# Copilot instructions for `agentic-optimizer`

A local **GitHub Copilot CLI** session drives one or more **remote PyTorch training runs** through a local
**MCP stdio server** and a reachable **HTTP broker**, to stream telemetry/MLflow and interject mid-run
(hyperparameters, throughput/hardware, interrogations, label-noise handling, optional HPO).

> This file is about working *on* this repository. `agent/AGENTS.md` is a different audience: it is the
> runtime playbook for the agent *operating a live run* via the MCP tools. Don't conflate the two.

## Build / test / lint

Windows + PowerShell dev box; Python ≥ 3.10 (CI runs 3.10 and 3.12).

```powershell
pip install -e ".[dev]"                 # dev tooling — NOTE: intentionally has NO torch
python -m pytest -q                      # full suite (keep it green)
python -m pytest tests/test_integration.py::test_run_id_isolation   # single test
python -m pytest tests/test_bridge.py -q                            # single file
python -m ruff check .                   # lint (line-length 100, target py310)
python examples/live_demo.py             # end-to-end smoke over real HTTP; exits 0 on success
```

CI (`.github/workflows/ci.yml`) installs `[dev]` then runs `ruff check .` and `pytest -q`.

## Architecture (three processes, one broker)

Read `docs/architecture.md` for the full picture. The shape:

- **LOCAL** — Copilot CLI launches `agentic_optimizer.mcp_server` as a **stdio** subprocess (config in
  `agent/mcp-config.json`). It is a thin broker client; it is *not* network-exposed and does *not* run on the node.
- **BROKER** — `agentic_optimizer.controlplane`, the only network-reachable component. FastAPI + bearer token,
  in-memory by default, optional SQLite (`CONTROL_PLANE_PERSIST`). Console script `agentic-optimizer-broker`.
- **NODE** — your PyTorch loop + `agentic_optimizer.bridge.TrainingBridge`, which pushes telemetry and applies
  agent commands at safe sync points.

`agentic_optimizer.contract` holds the shared pydantic models and is the **single source of truth** — change it
first, then the broker/bridge/MCP sides. Command lifecycle: agent enqueues via an MCP tool → broker per-`run_id`
queue → bridge claims with a lease (background poller or sync point) → runs handler → posts `CommandResult` →
`wait_for_result`. `callback.py`/`driver.py` are a **secondary** legacy file-contract mode (`state.json`/
`control.json`); the MCP path is primary.

## Project-specific conventions (the non-obvious ones)

- **Keep the training node lightweight.** It must run on only `pydantic`, `httpx`, `torch`. `controlplane.py`
  imports FastAPI/uvicorn/starlette **lazily inside functions** (`create_app`, `from_app`) so that
  `import agentic_optimizer.bridge` never pulls FastAPI. Optional deps (`torch`, `optuna`, `mlflow`, `pynvml`)
  are likewise imported lazily at point of use. Don't hoist these to module top level.
- **Tests must run without torch.** CI's `[dev]` has no torch, so guard any torch use with
  `pytest.importorskip("torch")` (see `tests/test_telemetry.py`). `telemetry.gpu_telemetry()` returns `None`
  (never raises) when torch/NVML are absent.
- **Dependencies are role-based extras**, not one blob: `[broker]`, `[mcp]`, `[hpo]` (Optuna), `[gpu]`
  (`nvidia-ml-py`, imported as the `pynvml` module), `[torch]`, `[mlflow]`, `[all]`, `[dev]`. Put new deps in the
  narrowest extra that needs them.
- **MCP tools come in pairs.** Each tool is a standalone `*_impl(client, ...)` function (unit-testable in-process)
  wrapped by a thin `@mcp.tool()` in `build_server`. Impls are decorated with `_safe_impl`, so they return
  `{"error": ..., "available": False}` instead of raising. Tests and `examples/agent_sim.py` drive the `*_impl`
  functions directly — add new tools the same way.
- **Bridge command safety classes.** `_is_safe_async` decides execution: `pause`/`resume`/`flag_samples` and any
  interrogation/action registered via `bridge.register(name, fn, safe_async=True)` run **immediately** in the
  background poller thread and **must be read-only**; everything that mutates the loop (`set_hyperparameters`,
  `set_training_config`, `set_knob`, `set_augmentation`, `run_evaluation`) is **deferred** to a training-thread
  sync point (`drain_commands` at `on_epoch_end`). `_poll_once()` is split out for deterministic testing.
- **Backward-compatible contract changes.** New fields/params default to preserve today's behavior
  (`run_id="default"`, new `on_batch_end(..., sample_indices=None, sample_losses=None)` kwargs optional). Existing
  loops and tests call hooks with old signatures.
- **`run_id` everywhere.** All client/broker/tool calls thread a `run_id` (default `"default"`); multiple runs
  share one broker (`list_runs` / `select_run`).
- **Bodyless 204s.** The broker returns bare `204` for "nothing here" (real uvicorn/h11 drops the connection if a
  204 carries a body); clients treat `204` as `None`. Don't add a body to a 204.
- **In-process tests use `ControlPlaneClient.from_app(app)`** (starlette `TestClient`), because
  `httpx.ASGITransport` is async-only. Prefer this over real sockets in tests.
- **`__init__.py` lazy-exports** `TrainingBridge`, `HandlerRegistry`, `OptunaAdvisor`, `optuna_available`,
  `AgenticCallback` via `__getattr__` to keep the top-level import torch/fastapi-free.

## Env vars

- Local + node clients: `CONTROL_PLANE_URL`, `CONTROL_PLANE_TOKEN`, `CONTROL_PLANE_RUN_ID`.
- Broker: `CONTROL_PLANE_HOST`, `CONTROL_PLANE_PORT`, `CONTROL_PLANE_TOKEN`, `CONTROL_PLANE_PERSIST`,
  `CONTROL_PLANE_MAX_BODY_BYTES`, `CONTROL_PLANE_INSECURE=1` (allow non-loopback without a token — unsafe).

## Where things live

`src/agentic_optimizer/`: `contract.py`, `controlplane.py` (broker + httpx client), `bridge.py`
(`TrainingBridge` + `HandlerRegistry`), `mcp_server.py` (FastMCP tools + impls), `telemetry.py`,
`optuna_advisor.py`, `callback.py`/`driver.py` (legacy). Runnable `examples/` (`run_broker.py`,
`train_with_bridge.py`, `agent_sim.py`, `live_demo.py`). Docs in `README.md`, `docs/`, `agent/`, `docker/`,
`auth/`.
