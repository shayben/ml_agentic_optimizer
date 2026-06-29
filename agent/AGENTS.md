# Agent Instructions — Live Training Control via MCP

You are a local GitHub Copilot CLI agent controlling a remote PyTorch run through the `training-control` MCP
stdio server. The CLI and MCP server run locally; the training job runs remotely; both talk to the same broker.
Select the right run with `select_run(run_id)` when needed.

## Tools

Observe:
- `get_training_state()` — latest step/epoch, losses, optimizer settings, grad norm, throughput, GPU memory/util,
  pause state, `last_error`, and MLflow linkage when present.
- `get_metrics(limit)` — recent metric history.
- `get_mlflow_info()` — MLflow run ID, tracking URI, experiment, and latest metrics.
- `list_runs()` / `select_run(run_id)` — inspect and switch broker run namespaces.
- `list_knobs()` — custom job-advertised knobs.
- `get_suspicious_samples(limit)` — highest per-sample losses reported by the bridge.

Interject:
- `set_hyperparameters(lr, weight_decay, momentum, grad_clip)` — apply non-null optimizer controls.
- `set_training_config(batch_size, num_workers, grad_accum_steps, amp)` — ask the bridge callback to rebuild the
  DataLoader, adjust accumulation, or toggle AMP.
- `set_knob(name, value)` — set an advertised custom knob.
- `invoke(action, args)` / `interrogate(name, args)` — queue custom loop actions or data/model queries.
- `invoke_and_wait(...)` / `interrogate_and_wait(...)` — one-call queue + result collection.
- `flag_samples(indices)` — drive the job's `on_flagged_samples` callback (for example down-weight noisy labels).
- `run_evaluation(args)`, `pause_training()`, `resume_training()`, `wait_for_result(command_id)`.

Optional HPO tools (only with `[hpo]`): `hpo_configure`, `hpo_suggest`, `hpo_report`,
`hpo_report_intermediate`, `hpo_best`.

## Recommended workflow

1. **Inspect first.** Call `get_training_state`, `get_metrics`, and, for GPU runs, check `gpu.util_pct`, memory,
   throughput, and `last_error`.
2. **Diagnose.** Identify divergence, plateau, overfitting, low GPU utilization, input bottlenecks, or label noise.
3. **Act conservatively.**
   - Divergence / high grad norm → lower `lr`, add `grad_clip`.
   - Plateau → modest LR/regularization change or start an HPO loop.
   - Low GPU util with memory headroom → `set_training_config(batch_size=..., num_workers=..., amp=...)`.
   - Custom training behavior → use `list_knobs`, then `set_knob`, `invoke`, or `interrogate`.
4. **Investigate data quality.** Use `interrogate_and_wait("per_class_loss", ...)` or related handlers, then
   `get_suspicious_samples(limit)` and `flag_samples(indices)` when evidence supports label noise.
5. **HPO loop when useful.** `hpo_suggest` → apply with `set_hyperparameters` / `set_training_config` → observe
   objective via `get_metrics` → `hpo_report` → repeat. Add root-cause reasoning; do not blindly chase trials.
6. **Verify.** Use `wait_for_result` for queued mutations, then re-read telemetry/metrics to confirm the change
   landed and helped.

## Safety

- Apply one meaningful change at a time and observe its effect.
- Mutations apply at bridge safe sync points; read-only interrogations may return sooner.
- Do not invent custom knob/action names. Use `list_knobs` and documented registered handlers.
- `pause_training` has a bridge-side safety timeout; resume promptly.
- Treat broker access as privileged because the token can mutate live training.

## Secondary legacy mode

If you are deliberately run on the training node with a `state.json` and no MCP server, use the file-contract mode
(`control.json` per `src/agentic_optimizer/contract.py::ControlSignal`). The MCP broker flow is primary.
