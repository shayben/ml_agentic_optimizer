"""A remote-style PyTorch training job wired to the control plane via :class:`TrainingBridge`.

This is what runs on the AML node. It trains a tiny MLP on synthetic data (fast, no torchvision) and exposes
the live run to a local agent: it pushes telemetry every epoch, applies queued commands at epoch boundaries,
and registers a few **custom** influence points so the agent can interrogate and steer the run.

The synthetic dataset carries injected **label noise** so the end-to-end label-noise workflow is real:

* per-batch **per-sample losses** are streamed to the bridge so the agent can pull the worst offenders via
  ``get_suspicious_samples``;
* when the agent ``flag_samples`` them, ``on_flagged_samples`` zeroes their training weight (a robustness
  lever that actually changes the loss);
* ``set_training_config`` rebuilds the batch size live (a throughput / hardware-utilization lever).

Registered influence points:

* ``evaluate``           — run an eval pass (used by the ``run_evaluation`` built-in).
* ``per_class_loss``     — a read-only interrogation returning mean loss per class (``safe_async``).
* ``noise_report``       — a read-only RCA interrogation: how many flagged samples were truly noisy.
* knob ``label_smoothing`` — a custom loop knob the agent can set live.

Run standalone against a broker:

    python examples/train_with_bridge.py --broker http://127.0.0.1:8765 --epochs 20
"""
from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from agentic_optimizer.bridge import TrainingBridge
from agentic_optimizer.controlplane import ControlPlaneClient
from agentic_optimizer.telemetry import compute_grad_norm


def make_data(n=1024, dim=20, classes=4, seed=0, noise_frac=0.0):
    """Gaussian-blob classification data. ``noise_frac`` flips that fraction of labels (label noise).

    Returns ``(x, y, centers, noisy)`` where ``noisy`` is a bool mask of the corrupted samples.
    """
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(classes, dim, generator=g) * 2.5
    y = torch.randint(0, classes, (n,), generator=g)
    x = centers[y] + torch.randn(n, dim, generator=g)
    noisy = torch.zeros(n, dtype=torch.bool)
    if noise_frac > 0:
        k = int(n * noise_frac)
        idx = torch.randperm(n, generator=g)[:k]
        # flip to a different class so the corrupted samples become hard (high-loss) outliers
        y[idx] = (y[idx] + torch.randint(1, classes, (k,), generator=g)) % classes
        noisy[idx] = True
    return x, y, centers, noisy


class MLP(nn.Module):
    def __init__(self, dim=20, classes=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, classes))

    def forward(self, x):
        return self.net(x)


def run_training(
    client: ControlPlaneClient,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 0.2,
    epoch_pause_s: float = 0.25,
    mlflow: bool = False,
    verbose: bool = True,
    run_id: str = "default",
    poll_interval: float = 0.0,
    noise_frac: float = 0.08,
    checkpoint_dir: str | None = None,
) -> TrainingBridge:
    """Train the MLP, bridging the live run to ``client``. Returns the bridge (for inspection)."""
    torch.manual_seed(0)
    x, y, _, noisy = make_data(noise_frac=noise_frac)
    n, classes = x.size(0), 4
    xv, yv, _, _ = make_data(n=256, seed=1)

    model = MLP(classes=classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # A real LR scheduler so the agent can observe scheduler state (get_scheduler) and reconfigure it
    # (set_scheduler). step_size is large enough not to fire during the short demo, so it never
    # clobbers the agent's manual LR changes.
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    state = {"label_smoothing": 0.0}

    def on_scheduler_reconfig(args):
        step_size = int(args.get("step_size", scheduler.step_size))
        gamma = float(args.get("gamma", scheduler.gamma))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Per-sample weights: flagging a sample as label noise zeroes its weight, so it stops driving the
    # loss. This is the robustness lever the agent reaches via flag_samples -> on_flagged_samples.
    sample_weights = torch.ones(n)

    # Live training-loop config the agent can change via set_training_config. ``batch_size`` is read at
    # the top of every epoch, so changing it rebuilds the (synthetic) batching on the fly.
    cfg = {"batch_size": int(batch_size), "num_workers": 0}

    def on_training_config(tc):
        if tc.batch_size is not None:
            cfg["batch_size"] = max(1, int(tc.batch_size))
        if tc.num_workers is not None:
            cfg["num_workers"] = max(0, int(tc.num_workers))

    def on_flagged_samples(indices):
        valid = [i for i in indices if 0 <= i < n]
        if valid:
            sample_weights[valid] = 0.0

    bridge = TrainingBridge(
        optimizer,
        client,
        model=model,
        max_epochs=epochs,
        mlflow=mlflow,
        run_id=run_id,
        poll_interval=poll_interval,
        on_training_config=on_training_config,
        on_flagged_samples=on_flagged_samples,
        scheduler=scheduler,
        on_scheduler_reconfig=on_scheduler_reconfig,
        # Guardrails bound any agent LR change to a safe range (out-of-range requests are clamped).
        guardrails={"bounds": {"lr": {"min": 1e-5, "max": 0.5}}},
        # Label noise makes losses spiky; record anomalies but don't auto-pause this scripted demo.
        auto_pause_on_anomaly=False,
        checkpoint_dir=checkpoint_dir,
    )

    # --- custom influence points the agent can reach ---
    # These are read-only (no_grad, no module-state mutation), so they are safe to answer from the
    # bridge's background poller thread (safe_async=True) for low-latency interrogation.
    def evaluate(args, ctx):
        with torch.no_grad():
            acc = (model(xv).argmax(1) == yv).float().mean().item()
        return {"val_acc": round(acc, 4)}

    def per_class_loss(args, ctx):
        out = {}
        with torch.no_grad():
            logits = model(xv)
            for c in range(classes):
                mask = yv == c
                if mask.any():
                    out[str(c)] = round(F.cross_entropy(logits[mask], yv[mask]).item(), 4)
        return out

    def noise_report(args, ctx):
        """RCA helper: of the samples the agent flagged, how many were genuinely corrupted."""
        flagged = sorted(bridge.flagged_indices)
        true_pos = int(noisy[flagged].sum().item()) if flagged else 0
        return {
            "flagged": len(flagged),
            "true_noisy_flagged": true_pos,
            "total_noisy": int(noisy.sum().item()),
            "active_samples": int((sample_weights > 0).sum().item()),
        }

    bridge.register("evaluate", evaluate)
    bridge.register("per_class_loss", per_class_loss, safe_async=True)
    bridge.register("noise_report", noise_report, safe_async=True)
    bridge.register_knob(
        "label_smoothing",
        lambda v: state.__setitem__("label_smoothing", float(v)),
        description="cross-entropy label smoothing in [0, 1)",
        value=0.0,
    )

    bridge.on_train_begin()
    for epoch in range(epochs):
        if bridge.should_stop():  # honour a graceful stop_training request from the agent
            break
        model.train()
        bs = cfg["batch_size"]
        steps_per_epoch = max(1, n // bs)
        perm = torch.randperm(n)
        last_loss = 0.0
        for i in range(steps_per_epoch):
            with bridge.section("data"):
                idx = perm[i * bs : (i + 1) * bs]
                xb, yb, wb = x[idx], y[idx], sample_weights[idx]
            optimizer.zero_grad()
            with bridge.section("forward"):
                per_sample = F.cross_entropy(
                    model(xb), yb, label_smoothing=state["label_smoothing"], reduction="none"
                )
                loss = (per_sample * wb).sum() / wb.sum().clamp(min=1.0)
            with bridge.section("backward"):
                loss.backward()
                gn = compute_grad_norm(model)
                bridge.clip_gradients(model)  # honours any agent-set grad_clip
            optimizer.step()
            last_loss = loss.item()
            # Stream per-sample losses so the agent can triage label noise via get_suspicious_samples.
            bridge.on_batch_end(
                last_loss,
                batch_size=int(xb.size(0)),
                grad_norm=gn,
                sample_indices=idx.tolist(),
                sample_losses=per_sample.detach().tolist(),
            )

        bridge.scheduler_step()  # advance the LR scheduler (agent can read/reconfigure it)
        val = evaluate({}, None)["val_acc"]
        bridge.on_epoch_end(
            epoch,
            metrics={"val_acc": val, "train_loss": last_loss, "batch_size": float(cfg["batch_size"])},
        )
        if verbose:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"[train] epoch {epoch}: val_acc={val:.3f} loss={last_loss:.3f} "
                f"lr={lr_now:.4f} ls={state['label_smoothing']:.2f} bs={cfg['batch_size']} "
                f"cmds={len(bridge.processed_commands)}"
            )
        time.sleep(epoch_pause_s)  # leave room for the agent to interject between epochs
    bridge.on_train_end()
    if verbose:
        active = int((sample_weights > 0).sum().item())
        print(f"[train] done. commands applied: {len(bridge.processed_commands)}; "
              f"flagged={sorted(bridge.flagged_indices)[:10]}; active_samples={active}/{n}")
    return bridge


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--broker", default=os.environ.get("CONTROL_PLANE_URL", "http://127.0.0.1:8765"))
    ap.add_argument("--token", default=os.environ.get("CONTROL_PLANE_TOKEN"))
    ap.add_argument("--tunnel-access-token",
                    default=os.environ.get("CONTROL_PLANE_TUNNEL_ACCESS_TOKEN"),
                    help="Dev Tunnels connect token for a non-anonymous tunnel "
                         "(sent as the X-Tunnel-Authorization header)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--run-id", default=os.environ.get("CONTROL_PLANE_RUN_ID", "default"))
    ap.add_argument("--poll-interval", type=float, default=0.0,
                    help="seconds between background command polls (0 = drain only at epoch sync points)")
    ap.add_argument("--noise-frac", type=float, default=0.08, help="fraction of labels to corrupt")
    ap.add_argument("--mlflow", action="store_true", help="link an active MLflow run into telemetry")
    args = ap.parse_args()

    client = ControlPlaneClient.from_url(
        args.broker, args.token, tunnel_access_token=args.tunnel_access_token
    )
    if not client.health():
        raise SystemExit(f"broker not reachable at {args.broker}")
    run_training(
        client,
        epochs=args.epochs,
        lr=args.lr,
        mlflow=args.mlflow,
        run_id=args.run_id,
        poll_interval=args.poll_interval,
        noise_frac=args.noise_frac,
    )


if __name__ == "__main__":
    main()
