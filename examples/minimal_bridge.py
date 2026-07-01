"""The smallest possible integration: drop ``attach`` into a vanilla PyTorch loop.

This is the friction-free path. Three lines wire a stock training loop to the control plane:

    from agentic_optimizer import attach
    bridge = attach(optimizer, model)        # 1) build (no-op until CONTROL_PLANE_URL is set)
    ...
    bridge.train_step(loss, batch_size=n)    # 2) one call replaces backward/step/zero_grad + telemetry
    bridge.epoch_end(epoch, val_acc=acc)     # 3) push epoch metrics + apply queued agent commands

``attach`` returns a :class:`~agentic_optimizer.bridge.NoOpBridge` when ``CONTROL_PLANE_URL`` is unset,
so this *exact* script runs unchanged with zero overhead off the control plane, and becomes live
(agent-steerable) the moment the env vars are present — no code changes.

Run it standalone (inert bridge):

    python examples/minimal_bridge.py

Make it live — point it at a broker and the same script is steerable by the agent:

    set CONTROL_PLANE_URL=http://127.0.0.1:8765    # PowerShell: $env:CONTROL_PLANE_URL = "..."
    set CONTROL_PLANE_TOKEN=...                     # if the broker requires a token
    python examples/minimal_bridge.py
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from agentic_optimizer import attach


def main() -> None:
    torch.manual_seed(0)
    # toy data: 4-way classification of Gaussian blobs
    centers = torch.randn(4, 20) * 2.5
    y = torch.randint(0, 4, (1024,))
    x = centers[y] + torch.randn(1024, 20)

    model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 4))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.9)

    # (1) one line to wire the loop to the control plane (inert NoOpBridge unless CONTROL_PLANE_URL is set)
    bridge = attach(optimizer, model)

    with bridge:  # on_train_begin / on_train_end (a no-op for NoOpBridge)
        for epoch in range(20):
            if bridge.should_stop():  # the agent can request a graceful stop
                break
            model.train()
            for i in range(0, x.size(0), 128):
                xb, yb = x[i : i + 128], y[i : i + 128]
                loss = F.cross_entropy(model(xb), yb)
                # (2) one call: loss.backward() + grad-clip + optimizer.step() + zero_grad() + telemetry
                bridge.train_step(loss, batch_size=xb.size(0))

            with torch.no_grad():
                acc = (model(x).argmax(1) == y).float().mean().item()
            # (3) one call: push epoch metrics and apply any commands the agent queued
            bridge.epoch_end(epoch, val_acc=acc, train_loss=loss.item())
            print(f"epoch {epoch}: val_acc={acc:.3f} lr={optimizer.param_groups[0]['lr']:.4f}")

    print(f"done ({type(bridge).__name__}).")


if __name__ == "__main__":
    main()
