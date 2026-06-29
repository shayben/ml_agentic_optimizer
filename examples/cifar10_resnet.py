"""CIFAR-10 demo: a small CNN trained with the GitHub Copilot CLI as the in-the-loop optimizer.

The training loop is an ordinary PyTorch loop. The only integration is ``AgenticCallback``: each epoch it
emits a ``state.json`` snapshot, asks the agent (driver) for a ``control.json`` decision, and applies it.

Two agents are selectable:
  * ``heuristic`` (default) — a tiny built-in rule, so the demo runs anywhere with no CLI/auth.
  * ``copilot``   — the real GitHub Copilot CLI via ``CopilotOptimizerDriver`` (needs ``copilot`` + auth).

Runs offline with ``--fake-data`` (random tensors); otherwise downloads CIFAR-10 via torchvision.

Examples
--------
    python examples/cifar10_resnet.py --fake-data --epochs 3 --limit-batches 5
    python examples/cifar10_resnet.py --epochs 10                      # real CIFAR-10, heuristic agent
    python examples/cifar10_resnet.py --agent copilot --epochs 5       # real Copilot CLI agent
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from agentic_optimizer import (
    AgenticCallback,
    ControlSignal,
    CopilotOptimizerDriver,
    FunctionDriver,
)
from agentic_optimizer.contract import TrainingState


class SmallCNN(nn.Module):
    """A compact 3-block CNN for 32x32 images (fast on CPU)."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def get_loaders(args) -> tuple[DataLoader, DataLoader]:
    if args.fake_data:
        g = torch.Generator().manual_seed(0)
        xtr = torch.randn(512, 3, 32, 32, generator=g)
        ytr = torch.randint(0, 10, (512,), generator=g)
        xva = torch.randn(128, 3, 32, 32, generator=g)
        yva = torch.randint(0, 10, (128,), generator=g)
        return (
            DataLoader(TensorDataset(xtr, ytr), batch_size=args.batch_size, shuffle=True),
            DataLoader(TensorDataset(xva, yva), batch_size=args.batch_size),
        )
    import torchvision
    import torchvision.transforms as T

    tf = T.Compose(
        [T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]
    )
    train = torchvision.datasets.CIFAR10(args.data_dir, train=True, download=True, transform=tf)
    val = torchvision.datasets.CIFAR10(args.data_dir, train=False, download=True, transform=tf)
    return (
        DataLoader(train, batch_size=args.batch_size, shuffle=True),
        DataLoader(val, batch_size=args.batch_size),
    )


def heuristic_agent(state: TrainingState) -> ControlSignal:
    """A minimal offline stand-in for the Copilot agent (LR rescue + plateau nudge)."""
    lh = state.loss_history
    pg = state.param_groups[0] if state.param_groups else None
    if pg is None or len(lh) < 4:
        return ControlSignal()
    lr = pg.lr
    recent = lh[-3:]
    if recent[-1] > 1.15 * min(recent):
        return ControlSignal(set_lr=round(lr * 0.5, 8), grad_clip=5.0,
                             notes="heuristic: loss rising -> halve lr + clip")
    window = lh[-5:]
    if len(window) >= 5 and (max(window) - min(window)) < 0.02:
        return ControlSignal(set_lr=round(lr * 1.1, 8), notes="heuristic: plateau -> nudge lr up")
    return ControlSignal()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += F.cross_entropy(out, y, reduction="sum").item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1), loss_sum / max(total, 1)


def build_driver(args):
    if args.agent == "copilot":
        prompt_path = Path(__file__).resolve().parents[1] / "agent" / "optimizer_prompt.md"
        return CopilotOptimizerDriver(
            workdir=str(Path(args.work_dir)),
            prompt_path=prompt_path if prompt_path.exists() else None,
            model=args.model or None,
            timeout_s=args.timeout,
        )
    return FunctionDriver(heuristic_agent)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--work-dir", default="./runs/demo")
    ap.add_argument("--fake-data", action="store_true", help="use random tensors (offline/CI)")
    ap.add_argument("--agent", choices=["heuristic", "copilot"], default="heuristic")
    ap.add_argument("--optimize-every", type=int, default=1)
    ap.add_argument("--limit-batches", type=int, default=0, help="cap train batches/epoch (0=all)")
    ap.add_argument("--async-mode", action="store_true")
    ap.add_argument("--model", default="", help="copilot model override (e.g. gpt-5)")
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = get_loaders(args)
    model = SmallCNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    cb = AgenticCallback(
        optimizer,
        driver=build_driver(args),
        optimize_every=args.optimize_every,
        state_path=str(Path(args.work_dir) / "state.json"),
        control_path=str(Path(args.work_dir) / "control.json"),
        max_epochs=args.epochs,
        model=model,
        async_mode=args.async_mode,
    )

    print(f"device={device} agent={args.agent} epochs={args.epochs} start_lr={args.lr}")
    cb.on_train_begin()
    for epoch in range(args.epochs):
        model.train()
        for i, (x, y) in enumerate(train_loader):
            if args.limit_batches and i >= args.limit_batches:
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            grad_norm = cb.compute_grad_norm(model)
            cb.clip_gradients(model)  # applies agent-set grad_clip, if any
            optimizer.step()
            cb.on_batch_end(loss.item(), batch_size=x.size(0), grad_norm=grad_norm)

        acc, vloss = evaluate(model, val_loader, device)
        ctrl = cb.on_epoch_end(epoch, metrics={"val_acc": acc, "val_loss": vloss})
        lr_now = optimizer.param_groups[0]["lr"]
        decision = ctrl.model_dump(exclude_none=True, exclude_defaults=True)
        suffix = f" | agent: {decision}" if decision else ""
        print(f"epoch {epoch}: val_acc={acc:.3f} val_loss={vloss:.3f} lr={lr_now:.5f}{suffix}")
    cb.on_train_end()
    print(f"done. agent consultations applied: {len(cb.applied_controls)}")
    if cb.flagged_indices:
        print(f"flagged (suspected-noisy) sample indices: {sorted(cb.flagged_indices)[:20]} ...")


if __name__ == "__main__":
    main()
