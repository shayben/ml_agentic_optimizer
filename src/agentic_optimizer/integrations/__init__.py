"""Framework integrations for live agentic optimizer control."""
from __future__ import annotations

from typing import Any

__all__ = ["HFBridgeCallback", "LightningBridgeCallback"]


def __getattr__(name: str) -> Any:
    if name == "LightningBridgeCallback":
        from .lightning import BridgeCallback

        return BridgeCallback
    if name == "HFBridgeCallback":
        from .hf import HFBridgeCallback

        return HFBridgeCallback
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
