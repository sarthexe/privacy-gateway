"""Health endpoint response models."""

from typing import Literal

from pydantic import BaseModel

ServiceStatus = Literal["available", "unavailable"]


class HealthResponse(BaseModel):
    """Non-sensitive service health summary."""

    status: Literal["ok", "degraded"]
    services: dict[str, ServiceStatus]