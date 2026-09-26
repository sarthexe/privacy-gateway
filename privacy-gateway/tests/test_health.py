"""Health endpoint tests that do not require live databases."""

from httpx import AsyncClient

from app.api.routes import health


async def test_health_returns_ok_when_dependencies_are_reachable(
    client: AsyncClient,
    monkeypatch,
) -> None:
    async def available(*_args: object) -> None:
        return None

    monkeypatch.setattr(health, "check_database", available)
    monkeypatch.setattr(health, "check_redis", available)

    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "services": {"postgres": "available", "redis": "available"},
    }


async def test_health_hides_dependency_error_details(
    client: AsyncClient,
    monkeypatch,
) -> None:
    async def unavailable(*_args: object) -> None:
        raise RuntimeError("synthetic connection failure")

    async def available(*_args: object) -> None:
        return None

    monkeypatch.setattr(health, "check_database", unavailable)
    monkeypatch.setattr(health, "check_redis", available)

    response = await client.get("/health")

    assert response.status_code == 503
    assert response.json() == {
        "status": "degraded",
        "services": {"postgres": "unavailable", "redis": "available"},
    }
    assert "synthetic connection failure" not in response.text
