"""Contract tests for the initial chat completions endpoint."""

from httpx import AsyncClient


async def test_chat_completions_returns_openai_error_until_routing_exists(
    client: AsyncClient,
) -> None:
    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": "example-model",
            "messages": [{"role": "user", "content": "Summarize the public dataset."}],
        },
    )

    assert response.status_code == 501
    assert response.json() == {
        "error": {
            "message": "No model provider is configured yet.",
            "type": "server_error",
            "param": None,
            "code": "model_not_configured",
        }
    }


async def test_chat_completions_rejects_unsupported_message_roles(
    client: AsyncClient,
) -> None:
    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": "example-model",
            "messages": [{"role": "tool", "content": "not supported in this skeleton"}],
        },
    )

    assert response.status_code == 422