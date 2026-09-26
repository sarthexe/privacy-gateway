"""OpenAI-compatible chat completions contract placeholder."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.models.chat import ChatCompletionRequest, OpenAIErrorResponse

router = APIRouter(tags=["chat completions"])


@router.post(
    "/chat/completions",
    status_code=501,
    response_model=None,
    responses={
        400: {"model": OpenAIErrorResponse},
        501: {"model": OpenAIErrorResponse},
    },
    summary="Create a chat completion",
)
async def create_chat_completion(payload: ChatCompletionRequest) -> JSONResponse:
    """Validate the OpenAI-style request while provider routing is unimplemented."""
    del payload
    return JSONResponse(
        status_code=501,
        content={
            "error": {
                "message": "No model provider is configured yet.",
                "type": "server_error",
                "param": None,
                "code": "model_not_configured",
            }
        },
    )
