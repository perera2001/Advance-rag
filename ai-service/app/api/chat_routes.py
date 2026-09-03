from fastapi import (
    APIRouter,
    HTTPException,
    status,
)

from app.memory.session import clear_session
from app.schemas.chat_schema import (
    ChatRequest,
    ChatResponse,
    ClearSessionResponse,
)
from app.services.chat_service import execute_chat


router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


@router.post(
    "",
    response_model=ChatResponse,
)
def chat(request: ChatRequest):
    try:
        return execute_chat(request)

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Chat processing failed.",
        ) from error


@router.delete(
    "/sessions/{session_id}",
    response_model=ClearSessionResponse,
)
def delete_chat_session(
    session_id: str,
):
    clear_session(session_id)

    return ClearSessionResponse(
        message="Chat session cleared successfully.",
        session_id=session_id,
    )
