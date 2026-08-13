from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(
        min_length=1,
        max_length=100,
    )
    question: str = Field(
        min_length=1,
        max_length=2000,
    )
    document_ids: list[str] = Field(
        min_length=1,
    )


class SourceResponse(BaseModel):
    document_id: str
    filename: str
    title: str
    page: int


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]


class ClearSessionResponse(BaseModel):
    message: str
    session_id: str
