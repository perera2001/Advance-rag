
from datetime import datetime

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    document_id: str
    filename: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    page_count: int
    chunk_count: int
    uploaded_at: datetime


class DocumentUploadResponse(BaseModel):
    message: str
    document: DocumentMetadata


class DocumentListResponse(BaseModel):
    documents: list[DocumentMetadata]


class DocumentDeleteResponse(BaseModel):
    message: str
    document_id: str