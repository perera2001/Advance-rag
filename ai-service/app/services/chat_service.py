from langchain_core.documents import Document

from app.memory.session import (
    add_conversation,
    get_session_history,
)
from app.rag.chain import run_rag_chain
from app.rag.vectorstore import document_exists
from app.schemas.chat_schema import (
    ChatRequest,
    ChatResponse,
    SourceResponse,
)


def validate_document_ids(
    document_ids: list[str],
) -> list[str]:
    unique_document_ids = list(
        dict.fromkeys(document_ids)
    )

    missing_document_ids = [
        document_id
        for document_id in unique_document_ids
        if not document_exists(document_id)
    ]

    if missing_document_ids:
        raise ValueError(
            "The following documents were not found: "
            + ", ".join(missing_document_ids)
        )

    return unique_document_ids


def create_sources(
    documents: list[Document],
) -> list[SourceResponse]:
    sources = []
    seen_sources = set()

    for document in documents:
        metadata = document.metadata

        source_key = (
            metadata["document_id"],
            metadata["page"],
        )

        if source_key in seen_sources:
            continue

        seen_sources.add(source_key)

        sources.append(
            SourceResponse(
                document_id=metadata["document_id"],
                filename=metadata["filename"],
                title=metadata["title"],
                page=metadata["page"],
            )
        )

    return sources


def execute_chat(
    request: ChatRequest,
) -> ChatResponse:
    document_ids = validate_document_ids(
        request.document_ids
    )

    history = get_session_history(
        request.session_id
    )

    answer, source_documents = run_rag_chain(
        question=request.question,
        document_ids=document_ids,
        chat_history=history.messages,
    )

    add_conversation(
        session_id=request.session_id,
        question=request.question,
        answer=answer,
    )

    return ChatResponse(
        answer=answer,
        sources=create_sources(source_documents),
    )
