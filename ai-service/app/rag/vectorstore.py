
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import settings
from app.rag.embeddings import embedding_model


vectorstore = Chroma(
    collection_name=settings.chroma_collection_name,
    embedding_function=embedding_model,
    persist_directory=str(settings.chroma_path),
)


def add_chunks(chunks: list[Document]) -> list[str]:
    chunk_ids = [
        chunk.metadata["chunk_id"]
        for chunk in chunks
    ]

    vectorstore.add_documents(
        documents=chunks,
        ids=chunk_ids,
    )

    return chunk_ids


def get_document_data(document_id: str) -> dict:
    return vectorstore.get(
        where={
            "document_id": document_id
        }
    )


def document_exists(document_id: str) -> bool:
    data = get_document_data(document_id)
    return len(data["ids"]) > 0


def delete_document(document_id: str) -> int:
    data = get_document_data(document_id)
    chunk_ids = data["ids"]

    if chunk_ids:
        vectorstore.delete(ids=chunk_ids)

    return len(chunk_ids)


def get_all_document_data() -> dict:
    return vectorstore.get()