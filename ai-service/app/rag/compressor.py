from langchain_core.documents import Document
from langchain_classic.retrievers.document_compressors import (
    EmbeddingsFilter,
)

from app.rag.embeddings import embedding_model


def compress_documents(
    question: str,
    documents: list[Document],
    top_k: int = 8,
) -> list[Document]:
    if not documents:
        return []

    embeddings_filter = EmbeddingsFilter(
        embeddings=embedding_model,
        k=top_k,
        similarity_threshold=0.60,
    )

    compressed_documents = (
        embeddings_filter.compress_documents(
            documents=documents,
            query=question,
        )
    )

    return list(compressed_documents)
