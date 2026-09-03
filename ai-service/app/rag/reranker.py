from langchain_core.documents import Document

from app.config import settings


def rerank_documents(
    question: str,
    documents: list[Document],
    top_n: int = 8,
) -> list[Document]:
    if not settings.enable_reranker:
        return documents

    if not settings.cohere_api_key:
        raise ValueError(
            "COHERE_API_KEY is required when ENABLE_RERANKER=true."
        )

    try:
        from langchain_cohere import CohereRerank
    except ImportError as error:
        raise RuntimeError(
            "Install langchain-cohere to enable reranking."
        ) from error

    reranker = CohereRerank(
        model="rerank-v3.5",
        cohere_api_key=settings.cohere_api_key,
        top_n=top_n,
    )

    return list(
        reranker.compress_documents(
            documents=documents,
            query=question,
        )
    )
