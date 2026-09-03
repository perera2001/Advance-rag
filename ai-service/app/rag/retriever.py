
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI

from app.config import settings
from app.rag.self_query_retriever import (
    create_self_query_retriever,
)
from app.rag.vectorstore import vectorstore


class GeneratedQueries(BaseModel):
    queries: list[str] = Field(
        description="Three alternative versions of the user question"
    )


multi_query_llm = ChatOpenAI(
    model=settings.openai_chat_model,
    api_key=settings.openai_api_key,
    temperature=0,
).with_structured_output(GeneratedQueries)


def generate_queries(question: str) -> list[str]:
    result = multi_query_llm.invoke(
        f"""
Generate exactly three alternative versions of the following question.

The alternatives must preserve the original meaning while using different
wording. Return only the three alternatives through the required structure.

Question:
{question}
"""
    )

    all_queries = [question, *result.queries]

    return list(
        dict.fromkeys(
            query.strip()
            for query in all_queries
            if query.strip()
        )
    )


def get_selected_chunks(
    document_ids: list[str],
) -> list[Document]:
    if len(document_ids) == 1:
        document_filter = {
            "document_id": document_ids[0]
        }
    else:
        document_filter = {
            "document_id": {
                "$in": document_ids
            }
        }

    data = vectorstore.get(
        where=document_filter,
        include=[
            "documents",
            "metadatas",
        ],
    )

    return [
        Document(
            page_content=content,
            metadata=metadata,
        )
        for content, metadata in zip(
            data["documents"],
            data["metadatas"],
        )
    ]


def create_bm25_retriever(
    documents: list[Document],
    top_k: int,
) -> BM25Retriever:
    retriever = BM25Retriever.from_documents(
        documents
    )
    retriever.k = top_k

    return retriever


def add_ranked_results(
    scores: dict[str, float],
    documents: dict[str, Document],
    ranked_results: list[Document],
    weight: float,
):
    rank_constant = 60

    for rank, document in enumerate(
        ranked_results,
        start=1,
    ):
        chunk_id = document.metadata["chunk_id"]

        documents[chunk_id] = document
        scores[chunk_id] = scores.get(
            chunk_id,
            0.0,
        ) + weight / (rank_constant + rank)


def retrieve_documents(
    question: str,
    document_ids: list[str],
    retrieval_k: int = 6,
    final_k: int = 12,
) -> list[Document]:
    selected_chunks = get_selected_chunks(
        document_ids
    )

    if not selected_chunks:
        return []

    queries = generate_queries(question)

    self_query_retriever = (
        create_self_query_retriever(
            document_ids=document_ids,
            top_k=retrieval_k,
        )
    )

    bm25_retriever = create_bm25_retriever(
        documents=selected_chunks,
        top_k=retrieval_k,
    )

    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}

    for query in queries:
        vector_results = (
            self_query_retriever.invoke(query)
        )
        bm25_results = bm25_retriever.invoke(query)

        add_ranked_results(
            scores=scores,
            documents=documents,
            ranked_results=vector_results,
            weight=0.5,
        )

        add_ranked_results(
            scores=scores,
            documents=documents,
            ranked_results=bm25_results,
            weight=0.5,
        )

    ranked_chunk_ids = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )

    return [
        documents[chunk_id]
        for chunk_id in ranked_chunk_ids[:final_k]
    ]