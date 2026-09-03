from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from app.config import settings
from app.rag.compressor import compress_documents
from app.rag.prompts import (
    answer_prompt,
    contextualize_question_prompt,
)
from app.rag.reranker import rerank_documents
from app.rag.retriever import retrieve_documents


chat_model = ChatOpenAI(
    model=settings.openai_chat_model,
    api_key=settings.openai_api_key,
    temperature=0,
)


def create_standalone_question(
    question: str,
    chat_history: list,
) -> str:
    if not chat_history:
        return question

    contextualize_chain = (
        contextualize_question_prompt
        | chat_model
        | StrOutputParser()
    )

    return contextualize_chain.invoke(
        {
            "question": question,
            "chat_history": chat_history,
        }
    ).strip()


def format_context(
    documents: list[Document],
) -> str:
    formatted_chunks = []

    for document in documents:
        metadata = document.metadata

        formatted_chunks.append(
            "\n".join(
                [
                    f"Title: {metadata.get('title', 'Unknown')}",
                    f"Filename: {metadata.get('filename', 'Unknown')}",
                    f"Page: {metadata.get('page', 'Unknown')}",
                    f"Content: {document.page_content}",
                ]
            )
        )

    return "\n\n---\n\n".join(formatted_chunks)


def run_rag_chain(
    question: str,
    document_ids: list[str],
    chat_history: list,
) -> tuple[str, list[Document]]:
    standalone_question = create_standalone_question(
        question=question,
        chat_history=chat_history,
    )

    retrieved_documents = retrieve_documents(
        question=standalone_question,
        document_ids=document_ids,
    )

    if not retrieved_documents:
        return (
            "I could not find enough information in "
            "the selected documents.",
            [],
        )

    reranked_documents = rerank_documents(
        question=standalone_question,
        documents=retrieved_documents,
        top_n=8,
    )

    compressed_documents = compress_documents(
        question=standalone_question,
        documents=reranked_documents,
        top_k=8,
    )

    if not compressed_documents:
        return (
            "I could not find enough information in "
            "the selected documents.",
            [],
        )

    context = format_context(
        compressed_documents
    )

    response_chain = (
        answer_prompt
        | chat_model
        | StrOutputParser()
    )

    answer = response_chain.invoke(
        {
            "question": question,
            "chat_history": chat_history,
            "context": context,
        }
    ).strip()

    return answer, compressed_documents
