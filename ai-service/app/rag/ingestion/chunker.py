
from uuid import uuid4

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker

from app.rag.embeddings import embedding_model
from app.rag.ingestion.metadata_extractor import ExtractedMetadata


semantic_chunker = SemanticChunker(
    embeddings=embedding_model,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=90,
)


def create_chunks(
    pages: list[Document],
    extracted_metadata: ExtractedMetadata,
) -> list[Document]:
    chunks = semantic_chunker.split_documents(pages)

    authors = ", ".join(extracted_metadata.authors)

    for chunk_index, chunk in enumerate(chunks):
        chunk.metadata.update(
            {
                "chunk_id": str(uuid4()),
                "chunk_index": chunk_index,
                "title": extracted_metadata.title,
                "authors": authors,
            }
        )

        if extracted_metadata.year is not None:
            chunk.metadata["year"] = extracted_metadata.year

    return chunks