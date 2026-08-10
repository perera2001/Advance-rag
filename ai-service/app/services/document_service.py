import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.config import settings
from app.rag.ingestion.chunker import create_chunks
from app.rag.ingestion.loader import load_pdf
from app.rag.ingestion.metadata_extractor import extract_metadata
from app.rag.vectorstore import (
    add_chunks,
    delete_document,
    get_all_document_data,
    get_document_data,
)
from app.schemas.document_schema import DocumentMetadata


def get_next_pdf_filename() -> str:
    largest_number = 0

    for file_path in settings.pdf_path.glob("paper*.pdf"):
        match = re.fullmatch(
            r"paper(\d+)\.pdf",
            file_path.name,
            re.IGNORECASE,
        )

        if match:
            largest_number = max(
                largest_number,
                int(match.group(1)),
            )

    return f"paper{largest_number + 1}.pdf"


async def process_document(
    uploaded_file: UploadFile,
) -> DocumentMetadata:
    document_id = str(uuid4())
    stored_filename = get_next_pdf_filename()
    file_path = settings.pdf_path / stored_filename

    try:
        with file_path.open("wb") as destination:
            shutil.copyfileobj(
                uploaded_file.file,
                destination,
            )

        pages = load_pdf(
            file_path=file_path,
            document_id=document_id,
        )

        if not pages:
            raise ValueError(
                "The PDF does not contain readable pages."
            )

        original_filename = (
            uploaded_file.filename or stored_filename
        )

        extracted_metadata = extract_metadata(
            pages=pages,
            fallback_title=Path(original_filename).stem,
        )

        chunks = create_chunks(
            pages=pages,
            extracted_metadata=extracted_metadata,
        )

        if not chunks:
            raise ValueError(
                "No chunks could be created from the PDF."
            )

        uploaded_at = datetime.now(timezone.utc)

        for chunk in chunks:
            chunk.metadata.update(
                {
                    "original_filename": original_filename,
                    "page_count": len(pages),
                    "chunk_count": len(chunks),
                    "uploaded_at": uploaded_at.isoformat(),
                }
            )

        add_chunks(chunks)

        return DocumentMetadata(
            document_id=document_id,
            filename=stored_filename,
            title=extracted_metadata.title,
            authors=extracted_metadata.authors,
            year=extracted_metadata.year,
            page_count=len(pages),
            chunk_count=len(chunks),
            uploaded_at=uploaded_at,
        )

    except Exception:
        delete_document(document_id)

        if file_path.exists():
            file_path.unlink()

        raise

    finally:
        await uploaded_file.close()


def list_documents() -> list[DocumentMetadata]:
    data = get_all_document_data()
    documents: dict[str, DocumentMetadata] = {}

    for metadata in data.get("metadatas", []):
        if not metadata:
            continue

        document_id = metadata.get("document_id")

        if not document_id or document_id in documents:
            continue

        authors_text = metadata.get("authors", "")
        authors = [
            author.strip()
            for author in authors_text.split(",")
            if author.strip()
        ]

        documents[document_id] = DocumentMetadata(
            document_id=document_id,
            filename=metadata["filename"],
            title=metadata["title"],
            authors=authors,
            year=metadata.get("year"),
            page_count=metadata["page_count"],
            chunk_count=metadata["chunk_count"],
            uploaded_at=datetime.fromisoformat(
                metadata["uploaded_at"]
            ),
        )

    return sorted(
        documents.values(),
        key=lambda document: document.uploaded_at,
        reverse=True,
    )


def remove_document(document_id: str) -> str | None:
    data = get_document_data(document_id)

    if not data["ids"]:
        return None

    metadata = data["metadatas"][0]
    stored_filename = metadata["filename"]

    delete_document(document_id)

    file_path = settings.pdf_path / stored_filename

    if file_path.exists():
        file_path.unlink()

    return stored_filename