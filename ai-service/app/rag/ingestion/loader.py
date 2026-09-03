
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document


def load_pdf(
    file_path: Path,
    document_id: str,
) -> list[Document]:
    loader = PyMuPDFLoader(str(file_path))
    pages = loader.load()

    for page_number, page in enumerate(pages, start=1):
        page.metadata = {
            "document_id": document_id,
            "filename": file_path.name,
            "source": file_path.name,
            "page": page_number,
        }

    return pages