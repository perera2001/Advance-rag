
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.schemas.document_schema import (
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentUploadResponse,
)
from app.services.document_service import (
    list_documents,
    process_document,
    remove_document,
)


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: UploadFile = File(...),
):
    filename = file.filename or ""

    if Path(filename).suffix.lower() != ".pdf":
        await file.close()

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed.",
        )

    try:
        document = await process_document(file)

        return DocumentUploadResponse(
            message="Document processed successfully.",
            document=document,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document processing failed.",
        ) from error


@router.get(
    "",
    response_model=DocumentListResponse,
)
def get_documents():
    return DocumentListResponse(
        documents=list_documents()
    )


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
)
def delete_document(document_id: str):
    filename = remove_document(document_id)

    if filename is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    return DocumentDeleteResponse(
        message="Document deleted successfully.",
        document_id=document_id,
    )