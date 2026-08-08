from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from app.config import settings


class ExtractedMetadata(BaseModel):
    title: str = Field(
        description="The complete title of the research paper"
    )
    authors: list[str] = Field(
        default_factory=list,
        description="Names of the research paper authors",
    )
    year: int | None = Field(
        default=None,
        description="Publication year",
    )


metadata_model = ChatOpenAI(
    model=settings.openai_chat_model,
    api_key=settings.openai_api_key,
    temperature=0,
).with_structured_output(ExtractedMetadata)


def extract_metadata(
    pages: list[Document],
    fallback_title: str,
) -> ExtractedMetadata:
    first_pages_text = "\n\n".join(
        page.page_content for page in pages[:2]
    )[:12000]

    prompt = f"""
Extract metadata from the following research paper text.

Return:
- The complete research paper title
- All author names
- The publication year, if available

Do not invent missing information.
If the title cannot be identified, use: {fallback_title}
If authors cannot be identified, return an empty list.
If the year cannot be identified, return null.

Research paper text:
{first_pages_text}
"""

    return metadata_model.invoke(prompt)
