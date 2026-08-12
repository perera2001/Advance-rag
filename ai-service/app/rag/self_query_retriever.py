from langchain_classic.chains.query_constructor.base import AttributeInfo
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.query_constructors.chroma import ChromaTranslator
from langchain_openai import ChatOpenAI

from app.config import settings
from app.rag.vectorstore import vectorstore


DOCUMENT_DESCRIPTION = """
Research-paper chunks containing titles, authors, publication years,
page numbers, filenames, and document identifiers.
"""


METADATA_FIELDS = [
    AttributeInfo(
        name="title",
        description="The title of the research paper",
        type="string",
    ),
    AttributeInfo(
        name="authors",
        description="Comma-separated research paper author names",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="The publication year of the research paper",
        type="integer",
    ),
    AttributeInfo(
        name="filename",
        description="Stored PDF filename such as paper1.pdf",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="The page number in the PDF",
        type="integer",
    ),
]


class SelectionAwareSelfQueryRetriever(SelfQueryRetriever):
    def _prepare_query(
        self,
        query,
        structured_query,
    ):
        selection_filter = self.search_kwargs.get("filter")

        new_query, search_kwargs = super()._prepare_query(
            query,
            structured_query,
        )

        generated_filter = search_kwargs.get("filter")

        if selection_filter and generated_filter:
            search_kwargs["filter"] = {
                "$and": [
                    selection_filter,
                    generated_filter,
                ]
            }
        elif selection_filter:
            search_kwargs["filter"] = selection_filter

        return new_query, search_kwargs


query_llm = ChatOpenAI(
    model=settings.openai_chat_model,
    api_key=settings.openai_api_key,
    temperature=0,
)


def create_self_query_retriever(
    document_ids: list[str],
    top_k: int = 6,
) -> SelectionAwareSelfQueryRetriever:
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

    return SelectionAwareSelfQueryRetriever.from_llm(
        llm=query_llm,
        vectorstore=vectorstore,
        document_contents=DOCUMENT_DESCRIPTION,
        metadata_field_info=METADATA_FIELDS,
        structured_query_translator=ChromaTranslator(),
        search_kwargs={
            "k": top_k,
            "filter": document_filter,
        },
        enable_limit=True,
        verbose=False,
    )
