
from langchain_openai import OpenAIEmbeddings

from app.config import settings


embedding_model = OpenAIEmbeddings(
    model=settings.openai_embedding_model,
    api_key=settings.openai_api_key,
)