
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings(BaseSettings):
    openai_api_key: str
    cohere_api_key: str | None = None

    enable_reranker: bool = False

    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    chroma_collection_name: str = "research_papers"
    chroma_db_path: str = "data/chroma_db"
    pdf_directory: str = "data/pdfs"

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def chroma_path(self) -> Path:
        return BASE_DIR / self.chroma_db_path

    @property
    def pdf_path(self) -> Path:
        return BASE_DIR / self.pdf_directory


settings = Settings()

settings.chroma_path.mkdir(parents=True, exist_ok=True)
settings.pdf_path.mkdir(parents=True, exist_ok=True)