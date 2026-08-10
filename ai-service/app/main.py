from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.document_routes import router as document_router


app = FastAPI(
    title="Advanced RAG AI Service",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(document_router)


@app.get("/")
def home():
    return {
        "message": "Advanced RAG AI service is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }