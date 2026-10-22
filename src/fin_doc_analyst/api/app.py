"""FastAPI surface: upload, ask, list documents, and health."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

import structlog
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator

from fin_doc_analyst import __version__
from fin_doc_analyst.agent.service import AnalystService
from fin_doc_analyst.ingestion.pdf import PdfParseError
from fin_doc_analyst.schemas import AnalysisResult, DocumentInfo


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    doc_ids: list[str] | None = None

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question is required")
        return value.strip()


class HealthResponse(BaseModel):
    status: str
    version: str
    llm_provider: str
    embedding_provider: str
    vector_backend: str
    documents: int


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    if app.state.service is None:
        app.state.service = AnalystService.from_settings()
    yield


def create_app(service: AnalystService | None = None) -> FastAPI:
    app = FastAPI(
        title="fin-doc-analyst",
        version=__version__,
        description=(
            "Agentic RAG analyst for synthetic bank statements, loan agreements, "
            "and insurance policies."
        ),
        lifespan=_lifespan,
    )
    app.state.service = service

    @app.middleware("http")
    async def request_context(request, call_next):  # type: ignore[no-untyped-def]
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=uuid4().hex[:12])
        try:
            return await call_next(request)
        finally:
            structlog.contextvars.clear_contextvars()

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        current: AnalystService = app.state.service
        try:
            ok = current.store.ping()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=503, detail="store unavailable") from exc
        if not ok:
            raise HTTPException(status_code=503, detail="store unavailable")
        settings = current.settings
        return HealthResponse(
            status="ok",
            version=__version__,
            llm_provider=settings.resolved_llm_provider(),
            embedding_provider=settings.resolved_embedding_provider(),
            vector_backend=settings.vector_backend,
            documents=len(current.list_documents()),
        )

    @app.get("/documents", response_model=list[DocumentInfo])
    def list_documents() -> list[DocumentInfo]:
        return app.state.service.list_documents()

    @app.post("/documents", response_model=DocumentInfo)
    async def upload_document(file: UploadFile = File(...)) -> DocumentInfo:
        name = file.filename or "upload.pdf"
        limit = app.state.service.settings.max_upload_bytes
        # Read one byte past the limit so an oversized body is rejected without buffering the rest.
        data = await file.read(limit + 1)
        try:
            return app.state.service.ingest_pdf(name, data, persist=True)
        except PdfParseError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/ask", response_model=AnalysisResult)
    def ask(body: AskRequest) -> AnalysisResult:
        try:
            return app.state.service.ask(body.question, body.doc_ids)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


app = create_app()
