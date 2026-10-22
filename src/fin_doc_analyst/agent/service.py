"""Application service used by FastAPI, Streamlit, and the eval runner."""

from __future__ import annotations

from pathlib import Path

from fin_doc_analyst.agent.graph import build_graph
from fin_doc_analyst.agent.state import AgentState
from fin_doc_analyst.config import Settings, configure_tracing, get_settings
from fin_doc_analyst.guardrails.pii import contains_raw_pii, redact
from fin_doc_analyst.ingestion.pdf import PdfParseError
from fin_doc_analyst.ingestion.pipeline import ingest_pdf
from fin_doc_analyst.llm.factory import build_llm
from fin_doc_analyst.logging_config import configure_logging, get_logger
from fin_doc_analyst.retrieval.embeddings import build_embedder
from fin_doc_analyst.retrieval.factory import build_store
from fin_doc_analyst.schemas import AnalysisResult, Citation, DocumentInfo, RiskFlag, ToolTrace


class AnalystService:
    def __init__(self, settings: Settings, store: object, llm: object) -> None:
        self.settings = settings
        self.store = store
        self.llm = llm
        self.graph = build_graph(store, llm)
        configure_logging(settings.log_level)
        configure_tracing(settings)
        self._log = get_logger("service")

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        load_samples: bool | None = None,
    ) -> AnalystService:
        resolved = settings or get_settings()
        embedder = build_embedder(resolved)
        store = build_store(resolved, embedder)
        service = cls(resolved, store, build_llm(resolved))
        should_load = resolved.load_samples if load_samples is None else load_samples
        if should_load and resolved.samples_dir.exists():
            service.load_directory(resolved.samples_dir, persist=False)
        if resolved.uploads_dir.exists():
            service.load_directory(resolved.uploads_dir, persist=False)
        return service

    def load_directory(self, path: Path, persist: bool = False) -> None:
        for pdf in sorted(path.glob("*.pdf")):
            self.ingest_pdf(pdf.name, pdf.read_bytes(), persist=persist)

    def ingest_pdf(self, filename: str, data: bytes, persist: bool = True) -> DocumentInfo:
        if not filename.lower().endswith(".pdf"):
            raise PdfParseError("Only PDF uploads are supported")
        if not data.startswith(b"%PDF"):
            raise PdfParseError("File is not a PDF")
        if len(data) > self.settings.max_upload_bytes:
            limit = self.settings.max_upload_bytes
            raise PdfParseError(f"PDF exceeds the upload limit of {limit} bytes")
        info, chunks = ingest_pdf(filename, data)
        self.store.add(info, chunks)
        if persist:
            self.settings.uploads_dir.mkdir(parents=True, exist_ok=True)
            target = self.settings.uploads_dir / Path(filename).name
            target.write_bytes(data)
        self._log.info(
            "document.ingested", doc_id=info.doc_id, pages=info.pages, chunks=info.chunks
        )
        return info

    def list_documents(self) -> list[DocumentInfo]:
        return self.store.list_documents()

    def ask(
        self,
        question: str,
        doc_ids: list[str] | None = None,
        include_context: bool = False,
    ) -> AnalysisResult:
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("question is required")
        initial: AgentState = {
            "question": clean_question,
            "doc_ids": doc_ids or [],
            "route": "",
            "retrieved": [],
            "graded": [],
            "quarantined": [],
            "tool_results": [],
            "risks": [],
            "citations": [],
            "answer": "",
            "refused": False,
            "refusal_reason": "",
            "injection_detected": False,
            "grounded": False,
            "used_contexts": [],
            "retrieved_contexts": [],
        }
        final = self.graph.invoke(initial)
        result = _result_from_state(final)
        result = _redact_result(result)
        if contains_raw_pii(result.answer):
            result.answer = redact(result.answer)
        self._log.info(
            "question.answered",
            route=result.route,
            refused=result.refused,
            reason=result.refusal_reason,
            injection=result.injection_detected,
            citations=len(result.citations),
            tools=[tool.name for tool in result.tool_results],
        )
        if not include_context:
            result.retrieved_contexts = []
            result.used_contexts = []
        return result


def _result_from_state(state: dict[str, object]) -> AnalysisResult:
    citations = [Citation.model_validate(item) for item in state.get("citations") or []]
    risks = [RiskFlag.model_validate(item) for item in state.get("risks") or []]
    tools = [ToolTrace.model_validate(item) for item in state.get("tool_results") or []]
    return AnalysisResult(
        answer=str(state.get("answer") or ""),
        citations=citations,
        risks=risks,
        refused=bool(state.get("refused")),
        refusal_reason=str(state.get("refusal_reason") or ""),
        route=str(state.get("route") or ""),
        tool_results=tools,
        injection_detected=bool(state.get("injection_detected")),
        grounded=bool(state.get("grounded")),
        retrieved_chunk_ids=[
            str(item["chunk_id"]) for item in state.get("retrieved") or [] if isinstance(item, dict)
        ],
        used_chunk_ids=[
            str(item["chunk_id"]) for item in state.get("graded") or [] if isinstance(item, dict)
        ],
        retrieved_contexts=[str(item) for item in state.get("retrieved_contexts") or []],
        used_contexts=[str(item) for item in state.get("used_contexts") or []],
    )


def _redact_result(result: AnalysisResult) -> AnalysisResult:
    result.answer = redact(result.answer)
    for citation in result.citations:
        citation.excerpt = redact(citation.excerpt)
    for risk in result.risks:
        risk.excerpt = redact(risk.excerpt)
        risk.explanation = redact(risk.explanation)
    return result
