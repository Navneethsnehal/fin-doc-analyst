"""Local demo. Uses the in-process engine, or the API when API_BASE_URL is set."""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

from fin_doc_analyst.agent.service import AnalystService
from fin_doc_analyst.config import Settings


def _local_service() -> AnalystService:
    settings = Settings(
        llm_provider=os.environ.get("LLM_PROVIDER", "auto"),
        embedding_provider=os.environ.get("EMBEDDING_PROVIDER", "auto"),
        vector_backend=os.environ.get("VECTOR_BACKEND", "memory"),
        log_level="INFO",
    )
    return AnalystService.from_settings(settings, load_samples=True)


def _ask_local(question: str) -> dict[str, Any]:
    service = _cached_service()
    return service.ask(question).model_dump()


@st.cache_resource(show_spinner=False)
def _cached_service() -> AnalystService:
    return _local_service()


def _ask_api(base_url: str, question: str) -> dict[str, Any]:
    response = httpx.post(f"{base_url.rstrip('/')}/ask", json={"question": question}, timeout=60)
    response.raise_for_status()
    return response.json()


def _list_api(base_url: str) -> list[dict[str, Any]]:
    response = httpx.get(f"{base_url.rstrip('/')}/documents", timeout=30)
    response.raise_for_status()
    return response.json()


def _upload_api(base_url: str, name: str, data: bytes) -> None:
    response = httpx.post(
        f"{base_url.rstrip('/')}/documents",
        files={"file": (name, data, "application/pdf")},
        timeout=60,
    )
    response.raise_for_status()


def main() -> None:
    st.set_page_config(page_title="fin-doc-analyst", page_icon="📄", layout="wide")
    st.title("fin-doc-analyst")
    st.caption(
        "Agentic RAG over synthetic bank, loan, and insurance documents. "
        "Answers cite a page. Unsupported questions are refused. "
        "Nothing in the sample corpus is a real customer's data."
    )
    api_base = os.environ.get("API_BASE_URL", "").strip()
    with st.sidebar:
        st.subheader("Documents")
        if api_base:
            st.write(f"API: `{api_base}`")
            docs = _list_api(api_base)
        else:
            st.write("In-process engine, fake LLM unless a provider key is configured.")
            docs = [doc.model_dump() for doc in _cached_service().list_documents()]
        for doc in docs:
            st.markdown(f"**{doc['doc_name']}**  \n{doc['pages']} pages · {doc['chunks']} chunks")
        upload = st.file_uploader("Upload a PDF", type=["pdf"])
        if upload is not None and st.button("Index upload"):
            data = upload.getvalue()
            if api_base:
                _upload_api(api_base, upload.name, data)
            else:
                _cached_service().ingest_pdf(upload.name, data, persist=True)
            st.success(f"Indexed {upload.name}")
            st.rerun()

    question = st.text_area(
        "Question",
        placeholder="What is the ending balance? Which clauses renew automatically?",
        height=100,
    )
    if st.button("Ask", type="primary") and question.strip():
        with st.spinner("Reading the documents..."):
            payload = _ask_api(api_base, question) if api_base else _ask_local(question)
        if payload.get("refused"):
            st.warning(payload["answer"])
        else:
            st.markdown(payload["answer"])
        columns = st.columns(3)
        columns[0].metric("Route", payload.get("route") or "—")
        columns[1].metric("Grounded", "yes" if payload.get("grounded") else "no")
        columns[2].metric(
            "Injection",
            "blocked" if payload.get("injection_detected") else "none",
        )
        tools = payload.get("tool_results") or []
        if tools:
            st.subheader("Tools")
            for tool in tools:
                st.code(
                    f"{tool['name']}({tool['arguments']}) -> {tool['output']}",
                    language="json",
                )
        risks = payload.get("risks") or []
        if risks:
            st.subheader("Clause risks")
            for risk in risks:
                st.markdown(
                    f"**{risk['severity']} · {risk['clause_type']}** · "
                    f"{risk['doc_name']} p.{risk['page']}  \n"
                    f"_{risk['excerpt']}_  \n{risk['explanation']}"
                )
        citations = payload.get("citations") or []
        if citations:
            st.subheader("Citations")
            for citation in citations:
                st.markdown(
                    f"- **{citation['doc_name']}** p.{citation['page']} "
                    f"({citation['section']}) — {citation['excerpt'][:180]}"
                )


if __name__ == "__main__":
    main()
