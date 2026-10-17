"""Shared data contracts for ingestion, retrieval, and the agent."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_name: str
    page: int
    section: str
    text: str


class DocumentInfo(BaseModel):
    doc_id: str
    doc_name: str
    pages: int
    chunks: int
    synthetic: bool = True


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    doc_name: str
    page: int
    section: str
    excerpt: str


class RiskFlag(BaseModel):
    clause_type: str
    severity: str
    excerpt: str
    explanation: str
    doc_id: str
    doc_name: str
    page: int
    section: str
    chunk_id: str


class ToolTrace(BaseModel):
    name: str
    arguments: dict[str, object]
    output: dict[str, object]


class AnalysisResult(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    risks: list[RiskFlag] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: str = ""
    route: str = ""
    tool_results: list[ToolTrace] = Field(default_factory=list)
    injection_detected: bool = False
    grounded: bool = False
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    used_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_contexts: list[str] = Field(default_factory=list)
    used_contexts: list[str] = Field(default_factory=list)
