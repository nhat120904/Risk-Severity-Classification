from __future__ import annotations

from pydantic import BaseModel
from typing import List
from src.core.schemas import Risk


class HealthResponse(BaseModel):
    status: str


class ClassifiedItem(BaseModel):
    deficiency: str
    root_cause: str
    corrective: str
    preventive: str
    risk_llm: Risk
    risk_final: Risk
    rationale: str
    evidence: List[str]


class ClassifyResponse(BaseModel):
    count: int
    items: List[ClassifiedItem]
    rag_used: bool | None = None
    notice: str | None = None


