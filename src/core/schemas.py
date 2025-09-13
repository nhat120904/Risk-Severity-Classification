from enum import Enum
from pydantic import BaseModel, Field


class Risk(str, Enum):
    High = "High"
    Medium = "Medium"
    Low = "Low"


class ClfOut(BaseModel):
    risk: Risk
    rationale: str = Field(..., description="short reason for the label")
    evidence: list[str] = Field(default_factory=list)


class DefRecord(BaseModel):
    deficiency: str = Field(...)
    root_cause: str = Field("")
    corrective: str = Field("")
    preventive: str = Field("")


