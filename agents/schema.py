# agents/schemas.py
from pydantic import BaseModel, Field
from typing import Literal


class RawSignal(BaseModel):
    competitor: str
    signal_type: str = Field(description="One of: pricing, feature, sentiment, hiring, news")
    description: str = Field(description="One sentence summary of the signal")
    source: str = Field(description="Where this signal came from")
    raw_evidence: str = Field(description="Direct quote or data point supporting this signal")
    confidence: float = Field(description="Confidence score 0-10", ge=0, le=10)


class RawSignalList(BaseModel):
    signals: list[RawSignal]


class AnalyzedSignal(BaseModel):
    competitor: str
    signal_type: str
    description: str
    assessment: Literal["THREAT", "OPPORTUNITY", "NEUTRAL"]
    affected_markets: list[str] = Field(description="SwiftMart markets affected e.g. ['delhi', 'bangalore']")
    impact_score: int = Field(description="Impact score 1-10", ge=1, le=10)
    reasoning: str = Field(description="Why this matters to SwiftMart specifically")
    recommended_action: str = Field(description="Concrete action SwiftMart should take")
    confidence: float = Field(ge=0, le=10)


class AnalyzedSignalList(BaseModel):
    signals: list[AnalyzedSignal]