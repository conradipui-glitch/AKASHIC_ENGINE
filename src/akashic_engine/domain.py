"""Canonical domain models for AKASHIC_ENGINE.

The core distinction is epistemic: observed/source-grounded information,
derived interpretation, and symbolic hypotheses are represented explicitly and
must never be silently promoted between layers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EpistemicLevel(StrEnum):
    """How strongly a record may be treated as knowledge about the user/world."""

    OBSERVED = "observed"
    INTERPRETATION = "interpretation"
    SYMBOLIC_HYPOTHESIS = "symbolic_hypothesis"


class ValidationState(StrEnum):
    UNVERIFIED = "unverified"
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    VALIDATED = "validated"


class EventType(StrEnum):
    RECORD = "record"
    SUPERSEDE = "supersede"
    RETRACT = "retract"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class FrozenModel(BaseModel):
    """Immutable value object used for canonical/derived records."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class EvidenceSpan(FrozenModel):
    evidence_id: str = Field(default_factory=lambda: uuid4().hex)
    source_id: str
    source_version_id: str
    content_hash: str
    start_offset: int | None = None
    end_offset: int | None = None
    excerpt: str | None = None
    extraction_method: str = "direct"
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("end_offset")
    @classmethod
    def end_not_before_start(cls, value: int | None, info):
        start = info.data.get("start_offset")
        if value is not None and start is not None and value < start:
            raise ValueError("end_offset must be >= start_offset")
        return value


class Claim(FrozenModel):
    claim_id: str = Field(default_factory=lambda: uuid4().hex)
    statement: str
    claim_type: str = "general"
    producer: str
    evidence_refs: tuple[str, ...] = ()
    supports: tuple[str, ...] = ()
    contradicts: tuple[str, ...] = ()
    epistemic_level: EpistemicLevel
    validation_state: ValidationState = ValidationState.UNVERIFIED
    created_at: datetime = Field(default_factory=utc_now)


class EngineEvent(FrozenModel):
    """Append-only event. seq is assigned by the EventStore adapter."""

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    seq: int | None = None
    event_type: EventType = EventType.RECORD
    entity_type: str
    entity_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    epistemic_level: EpistemicLevel
    producer: str
    intent: str
    target_event_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class PatternConfidenceComponents(FrozenModel):
    """Reproducible inputs for longitudinal pattern confidence.

    Values are normalized to [0, 1]. The final score is deterministic and does
    not accept a model's self-reported confidence as evidence.
    """

    evidence_quality: float = Field(ge=0.0, le=1.0)
    evidence_diversity: float = Field(ge=0.0, le=1.0)
    recurrence: float = Field(ge=0.0, le=1.0)
    temporal_spread: float = Field(ge=0.0, le=1.0)
    cross_domain_presence: float = Field(ge=0.0, le=1.0)
    user_confirmation: float = Field(ge=0.0, le=1.0)
    contradiction: float = Field(default=0.0, ge=0.0, le=1.0)

    def score(self) -> float:
        positive = (
            0.22 * self.evidence_quality
            + 0.18 * self.evidence_diversity
            + 0.20 * self.recurrence
            + 0.14 * self.temporal_spread
            + 0.10 * self.cross_domain_presence
            + 0.16 * self.user_confirmation
        )
        score = positive * (1.0 - 0.65 * self.contradiction)
        return round(max(0.0, min(1.0, score)), 6)


class PatternProjection(FrozenModel):
    pattern_id: str = Field(default_factory=lambda: uuid4().hex)
    pattern_type: str
    description: str
    evidence_refs: tuple[str, ...]
    first_observed: datetime
    last_observed: datetime
    recurrence_count: int = Field(ge=1)
    domain_tags: tuple[str, ...] = ()
    contradictions: tuple[str, ...] = ()
    confidence_components: PatternConfidenceComponents
    status: str = "candidate"

    @property
    def confidence(self) -> float:
        return self.confidence_components.score()


class MethodManifest(FrozenModel):
    method_id: str
    version: str
    title: str
    category: str
    required_inputs: tuple[str, ...] = ()
    optional_inputs: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    plugin_dependencies: tuple[str, ...] = ()
    allowed_epistemic_levels: tuple[EpistemicLevel, ...]
    produces: tuple[str, ...] = ()

    def missing_inputs(self, supplied: dict[str, Any]) -> tuple[str, ...]:
        return tuple(name for name in self.required_inputs if not supplied.get(name))

    def validate_inputs(self, supplied: dict[str, Any]) -> None:
        missing = self.missing_inputs(supplied)
        if missing:
            raise ValueError(f"Missing required inputs: {', '.join(missing)}")
