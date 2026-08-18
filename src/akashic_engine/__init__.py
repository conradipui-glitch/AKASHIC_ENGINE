"""AKASHIC_ENGINE core package."""

from .domain import (
    Claim,
    EngineEvent,
    EpistemicLevel,
    EventType,
    EvidenceSpan,
    MethodManifest,
    PatternConfidenceComponents,
    PatternProjection,
    ValidationState,
)
from .attunement import (
    AttunementBundle,
    AttunementCandidate,
    AttunementComponent,
    AttunementFeatures,
    AttunementPolicy,
    AttunementPolicyError,
    AttunementResult,
    AttunementWeights,
    ExplainableAttunementEngine,
    UnknownReaderRoleError,
)
from .evidence_graph import (
    ClaimExplanation,
    ClaimRelation,
    ClaimRelationType,
    InMemoryEvidenceGraph,
)
from .event_store import (
    EpistemicBoundaryError,
    EventNotFoundError,
    InMemoryEventStore,
    InvalidEventTransitionError,
)

__all__ = [
    "AttunementBundle",
    "AttunementCandidate",
    "AttunementComponent",
    "AttunementFeatures",
    "AttunementPolicy",
    "AttunementPolicyError",
    "AttunementResult",
    "AttunementWeights",
    "Claim",
    "ClaimExplanation",
    "ClaimRelation",
    "ClaimRelationType",
    "ExplainableAttunementEngine",
    "UnknownReaderRoleError",
    "EngineEvent",
    "EpistemicBoundaryError",
    "EpistemicLevel",
    "EventNotFoundError",
    "EventType",
    "EvidenceSpan",
    "InMemoryEvidenceGraph",
    "InMemoryEventStore",
    "InvalidEventTransitionError",
    "MethodManifest",
    "PatternConfidenceComponents",
    "PatternProjection",
    "ValidationState",
]

__version__ = "0.1.0-dev0"
