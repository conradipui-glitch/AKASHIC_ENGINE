"""Provider-agnostic core interfaces."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, Sequence

from .attunement import AttunementBundle, AttunementCandidate
from .domain import Claim, EngineEvent, EpistemicLevel, EvidenceSpan
from .evidence_bundle import EvidenceBundle
from .evidence_graph import ClaimExplanation, ClaimRelation, ClaimRelationType
from .pattern_engine import PatternAnalysis, PatternOccurrence
from .ingestion import IngestionEnvelope, IngestionReceipt
from .source_store import SourceRecord, SourceVersion


class EventStore(Protocol):
    def append(self, event: EngineEvent) -> EngineEvent: ...

    def read(self, *, after_seq: int = 0) -> Sequence[EngineEvent]: ...

    def replay(self, *, entity_type: str, entity_id: str) -> Sequence[EngineEvent]: ...

    def supersede(self, target_event_id: str, replacement: EngineEvent) -> EngineEvent: ...

    def retract(self, target_event_id: str, *, producer: str, intent: str, reason: str) -> EngineEvent: ...

    def current_records(
        self,
        *,
        entity_type: str | None = None,
        entity_id: str | None = None,
        epistemic_levels: set[EpistemicLevel] | None = None,
    ) -> Sequence[EngineEvent]: ...


class IngestionService(Protocol):
    def ingest(self, envelope: IngestionEnvelope) -> IngestionReceipt: ...


class SourceStore(Protocol):
    def register_source(
        self,
        *,
        namespace: str,
        external_key: str,
        source_type: str,
        uri: str | None = None,
    ) -> SourceRecord: ...

    def ingest_version(
        self,
        source_id: str,
        *,
        content: bytes,
        mime_type: str,
        observed_at: datetime | None = None,
    ) -> SourceVersion: ...

    def get_source(self, source_id: str) -> SourceRecord: ...

    def get_version(self, source_version_id: str) -> SourceVersion: ...

    def read_content(self, source_version_id: str) -> bytes: ...

    def assert_evidence_span(self, span: EvidenceSpan) -> None: ...


class EvidenceGraph(Protocol):
    def add_evidence(self, span: EvidenceSpan) -> EvidenceSpan: ...

    def add_claim(self, claim: Claim) -> Claim: ...

    def add_relation(
        self,
        source_claim_id: str,
        target_claim_id: str,
        relation: ClaimRelationType,
    ) -> ClaimRelation: ...

    def explain_claim(self, claim_id: str) -> ClaimExplanation: ...


class PatternEngine(Protocol):
    def build(
        self,
        *,
        pattern_type: str,
        description: str,
        evidence_bundle: EvidenceBundle,
        occurrences: tuple[PatternOccurrence, ...],
        status: str = "candidate",
    ) -> PatternAnalysis: ...


class AttunementEngine(Protocol):
    def retrieve(
        self,
        *,
        query: str,
        reader_role: str,
        budget: int,
        candidates: tuple[AttunementCandidate, ...],
    ) -> AttunementBundle: ...


class Reader(Protocol):
    reader_id: str
    version: str

    def read(self, context: Any) -> Any: ...


class ModelProvider(Protocol):
    def generate(self, *, prompt: str, **kwargs: Any) -> Any: ...

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class KnowledgeProvider(Protocol):
    def search(self, query: str, *, limit: int = 10) -> Any: ...
