"""Provider-agnostic core interfaces."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

from .attunement import AttunementBundle, AttunementCandidate
from .domain import Claim, EngineEvent, EpistemicLevel, EvidenceSpan


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


class EvidenceGraph(Protocol):
    def add_evidence(self, span: EvidenceSpan) -> EvidenceSpan: ...

    def add_claim(self, claim: Claim) -> Claim: ...


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
