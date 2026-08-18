"""Authoritative evidence graph for source-grounded claims."""

from __future__ import annotations

from enum import StrEnum
from threading import RLock

from pydantic import Field

from .domain import Claim, EpistemicLevel, EvidenceSpan, FrozenModel, ValidationState


class EvidenceGraphError(RuntimeError):
    pass


class DuplicateGraphNodeError(EvidenceGraphError):
    pass


class UnknownGraphNodeError(EvidenceGraphError):
    pass


class InvalidGraphRelationError(EvidenceGraphError):
    pass


class ClaimRelationType(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"


class ClaimRelation(FrozenModel):
    source_claim_id: str
    target_claim_id: str
    relation: ClaimRelationType


class ClaimExplanation(FrozenModel):
    claim: Claim
    evidence: tuple[EvidenceSpan, ...] = ()
    incoming_relations: tuple[ClaimRelation, ...] = ()
    outgoing_relations: tuple[ClaimRelation, ...] = ()


class InMemoryEvidenceGraph:
    """Reference graph that keeps evidence and claim relations explicit.

    This graph is authoritative for provenance. Semantic/concept links belong in
    a separate derived ConceptGraph and must never be treated as evidence.
    """

    def __init__(self) -> None:
        self._evidence: dict[str, EvidenceSpan] = {}
        self._evidence_identity: dict[tuple[object, ...], str] = {}
        self._claims: dict[str, Claim] = {}
        self._relations: set[ClaimRelation] = set()
        self._lock = RLock()

    def add_evidence(self, span: EvidenceSpan) -> EvidenceSpan:
        with self._lock:
            if span.evidence_id in self._evidence:
                raise DuplicateGraphNodeError(f"Duplicate evidence_id: {span.evidence_id}")
            identity = self._evidence_identity_key(span)
            duplicate_id = self._evidence_identity.get(identity)
            if duplicate_id is not None:
                raise DuplicateGraphNodeError(
                    f"Duplicate evidence span: {span.evidence_id} duplicates {duplicate_id}"
                )
            self._evidence[span.evidence_id] = span
            self._evidence_identity[identity] = span.evidence_id
            return span

    def add_claim(self, claim: Claim) -> Claim:
        with self._lock:
            if claim.claim_id in self._claims:
                raise DuplicateGraphNodeError(f"Duplicate claim_id: {claim.claim_id}")

            if len(set(claim.evidence_refs)) != len(claim.evidence_refs):
                raise InvalidGraphRelationError(
                    "A claim cannot count the same evidence reference more than once"
                )

            missing = tuple(ref for ref in claim.evidence_refs if ref not in self._evidence)
            if missing:
                raise UnknownGraphNodeError(
                    f"Claim references unknown evidence: {', '.join(missing)}"
                )
            if claim.validation_state in {ValidationState.SUPPORTED, ValidationState.VALIDATED} and not claim.evidence_refs:
                raise InvalidGraphRelationError(
                    "Supported/validated claims require at least one evidence reference"
                )

            self._claims[claim.claim_id] = claim
            return claim

    def add_relation(
        self,
        source_claim_id: str,
        target_claim_id: str,
        relation: ClaimRelationType,
    ) -> ClaimRelation:
        with self._lock:
            if source_claim_id == target_claim_id:
                raise InvalidGraphRelationError("A claim cannot support/contradict itself")
            source = self._require_claim(source_claim_id)
            target = self._require_claim(target_claim_id)
            if self._epistemic_rank(source.epistemic_level) > self._epistemic_rank(
                target.epistemic_level
            ):
                raise InvalidGraphRelationError(
                    "A weaker epistemic claim cannot authoritatively support/contradict a stronger claim"
                )
            edge = ClaimRelation(
                source_claim_id=source_claim_id,
                target_claim_id=target_claim_id,
                relation=relation,
            )
            self._relations.add(edge)
            return edge

    def get_claim(self, claim_id: str) -> Claim:
        with self._lock:
            return self._require_claim(claim_id)

    def get_evidence(self, evidence_id: str) -> EvidenceSpan:
        with self._lock:
            try:
                return self._evidence[evidence_id]
            except KeyError as exc:
                raise UnknownGraphNodeError(evidence_id) from exc

    def evidence_for_claim(self, claim_id: str) -> tuple[EvidenceSpan, ...]:
        with self._lock:
            claim = self._require_claim(claim_id)
            return tuple(self._evidence[ref] for ref in claim.evidence_refs)

    def explain_claim(self, claim_id: str) -> ClaimExplanation:
        with self._lock:
            claim = self._require_claim(claim_id)
            incoming = tuple(
                sorted(
                    (edge for edge in self._relations if edge.target_claim_id == claim_id),
                    key=self._relation_sort_key,
                )
            )
            outgoing = tuple(
                sorted(
                    (edge for edge in self._relations if edge.source_claim_id == claim_id),
                    key=self._relation_sort_key,
                )
            )
            return ClaimExplanation(
                claim=claim,
                evidence=tuple(self._evidence[ref] for ref in claim.evidence_refs),
                incoming_relations=incoming,
                outgoing_relations=outgoing,
            )

    def _require_claim(self, claim_id: str) -> Claim:
        try:
            return self._claims[claim_id]
        except KeyError as exc:
            raise UnknownGraphNodeError(claim_id) from exc

    @staticmethod
    def _evidence_identity_key(span: EvidenceSpan) -> tuple[object, ...]:
        return (
            span.source_id,
            span.source_version_id,
            span.content_hash,
            span.start_offset,
            span.end_offset,
            span.excerpt,
        )

    @staticmethod
    def _epistemic_rank(level: EpistemicLevel) -> int:
        return {
            EpistemicLevel.OBSERVED: 0,
            EpistemicLevel.INTERPRETATION: 1,
            EpistemicLevel.SYMBOLIC_HYPOTHESIS: 2,
        }[level]

    @staticmethod
    def _relation_sort_key(edge: ClaimRelation) -> tuple[str, str, str]:
        return (edge.relation.value, edge.source_claim_id, edge.target_claim_id)
