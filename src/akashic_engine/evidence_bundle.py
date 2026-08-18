"""Replayable evidence snapshots built from the authoritative EvidenceGraph."""

from __future__ import annotations

import hashlib
import json
from typing import Protocol

from pydantic import Field

from .domain import Claim, EvidenceSpan, FrozenModel
from .evidence_graph import ClaimExplanation, ClaimRelation


class EvidenceBundleError(RuntimeError):
    pass


class DuplicateBundleRequestError(EvidenceBundleError):
    pass


class EvidenceGraphReader(Protocol):
    def get_claim(self, claim_id: str) -> Claim: ...

    def explain_claim(self, claim_id: str) -> ClaimExplanation: ...


class EvidenceBundleItem(FrozenModel):
    claim: Claim
    evidence: tuple[EvidenceSpan, ...] = ()
    incoming_relations: tuple[ClaimRelation, ...] = ()
    outgoing_relations: tuple[ClaimRelation, ...] = ()


class EvidenceBundle(FrozenModel):
    """Immutable evidence snapshot suitable for replay and downstream scoring."""

    bundle_version: str = "1"
    requested_claim_ids: tuple[str, ...]
    items: tuple[EvidenceBundleItem, ...]
    evidence: tuple[EvidenceSpan, ...]
    relations: tuple[ClaimRelation, ...]
    fingerprint: str = Field(min_length=64, max_length=64)

    def claim_map(self) -> dict[str, Claim]:
        return {item.claim.claim_id: item.claim for item in self.items}

    def evidence_map(self) -> dict[str, EvidenceSpan]:
        return {span.evidence_id: span for span in self.evidence}

    def relation_set(self) -> set[ClaimRelation]:
        return set(self.relations)


class EvidenceBundleBuilder:
    """Build a deterministic, closed one-hop provenance snapshot from a graph."""

    def __init__(self, graph: EvidenceGraphReader) -> None:
        self.graph = graph

    def build(
        self,
        claim_ids: tuple[str, ...],
        *,
        include_relation_context: bool = True,
    ) -> EvidenceBundle:
        if not claim_ids:
            raise EvidenceBundleError("At least one claim is required")
        if len(set(claim_ids)) != len(claim_ids):
            raise DuplicateBundleRequestError("Duplicate requested claim IDs are not allowed")

        requested = tuple(sorted(claim_ids))
        requested_explanations = {
            claim_id: self.graph.explain_claim(claim_id) for claim_id in requested
        }

        boundary_relations: set[ClaimRelation] = set()
        for explanation in requested_explanations.values():
            boundary_relations.update(explanation.incoming_relations)
            boundary_relations.update(explanation.outgoing_relations)

        claim_context_ids = set(requested)
        if include_relation_context:
            for edge in boundary_relations:
                claim_context_ids.add(edge.source_claim_id)
                claim_context_ids.add(edge.target_claim_id)

        explanations = {
            claim_id: self.graph.explain_claim(claim_id)
            for claim_id in sorted(claim_context_ids)
        }

        # Freeze the one-hop node set first, then retain only edges whose two
        # endpoints are present. This prevents dangling second-hop relations.
        closed_relations: set[ClaimRelation] = set()
        for explanation in explanations.values():
            for edge in (*explanation.incoming_relations, *explanation.outgoing_relations):
                if (
                    edge.source_claim_id in claim_context_ids
                    and edge.target_claim_id in claim_context_ids
                ):
                    closed_relations.add(edge)

        items: list[EvidenceBundleItem] = []
        evidence_by_id: dict[str, EvidenceSpan] = {}
        for claim_id in sorted(claim_context_ids):
            explanation = explanations[claim_id]
            incoming = tuple(
                sorted(
                    (
                        edge
                        for edge in explanation.incoming_relations
                        if edge in closed_relations
                    ),
                    key=self._relation_sort_key,
                )
            )
            outgoing = tuple(
                sorted(
                    (
                        edge
                        for edge in explanation.outgoing_relations
                        if edge in closed_relations
                    ),
                    key=self._relation_sort_key,
                )
            )
            item = EvidenceBundleItem(
                claim=explanation.claim,
                evidence=tuple(sorted(explanation.evidence, key=lambda span: span.evidence_id)),
                incoming_relations=incoming,
                outgoing_relations=outgoing,
            )
            items.append(item)
            for span in item.evidence:
                evidence_by_id[span.evidence_id] = span

        sorted_relations = tuple(sorted(closed_relations, key=self._relation_sort_key))
        sorted_evidence = tuple(evidence_by_id[key] for key in sorted(evidence_by_id))
        fingerprint = self._fingerprint(
            requested_claim_ids=requested,
            items=tuple(items),
            evidence=sorted_evidence,
            relations=sorted_relations,
        )
        return EvidenceBundle(
            requested_claim_ids=requested,
            items=tuple(items),
            evidence=sorted_evidence,
            relations=sorted_relations,
            fingerprint=fingerprint,
        )

    @staticmethod
    def _relation_sort_key(edge: ClaimRelation) -> tuple[str, str, str]:
        return (edge.relation.value, edge.source_claim_id, edge.target_claim_id)

    @staticmethod
    def _fingerprint(
        *,
        requested_claim_ids: tuple[str, ...],
        items: tuple[EvidenceBundleItem, ...],
        evidence: tuple[EvidenceSpan, ...],
        relations: tuple[ClaimRelation, ...],
    ) -> str:
        payload = {
            "bundle_version": "1",
            "requested_claim_ids": requested_claim_ids,
            "items": [item.model_dump(mode="json") for item in items],
            "evidence": [span.model_dump(mode="json") for span in evidence],
            "relations": [edge.model_dump(mode="json") for edge in relations],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
