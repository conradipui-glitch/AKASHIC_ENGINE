import pytest

from akashic_engine import Claim, EpistemicLevel, EvidenceSpan, ValidationState
from akashic_engine.evidence_graph import (
    ClaimRelationType,
    DuplicateGraphNodeError,
    InMemoryEvidenceGraph,
    InvalidGraphRelationError,
    UnknownGraphNodeError,
)


def span(evidence_id: str = "e1") -> EvidenceSpan:
    return EvidenceSpan(
        evidence_id=evidence_id,
        source_id="conversation-1",
        source_version_id="v1",
        content_hash="abc",
        excerpt="User described the event.",
    )


def claim(
    claim_id: str,
    *,
    evidence_refs: tuple[str, ...] = (),
    state: ValidationState = ValidationState.UNVERIFIED,
) -> Claim:
    return Claim(
        claim_id=claim_id,
        statement=f"claim {claim_id}",
        producer="test",
        evidence_refs=evidence_refs,
        epistemic_level=EpistemicLevel.OBSERVED,
        validation_state=state,
    )


def test_claim_explanation_keeps_exact_evidence_provenance():
    graph = InMemoryEvidenceGraph()
    evidence = graph.add_evidence(span())
    stored_claim = graph.add_claim(
        claim("c1", evidence_refs=(evidence.evidence_id,), state=ValidationState.SUPPORTED)
    )

    explanation = graph.explain_claim(stored_claim.claim_id)

    assert explanation.claim == stored_claim
    assert explanation.evidence == (evidence,)


def test_claim_rejects_unknown_evidence_reference():
    graph = InMemoryEvidenceGraph()

    with pytest.raises(UnknownGraphNodeError, match="missing"):
        graph.add_claim(claim("c1", evidence_refs=("missing",)))


def test_supported_claim_requires_evidence():
    graph = InMemoryEvidenceGraph()

    with pytest.raises(InvalidGraphRelationError, match="require"):
        graph.add_claim(claim("c1", state=ValidationState.SUPPORTED))


def test_support_and_contradiction_relations_are_explicit():
    graph = InMemoryEvidenceGraph()
    graph.add_claim(claim("c1"))
    graph.add_claim(claim("c2"))
    graph.add_claim(claim("c3"))

    graph.add_relation("c2", "c1", ClaimRelationType.SUPPORTS)
    graph.add_relation("c3", "c1", ClaimRelationType.CONTRADICTS)

    explanation = graph.explain_claim("c1")
    assert {edge.relation for edge in explanation.incoming_relations} == {
        ClaimRelationType.SUPPORTS,
        ClaimRelationType.CONTRADICTS,
    }


def test_claim_cannot_relate_to_itself_and_duplicate_nodes_fail():
    graph = InMemoryEvidenceGraph()
    graph.add_claim(claim("c1"))

    with pytest.raises(InvalidGraphRelationError, match="itself"):
        graph.add_relation("c1", "c1", ClaimRelationType.SUPPORTS)

    with pytest.raises(DuplicateGraphNodeError):
        graph.add_claim(claim("c1"))
