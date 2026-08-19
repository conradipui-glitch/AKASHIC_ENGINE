import pytest
from pydantic import ValidationError

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
    level: EpistemicLevel = EpistemicLevel.OBSERVED,
) -> Claim:
    return Claim(
        claim_id=claim_id,
        statement=f"claim {claim_id}",
        producer="test",
        evidence_refs=evidence_refs,
        epistemic_level=level,
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


def test_exact_duplicate_evidence_span_cannot_inflate_graph():
    graph = InMemoryEvidenceGraph()
    graph.add_evidence(span("e1"))

    with pytest.raises(DuplicateGraphNodeError, match="duplicates e1"):
        graph.add_evidence(span("e2"))


def test_claim_cannot_count_same_evidence_reference_twice():
    graph = InMemoryEvidenceGraph()
    graph.add_evidence(span("e1"))

    with pytest.raises(InvalidGraphRelationError, match="more than once"):
        graph.add_claim(
            claim(
                "c1",
                evidence_refs=("e1", "e1"),
                state=ValidationState.SUPPORTED,
            )
        )


def test_weaker_epistemic_claim_cannot_authoritatively_relate_to_stronger_claim():
    graph = InMemoryEvidenceGraph()
    graph.add_claim(claim("observed", level=EpistemicLevel.OBSERVED))
    graph.add_claim(
        claim("symbolic", level=EpistemicLevel.SYMBOLIC_HYPOTHESIS)
    )

    with pytest.raises(InvalidGraphRelationError, match="weaker epistemic"):
        graph.add_relation(
            "symbolic",
            "observed",
            ClaimRelationType.SUPPORTS,
        )


def test_claim_relation_fields_are_not_a_second_source_of_truth():
    with pytest.raises(ValidationError):
        Claim(
            claim_id="c1",
            statement="claim",
            producer="test",
            epistemic_level=EpistemicLevel.OBSERVED,
            supports=("c2",),
        )


def test_derived_from_points_from_weaker_claim_to_stronger_provenance():
    graph = InMemoryEvidenceGraph()
    graph.add_claim(claim("observed", level=EpistemicLevel.OBSERVED))
    graph.add_claim(claim("interpretation", level=EpistemicLevel.INTERPRETATION))

    edge = graph.add_relation(
        "interpretation",
        "observed",
        ClaimRelationType.DERIVED_FROM,
    )

    assert edge.source_claim_id == "interpretation"
    assert edge.target_claim_id == "observed"


def test_derived_from_rejects_inverted_epistemic_direction():
    graph = InMemoryEvidenceGraph()
    graph.add_claim(claim("observed", level=EpistemicLevel.OBSERVED))
    graph.add_claim(claim("interpretation", level=EpistemicLevel.INTERPRETATION))

    with pytest.raises(InvalidGraphRelationError, match="DERIVED_FROM"):
        graph.add_relation(
            "observed",
            "interpretation",
            ClaimRelationType.DERIVED_FROM,
        )
