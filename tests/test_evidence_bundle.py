import pytest

from akashic_engine import (
    Claim,
    ClaimRelationType,
    EpistemicLevel,
    EvidenceBundleBuilder,
    EvidenceSpan,
    InMemoryEvidenceGraph,
    ValidationState,
)
from akashic_engine.evidence_bundle import DuplicateBundleRequestError


def span(evidence_id: str, *, source_id: str, excerpt: str) -> EvidenceSpan:
    return EvidenceSpan(
        evidence_id=evidence_id,
        source_id=source_id,
        source_version_id="v1",
        content_hash=f"hash-{evidence_id}",
        excerpt=excerpt,
    )


def claim(claim_id: str, evidence_id: str, *, statement: str | None = None) -> Claim:
    return Claim(
        claim_id=claim_id,
        statement=statement or f"statement {claim_id}",
        producer="test",
        evidence_refs=(evidence_id,),
        epistemic_level=EpistemicLevel.OBSERVED,
        validation_state=ValidationState.SUPPORTED,
    )


def graph_with_relation() -> InMemoryEvidenceGraph:
    graph = InMemoryEvidenceGraph()
    graph.add_evidence(span("e1", source_id="conversation-1", excerpt="first"))
    graph.add_evidence(span("e2", source_id="conversation-2", excerpt="second"))
    graph.add_claim(claim("c1", "e1"))
    graph.add_claim(claim("c2", "e2"))
    graph.add_relation("c2", "c1", ClaimRelationType.CONTRADICTS)
    return graph


def test_bundle_fingerprint_is_deterministic_and_request_order_independent():
    graph = graph_with_relation()
    builder = EvidenceBundleBuilder(graph)

    first = builder.build(("c1", "c2"))
    second = builder.build(("c2", "c1"))

    assert first.fingerprint == second.fingerprint
    assert first.requested_claim_ids == ("c1", "c2")


def test_bundle_includes_one_hop_relation_context_and_its_evidence():
    bundle = EvidenceBundleBuilder(graph_with_relation()).build(("c1",))

    assert set(bundle.claim_map()) == {"c1", "c2"}
    assert set(bundle.evidence_map()) == {"e1", "e2"}
    assert len(bundle.relations) == 1
    assert bundle.relations[0].relation is ClaimRelationType.CONTRADICTS


def test_bundle_without_relation_context_has_no_dangling_relation_edges():
    bundle = EvidenceBundleBuilder(graph_with_relation()).build(
        ("c1",), include_relation_context=False
    )

    assert set(bundle.claim_map()) == {"c1"}
    assert set(bundle.evidence_map()) == {"e1"}
    assert bundle.relations == ()


def test_one_hop_bundle_is_closed_and_does_not_leak_second_hop_edges():
    graph = InMemoryEvidenceGraph()
    for index in range(1, 4):
        graph.add_evidence(
            span(f"e{index}", source_id=f"source-{index}", excerpt=f"e{index}")
        )
        graph.add_claim(claim(f"c{index}", f"e{index}"))
    graph.add_relation("c2", "c1", ClaimRelationType.SUPPORTS)
    graph.add_relation("c3", "c2", ClaimRelationType.SUPPORTS)

    bundle = EvidenceBundleBuilder(graph).build(("c1",))

    assert set(bundle.claim_map()) == {"c1", "c2"}
    assert {
        (edge.source_claim_id, edge.target_claim_id) for edge in bundle.relations
    } == {("c2", "c1")}
    assert all(
        edge.source_claim_id in bundle.claim_map()
        and edge.target_claim_id in bundle.claim_map()
        for edge in bundle.relations
    )


def test_duplicate_requested_claims_are_rejected():
    builder = EvidenceBundleBuilder(graph_with_relation())
    with pytest.raises(DuplicateBundleRequestError):
        builder.build(("c1", "c1"))


def test_bundle_fingerprint_changes_when_provenance_changes():
    first_graph = InMemoryEvidenceGraph()
    first_graph.add_evidence(span("e1", source_id="source-a", excerpt="alpha"))
    first_graph.add_claim(claim("c1", "e1"))

    second_graph = InMemoryEvidenceGraph()
    second_graph.add_evidence(span("e1", source_id="source-b", excerpt="alpha"))
    second_graph.add_claim(claim("c1", "e1"))

    first = EvidenceBundleBuilder(first_graph).build(("c1",))
    second = EvidenceBundleBuilder(second_graph).build(("c1",))

    assert first.fingerprint != second.fingerprint
