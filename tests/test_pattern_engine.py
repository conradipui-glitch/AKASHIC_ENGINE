from datetime import datetime, timedelta, timezone

import pytest

from akashic_engine import (
    Claim,
    ClaimRelationType,
    EpistemicLevel,
    EvidenceBundleBuilder,
    EvidenceSpan,
    InMemoryEvidenceGraph,
    LongitudinalPatternEngine,
    PatternAttunementSignals,
    PatternOccurrence,
    ValidationState,
)
from akashic_engine.pattern_engine import (
    EpistemicPatternBoundaryError,
    InvalidPatternEvidenceError,
)


NOW = datetime(2026, 8, 19, tzinfo=timezone.utc)


def add_claim(
    graph: InMemoryEvidenceGraph,
    *,
    claim_id: str,
    evidence_id: str,
    source_id: str,
    level: EpistemicLevel = EpistemicLevel.OBSERVED,
) -> None:
    graph.add_evidence(
        EvidenceSpan(
            evidence_id=evidence_id,
            source_id=source_id,
            source_version_id="v1",
            content_hash=f"hash-{evidence_id}",
            excerpt=f"evidence {evidence_id}",
        )
    )
    graph.add_claim(
        Claim(
            claim_id=claim_id,
            statement=f"claim {claim_id}",
            producer="test",
            evidence_refs=(evidence_id,),
            epistemic_level=level,
            validation_state=ValidationState.SUPPORTED,
        )
    )


def occurrence(
    occurrence_id: str,
    claim_id: str,
    evidence_id: str,
    *,
    days: int,
    domain: str,
    dependency_group_id: str,
    quality: float = 0.8,
    confirmed: bool = False,
) -> PatternOccurrence:
    return PatternOccurrence(
        occurrence_id=occurrence_id,
        claim_id=claim_id,
        evidence_refs=(evidence_id,),
        observed_at=NOW + timedelta(days=days),
        domain=domain,
        dependency_group_id=dependency_group_id,
        evidence_quality=quality,
        user_confirmed=confirmed,
    )


def build_three_claim_graph() -> InMemoryEvidenceGraph:
    graph = InMemoryEvidenceGraph()
    add_claim(graph, claim_id="c1", evidence_id="e1", source_id="conversation-1")
    add_claim(graph, claim_id="c2", evidence_id="e2", source_id="conversation-2")
    add_claim(graph, claim_id="c3", evidence_id="e3", source_id="journal-1")
    return graph


def analyze(graph, occurrences):
    bundle = EvidenceBundleBuilder(graph).build(tuple(sorted({o.claim_id for o in occurrences})))
    return LongitudinalPatternEngine().build(
        pattern_type="recurrence",
        description="same structure repeats",
        evidence_bundle=bundle,
        occurrences=tuple(occurrences),
    )


def test_dependency_group_prevents_duplicate_occurrences_inflating_recurrence():
    graph = build_three_claim_graph()
    analysis = analyze(
        graph,
        (
            occurrence("o1", "c1", "e1", days=0, domain="work", dependency_group_id="episode-1"),
            occurrence("o2", "c2", "e2", days=1, domain="work", dependency_group_id="episode-1"),
            occurrence("o3", "c3", "e3", days=30, domain="life", dependency_group_id="episode-2"),
        ),
    )

    assert analysis.projection.recurrence_count == 2
    assert len(analysis.occurrence_groups) == 2
    assert analysis.projection.confidence_components.recurrence == 0.25


def test_source_diversity_counts_source_lineages_not_raw_evidence_refs():
    graph = InMemoryEvidenceGraph()
    add_claim(graph, claim_id="c1", evidence_id="e1", source_id="same-source")
    add_claim(graph, claim_id="c2", evidence_id="e2", source_id="same-source")
    analysis = analyze(
        graph,
        (
            occurrence("o1", "c1", "e1", days=0, domain="work", dependency_group_id="g1"),
            occurrence("o2", "c2", "e2", days=20, domain="life", dependency_group_id="g2"),
        ),
    )

    assert analysis.projection.confidence_components.evidence_diversity == 0.333333


def test_temporal_and_cross_domain_metrics_are_evidence_derived():
    graph = build_three_claim_graph()
    analysis = analyze(
        graph,
        (
            occurrence("o1", "c1", "e1", days=0, domain="work", dependency_group_id="g1"),
            occurrence("o2", "c2", "e2", days=90, domain="relationships", dependency_group_id="g2"),
            occurrence("o3", "c3", "e3", days=180, domain="health", dependency_group_id="g3"),
        ),
    )

    components = analysis.projection.confidence_components
    assert components.temporal_spread == 1.0
    assert components.cross_domain_presence == 1.0
    assert analysis.projection.domain_tags == ("health", "relationships", "work")


def test_graph_contradiction_automatically_penalizes_pattern_confidence():
    clean_graph = build_three_claim_graph()
    clean = analyze(
        clean_graph,
        (
            occurrence("o1", "c1", "e1", days=0, domain="work", dependency_group_id="g1"),
            occurrence("o2", "c2", "e2", days=30, domain="life", dependency_group_id="g2"),
        ),
    )

    contradicted_graph = build_three_claim_graph()
    contradicted_graph.add_relation("c3", "c1", ClaimRelationType.CONTRADICTS)
    bundle = EvidenceBundleBuilder(contradicted_graph).build(("c1", "c2"))
    contradicted = LongitudinalPatternEngine().build(
        pattern_type="recurrence",
        description="same structure repeats",
        evidence_bundle=bundle,
        occurrences=(
            occurrence("o1", "c1", "e1", days=0, domain="work", dependency_group_id="g1"),
            occurrence("o2", "c2", "e2", days=30, domain="life", dependency_group_id="g2"),
        ),
    )

    assert contradicted.projection.confidence_components.contradiction == 0.5
    assert contradicted.projection.confidence < clean.projection.confidence
    assert contradicted.projection.contradictions == ("c3",)


def test_symbolic_hypothesis_cannot_become_factual_pattern_evidence():
    graph = InMemoryEvidenceGraph()
    add_claim(
        graph,
        claim_id="symbolic",
        evidence_id="e1",
        source_id="reading-1",
        level=EpistemicLevel.SYMBOLIC_HYPOTHESIS,
    )
    bundle = EvidenceBundleBuilder(graph).build(("symbolic",))

    with pytest.raises(EpistemicPatternBoundaryError):
        LongitudinalPatternEngine().build(
            pattern_type="recurrence",
            description="symbolic pattern",
            evidence_bundle=bundle,
            occurrences=(
                occurrence(
                    "o1",
                    "symbolic",
                    "e1",
                    days=0,
                    domain="symbolic",
                    dependency_group_id="g1",
                ),
            ),
        )


def test_occurrence_evidence_must_be_provenance_for_its_claim():
    graph = build_three_claim_graph()
    bundle = EvidenceBundleBuilder(graph).build(("c1", "c2"))

    with pytest.raises(InvalidPatternEvidenceError, match="not provenance"):
        LongitudinalPatternEngine().build(
            pattern_type="recurrence",
            description="bad wiring",
            evidence_bundle=bundle,
            occurrences=(
                occurrence("o1", "c1", "e2", days=0, domain="work", dependency_group_id="g1"),
            ),
        )


def test_user_confirmation_counts_per_dependency_group_not_per_occurrence():
    graph = build_three_claim_graph()
    analysis = analyze(
        graph,
        (
            occurrence("o1", "c1", "e1", days=0, domain="work", dependency_group_id="g1", confirmed=True),
            occurrence("o2", "c2", "e2", days=1, domain="work", dependency_group_id="g1", confirmed=True),
            occurrence("o3", "c3", "e3", days=30, domain="life", dependency_group_id="g2"),
        ),
    )

    assert analysis.projection.confidence_components.user_confirmation == 0.5


def test_analysis_fingerprint_is_deterministic():
    graph = build_three_claim_graph()
    occurrences = (
        occurrence("o1", "c1", "e1", days=0, domain="work", dependency_group_id="g1"),
        occurrence("o2", "c2", "e2", days=30, domain="life", dependency_group_id="g2"),
    )
    first = analyze(graph, occurrences)
    second = LongitudinalPatternEngine().build(
        pattern_type="recurrence",
        description="same structure repeats",
        evidence_bundle=EvidenceBundleBuilder(graph).build(("c1", "c2")),
        occurrences=occurrences,
    )

    assert first.projection.pattern_id == second.projection.pattern_id
    assert first.evidence_bundle_fingerprint == second.evidence_bundle_fingerprint
    assert first.analysis_fingerprint == second.analysis_fingerprint


def test_pattern_to_attunement_bridge_uses_derived_pattern_features():
    graph = build_three_claim_graph()
    analysis = analyze(
        graph,
        (
            occurrence("o1", "c1", "e1", days=0, domain="work", dependency_group_id="g1"),
            occurrence("o2", "c2", "e2", days=30, domain="life", dependency_group_id="g2"),
            occurrence("o3", "c3", "e3", days=60, domain="health", dependency_group_id="g3"),
        ),
    )
    candidate = analysis.to_attunement_candidate(
        PatternAttunementSignals(
            semantic_similarity=0.9,
            emotional_significance=0.4,
            unresolvedness=0.7,
            temporal_relevance=0.8,
            user_importance=0.6,
        )
    )

    components = analysis.projection.confidence_components
    assert candidate.epistemic_level is EpistemicLevel.INTERPRETATION
    assert candidate.features.recurrence == components.recurrence
    assert candidate.features.cross_domain_presence == components.cross_domain_presence
    assert candidate.features.contradiction == components.contradiction
    assert candidate.evidence_refs == analysis.projection.evidence_refs
