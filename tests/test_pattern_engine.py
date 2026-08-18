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
    days: int = 0,
    domains: tuple[str, ...] = (),
    level: EpistemicLevel = EpistemicLevel.OBSERVED,
    state: ValidationState = ValidationState.SUPPORTED,
    claim_type: str = "general",
    content_hash: str | None = None,
) -> None:
    graph.add_evidence(
        EvidenceSpan(
            evidence_id=evidence_id,
            source_id=source_id,
            source_version_id="v1",
            content_hash=content_hash or f"source-hash-{source_id}",
            excerpt=f"evidence {evidence_id}",
            observed_at=NOW + timedelta(days=days),
        )
    )
    graph.add_claim(
        Claim(
            claim_id=claim_id,
            statement=f"claim {claim_id}",
            claim_type=claim_type,
            producer="test",
            evidence_refs=(evidence_id,),
            domain_tags=domains,
            epistemic_level=level,
            validation_state=state,
        )
    )


def occurrence(
    occurrence_id: str,
    claim_id: str,
    evidence_id: str,
    *,
    dependency_group_id: str,
    confirmation_claim_id: str | None = None,
) -> PatternOccurrence:
    return PatternOccurrence(
        occurrence_id=occurrence_id,
        claim_id=claim_id,
        evidence_refs=(evidence_id,),
        dependency_group_id=dependency_group_id,
        confirmation_claim_id=confirmation_claim_id,
    )


def build_three_claim_graph() -> InMemoryEvidenceGraph:
    graph = InMemoryEvidenceGraph()
    add_claim(
        graph,
        claim_id="c1",
        evidence_id="e1",
        source_id="conversation-1",
        days=0,
        domains=("work",),
    )
    add_claim(
        graph,
        claim_id="c2",
        evidence_id="e2",
        source_id="conversation-2",
        days=90,
        domains=("relationships",),
    )
    add_claim(
        graph,
        claim_id="c3",
        evidence_id="e3",
        source_id="journal-1",
        days=180,
        domains=("health",),
    )
    return graph


def analyze(graph, occurrences):
    bundle = EvidenceBundleBuilder(graph).build(tuple(sorted({o.claim_id for o in occurrences})))
    return LongitudinalPatternEngine().build(
        pattern_type="recurrence",
        description="same structure repeats",
        evidence_bundle=bundle,
        occurrences=tuple(occurrences),
    )


def test_declared_dependency_group_prevents_duplicate_occurrences_inflating_recurrence():
    graph = build_three_claim_graph()
    analysis = analyze(
        graph,
        (
            occurrence("o1", "c1", "e1", dependency_group_id="episode-1"),
            occurrence("o2", "c2", "e2", dependency_group_id="episode-1"),
            occurrence("o3", "c3", "e3", dependency_group_id="episode-2"),
        ),
    )

    assert analysis.projection.recurrence_count == 2
    assert len(analysis.occurrence_groups) == 2
    assert analysis.projection.confidence_components.recurrence == 0.25


def test_same_source_cannot_be_split_into_fake_independent_groups():
    graph = InMemoryEvidenceGraph()
    add_claim(
        graph,
        claim_id="c1",
        evidence_id="e1",
        source_id="same-source",
        days=0,
        domains=("work",),
    )
    add_claim(
        graph,
        claim_id="c2",
        evidence_id="e2",
        source_id="same-source",
        days=120,
        domains=("life",),
    )
    analysis = analyze(
        graph,
        (
            occurrence("o1", "c1", "e1", dependency_group_id="fake-g1"),
            occurrence("o2", "c2", "e2", dependency_group_id="fake-g2"),
        ),
    )

    assert analysis.projection.recurrence_count == 1
    assert analysis.projection.confidence_components.recurrence == 0.0
    assert analysis.projection.confidence_components.temporal_spread == 0.0


def test_same_content_hash_under_alias_sources_is_still_dependent():
    graph = InMemoryEvidenceGraph()
    add_claim(
        graph,
        claim_id="c1",
        evidence_id="e1",
        source_id="alias-a",
        days=0,
        content_hash="same-source-version-hash",
    )
    add_claim(
        graph,
        claim_id="c2",
        evidence_id="e2",
        source_id="alias-b",
        days=90,
        content_hash="same-source-version-hash",
    )
    analysis = analyze(
        graph,
        (
            occurrence("o1", "c1", "e1", dependency_group_id="g1"),
            occurrence("o2", "c2", "e2", dependency_group_id="g2"),
        ),
    )

    assert analysis.projection.recurrence_count == 1


def test_source_diversity_counts_source_lineages_not_raw_evidence_refs():
    graph = InMemoryEvidenceGraph()
    add_claim(graph, claim_id="c1", evidence_id="e1", source_id="same-source")
    add_claim(graph, claim_id="c2", evidence_id="e2", source_id="same-source", days=20)
    analysis = analyze(
        graph,
        (
            occurrence("o1", "c1", "e1", dependency_group_id="g1"),
            occurrence("o2", "c2", "e2", dependency_group_id="g2"),
        ),
    )

    assert analysis.projection.confidence_components.evidence_diversity == 0.333333


def test_temporal_and_cross_domain_metrics_are_evidence_derived():
    graph = build_three_claim_graph()
    analysis = analyze(
        graph,
        (
            occurrence("o1", "c1", "e1", dependency_group_id="g1"),
            occurrence("o2", "c2", "e2", dependency_group_id="g2"),
            occurrence("o3", "c3", "e3", dependency_group_id="g3"),
        ),
    )

    components = analysis.projection.confidence_components
    assert components.temporal_spread == 1.0
    assert components.cross_domain_presence == 1.0
    assert analysis.projection.domain_tags == ("health", "relationships", "work")


def test_single_dependency_group_cannot_fake_cross_domain_recurrence():
    graph = build_three_claim_graph()
    analysis = analyze(
        graph,
        (
            occurrence("o1", "c1", "e1", dependency_group_id="one-episode"),
            occurrence("o2", "c2", "e2", dependency_group_id="one-episode"),
            occurrence("o3", "c3", "e3", dependency_group_id="one-episode"),
        ),
    )

    assert analysis.projection.recurrence_count == 1
    assert analysis.projection.confidence_components.cross_domain_presence == 0.333333
    assert analysis.projection.confidence_components.temporal_spread == 0.0


def test_evidence_quality_is_derived_from_claim_state_not_occurrence_input():
    graph = InMemoryEvidenceGraph()
    add_claim(
        graph,
        claim_id="supported",
        evidence_id="e1",
        source_id="s1",
        state=ValidationState.SUPPORTED,
    )
    add_claim(
        graph,
        claim_id="unverified",
        evidence_id="e2",
        source_id="s2",
        state=ValidationState.UNVERIFIED,
    )
    analysis = analyze(
        graph,
        (
            occurrence("o1", "supported", "e1", dependency_group_id="g1"),
            occurrence("o2", "unverified", "e2", dependency_group_id="g2"),
        ),
    )

    assert analysis.occurrence_groups[0].evidence_quality in {0.35, 0.8}
    assert analysis.projection.confidence_components.evidence_quality == 0.575


def test_missing_provenance_time_is_rejected():
    graph = InMemoryEvidenceGraph()
    graph.add_evidence(
        EvidenceSpan(
            evidence_id="e1",
            source_id="s1",
            source_version_id="v1",
            content_hash="h1",
            excerpt="undated",
        )
    )
    graph.add_claim(
        Claim(
            claim_id="c1",
            statement="claim",
            producer="test",
            evidence_refs=("e1",),
            epistemic_level=EpistemicLevel.OBSERVED,
            validation_state=ValidationState.SUPPORTED,
        )
    )

    with pytest.raises(InvalidPatternEvidenceError, match="observed_at"):
        analyze(graph, (occurrence("o1", "c1", "e1", dependency_group_id="g1"),))


def test_graph_contradiction_automatically_penalizes_pattern_confidence():
    clean_graph = build_three_claim_graph()
    clean = analyze(
        clean_graph,
        (
            occurrence("o1", "c1", "e1", dependency_group_id="g1"),
            occurrence("o2", "c2", "e2", dependency_group_id="g2"),
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
            occurrence("o1", "c1", "e1", dependency_group_id="g1"),
            occurrence("o2", "c2", "e2", dependency_group_id="g2"),
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
            occurrences=(occurrence("o1", "symbolic", "e1", dependency_group_id="g1"),),
        )


def test_occurrence_evidence_must_be_provenance_for_its_claim():
    graph = build_three_claim_graph()
    bundle = EvidenceBundleBuilder(graph).build(("c1", "c2"))

    with pytest.raises(InvalidPatternEvidenceError, match="not provenance"):
        LongitudinalPatternEngine().build(
            pattern_type="recurrence",
            description="bad wiring",
            evidence_bundle=bundle,
            occurrences=(occurrence("o1", "c1", "e2", dependency_group_id="g1"),),
        )


def test_user_confirmation_requires_evidence_backed_confirmation_claim_and_relation():
    graph = InMemoryEvidenceGraph()
    add_claim(
        graph,
        claim_id="c1",
        evidence_id="e1",
        source_id="conversation-1",
        domains=("work",),
    )
    add_claim(
        graph,
        claim_id="confirm-c1",
        evidence_id="ec1",
        source_id="confirmation-1",
        claim_type="user_confirmation",
        state=ValidationState.VALIDATED,
    )
    graph.add_relation("confirm-c1", "c1", ClaimRelationType.SUPPORTS)
    bundle = EvidenceBundleBuilder(graph).build(("c1",))
    analysis = LongitudinalPatternEngine().build(
        pattern_type="recurrence",
        description="confirmed pattern",
        evidence_bundle=bundle,
        occurrences=(
            occurrence(
                "o1",
                "c1",
                "e1",
                dependency_group_id="g1",
                confirmation_claim_id="confirm-c1",
            ),
        ),
    )

    assert analysis.projection.confidence_components.user_confirmation == 1.0


def test_unlinked_confirmation_claim_cannot_inflate_confidence():
    graph = InMemoryEvidenceGraph()
    add_claim(graph, claim_id="c1", evidence_id="e1", source_id="s1")
    add_claim(
        graph,
        claim_id="confirm",
        evidence_id="e2",
        source_id="s2",
        claim_type="user_confirmation",
        state=ValidationState.VALIDATED,
    )
    bundle = EvidenceBundleBuilder(graph).build(("c1", "confirm"))

    with pytest.raises(InvalidPatternEvidenceError, match="explicitly support"):
        LongitudinalPatternEngine().build(
            pattern_type="recurrence",
            description="fake confirmation",
            evidence_bundle=bundle,
            occurrences=(
                occurrence(
                    "o1",
                    "c1",
                    "e1",
                    dependency_group_id="g1",
                    confirmation_claim_id="confirm",
                ),
            ),
        )


def test_analysis_fingerprint_is_deterministic():
    graph = build_three_claim_graph()
    occurrences = (
        occurrence("o1", "c1", "e1", dependency_group_id="g1"),
        occurrence("o2", "c2", "e2", dependency_group_id="g2"),
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
            occurrence("o1", "c1", "e1", dependency_group_id="g1"),
            occurrence("o2", "c2", "e2", dependency_group_id="g2"),
            occurrence("o3", "c3", "e3", dependency_group_id="g3"),
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
