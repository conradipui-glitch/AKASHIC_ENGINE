import pytest

from akashic_engine import EpistemicLevel
from akashic_engine.attunement import (
    AttunementCandidate,
    AttunementFeatures,
    ExplainableAttunementEngine,
)


def features(**overrides: float) -> AttunementFeatures:
    values = dict(
        semantic_similarity=0.7,
        recurrence=0.5,
        emotional_significance=0.4,
        unresolvedness=0.4,
        temporal_relevance=0.5,
        cross_domain_presence=0.3,
        user_importance=0.5,
        evidence_strength=0.7,
        contradiction=0.0,
    )
    values.update(overrides)
    return AttunementFeatures(**values)


def candidate(
    candidate_id: str,
    *,
    level: EpistemicLevel = EpistemicLevel.OBSERVED,
    source_seq: int | None = None,
    **feature_overrides: float,
) -> AttunementCandidate:
    return AttunementCandidate(
        candidate_id=candidate_id,
        epistemic_level=level,
        features=features(**feature_overrides),
        evidence_refs=(f"e-{candidate_id}",),
        source_seq=source_seq,
    )


def test_score_breakdown_is_reproducible_and_sums_to_positive_score():
    engine = ExplainableAttunementEngine()
    result = engine.score_candidate(candidate("a"))

    assert result.score == engine.score_candidate(candidate("a")).score
    assert round(sum(item.contribution for item in result.components), 6) == result.positive_score
    assert {item.name for item in result.components} == {
        "semantic_similarity",
        "recurrence",
        "emotional_significance",
        "unresolvedness",
        "temporal_relevance",
        "cross_domain_presence",
        "user_importance",
        "evidence_strength",
    }


def test_contradiction_penalty_is_visible_and_lowers_score():
    engine = ExplainableAttunementEngine()
    clean = engine.score_candidate(candidate("clean", contradiction=0.0))
    contradicted = engine.score_candidate(candidate("bad", contradiction=0.8))

    assert contradicted.contradiction_penalty > 0
    assert contradicted.score < clean.score


def test_pattern_role_prefers_stronger_recurrence_when_other_features_match():
    engine = ExplainableAttunementEngine()
    low = candidate("low", recurrence=0.1)
    high = candidate("high", recurrence=0.9)

    bundle = engine.retrieve(
        query="what keeps repeating?",
        reader_role="pattern",
        budget=2,
        candidates=(low, high),
    )

    assert [item.candidate_id for item in bundle.results] == ["high", "low"]


def test_skeptic_filters_symbolic_hypotheses_but_symbolic_reader_can_see_them():
    engine = ExplainableAttunementEngine()
    observed = candidate("observed", level=EpistemicLevel.OBSERVED)
    symbolic = candidate(
        "symbolic",
        level=EpistemicLevel.SYMBOLIC_HYPOTHESIS,
        semantic_similarity=1.0,
    )

    skeptic = engine.retrieve(
        query="challenge this",
        reader_role="skeptic",
        budget=5,
        candidates=(observed, symbolic),
    )
    symbolic_view = engine.retrieve(
        query="symbolic reading",
        reader_role="symbolic",
        budget=5,
        candidates=(observed, symbolic),
    )

    assert [item.candidate_id for item in skeptic.results] == ["observed"]
    assert {item.candidate_id for item in symbolic_view.results} == {"observed", "symbolic"}


def test_budget_and_tie_breaking_are_deterministic():
    engine = ExplainableAttunementEngine()
    older = candidate("b", source_seq=1)
    newer = candidate("z", source_seq=2)
    same_seq_alpha = candidate("a", source_seq=2)

    bundle = engine.retrieve(
        query="q",
        reader_role="pattern",
        budget=2,
        candidates=(older, newer, same_seq_alpha),
    )

    assert [item.candidate_id for item in bundle.results] == ["a", "z"]


def test_budget_must_be_positive():
    engine = ExplainableAttunementEngine()
    with pytest.raises(ValueError, match="budget"):
        engine.retrieve(
            query="q",
            reader_role="pattern",
            budget=0,
            candidates=(),
        )
