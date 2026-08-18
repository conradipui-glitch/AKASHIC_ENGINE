from datetime import datetime, timezone

import pytest

from akashic_engine import (
    EpistemicLevel,
    MethodManifest,
    PatternConfidenceComponents,
    PatternProjection,
)


def test_pattern_confidence_is_deterministic_and_penalizes_contradiction():
    clean = PatternConfidenceComponents(
        evidence_quality=0.9,
        evidence_diversity=0.8,
        recurrence=0.9,
        temporal_spread=0.7,
        cross_domain_presence=0.6,
        user_confirmation=0.5,
        contradiction=0.0,
    )
    contradicted = clean.model_copy(update={"contradiction": 0.8})

    assert clean.score() == clean.score()
    assert contradicted.score() < clean.score()


def test_pattern_projection_exposes_computed_not_llm_owned_confidence():
    now = datetime.now(timezone.utc)
    components = PatternConfidenceComponents(
        evidence_quality=1,
        evidence_diversity=1,
        recurrence=1,
        temporal_spread=1,
        cross_domain_presence=1,
        user_confirmation=1,
    )
    pattern = PatternProjection(
        pattern_type="recurrence",
        description="same structural conflict appears across projects",
        evidence_refs=("e1", "e2"),
        first_observed=now,
        last_observed=now,
        recurrence_count=2,
        confidence_components=components,
    )

    assert pattern.confidence == 1.0


def test_method_manifest_blocks_execution_when_required_inputs_are_missing():
    manifest = MethodManifest(
        method_id="astrology.natal",
        version="1",
        title="Natal astrology",
        category="astrology",
        required_inputs=("birth_date", "birth_time", "birth_place"),
        allowed_epistemic_levels=(EpistemicLevel.INTERPRETATION,),
    )

    assert manifest.missing_inputs({"birth_date": "2000-01-01"}) == (
        "birth_time",
        "birth_place",
    )
    with pytest.raises(ValueError, match="birth_time"):
        manifest.validate_inputs({"birth_date": "2000-01-01"})


def test_method_manifest_accepts_valid_false_and_zero_values():
    manifest = MethodManifest(
        method_id="test.flags",
        version="1",
        title="Flags",
        category="test",
        required_inputs=("enabled", "offset"),
        allowed_epistemic_levels=(EpistemicLevel.INTERPRETATION,),
    )

    assert manifest.missing_inputs({"enabled": False, "offset": 0}) == ()


def test_evidence_observed_at_requires_timezone_awareness():
    from akashic_engine import EvidenceSpan

    with pytest.raises(ValueError, match="timezone-aware"):
        EvidenceSpan(
            source_id="s1",
            source_version_id="v1",
            content_hash="h1",
            observed_at=datetime(2026, 8, 19),
        )


def test_claim_domain_tags_are_normalized_and_deduplicated():
    from akashic_engine import Claim

    claim = Claim(
        statement="domain-tagged claim",
        producer="test",
        domain_tags=(" work ", "health", "work"),
        epistemic_level=EpistemicLevel.OBSERVED,
    )

    assert claim.domain_tags == ("health", "work")
