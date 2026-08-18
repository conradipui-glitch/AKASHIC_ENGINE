"""Deterministic, explainable ATTUNE scoring reference implementation."""

from __future__ import annotations

from pydantic import Field, model_validator

from .domain import EpistemicLevel, FrozenModel


class AttunementFeatures(FrozenModel):
    """Normalized measurements used by ATTUNE.

    semantic_similarity is deliberately supplied by an embedding/retrieval
    provider. ATTUNE combines already-measured signals; it does not own a model.
    """

    semantic_similarity: float = Field(ge=0.0, le=1.0)
    recurrence: float = Field(ge=0.0, le=1.0)
    emotional_significance: float = Field(ge=0.0, le=1.0)
    unresolvedness: float = Field(ge=0.0, le=1.0)
    temporal_relevance: float = Field(ge=0.0, le=1.0)
    cross_domain_presence: float = Field(ge=0.0, le=1.0)
    user_importance: float = Field(ge=0.0, le=1.0)
    evidence_strength: float = Field(ge=0.0, le=1.0)
    contradiction: float = Field(default=0.0, ge=0.0, le=1.0)


class AttunementWeights(FrozenModel):
    semantic_similarity: float = Field(default=0.26, ge=0.0)
    recurrence: float = Field(default=0.16, ge=0.0)
    emotional_significance: float = Field(default=0.10, ge=0.0)
    unresolvedness: float = Field(default=0.12, ge=0.0)
    temporal_relevance: float = Field(default=0.10, ge=0.0)
    cross_domain_presence: float = Field(default=0.08, ge=0.0)
    user_importance: float = Field(default=0.10, ge=0.0)
    evidence_strength: float = Field(default=0.08, ge=0.0)

    @model_validator(mode="after")
    def positive_total(self) -> "AttunementWeights":
        if self.total <= 0:
            raise ValueError("At least one ATTUNE weight must be positive")
        return self

    @property
    def total(self) -> float:
        return sum(
            (
                self.semantic_similarity,
                self.recurrence,
                self.emotional_significance,
                self.unresolvedness,
                self.temporal_relevance,
                self.cross_domain_presence,
                self.user_importance,
                self.evidence_strength,
            )
        )


class AttunementCandidate(FrozenModel):
    candidate_id: str
    epistemic_level: EpistemicLevel
    features: AttunementFeatures
    evidence_refs: tuple[str, ...] = ()
    source_seq: int | None = Field(default=None, ge=1)


class AttunementComponent(FrozenModel):
    name: str
    feature_value: float
    normalized_weight: float
    contribution: float


class AttunementResult(FrozenModel):
    candidate_id: str
    epistemic_level: EpistemicLevel
    score: float = Field(ge=0.0, le=1.0)
    positive_score: float = Field(ge=0.0, le=1.0)
    contradiction_penalty: float = Field(ge=0.0, le=1.0)
    components: tuple[AttunementComponent, ...]
    evidence_refs: tuple[str, ...] = ()
    source_seq: int | None = None


class AttunementBundle(FrozenModel):
    query: str
    reader_role: str
    allowed_epistemic_levels: tuple[EpistemicLevel, ...]
    results: tuple[AttunementResult, ...]


class AttunementPolicy(FrozenModel):
    allowed_epistemic_levels: tuple[EpistemicLevel, ...]
    contradiction_penalty_factor: float = Field(default=0.65, ge=0.0, le=1.0)


_DEFAULT_POLICIES: dict[str, AttunementPolicy] = {
    "skeptic": AttunementPolicy(
        allowed_epistemic_levels=(
            EpistemicLevel.OBSERVED,
            EpistemicLevel.INTERPRETATION,
        ),
        contradiction_penalty_factor=0.80,
    ),
    "pattern": AttunementPolicy(
        allowed_epistemic_levels=(
            EpistemicLevel.OBSERVED,
            EpistemicLevel.INTERPRETATION,
        ),
        contradiction_penalty_factor=0.65,
    ),
    "symbolic": AttunementPolicy(
        allowed_epistemic_levels=(
            EpistemicLevel.OBSERVED,
            EpistemicLevel.INTERPRETATION,
            EpistemicLevel.SYMBOLIC_HYPOTHESIS,
        ),
        contradiction_penalty_factor=0.65,
    ),
    "synthesis": AttunementPolicy(
        allowed_epistemic_levels=(
            EpistemicLevel.OBSERVED,
            EpistemicLevel.INTERPRETATION,
            EpistemicLevel.SYMBOLIC_HYPOTHESIS,
        ),
        contradiction_penalty_factor=0.65,
    ),
}


class ExplainableAttunementEngine:
    """Ranks pre-retrieved candidates and exposes every score contribution."""

    def __init__(self, *, weights: AttunementWeights | None = None) -> None:
        self.weights = weights or AttunementWeights()

    def policy_for_role(self, reader_role: str) -> AttunementPolicy:
        return _DEFAULT_POLICIES.get(
            reader_role,
            AttunementPolicy(
                allowed_epistemic_levels=(
                    EpistemicLevel.OBSERVED,
                    EpistemicLevel.INTERPRETATION,
                )
            ),
        )

    def score_candidate(
        self,
        candidate: AttunementCandidate,
        *,
        policy: AttunementPolicy | None = None,
    ) -> AttunementResult:
        policy = policy or self.policy_for_role("default")
        pairs = (
            ("semantic_similarity", candidate.features.semantic_similarity, self.weights.semantic_similarity),
            ("recurrence", candidate.features.recurrence, self.weights.recurrence),
            ("emotional_significance", candidate.features.emotional_significance, self.weights.emotional_significance),
            ("unresolvedness", candidate.features.unresolvedness, self.weights.unresolvedness),
            ("temporal_relevance", candidate.features.temporal_relevance, self.weights.temporal_relevance),
            ("cross_domain_presence", candidate.features.cross_domain_presence, self.weights.cross_domain_presence),
            ("user_importance", candidate.features.user_importance, self.weights.user_importance),
            ("evidence_strength", candidate.features.evidence_strength, self.weights.evidence_strength),
        )
        total = self.weights.total
        components = tuple(
            AttunementComponent(
                name=name,
                feature_value=value,
                normalized_weight=round(weight / total, 6),
                contribution=round(value * (weight / total), 6),
            )
            for name, value, weight in pairs
        )
        positive_score = min(1.0, sum(item.contribution for item in components))
        penalty = candidate.features.contradiction * policy.contradiction_penalty_factor
        final_score = positive_score * (1.0 - penalty)
        return AttunementResult(
            candidate_id=candidate.candidate_id,
            epistemic_level=candidate.epistemic_level,
            score=round(max(0.0, min(1.0, final_score)), 6),
            positive_score=round(positive_score, 6),
            contradiction_penalty=round(penalty, 6),
            components=components,
            evidence_refs=candidate.evidence_refs,
            source_seq=candidate.source_seq,
        )

    def retrieve(
        self,
        *,
        query: str,
        reader_role: str,
        budget: int,
        candidates: tuple[AttunementCandidate, ...],
    ) -> AttunementBundle:
        if budget < 1:
            raise ValueError("budget must be >= 1")
        policy = self.policy_for_role(reader_role)
        allowed = set(policy.allowed_epistemic_levels)
        scored = [
            self.score_candidate(candidate, policy=policy)
            for candidate in candidates
            if candidate.epistemic_level in allowed
        ]
        scored.sort(
            key=lambda item: (
                -item.score,
                -(item.source_seq or 0),
                item.candidate_id,
            )
        )
        return AttunementBundle(
            query=query,
            reader_role=reader_role,
            allowed_epistemic_levels=policy.allowed_epistemic_levels,
            results=tuple(scored[:budget]),
        )
