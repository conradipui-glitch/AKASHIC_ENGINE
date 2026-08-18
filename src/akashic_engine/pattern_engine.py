"""Deterministic longitudinal pattern construction over replayable evidence bundles."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime

from pydantic import Field, field_validator

from .attunement import AttunementCandidate, AttunementFeatures
from .domain import (
    EpistemicLevel,
    FrozenModel,
    PatternConfidenceComponents,
    PatternProjection,
)
from .evidence_bundle import EvidenceBundle
from .evidence_graph import ClaimRelationType


class PatternEngineError(RuntimeError):
    pass


class InvalidPatternEvidenceError(PatternEngineError):
    pass


class EpistemicPatternBoundaryError(InvalidPatternEvidenceError):
    pass


class PatternOccurrence(FrozenModel):
    """One observed occurrence proposed as evidence for a longitudinal pattern."""

    occurrence_id: str
    claim_id: str
    evidence_refs: tuple[str, ...]
    observed_at: datetime
    domain: str
    dependency_group_id: str
    evidence_quality: float = Field(ge=0.0, le=1.0)
    user_confirmed: bool = False

    @field_validator("evidence_refs")
    @classmethod
    def nonempty_unique_evidence(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("Pattern occurrences require evidence references")
        if len(set(value)) != len(value):
            raise ValueError("Pattern occurrences cannot repeat an evidence reference")
        return value


class PatternOccurrenceGroup(FrozenModel):
    dependency_group_id: str
    occurrence_ids: tuple[str, ...]
    claim_ids: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    source_group_ids: tuple[str, ...]
    domains: tuple[str, ...]
    first_observed: datetime
    last_observed: datetime
    evidence_quality: float = Field(ge=0.0, le=1.0)
    user_confirmed: bool
    contradicted: bool
    contradicting_claim_ids: tuple[str, ...] = ()


class PatternAttunementSignals(FrozenModel):
    """Query/session-specific ATTUNE features not derivable from pattern evidence."""

    semantic_similarity: float = Field(ge=0.0, le=1.0)
    emotional_significance: float = Field(ge=0.0, le=1.0)
    unresolvedness: float = Field(ge=0.0, le=1.0)
    temporal_relevance: float = Field(ge=0.0, le=1.0)
    user_importance: float = Field(ge=0.0, le=1.0)


class PatternAnalysis(FrozenModel):
    projection: PatternProjection
    evidence_bundle_fingerprint: str = Field(min_length=64, max_length=64)
    occurrence_groups: tuple[PatternOccurrenceGroup, ...]
    analysis_fingerprint: str = Field(min_length=64, max_length=64)

    def to_attunement_candidate(
        self,
        signals: PatternAttunementSignals,
        *,
        candidate_id: str | None = None,
    ) -> AttunementCandidate:
        components = self.projection.confidence_components
        evidence_strength = round(
            0.60 * components.evidence_quality + 0.40 * components.evidence_diversity,
            6,
        )
        return AttunementCandidate(
            candidate_id=candidate_id or self.projection.pattern_id,
            epistemic_level=EpistemicLevel.INTERPRETATION,
            features=AttunementFeatures(
                semantic_similarity=signals.semantic_similarity,
                recurrence=components.recurrence,
                emotional_significance=signals.emotional_significance,
                unresolvedness=signals.unresolvedness,
                temporal_relevance=signals.temporal_relevance,
                cross_domain_presence=components.cross_domain_presence,
                user_importance=signals.user_importance,
                evidence_strength=evidence_strength,
                contradiction=components.contradiction,
            ),
            evidence_refs=self.projection.evidence_refs,
        )


class LongitudinalPatternEngine:
    """Build PatternProjection from independent, source-grounded occurrences."""

    def build(
        self,
        *,
        pattern_type: str,
        description: str,
        evidence_bundle: EvidenceBundle,
        occurrences: tuple[PatternOccurrence, ...],
        status: str = "candidate",
    ) -> PatternAnalysis:
        if not occurrences:
            raise PatternEngineError("At least one occurrence is required")
        occurrence_ids = [item.occurrence_id for item in occurrences]
        if len(set(occurrence_ids)) != len(occurrence_ids):
            raise InvalidPatternEvidenceError("Duplicate occurrence IDs are not allowed")

        claims = evidence_bundle.claim_map()
        evidence = evidence_bundle.evidence_map()
        incoming_contradictions: dict[str, set[str]] = defaultdict(set)
        for edge in evidence_bundle.relations:
            if edge.relation is ClaimRelationType.CONTRADICTS:
                incoming_contradictions[edge.target_claim_id].add(edge.source_claim_id)

        for occurrence in occurrences:
            try:
                claim = claims[occurrence.claim_id]
            except KeyError as exc:
                raise InvalidPatternEvidenceError(
                    f"Occurrence references claim outside evidence bundle: {occurrence.claim_id}"
                ) from exc
            if claim.epistemic_level is EpistemicLevel.SYMBOLIC_HYPOTHESIS:
                raise EpistemicPatternBoundaryError(
                    "Symbolic hypotheses cannot become evidence for a factual/interpretive pattern"
                )
            claim_evidence = set(claim.evidence_refs)
            for evidence_id in occurrence.evidence_refs:
                if evidence_id not in evidence:
                    raise InvalidPatternEvidenceError(
                        f"Occurrence references evidence outside bundle: {evidence_id}"
                    )
                if evidence_id not in claim_evidence:
                    raise InvalidPatternEvidenceError(
                        f"Evidence {evidence_id} is not provenance for claim {claim.claim_id}"
                    )

        grouped: dict[str, list[PatternOccurrence]] = defaultdict(list)
        for occurrence in occurrences:
            grouped[occurrence.dependency_group_id].append(occurrence)

        group_summaries: list[PatternOccurrenceGroup] = []
        all_contradictors: set[str] = set()
        for dependency_group_id in sorted(grouped):
            group = grouped[dependency_group_id]
            evidence_refs = sorted({ref for item in group for ref in item.evidence_refs})
            source_ids = sorted({evidence[ref].source_id for ref in evidence_refs})
            claim_ids = sorted({item.claim_id for item in group})
            contradictors = sorted(
                {
                    source_claim_id
                    for claim_id in claim_ids
                    for source_claim_id in incoming_contradictions.get(claim_id, set())
                }
            )
            all_contradictors.update(contradictors)
            group_summaries.append(
                PatternOccurrenceGroup(
                    dependency_group_id=dependency_group_id,
                    occurrence_ids=tuple(sorted(item.occurrence_id for item in group)),
                    claim_ids=tuple(claim_ids),
                    evidence_refs=tuple(evidence_refs),
                    source_group_ids=tuple(source_ids),
                    domains=tuple(sorted({item.domain for item in group})),
                    first_observed=min(item.observed_at for item in group),
                    last_observed=max(item.observed_at for item in group),
                    evidence_quality=max(item.evidence_quality for item in group),
                    user_confirmed=any(item.user_confirmed for item in group),
                    contradicted=bool(contradictors),
                    contradicting_claim_ids=tuple(contradictors),
                )
            )

        groups = tuple(group_summaries)
        independent_count = len(groups)
        unique_sources = {source for group in groups for source in group.source_group_ids}
        unique_domains = {domain for group in groups for domain in group.domains}
        first_observed = min(group.first_observed for group in groups)
        last_observed = max(group.last_observed for group in groups)
        temporal_days = max(0.0, (last_observed - first_observed).total_seconds() / 86400.0)

        components = PatternConfidenceComponents(
            evidence_quality=round(
                sum(group.evidence_quality for group in groups) / independent_count, 6
            ),
            evidence_diversity=round(min(1.0, len(unique_sources) / 3.0), 6),
            recurrence=round(min(1.0, max(0, independent_count - 1) / 4.0), 6),
            temporal_spread=round(min(1.0, temporal_days / 180.0), 6),
            cross_domain_presence=round(min(1.0, len(unique_domains) / 3.0), 6),
            user_confirmation=round(
                sum(1 for group in groups if group.user_confirmed) / independent_count, 6
            ),
            contradiction=round(
                sum(1 for group in groups if group.contradicted) / independent_count, 6
            ),
        )
        pattern_id = self._pattern_id(
            pattern_type=pattern_type,
            description=description,
            bundle_fingerprint=evidence_bundle.fingerprint,
            groups=groups,
        )
        projection = PatternProjection(
            pattern_id=pattern_id,
            pattern_type=pattern_type,
            description=description,
            evidence_refs=tuple(sorted({ref for group in groups for ref in group.evidence_refs})),
            first_observed=first_observed,
            last_observed=last_observed,
            recurrence_count=independent_count,
            domain_tags=tuple(sorted(unique_domains)),
            contradictions=tuple(sorted(all_contradictors)),
            confidence_components=components,
            status=status,
        )
        analysis_fingerprint = self._fingerprint(
            projection=projection,
            bundle_fingerprint=evidence_bundle.fingerprint,
            groups=groups,
        )
        return PatternAnalysis(
            projection=projection,
            evidence_bundle_fingerprint=evidence_bundle.fingerprint,
            occurrence_groups=groups,
            analysis_fingerprint=analysis_fingerprint,
        )

    @staticmethod
    def _pattern_id(
        *,
        pattern_type: str,
        description: str,
        bundle_fingerprint: str,
        groups: tuple[PatternOccurrenceGroup, ...],
    ) -> str:
        payload = {
            "pattern_type": pattern_type,
            "description": description,
            "evidence_bundle_fingerprint": bundle_fingerprint,
            "occurrence_groups": [group.model_dump(mode="json") for group in groups],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return "pat_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def _fingerprint(
        *,
        projection: PatternProjection,
        bundle_fingerprint: str,
        groups: tuple[PatternOccurrenceGroup, ...],
    ) -> str:
        payload = {
            "projection": projection.model_dump(mode="json"),
            "evidence_bundle_fingerprint": bundle_fingerprint,
            "occurrence_groups": [group.model_dump(mode="json") for group in groups],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
