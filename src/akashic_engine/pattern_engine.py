"""Deterministic longitudinal pattern construction over replayable evidence bundles."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime

from pydantic import Field, field_validator

from .attunement import AttunementCandidate, AttunementFeatures
from .domain import (
    Claim,
    EpistemicLevel,
    EvidenceSpan,
    FrozenModel,
    PatternConfidenceComponents,
    PatternProjection,
    ValidationState,
)
from .evidence_bundle import EvidenceBundle
from .evidence_graph import ClaimRelation, ClaimRelationType


class PatternEngineError(RuntimeError):
    pass


class InvalidPatternEvidenceError(PatternEngineError):
    pass


class EpistemicPatternBoundaryError(InvalidPatternEvidenceError):
    pass


class PatternOccurrence(FrozenModel):
    """A provenance selection proposed as one occurrence of a pattern.

    dependency_group_id is only a conservative merge hint. The engine also
    derives dependence from claims, evidence, source lineages, and content
    hashes, so callers cannot manufacture independence merely by changing it.
    """

    occurrence_id: str = Field(min_length=1)
    claim_id: str = Field(min_length=1)
    evidence_refs: tuple[str, ...]
    dependency_group_id: str = Field(min_length=1)
    confirmation_claim_id: str | None = None

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
    declared_dependency_group_ids: tuple[str, ...]
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


class _OccurrenceContext(FrozenModel):
    occurrence: PatternOccurrence
    claim: Claim
    evidence: tuple[EvidenceSpan, ...]
    observed_at: datetime
    source_ids: tuple[str, ...]
    lineage_tokens: tuple[str, ...]
    domains: tuple[str, ...]
    evidence_quality: float
    user_confirmed: bool


class LongitudinalPatternEngine:
    """Build PatternProjection from independently grounded occurrences."""

    _VALIDATION_QUALITY = {
        ValidationState.UNVERIFIED: 0.35,
        ValidationState.SUPPORTED: 0.80,
        ValidationState.CONTRADICTED: 0.10,
        ValidationState.VALIDATED: 1.00,
    }
    _EPISTEMIC_QUALITY = {
        EpistemicLevel.OBSERVED: 1.00,
        EpistemicLevel.INTERPRETATION: 0.75,
    }

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
        relation_set = evidence_bundle.relation_set()
        incoming_contradictions: dict[str, set[str]] = defaultdict(set)
        for edge in evidence_bundle.relations:
            if edge.relation is ClaimRelationType.CONTRADICTS:
                incoming_contradictions[edge.target_claim_id].add(edge.source_claim_id)

        contexts = tuple(
            self._context_for_occurrence(
                occurrence,
                claims=claims,
                evidence=evidence,
                relation_set=relation_set,
            )
            for occurrence in occurrences
        )

        components_by_root = self._dependency_components(contexts)
        group_summaries: list[PatternOccurrenceGroup] = []
        all_contradictors: set[str] = set()
        for member_contexts in components_by_root:
            evidence_refs = sorted(
                {ref for context in member_contexts for ref in context.occurrence.evidence_refs}
            )
            source_ids = sorted(
                {source for context in member_contexts for source in context.source_ids}
            )
            claim_ids = sorted({context.occurrence.claim_id for context in member_contexts})
            contradictors = sorted(
                {
                    source_claim_id
                    for claim_id in claim_ids
                    for source_claim_id in incoming_contradictions.get(claim_id, set())
                }
            )
            all_contradictors.update(contradictors)
            group_id = self._dependency_component_id(member_contexts)
            group_summaries.append(
                PatternOccurrenceGroup(
                    dependency_group_id=group_id,
                    declared_dependency_group_ids=tuple(
                        sorted(
                            {
                                context.occurrence.dependency_group_id
                                for context in member_contexts
                            }
                        )
                    ),
                    occurrence_ids=tuple(
                        sorted(context.occurrence.occurrence_id for context in member_contexts)
                    ),
                    claim_ids=tuple(claim_ids),
                    evidence_refs=tuple(evidence_refs),
                    source_group_ids=tuple(source_ids),
                    domains=tuple(
                        sorted({domain for context in member_contexts for domain in context.domains})
                    ),
                    first_observed=min(context.observed_at for context in member_contexts),
                    last_observed=max(context.observed_at for context in member_contexts),
                    evidence_quality=round(
                        sum(context.evidence_quality for context in member_contexts)
                        / len(member_contexts),
                        6,
                    ),
                    user_confirmed=any(context.user_confirmed for context in member_contexts),
                    contradicted=bool(contradictors),
                    contradicting_claim_ids=tuple(contradictors),
                )
            )

        groups = tuple(sorted(group_summaries, key=lambda group: group.dependency_group_id))
        independent_count = len(groups)
        unique_sources = {source for group in groups for source in group.source_group_ids}
        unique_domains = {domain for group in groups for domain in group.domains}
        first_observed = min(group.first_observed for group in groups)
        last_observed = max(group.last_observed for group in groups)

        # Temporal spread measures separation between independent groups, not the
        # internal duration of a single source/dependency component.
        if independent_count < 2:
            temporal_days = 0.0
        else:
            representative_times = [group.first_observed for group in groups]
            temporal_days = max(0.0, (max(representative_times) - min(representative_times)).total_seconds() / 86400.0)

        # A single dependency group cannot prove cross-domain recurrence merely
        # by carrying several labels.
        effective_domain_count = min(len(unique_domains), independent_count)

        confidence_components = PatternConfidenceComponents(
            evidence_quality=round(
                sum(group.evidence_quality for group in groups) / independent_count, 6
            ),
            evidence_diversity=round(min(1.0, len(unique_sources) / 3.0), 6),
            recurrence=round(min(1.0, max(0, independent_count - 1) / 4.0), 6),
            temporal_spread=round(min(1.0, temporal_days / 180.0), 6),
            cross_domain_presence=round(min(1.0, effective_domain_count / 3.0), 6),
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
            confidence_components=confidence_components,
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

    def _context_for_occurrence(
        self,
        occurrence: PatternOccurrence,
        *,
        claims: dict[str, Claim],
        evidence: dict[str, EvidenceSpan],
        relation_set: set[ClaimRelation],
    ) -> _OccurrenceContext:
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
        spans: list[EvidenceSpan] = []
        for evidence_id in occurrence.evidence_refs:
            try:
                span = evidence[evidence_id]
            except KeyError as exc:
                raise InvalidPatternEvidenceError(
                    f"Occurrence references evidence outside bundle: {evidence_id}"
                ) from exc
            if evidence_id not in claim_evidence:
                raise InvalidPatternEvidenceError(
                    f"Evidence {evidence_id} is not provenance for claim {claim.claim_id}"
                )
            if span.observed_at is None:
                raise InvalidPatternEvidenceError(
                    f"Evidence {evidence_id} requires provenance observed_at for longitudinal analysis"
                )
            spans.append(span)

        user_confirmed = False
        if occurrence.confirmation_claim_id is not None:
            user_confirmed = self._validate_confirmation(
                occurrence,
                claims=claims,
                relation_set=relation_set,
            )

        validation_quality = self._VALIDATION_QUALITY[claim.validation_state]
        epistemic_quality = self._EPISTEMIC_QUALITY[claim.epistemic_level]
        evidence_quality = round(validation_quality * epistemic_quality, 6)
        source_ids = tuple(sorted({span.source_id for span in spans}))
        lineage_tokens = tuple(
            sorted(
                {
                    *(f"source:{span.source_id}" for span in spans),
                    *(f"source-version:{span.source_id}:{span.source_version_id}" for span in spans),
                    *(f"content:{span.content_hash}" for span in spans),
                }
            )
        )
        return _OccurrenceContext(
            occurrence=occurrence,
            claim=claim,
            evidence=tuple(sorted(spans, key=lambda span: span.evidence_id)),
            observed_at=min(span.observed_at for span in spans if span.observed_at is not None),
            source_ids=source_ids,
            lineage_tokens=lineage_tokens,
            domains=claim.domain_tags,
            evidence_quality=evidence_quality,
            user_confirmed=user_confirmed,
        )

    @staticmethod
    def _validate_confirmation(
        occurrence: PatternOccurrence,
        *,
        claims: dict[str, Claim],
        relation_set: set[ClaimRelation],
    ) -> bool:
        confirmation_id = occurrence.confirmation_claim_id
        assert confirmation_id is not None
        try:
            confirmation = claims[confirmation_id]
        except KeyError as exc:
            raise InvalidPatternEvidenceError(
                f"Confirmation claim is outside evidence bundle: {confirmation_id}"
            ) from exc
        if confirmation.claim_type != "user_confirmation":
            raise InvalidPatternEvidenceError(
                "Pattern confirmation requires claim_type=user_confirmation"
            )
        if confirmation.epistemic_level is not EpistemicLevel.OBSERVED:
            raise InvalidPatternEvidenceError("Pattern confirmation must be OBSERVED")
        if confirmation.validation_state not in {
            ValidationState.SUPPORTED,
            ValidationState.VALIDATED,
        }:
            raise InvalidPatternEvidenceError(
                "Pattern confirmation must be evidence-supported or validated"
            )
        expected_edge = ClaimRelation(
            source_claim_id=confirmation_id,
            target_claim_id=occurrence.claim_id,
            relation=ClaimRelationType.SUPPORTS,
        )
        if expected_edge not in relation_set:
            raise InvalidPatternEvidenceError(
                "Pattern confirmation must explicitly support the occurrence claim"
            )
        return True

    def _dependency_components(
        self, contexts: tuple[_OccurrenceContext, ...]
    ) -> tuple[tuple[_OccurrenceContext, ...], ...]:
        ordered = tuple(sorted(contexts, key=lambda context: context.occurrence.occurrence_id))
        parent = list(range(len(ordered)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            root_left, root_right = find(left), find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        token_owner: dict[str, int] = {}
        for index, context in enumerate(ordered):
            occurrence = context.occurrence
            tokens = {
                f"declared:{occurrence.dependency_group_id}",
                f"claim:{occurrence.claim_id}",
                *(f"evidence:{ref}" for ref in occurrence.evidence_refs),
                *context.lineage_tokens,
            }
            for token in sorted(tokens):
                previous = token_owner.get(token)
                if previous is None:
                    token_owner[token] = index
                else:
                    union(index, previous)

        grouped: dict[int, list[_OccurrenceContext]] = defaultdict(list)
        for index, context in enumerate(ordered):
            grouped[find(index)].append(context)
        return tuple(
            tuple(sorted(group, key=lambda context: context.occurrence.occurrence_id))
            for _, group in sorted(
                grouped.items(),
                key=lambda item: min(
                    context.occurrence.occurrence_id for context in item[1]
                ),
            )
        )

    @staticmethod
    def _dependency_component_id(contexts: tuple[_OccurrenceContext, ...]) -> str:
        payload = {
            "occurrence_ids": sorted(
                context.occurrence.occurrence_id for context in contexts
            ),
            "declared_dependency_group_ids": sorted(
                {context.occurrence.dependency_group_id for context in contexts}
            ),
            "lineage_tokens": sorted(
                {token for context in contexts for token in context.lineage_tokens}
            ),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return "dep_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]

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
