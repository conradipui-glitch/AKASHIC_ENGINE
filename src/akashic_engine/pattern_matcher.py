"""Conservative matching and promotion of semantic candidates into occurrences.

Semantic extraction stops at unverified PatternOccurrenceCandidate records. This
module is the only bridge that may promote those candidates into real
PatternOccurrence values accepted by LongitudinalPatternEngine.

Matcher output is treated as an untrusted grouping proposal. AKASHIC_ENGINE owns
matcher provenance, policy, candidate integrity checks, independence checks, and
occurrence identities.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from threading import RLock
from typing import Protocol

from pydantic import Field, ValidationError, field_validator

from .domain import Claim, EpistemicLevel, EvidenceSpan, FrozenModel, ValidationState
from .evidence_graph import ClaimRelationType, UnknownGraphNodeError
from .extraction import PatternOccurrenceCandidate, SemanticExtractionReceipt
from .pattern_engine import PatternOccurrence


class PatternMatchingError(RuntimeError):
    pass


class InvalidPatternMatchInputError(PatternMatchingError):
    pass


class InvalidPatternMatchProposalError(PatternMatchingError):
    pass


class DuplicatePatternMatchProposalError(InvalidPatternMatchProposalError):
    pass


class PatternPromotionConflictError(PatternMatchingError):
    pass


_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def _identifier(value: str, field_name: str) -> str:
    cleaned = unicodedata.normalize("NFC", value.strip()).lower()
    if not cleaned or not _IDENTIFIER_RE.fullmatch(cleaned):
        raise ValueError(
            f"{field_name} must use lowercase letters, digits, '.', '_' or '-'"
        )
    return cleaned


def _sha256_json(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class PatternMatcherManifest(FrozenModel):
    """Engine-owned identity for one matcher configuration."""

    matcher_id: str
    matcher_version: str
    configuration_fingerprint: str = Field(min_length=64, max_length=64)

    @field_validator("matcher_id")
    @classmethod
    def normalize_matcher_id(cls, value: str) -> str:
        return _identifier(value, "matcher_id")

    @field_validator("matcher_version")
    @classmethod
    def normalize_matcher_version(cls, value: str) -> str:
        cleaned = unicodedata.normalize("NFC", value.strip())
        if not cleaned:
            raise ValueError("matcher_version cannot be empty")
        return cleaned

    @field_validator("configuration_fingerprint")
    @classmethod
    def validate_configuration_fingerprint(cls, value: str) -> str:
        lowered = value.lower()
        if not re.fullmatch(r"[0-9a-f]{64}", lowered):
            raise ValueError("configuration_fingerprint must be a SHA-256 hex digest")
        return lowered

    @property
    def fingerprint(self) -> str:
        return _sha256_json(self.model_dump(mode="json"))


class ValidatedPatternCandidate(FrozenModel):
    """Closed candidate view supplied to a matcher after engine validation."""

    candidate: PatternOccurrenceCandidate
    claim: Claim
    source_claims: tuple[Claim, ...]
    evidence: tuple[EvidenceSpan, ...]
    source_ids: tuple[str, ...]
    lineage_tokens: tuple[str, ...]


class PatternMatchContext(FrozenModel):
    candidates: tuple[ValidatedPatternCandidate, ...]
    input_fingerprint: str = Field(min_length=64, max_length=64)


class PatternMatchProposal(FrozenModel):
    """Untrusted proposal that a set of candidates represents one pattern."""

    candidate_ids: tuple[str, ...]
    pattern_key: str = Field(min_length=1, max_length=128)

    @field_validator("candidate_ids")
    @classmethod
    def validate_candidate_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(sorted(value))
        if not normalized:
            raise ValueError("candidate_ids cannot be empty")
        if any(not item for item in normalized):
            raise ValueError("candidate_ids cannot contain empty IDs")
        if any(len(item) > 256 for item in normalized):
            raise ValueError("candidate_ids cannot exceed 256 characters")
        if len(set(normalized)) != len(normalized):
            raise ValueError("candidate_ids must be unique")
        return normalized

    @field_validator("pattern_key")
    @classmethod
    def normalize_pattern_key(cls, value: str) -> str:
        return _identifier(value, "pattern_key")


class PatternPromotionPolicy(FrozenModel):
    """Engine-owned policy for turning match proposals into real occurrences."""

    policy_version: str = "1"
    max_extraction_receipts_per_input: int = Field(default=128, ge=1, le=2048)
    max_candidates_per_input: int = Field(default=256, ge=2, le=4096)
    max_candidates_per_proposal: int = Field(default=64, ge=2, le=1024)
    max_proposals_per_input: int = Field(default=32, ge=1, le=512)
    min_candidates_per_pattern: int = Field(default=2, ge=2, le=64)
    min_independent_groups: int = Field(default=2, ge=2, le=64)
    require_exact_hint_match: bool = True
    require_unique_independent_lineage: bool = True
    disallow_candidate_overlap: bool = True
    disallow_extraction_claim_overlap: bool = True
    allowed_claim_types: tuple[str, ...] = ("semantic_interpretation",)

    @field_validator("allowed_claim_types")
    @classmethod
    def normalize_claim_types(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(sorted(_identifier(item, "claim_type") for item in value))
        if not normalized:
            raise ValueError("allowed_claim_types cannot be empty")
        if any(item in {"source_text", "user_confirmation"} for item in normalized):
            raise ValueError("observed/reserved claim types cannot be promoted as semantic candidates")
        if len(set(normalized)) != len(normalized):
            raise ValueError("allowed_claim_types must be unique")
        return normalized

    @property
    def fingerprint(self) -> str:
        return _sha256_json(self.model_dump(mode="json"))


class PatternMatcher(Protocol):
    def match(self, context: PatternMatchContext) -> tuple[PatternMatchProposal, ...]: ...


class ExactHintPatternMatcher:
    """Deterministic reference matcher: exact non-null hint equality only.

    This matcher intentionally does not perform semantic canonicalization. It is
    a safe baseline and a useful replay oracle for future model/embedding-based
    matchers.
    """

    def match(self, context: PatternMatchContext) -> tuple[PatternMatchProposal, ...]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for item in context.candidates:
            hint = item.candidate.pattern_hint
            if hint is not None:
                grouped[hint].append(item.candidate.candidate_id)
        return tuple(
            PatternMatchProposal(candidate_ids=tuple(sorted(candidate_ids)), pattern_key=hint)
            for hint, candidate_ids in sorted(grouped.items())
            if len(candidate_ids) >= 2
        )


class PatternPromotionEvidenceGraph(Protocol):
    def get_claim(self, claim_id: str) -> Claim: ...

    def get_evidence(self, evidence_id: str) -> EvidenceSpan: ...

    def explain_claim(self, claim_id: str): ...


class PromotedPatternOccurrence(FrozenModel):
    pattern_key: str
    candidate_id: str
    occurrence: PatternOccurrence
    matcher_manifest_fingerprint: str = Field(min_length=64, max_length=64)
    policy_fingerprint: str = Field(min_length=64, max_length=64)
    promotion_fingerprint: str = Field(min_length=64, max_length=64)


class PatternPromotionReceipt(FrozenModel):
    matcher_manifest_fingerprint: str = Field(min_length=64, max_length=64)
    policy_fingerprint: str = Field(min_length=64, max_length=64)
    input_fingerprint: str = Field(min_length=64, max_length=64)
    proposal_fingerprint: str = Field(min_length=64, max_length=64)
    promotions: tuple[PromotedPatternOccurrence, ...]
    receipt_fingerprint: str = Field(min_length=64, max_length=64)

    def occurrences_by_pattern_key(self) -> dict[str, tuple[PatternOccurrence, ...]]:
        grouped: dict[str, list[PatternOccurrence]] = defaultdict(list)
        for item in self.promotions:
            grouped[item.pattern_key].append(item.occurrence)
        return {
            key: tuple(sorted(items, key=lambda occurrence: occurrence.occurrence_id))
            for key, items in sorted(grouped.items())
        }

    def claim_ids_by_pattern_key(self) -> dict[str, tuple[str, ...]]:
        return {
            key: tuple(sorted({item.claim_id for item in occurrences}))
            for key, occurrences in self.occurrences_by_pattern_key().items()
        }


class PatternPromotionEngine:
    """Validated semantic candidates -> real PatternOccurrence values."""

    def __init__(
        self,
        *,
        evidence_graph: PatternPromotionEvidenceGraph,
        matcher: PatternMatcher,
        matcher_manifest: PatternMatcherManifest,
        policy: PatternPromotionPolicy | None = None,
    ) -> None:
        self.evidence_graph = evidence_graph
        self.matcher = matcher
        self.matcher_manifest = matcher_manifest
        self.policy = policy or PatternPromotionPolicy()
        self._lock = RLock()

    def promote(
        self, extraction_receipts: tuple[SemanticExtractionReceipt, ...]
    ) -> PatternPromotionReceipt:
        with self._lock:
            manifest = PatternMatcherManifest.model_validate(
                self.matcher_manifest.model_dump(mode="python")
            )
            policy = PatternPromotionPolicy.model_validate(
                self.policy.model_dump(mode="python")
            )
            context = self._build_context(extraction_receipts, policy=policy)
            raw_proposals = self.matcher.match(context)
            # An external matcher may hold references to graph-backed services.
            # Rebuild the closed context after the callback so TOCTOU mutation of
            # claims/evidence/relations cannot be promoted from a stale snapshot.
            try:
                refreshed_context = self._build_context(
                    extraction_receipts, policy=policy
                )
            except InvalidPatternMatchInputError as exc:
                raise PatternPromotionConflictError(
                    "validated pattern candidate context changed during matcher callback"
                ) from exc
            if refreshed_context != context:
                raise PatternPromotionConflictError(
                    "validated pattern candidate context changed during matcher callback"
                )
            context = refreshed_context
            if not isinstance(raw_proposals, tuple):
                raise InvalidPatternMatchProposalError(
                    "matcher must return a tuple of PatternMatchProposal"
                )
            if len(raw_proposals) > policy.max_proposals_per_input:
                raise InvalidPatternMatchProposalError(
                    f"matcher returned {len(raw_proposals)} proposals; policy limit is "
                    f"{policy.max_proposals_per_input}"
                )
            proposals = tuple(self._revalidate_proposal(item) for item in raw_proposals)
            ordered = tuple(sorted(proposals, key=self._proposal_sort_key))
            fingerprints = tuple(self._proposal_fingerprint(item) for item in ordered)
            if len(set(fingerprints)) != len(fingerprints):
                raise DuplicatePatternMatchProposalError("duplicate pattern match proposals are not allowed")

            candidate_map = {
                item.candidate.candidate_id: item for item in context.candidates
            }
            seen_candidates: set[str] = set()
            seen_claims: set[str] = set()
            seen_pattern_keys: set[str] = set()
            promotions: dict[str, PromotedPatternOccurrence] = {}

            for proposal in ordered:
                if proposal.pattern_key in seen_pattern_keys:
                    raise InvalidPatternMatchProposalError(
                        f"pattern_key may appear only once per promotion run: {proposal.pattern_key}"
                    )
                seen_pattern_keys.add(proposal.pattern_key)
                selected = self._validate_match_proposal(
                    proposal,
                    candidate_map=candidate_map,
                    policy=policy,
                )
                proposal_claim_ids = {item.claim.claim_id for item in selected}
                if policy.disallow_candidate_overlap:
                    overlap = seen_candidates.intersection(proposal.candidate_ids)
                    if overlap:
                        raise InvalidPatternMatchProposalError(
                            "candidate cannot be assigned to multiple pattern proposals: "
                            + ", ".join(sorted(overlap))
                        )
                if policy.disallow_extraction_claim_overlap:
                    claim_overlap = seen_claims.intersection(proposal_claim_ids)
                    if claim_overlap:
                        raise InvalidPatternMatchProposalError(
                            "one semantic claim cannot be promoted into multiple patterns in one run: "
                            + ", ".join(sorted(claim_overlap))
                        )
                seen_candidates.update(proposal.candidate_ids)
                seen_claims.update(proposal_claim_ids)

                for item in selected:
                    promoted = self._promote_candidate(
                        item,
                        pattern_key=proposal.pattern_key,
                        manifest=manifest,
                        policy=policy,
                    )
                    existing = promotions.get(promoted.promotion_fingerprint)
                    if existing is not None and existing != promoted:
                        raise PatternPromotionConflictError(
                            "promotion fingerprint resolved to different data"
                        )
                    promotions[promoted.promotion_fingerprint] = promoted

            promotion_tuple = tuple(promotions[key] for key in sorted(promotions))
            proposal_fingerprint = _sha256_json(fingerprints)
            receipt_fingerprint = _sha256_json(
                {
                    "matcher_manifest_fingerprint": manifest.fingerprint,
                    "policy_fingerprint": policy.fingerprint,
                    "input_fingerprint": context.input_fingerprint,
                    "proposal_fingerprint": proposal_fingerprint,
                    "promotions": [item.model_dump(mode="json") for item in promotion_tuple],
                }
            )
            return PatternPromotionReceipt(
                matcher_manifest_fingerprint=manifest.fingerprint,
                policy_fingerprint=policy.fingerprint,
                input_fingerprint=context.input_fingerprint,
                proposal_fingerprint=proposal_fingerprint,
                promotions=promotion_tuple,
                receipt_fingerprint=receipt_fingerprint,
            )

    def _build_context(
        self,
        receipts: tuple[SemanticExtractionReceipt, ...],
        *,
        policy: PatternPromotionPolicy,
    ) -> PatternMatchContext:
        if not isinstance(receipts, tuple):
            raise InvalidPatternMatchInputError(
                "extraction_receipts must be a tuple of SemanticExtractionReceipt"
            )
        if not receipts:
            raise InvalidPatternMatchInputError("at least one semantic extraction receipt is required")
        if len(receipts) > policy.max_extraction_receipts_per_input:
            raise InvalidPatternMatchInputError(
                f"received {len(receipts)} extraction receipts; policy limit is "
                f"{policy.max_extraction_receipts_per_input}"
            )

        receipt_fingerprints: set[str] = set()
        candidate_map: dict[str, ValidatedPatternCandidate] = {}
        claim_candidate_ids: dict[str, set[str]] = defaultdict(set)
        for raw_receipt in receipts:
            receipt = self._revalidate_receipt(raw_receipt)
            if receipt.receipt_fingerprint in receipt_fingerprints:
                raise InvalidPatternMatchInputError("duplicate semantic extraction receipts are not allowed")
            receipt_fingerprints.add(receipt.receipt_fingerprint)
            self._verify_extraction_receipt_fingerprint(receipt)
            derived_claim_ids = set(receipt.derived_claim_ids)
            for raw_candidate in receipt.occurrence_candidates:
                candidate = self._revalidate_candidate(raw_candidate)
                if candidate.extraction_claim_id not in derived_claim_ids:
                    raise InvalidPatternMatchInputError(
                        f"candidate {candidate.candidate_id} references claim outside its extraction receipt"
                    )
                validated = self._validate_candidate(candidate, policy=policy)
                existing = candidate_map.get(candidate.candidate_id)
                if existing is not None and existing != validated:
                    raise PatternPromotionConflictError(
                        f"candidate ID resolved to different data: {candidate.candidate_id}"
                    )
                candidate_map[candidate.candidate_id] = validated
                claim_candidate_ids[validated.claim.claim_id].add(candidate.candidate_id)

        ambiguous_claims = {
            claim_id: ids for claim_id, ids in claim_candidate_ids.items() if len(ids) > 1
        }
        if ambiguous_claims:
            raise InvalidPatternMatchInputError(
                "multiple candidate variants for one semantic claim are ambiguous; "
                "a matcher must not choose between extractor-supplied hints"
            )
        if len(candidate_map) < policy.min_candidates_per_pattern:
            raise InvalidPatternMatchInputError(
                f"promotion requires at least {policy.min_candidates_per_pattern} candidates"
            )
        if len(candidate_map) > policy.max_candidates_per_input:
            raise InvalidPatternMatchInputError(
                f"received {len(candidate_map)} candidates; policy limit is {policy.max_candidates_per_input}"
            )

        candidates = tuple(candidate_map[key] for key in sorted(candidate_map))
        input_fingerprint = _sha256_json(
            {
                "candidates": [
                    {
                        "candidate": item.candidate.model_dump(mode="json"),
                        "claim": item.claim.model_dump(mode="json", exclude={"created_at"}),
                        "source_claims": [
                            claim.model_dump(mode="json", exclude={"created_at"})
                            for claim in item.source_claims
                        ],
                        "evidence": [
                            span.model_dump(mode="json", exclude={"created_at"})
                            for span in item.evidence
                        ],
                        "source_ids": item.source_ids,
                        "lineage_tokens": item.lineage_tokens,
                    }
                    for item in candidates
                ]
            }
        )
        return PatternMatchContext(candidates=candidates, input_fingerprint=input_fingerprint)

    def _validate_candidate(
        self,
        candidate: PatternOccurrenceCandidate,
        *,
        policy: PatternPromotionPolicy,
    ) -> ValidatedPatternCandidate:
        expected_fingerprint = _sha256_json(
            {
                "pattern_hint": candidate.pattern_hint,
                "extraction_claim_id": candidate.extraction_claim_id,
                "source_claim_ids": candidate.source_claim_ids,
                "evidence_refs": candidate.evidence_refs,
                "dependency_group_id": candidate.dependency_group_id,
            }
        )
        if candidate.candidate_fingerprint != expected_fingerprint:
            raise InvalidPatternMatchInputError(
                f"candidate fingerprint mismatch: {candidate.candidate_id}"
            )
        if candidate.candidate_id != "ocx_" + expected_fingerprint[:40]:
            raise InvalidPatternMatchInputError(
                f"candidate ID is not canonical: {candidate.candidate_id}"
            )
        try:
            claim = self.evidence_graph.get_claim(candidate.extraction_claim_id)
        except UnknownGraphNodeError as exc:
            raise InvalidPatternMatchInputError(
                f"candidate references missing extraction claim: {candidate.extraction_claim_id}"
            ) from exc
        if claim.epistemic_level is not EpistemicLevel.INTERPRETATION:
            raise InvalidPatternMatchInputError("promotion candidate claim must be INTERPRETATION")
        if claim.claim_type not in policy.allowed_claim_types:
            raise InvalidPatternMatchInputError(
                f"promotion candidate claim_type is not allowed: {claim.claim_type}"
            )
        if claim.validation_state is ValidationState.CONTRADICTED:
            raise InvalidPatternMatchInputError("contradicted interpretations cannot be promoted")
        if tuple(sorted(claim.evidence_refs)) != tuple(sorted(candidate.evidence_refs)):
            raise InvalidPatternMatchInputError(
                "candidate evidence_refs do not match extraction claim provenance"
            )

        evidence: list[EvidenceSpan] = []
        for evidence_id in candidate.evidence_refs:
            try:
                evidence.append(self.evidence_graph.get_evidence(evidence_id))
            except UnknownGraphNodeError as exc:
                raise InvalidPatternMatchInputError(
                    f"candidate references missing evidence: {evidence_id}"
                ) from exc
        source_ids = tuple(sorted({span.source_id for span in evidence}))
        expected_dependency = "dep_" + _sha256_json({"source_ids": source_ids})[:32]
        if candidate.dependency_group_id != expected_dependency:
            raise InvalidPatternMatchInputError(
                f"candidate dependency group is not canonical: {candidate.candidate_id}"
            )

        explanation = self.evidence_graph.explain_claim(claim.claim_id)
        derived_targets = {
            edge.target_claim_id
            for edge in explanation.outgoing_relations
            if edge.relation is ClaimRelationType.DERIVED_FROM
        }
        if derived_targets != set(candidate.source_claim_ids):
            raise InvalidPatternMatchInputError(
                "candidate source_claim_ids do not exactly match DERIVED_FROM provenance"
            )

        source_claims: list[Claim] = []
        source_evidence: set[str] = set()
        for source_claim_id in candidate.source_claim_ids:
            try:
                source_claim = self.evidence_graph.get_claim(source_claim_id)
            except UnknownGraphNodeError as exc:
                raise InvalidPatternMatchInputError(
                    f"candidate references missing source claim: {source_claim_id}"
                ) from exc
            if source_claim.epistemic_level is not EpistemicLevel.OBSERVED:
                raise InvalidPatternMatchInputError("candidate provenance claim must be OBSERVED")
            if source_claim.claim_type != "source_text":
                raise InvalidPatternMatchInputError(
                    "candidate provenance claim must have claim_type=source_text"
                )
            if source_claim.validation_state not in {
                ValidationState.SUPPORTED,
                ValidationState.VALIDATED,
            }:
                raise InvalidPatternMatchInputError(
                    "candidate provenance claim must be evidence-supported"
                )
            source_claims.append(source_claim)
            source_evidence.update(source_claim.evidence_refs)
        if source_evidence != set(candidate.evidence_refs):
            raise InvalidPatternMatchInputError(
                "candidate evidence set does not exactly match source-claim provenance"
            )

        return ValidatedPatternCandidate(
            candidate=candidate,
            claim=claim,
            source_claims=tuple(sorted(source_claims, key=lambda item: item.claim_id)),
            evidence=tuple(sorted(evidence, key=lambda item: item.evidence_id)),
            source_ids=source_ids,
            lineage_tokens=tuple(
                sorted(
                    {
                        *(f"source:{span.source_id}" for span in evidence),
                        *(
                            f"source-version:{span.source_id}:{span.source_version_id}"
                            for span in evidence
                        ),
                        *(f"content:{span.content_hash}" for span in evidence),
                    }
                )
            ),
        )

    @staticmethod
    def _independent_components(
        selected: tuple[ValidatedPatternCandidate, ...]
    ) -> tuple[tuple[ValidatedPatternCandidate, ...], ...]:
        ordered = tuple(sorted(selected, key=lambda item: item.candidate.candidate_id))
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
        for index, item in enumerate(ordered):
            tokens = {
                f"dependency:{item.candidate.dependency_group_id}",
                f"claim:{item.claim.claim_id}",
                *(f"evidence:{ref}" for ref in item.candidate.evidence_refs),
                *item.lineage_tokens,
            }
            for token in tokens:
                owner = token_owner.get(token)
                if owner is None:
                    token_owner[token] = index
                else:
                    union(index, owner)

        grouped: dict[int, list[ValidatedPatternCandidate]] = defaultdict(list)
        for index, item in enumerate(ordered):
            grouped[find(index)].append(item)
        return tuple(
            tuple(sorted(items, key=lambda item: item.candidate.candidate_id))
            for _, items in sorted(
                grouped.items(),
                key=lambda pair: min(item.candidate.candidate_id for item in pair[1]),
            )
        )

    def _validate_match_proposal(
        self,
        proposal: PatternMatchProposal,
        *,
        candidate_map: dict[str, ValidatedPatternCandidate],
        policy: PatternPromotionPolicy,
    ) -> tuple[ValidatedPatternCandidate, ...]:
        if len(proposal.candidate_ids) < policy.min_candidates_per_pattern:
            raise InvalidPatternMatchProposalError(
                f"pattern proposal requires at least {policy.min_candidates_per_pattern} candidates"
            )
        if len(proposal.candidate_ids) > policy.max_candidates_per_proposal:
            raise InvalidPatternMatchProposalError(
                f"pattern proposal contains {len(proposal.candidate_ids)} candidates; policy limit is "
                f"{policy.max_candidates_per_proposal}"
            )
        missing = tuple(candidate_id for candidate_id in proposal.candidate_ids if candidate_id not in candidate_map)
        if missing:
            raise InvalidPatternMatchProposalError(
                "pattern proposal references candidates outside validated input: " + ", ".join(missing)
            )
        selected = tuple(candidate_map[item] for item in proposal.candidate_ids)
        components = self._independent_components(selected)
        if len(components) < policy.min_independent_groups:
            raise InvalidPatternMatchProposalError(
                f"pattern proposal has only {len(components)} independent lineage groups; policy requires "
                f"{policy.min_independent_groups}"
            )
        if policy.require_unique_independent_lineage and len(components) != len(selected):
            raise InvalidPatternMatchProposalError(
                "one pattern proposal may contain at most one candidate per independent lineage group"
            )
        if policy.require_exact_hint_match:
            hints = {item.candidate.pattern_hint for item in selected}
            if hints != {proposal.pattern_key}:
                raise InvalidPatternMatchProposalError(
                    "v1 promotion requires every candidate pattern_hint to exactly match pattern_key"
                )
        return selected

    @staticmethod
    def _promote_candidate(
        item: ValidatedPatternCandidate,
        *,
        pattern_key: str,
        manifest: PatternMatcherManifest,
        policy: PatternPromotionPolicy,
    ) -> PromotedPatternOccurrence:
        occurrence_id = "occ_" + _sha256_json(
            {
                "pattern_key": pattern_key,
                "candidate_id": item.candidate.candidate_id,
                "claim_id": item.claim.claim_id,
                "evidence_refs": item.candidate.evidence_refs,
                "dependency_group_id": item.candidate.dependency_group_id,
            }
        )[:40]
        occurrence = PatternOccurrence(
            occurrence_id=occurrence_id,
            claim_id=item.claim.claim_id,
            evidence_refs=item.candidate.evidence_refs,
            dependency_group_id=item.candidate.dependency_group_id,
        )
        promotion_fingerprint = _sha256_json(
            {
                "pattern_key": pattern_key,
                "candidate_id": item.candidate.candidate_id,
                "candidate_fingerprint": item.candidate.candidate_fingerprint,
                "occurrence": occurrence.model_dump(mode="json"),
                "matcher_manifest_fingerprint": manifest.fingerprint,
                "policy_fingerprint": policy.fingerprint,
            }
        )
        return PromotedPatternOccurrence(
            pattern_key=pattern_key,
            candidate_id=item.candidate.candidate_id,
            occurrence=occurrence,
            matcher_manifest_fingerprint=manifest.fingerprint,
            policy_fingerprint=policy.fingerprint,
            promotion_fingerprint=promotion_fingerprint,
        )

    @staticmethod
    def _revalidate_receipt(item: object) -> SemanticExtractionReceipt:
        if not isinstance(item, SemanticExtractionReceipt):
            raise InvalidPatternMatchInputError(
                "promotion input must contain SemanticExtractionReceipt values"
            )
        try:
            return SemanticExtractionReceipt.model_validate(item.model_dump(mode="python"))
        except (ValidationError, TypeError, ValueError) as exc:
            raise InvalidPatternMatchInputError(
                "semantic extraction receipt fails canonical validation"
            ) from exc

    @staticmethod
    def _revalidate_candidate(item: object) -> PatternOccurrenceCandidate:
        if not isinstance(item, PatternOccurrenceCandidate):
            raise InvalidPatternMatchInputError(
                "semantic extraction receipt contains a non-candidate item"
            )
        try:
            return PatternOccurrenceCandidate.model_validate(item.model_dump(mode="python"))
        except (ValidationError, TypeError, ValueError) as exc:
            raise InvalidPatternMatchInputError(
                "pattern occurrence candidate fails canonical validation"
            ) from exc

    @staticmethod
    def _verify_extraction_receipt_fingerprint(receipt: SemanticExtractionReceipt) -> None:
        expected = _sha256_json(
            {
                "ingestion_receipt_fingerprint": receipt.ingestion_receipt_fingerprint,
                "extractor_manifest_fingerprint": receipt.extractor_manifest_fingerprint,
                "policy_fingerprint": receipt.policy_fingerprint,
                "input_fingerprint": receipt.input_fingerprint,
                "proposal_fingerprint": receipt.proposal_fingerprint,
                "derived_claim_ids": receipt.derived_claim_ids,
                "occurrence_candidates": [
                    item.model_dump(mode="json") for item in receipt.occurrence_candidates
                ],
            }
        )
        if expected != receipt.receipt_fingerprint:
            raise InvalidPatternMatchInputError(
                "semantic extraction receipt fingerprint mismatch"
            )

    @staticmethod
    def _revalidate_proposal(item: object) -> PatternMatchProposal:
        if not isinstance(item, PatternMatchProposal):
            raise InvalidPatternMatchProposalError(
                "matcher returned an item that is not PatternMatchProposal"
            )
        try:
            return PatternMatchProposal.model_validate(item.model_dump(mode="python"))
        except (ValidationError, TypeError, ValueError) as exc:
            raise InvalidPatternMatchProposalError(
                "matcher returned a PatternMatchProposal that fails canonical validation"
            ) from exc

    @staticmethod
    def _proposal_fingerprint(proposal: PatternMatchProposal) -> str:
        return _sha256_json(proposal.model_dump(mode="json"))

    @classmethod
    def _proposal_sort_key(cls, proposal: PatternMatchProposal) -> str:
        return cls._proposal_fingerprint(proposal)
