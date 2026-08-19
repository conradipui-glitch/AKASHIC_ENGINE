"""Evidence-grounded semantic claim and occurrence extraction.

Extractors may propose interpretations, but AKASHIC_ENGINE owns provenance,
epistemic level, claim identity, graph relations, and occurrence construction.
A proposal is therefore an untrusted suggestion until it passes this module's
validation firewall.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from threading import RLock
from typing import Protocol

from pydantic import Field, ValidationError, field_validator

from .domain import Claim, EpistemicLevel, EvidenceSpan, FrozenModel, ValidationState
from .evidence_graph import (
    ClaimRelation,
    ClaimRelationType,
    DuplicateGraphNodeError,
    UnknownGraphNodeError,
)
from .ingestion import IngestionReceipt


class ExtractionError(RuntimeError):
    pass


class InvalidExtractionInputError(ExtractionError):
    pass


class InvalidClaimProposalError(ExtractionError):
    pass


class DuplicateClaimProposalError(InvalidClaimProposalError):
    pass


class ExtractionReplayConflictError(ExtractionError):
    pass


_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def _identifier(value: str, field_name: str) -> str:
    cleaned = unicodedata.normalize("NFC", value.strip()).lower()
    if not cleaned or not _IDENTIFIER_RE.fullmatch(cleaned):
        raise ValueError(
            f"{field_name} must use lowercase letters, digits, '.', '_' or '-'"
        )
    return cleaned


def _statement(value: str) -> str:
    cleaned = unicodedata.normalize("NFC", value.strip())
    if not cleaned:
        raise ValueError("statement cannot be empty")
    return cleaned


def _sha256_json(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ExtractorManifest(FrozenModel):
    """Versioned identity for one semantic extractor configuration."""

    extractor_id: str
    extractor_version: str
    configuration_fingerprint: str = Field(min_length=64, max_length=64)

    @field_validator("extractor_id")
    @classmethod
    def normalize_extractor_id(cls, value: str) -> str:
        return _identifier(value, "extractor_id")

    @field_validator("extractor_version")
    @classmethod
    def normalize_extractor_version(cls, value: str) -> str:
        cleaned = unicodedata.normalize("NFC", value.strip())
        if not cleaned:
            raise ValueError("extractor_version cannot be empty")
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


class ExtractionSourceClaim(FrozenModel):
    claim: Claim
    evidence: tuple[EvidenceSpan, ...]


class ClaimExtractionContext(FrozenModel):
    """Closed observed input supplied to an untrusted semantic extractor."""

    # Audit reference only. Delivery metadata does not participate in semantic
    # identity; input_fingerprint below is based on canonical source provenance.
    ingestion_receipt_fingerprint: str = Field(min_length=64, max_length=64)
    source_id: str
    source_version_id: str
    source_claims: tuple[ExtractionSourceClaim, ...]
    input_fingerprint: str = Field(min_length=64, max_length=64)


class ClaimExtractionProposal(FrozenModel):
    """Untrusted semantic suggestion returned by an extractor.

    The extractor cannot choose evidence IDs, epistemic level, claim type,
    domain tags, validation state, producer, or graph relations.
    ``pattern_hint`` is non-authoritative metadata only. It must be validated
    by a later matcher before any PatternOccurrence is created.
    """

    source_claim_ids: tuple[str, ...]
    statement: str = Field(min_length=1, max_length=4096)
    pattern_hint: str | None = Field(default=None, max_length=128)

    @field_validator("source_claim_ids")
    @classmethod
    def validate_source_claim_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(sorted(value))
        if not normalized:
            raise ValueError("source_claim_ids cannot be empty")
        if any(not item for item in normalized):
            raise ValueError("source_claim_ids cannot contain empty IDs")
        if any(len(item) > 256 for item in normalized):
            raise ValueError("source_claim_ids cannot exceed 256 characters")
        if len(set(normalized)) != len(normalized):
            raise ValueError("source_claim_ids must be unique")
        return normalized

    @field_validator("statement")
    @classmethod
    def normalize_statement(cls, value: str) -> str:
        return _statement(value)

    @field_validator("pattern_hint")
    @classmethod
    def normalize_pattern_hint(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _identifier(value, "pattern_hint")


class PatternOccurrenceCandidate(FrozenModel):
    """Unverified occurrence candidate that cannot enter PatternEngine directly.

    ``pattern_hint`` is extractor-supplied routing metadata only. A later
    matcher/validator must promote this record into a real PatternOccurrence.
    """

    candidate_id: str
    pattern_hint: str | None = None
    extraction_claim_id: str
    source_claim_ids: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    dependency_group_id: str
    candidate_fingerprint: str = Field(min_length=64, max_length=64)


class SemanticExtractionReceipt(FrozenModel):
    ingestion_receipt_fingerprint: str = Field(min_length=64, max_length=64)
    extractor_manifest_fingerprint: str = Field(min_length=64, max_length=64)
    policy_fingerprint: str = Field(min_length=64, max_length=64)
    input_fingerprint: str = Field(min_length=64, max_length=64)
    proposal_fingerprint: str = Field(min_length=64, max_length=64)
    derived_claim_ids: tuple[str, ...]
    occurrence_candidates: tuple[PatternOccurrenceCandidate, ...]
    receipt_fingerprint: str = Field(min_length=64, max_length=64)


class SemanticExtractionPolicy(FrozenModel):
    """Engine-owned bounds for accepting semantic proposals."""

    policy_version: str = "1"
    max_source_claims_per_input: int = Field(default=64, ge=1, le=1024)
    max_source_claims_per_proposal: int = Field(default=8, ge=1, le=128)
    max_proposals_per_input: int = Field(default=32, ge=1, le=256)
    derived_claim_type: str = "semantic_interpretation"

    @field_validator("derived_claim_type")
    @classmethod
    def fixed_claim_type(cls, value: str) -> str:
        cleaned = _identifier(value, "derived_claim_type")
        if cleaned in {"source_text", "user_confirmation"}:
            raise ValueError("reserved claim types cannot be used for semantic extraction")
        return cleaned

    @property
    def fingerprint(self) -> str:
        return _sha256_json(self.model_dump(mode="json"))


class ClaimExtractor(Protocol):
    def extract(self, context: ClaimExtractionContext) -> tuple[ClaimExtractionProposal, ...]: ...


class ExtractionEvidenceGraph(Protocol):
    def get_claim(self, claim_id: str) -> Claim: ...

    def get_evidence(self, evidence_id: str) -> EvidenceSpan: ...

    def add_claim(self, claim: Claim) -> Claim: ...

    def add_relation(
        self,
        source_claim_id: str,
        target_claim_id: str,
        relation: ClaimRelationType,
    ) -> ClaimRelation: ...


class _PreparedProposal(FrozenModel):
    proposal: ClaimExtractionProposal
    claim: Claim
    source_claim_ids: tuple[str, ...]
    candidate: PatternOccurrenceCandidate | None = None


class SemanticExtractionEngine:
    """Ingestion receipt -> INTERPRETATION claims + occurrence candidates.

    All untrusted extractor output is validated and prepared before the graph is
    mutated, preventing malformed later proposals from leaving partial memory.
    """

    def __init__(
        self,
        *,
        evidence_graph: ExtractionEvidenceGraph,
        extractor: ClaimExtractor,
        extractor_manifest: ExtractorManifest,
        policy: SemanticExtractionPolicy | None = None,
    ) -> None:
        self.evidence_graph = evidence_graph
        self.extractor = extractor
        self.extractor_manifest = extractor_manifest
        self.policy = policy or SemanticExtractionPolicy()
        self._lock = RLock()

    def extract(self, ingestion_receipt: IngestionReceipt) -> SemanticExtractionReceipt:
        with self._lock:
            # Engine-owned configuration is snapshotted before untrusted code runs.
            manifest = ExtractorManifest.model_validate(
                self.extractor_manifest.model_dump(mode="python")
            )
            policy = SemanticExtractionPolicy.model_validate(
                self.policy.model_dump(mode="python")
            )
            context = self._build_context(ingestion_receipt, policy=policy)
            raw_proposals = self.extractor.extract(context)
            if not isinstance(raw_proposals, tuple):
                raise InvalidClaimProposalError(
                    "extractor must return a tuple of ClaimExtractionProposal"
                )
            if len(raw_proposals) > policy.max_proposals_per_input:
                raise InvalidClaimProposalError(
                    f"extractor returned {len(raw_proposals)} proposals; policy limit is "
                    f"{policy.max_proposals_per_input}"
                )
            proposals = tuple(self._revalidate_proposal(item) for item in raw_proposals)

            ordered = tuple(sorted(proposals, key=self._proposal_sort_key))
            proposal_fingerprints = tuple(self._proposal_fingerprint(item) for item in ordered)
            if len(set(proposal_fingerprints)) != len(proposal_fingerprints):
                raise DuplicateClaimProposalError("duplicate semantic proposals are not allowed")

            source_map = {item.claim.claim_id: item for item in context.source_claims}
            # Phase 1: validate and prepare everything. No graph writes happen here.
            prepared = tuple(
                self._prepare_proposal(
                    proposal,
                    source_map=source_map,
                    context=context,
                    manifest=manifest,
                    policy=policy,
                )
                for proposal in ordered
            )

            claims_by_id: dict[str, Claim] = {}
            candidates: dict[str, PatternOccurrenceCandidate] = {}
            relations: set[tuple[str, str]] = set()
            for item in prepared:
                existing_claim = claims_by_id.get(item.claim.claim_id)
                if existing_claim is not None and existing_claim != item.claim:
                    raise ExtractionReplayConflictError(
                        f"Prepared claim ID resolved to different data: {item.claim.claim_id}"
                    )
                claims_by_id[item.claim.claim_id] = item.claim
                for source_claim_id in item.source_claim_ids:
                    relations.add((item.claim.claim_id, source_claim_id))
                if item.candidate is not None:
                    existing_candidate = candidates.get(item.candidate.candidate_id)
                    if existing_candidate is not None and existing_candidate != item.candidate:
                        raise ExtractionReplayConflictError(
                            "Occurrence candidate ID resolved to different data: "
                            f"{item.candidate.candidate_id}"
                        )
                    candidates[item.candidate.candidate_id] = item.candidate

            # Reject deterministic replay conflicts before introducing any new
            # claims from this batch. A storage transaction will later close the
            # remaining race between this preflight and the commit itself.
            for claim_id in sorted(claims_by_id):
                try:
                    existing = self.evidence_graph.get_claim(claim_id)
                except UnknownGraphNodeError:
                    continue
                if existing != claims_by_id[claim_id]:
                    raise ExtractionReplayConflictError(
                        f"Canonical extracted claim ID resolved to different data: {claim_id}"
                    )

            # Phase 2: commit the fully validated plan in deterministic order.
            for claim_id in sorted(claims_by_id):
                self._ensure_claim(claims_by_id[claim_id])
            for source_claim_id, target_claim_id in sorted(relations):
                self.evidence_graph.add_relation(
                    source_claim_id,
                    target_claim_id,
                    ClaimRelationType.DERIVED_FROM,
                )

            derived_tuple = tuple(sorted(claims_by_id))
            candidate_tuple = tuple(candidates[key] for key in sorted(candidates))
            proposal_fingerprint = _sha256_json(proposal_fingerprints)
            receipt_fingerprint = _sha256_json(
                {
                    "ingestion_receipt_fingerprint": ingestion_receipt.receipt_fingerprint,
                    "extractor_manifest_fingerprint": manifest.fingerprint,
                    "policy_fingerprint": policy.fingerprint,
                    "input_fingerprint": context.input_fingerprint,
                    "proposal_fingerprint": proposal_fingerprint,
                    "derived_claim_ids": derived_tuple,
                    "occurrence_candidates": [
                        item.model_dump(mode="json") for item in candidate_tuple
                    ],
                }
            )
            return SemanticExtractionReceipt(
                ingestion_receipt_fingerprint=ingestion_receipt.receipt_fingerprint,
                extractor_manifest_fingerprint=manifest.fingerprint,
                policy_fingerprint=policy.fingerprint,
                input_fingerprint=context.input_fingerprint,
                proposal_fingerprint=proposal_fingerprint,
                derived_claim_ids=derived_tuple,
                occurrence_candidates=candidate_tuple,
                receipt_fingerprint=receipt_fingerprint,
            )

    def _build_context(
        self, receipt: IngestionReceipt, *, policy: SemanticExtractionPolicy
    ) -> ClaimExtractionContext:
        if not receipt.claim_ids:
            raise InvalidExtractionInputError(
                "ingestion receipt contains no observed claims to extract from"
            )
        if len(set(receipt.claim_ids)) != len(receipt.claim_ids):
            raise InvalidExtractionInputError("ingestion receipt contains duplicate claim IDs")
        if len(set(receipt.evidence_ids)) != len(receipt.evidence_ids):
            raise InvalidExtractionInputError("ingestion receipt contains duplicate evidence IDs")
        if len(receipt.claim_ids) > policy.max_source_claims_per_input:
            raise InvalidExtractionInputError(
                f"ingestion receipt contains {len(receipt.claim_ids)} source claims; policy limit is "
                f"{policy.max_source_claims_per_input}"
            )

        source_claims: list[ExtractionSourceClaim] = []
        all_evidence_ids: set[str] = set()
        for claim_id in sorted(receipt.claim_ids):
            try:
                claim = self.evidence_graph.get_claim(claim_id)
            except UnknownGraphNodeError as exc:
                raise InvalidExtractionInputError(
                    f"ingestion receipt references missing claim: {claim_id}"
                ) from exc
            if claim.epistemic_level is not EpistemicLevel.OBSERVED:
                raise InvalidExtractionInputError(
                    f"extraction input claim must be OBSERVED: {claim_id}"
                )
            if claim.claim_type != "source_text":
                raise InvalidExtractionInputError(
                    f"extraction input claim must have claim_type=source_text: {claim_id}"
                )
            if claim.validation_state not in {
                ValidationState.SUPPORTED,
                ValidationState.VALIDATED,
            }:
                raise InvalidExtractionInputError(
                    f"extraction input claim must be evidence-supported: {claim_id}"
                )
            if not claim.evidence_refs:
                raise InvalidExtractionInputError(
                    f"extraction input claim has no evidence: {claim_id}"
                )

            spans: list[EvidenceSpan] = []
            for evidence_id in claim.evidence_refs:
                if evidence_id not in receipt.evidence_ids:
                    raise InvalidExtractionInputError(
                        f"claim {claim_id} references evidence outside ingestion receipt: {evidence_id}"
                    )
                try:
                    span = self.evidence_graph.get_evidence(evidence_id)
                except UnknownGraphNodeError as exc:
                    raise InvalidExtractionInputError(
                        f"ingestion receipt references missing evidence: {evidence_id}"
                    ) from exc
                if span.source_id != receipt.source_id:
                    raise InvalidExtractionInputError(
                        f"evidence {evidence_id} source_id does not match ingestion receipt"
                    )
                if span.source_version_id != receipt.source_version_id:
                    raise InvalidExtractionInputError(
                        f"evidence {evidence_id} source_version_id does not match ingestion receipt"
                    )
                spans.append(span)
                all_evidence_ids.add(evidence_id)
            source_claims.append(
                ExtractionSourceClaim(
                    claim=claim,
                    evidence=tuple(sorted(spans, key=lambda item: item.evidence_id)),
                )
            )

        if all_evidence_ids != set(receipt.evidence_ids):
            raise InvalidExtractionInputError(
                "ingestion receipt evidence set does not exactly match source claim provenance"
            )
        source_tuple = tuple(source_claims)
        input_fingerprint = _sha256_json(
            {
                # Delivery/audit metadata such as URI and receipt fingerprint are
                # intentionally excluded. The same canonical source state must
                # produce the same semantic input identity after replay.
                "source_id": receipt.source_id,
                "source_version_id": receipt.source_version_id,
                "source_claims": [
                    {
                        "claim": item.claim.model_dump(
                            mode="json", exclude={"created_at"}
                        ),
                        "evidence": [
                            span.model_dump(mode="json", exclude={"created_at"})
                            for span in item.evidence
                        ],
                    }
                    for item in source_tuple
                ],
            }
        )
        return ClaimExtractionContext(
            ingestion_receipt_fingerprint=receipt.receipt_fingerprint,
            source_id=receipt.source_id,
            source_version_id=receipt.source_version_id,
            source_claims=source_tuple,
            input_fingerprint=input_fingerprint,
        )

    def _prepare_proposal(
        self,
        proposal: ClaimExtractionProposal,
        *,
        source_map: dict[str, ExtractionSourceClaim],
        context: ClaimExtractionContext,
        manifest: ExtractorManifest,
        policy: SemanticExtractionPolicy,
    ) -> _PreparedProposal:
        if len(proposal.source_claim_ids) > policy.max_source_claims_per_proposal:
            raise InvalidClaimProposalError(
                f"proposal references {len(proposal.source_claim_ids)} source claims; policy limit is "
                f"{policy.max_source_claims_per_proposal}"
            )
        missing = tuple(item for item in proposal.source_claim_ids if item not in source_map)
        if missing:
            raise InvalidClaimProposalError(
                "proposal references claims outside extraction input: " + ", ".join(missing)
            )

        evidence_ids = tuple(
            sorted(
                {
                    evidence.evidence_id
                    for claim_id in proposal.source_claim_ids
                    for evidence in source_map[claim_id].evidence
                }
            )
        )
        producer = (
            f"extractor:{manifest.extractor_id}"
            f"@{manifest.extractor_version}:"
            f"config-{manifest.configuration_fingerprint[:12]}"
        )
        claim_id = "clx_" + _sha256_json(
            {
                "extractor_manifest_fingerprint": manifest.fingerprint,
                "policy_fingerprint": policy.fingerprint,
                "input_fingerprint": context.input_fingerprint,
                "source_claim_ids": proposal.source_claim_ids,
                "statement": proposal.statement,
                "evidence_refs": evidence_ids,
                "claim_type": policy.derived_claim_type,
                "epistemic_level": EpistemicLevel.INTERPRETATION.value,
            }
        )[:40]

        source_items = tuple(source_map[item] for item in proposal.source_claim_ids)
        # created_at is record metadata rather than event time. Reuse the
        # latest source-claim creation timestamp so repeated extraction against
        # one stored graph remains idempotent without conflating observed_at.
        created_at = max(item.claim.created_at for item in source_items)
        claim = Claim(
            claim_id=claim_id,
            statement=proposal.statement,
            claim_type=policy.derived_claim_type,
            producer=producer,
            evidence_refs=evidence_ids,
            domain_tags=(),
            epistemic_level=EpistemicLevel.INTERPRETATION,
            validation_state=ValidationState.UNVERIFIED,
            created_at=created_at,
        )
        candidate = None
        if proposal.pattern_hint is not None:
            candidate = self._occurrence_candidate(
                proposal=proposal,
                claim=claim,
                source_map=source_map,
            )
        return _PreparedProposal(
            proposal=proposal,
            claim=claim,
            source_claim_ids=proposal.source_claim_ids,
            candidate=candidate,
        )

    def _occurrence_candidate(
        self,
        *,
        proposal: ClaimExtractionProposal,
        claim: Claim,
        source_map: dict[str, ExtractionSourceClaim],
    ) -> PatternOccurrenceCandidate:
        source_ids = tuple(
            sorted(
                {
                    evidence.source_id
                    for claim_id in proposal.source_claim_ids
                    for evidence in source_map[claim_id].evidence
                }
            )
        )
        dependency_group_id = "dep_" + _sha256_json(
            {"source_ids": source_ids}
        )[:32]
        candidate_fingerprint = _sha256_json(
            {
                "pattern_hint": proposal.pattern_hint,
                "extraction_claim_id": claim.claim_id,
                "source_claim_ids": proposal.source_claim_ids,
                "evidence_refs": claim.evidence_refs,
                "dependency_group_id": dependency_group_id,
            }
        )
        return PatternOccurrenceCandidate(
            candidate_id="ocx_" + candidate_fingerprint[:40],
            pattern_hint=proposal.pattern_hint,
            extraction_claim_id=claim.claim_id,
            source_claim_ids=proposal.source_claim_ids,
            evidence_refs=claim.evidence_refs,
            dependency_group_id=dependency_group_id,
            candidate_fingerprint=candidate_fingerprint,
        )

    def _ensure_claim(self, claim: Claim) -> Claim:
        try:
            existing = self.evidence_graph.get_claim(claim.claim_id)
        except UnknownGraphNodeError:
            try:
                return self.evidence_graph.add_claim(claim)
            except DuplicateGraphNodeError:
                try:
                    raced = self.evidence_graph.get_claim(claim.claim_id)
                except UnknownGraphNodeError as exc:
                    raise ExtractionReplayConflictError(
                        f"Claim identity raced or conflicted: {claim.claim_id}"
                    ) from exc
                if raced != claim:
                    raise ExtractionReplayConflictError(
                        f"Canonical extracted claim ID resolved to different data: {claim.claim_id}"
                    )
                return raced
        if existing != claim:
            raise ExtractionReplayConflictError(
                f"Canonical extracted claim ID resolved to different data: {claim.claim_id}"
            )
        return existing

    @staticmethod
    def _revalidate_proposal(item: object) -> ClaimExtractionProposal:
        if not isinstance(item, ClaimExtractionProposal):
            raise InvalidClaimProposalError(
                "extractor returned an item that is not ClaimExtractionProposal"
            )
        try:
            return ClaimExtractionProposal.model_validate(
                item.model_dump(mode="python")
            )
        except (ValidationError, TypeError, ValueError) as exc:
            raise InvalidClaimProposalError(
                "extractor returned a ClaimExtractionProposal that fails canonical validation"
            ) from exc

    @staticmethod
    def _proposal_fingerprint(proposal: ClaimExtractionProposal) -> str:
        return _sha256_json(proposal.model_dump(mode="json"))

    @classmethod
    def _proposal_sort_key(cls, proposal: ClaimExtractionProposal) -> str:
        return cls._proposal_fingerprint(proposal)
