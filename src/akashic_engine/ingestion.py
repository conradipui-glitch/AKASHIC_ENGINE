"""Adapter identity contracts and deterministic runtime ingestion.

Adapters do not supply canonical SourceStore identifiers. They register a
versioned identity schema and provide values for that schema; AKASHIC_ENGINE
then derives the SourceStore namespace/external key and creates source-bound
evidence and literal observed claims.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from threading import RLock
from typing import Protocol

from pydantic import Field, field_validator

from .domain import Claim, EpistemicLevel, EvidenceSpan, FrozenModel, ValidationState
from .evidence_graph import (
    DuplicateGraphNodeError,
    UnknownGraphNodeError,
)
from .source_store import SourceRecord, SourceVersion


class IngestionError(RuntimeError):
    pass


class UnknownAdapterIdentityError(IngestionError):
    pass


class AdapterIdentityConflictError(IngestionError):
    pass


class InvalidAdapterIdentityError(IngestionError):
    pass


class DuplicateObservationError(IngestionError):
    pass


class IngestionReplayConflictError(IngestionError):
    pass


def _clean(value: str, field_name: str, *, lower: bool = False) -> str:
    cleaned = unicodedata.normalize("NFC", value.strip())
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty")
    return cleaned.lower() if lower else cleaned


_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def _identifier(value: str, field_name: str) -> str:
    cleaned = _clean(value, field_name, lower=True)
    if not _IDENTIFIER_RE.fullmatch(cleaned):
        raise ValueError(
            f"{field_name} must use only lowercase letters, digits, '.', '_' or '-'"
        )
    return cleaned


def _utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("observed_at must be timezone-aware")
    return value.astimezone(timezone.utc)


class AdapterIdentityManifest(FrozenModel):
    """Versioned declaration of how an adapter identifies one upstream object."""

    adapter_id: str
    authority_id: str
    identity_version: int = Field(ge=1)
    identity_fields: tuple[str, ...]
    allowed_source_types: tuple[str, ...] = ()

    @field_validator("adapter_id")
    @classmethod
    def normalize_adapter_id(cls, value: str) -> str:
        return _identifier(value, "adapter_id")

    @field_validator("authority_id")
    @classmethod
    def preserve_opaque_authority_id(cls, value: str) -> str:
        if value == "":
            raise ValueError("authority_id cannot be empty")
        return value

    @field_validator("identity_fields")
    @classmethod
    def normalize_identity_fields(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(sorted(_identifier(item, "identity_field") for item in value))
        if not normalized:
            raise ValueError("identity_fields cannot be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("identity_fields must be unique")
        return normalized

    @field_validator("allowed_source_types")
    @classmethod
    def normalize_source_types(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(sorted(_clean(item, "source_type", lower=True) for item in value))
        if len(set(normalized)) != len(normalized):
            raise ValueError("allowed_source_types must be unique")
        return normalized


class AdapterIdentityValue(FrozenModel):
    field: str
    value: str

    @field_validator("field")
    @classmethod
    def normalize_field(cls, value: str) -> str:
        return _identifier(value, "identity field")

    @field_validator("value")
    @classmethod
    def preserve_opaque_value(cls, value: str) -> str:
        if value == "":
            raise ValueError("identity value cannot be empty")
        return value


class DirectObservation(FrozenModel):
    """Literal observed text selected from the immutable source bytes.

    The caller cannot provide an arbitrary claim statement. The resulting
    OBSERVED claim statement is exactly ``excerpt`` after SourceStore binding.
    """

    excerpt: str
    start_offset: int | None = None
    end_offset: int | None = None

    @field_validator("excerpt")
    @classmethod
    def nonempty_excerpt(cls, value: str) -> str:
        if not value:
            raise ValueError("observation excerpt cannot be empty")
        return value


class IngestionEnvelope(FrozenModel):
    """Adapter-produced data before canonical source identity is assigned."""

    adapter_id: str
    authority_id: str
    identity_version: int = Field(ge=1)
    identity_values: tuple[AdapterIdentityValue, ...]
    source_type: str
    content: bytes
    mime_type: str
    observed_at: datetime | None = None
    uri: str | None = None
    observations: tuple[DirectObservation, ...] = ()

    @field_validator("adapter_id")
    @classmethod
    def normalize_adapter_id(cls, value: str) -> str:
        return _identifier(value, "adapter_id")

    @field_validator("authority_id")
    @classmethod
    def preserve_opaque_authority_id(cls, value: str) -> str:
        if value == "":
            raise ValueError("authority_id cannot be empty")
        return value

    @field_validator("source_type", "mime_type")
    @classmethod
    def normalize_labels(cls, value: str, info) -> str:
        return _clean(value, info.field_name, lower=True)

    @field_validator("observed_at")
    @classmethod
    def canonicalize_observed_at(cls, value: datetime | None) -> datetime | None:
        return _utc(value)

    @field_validator("uri")
    @classmethod
    def normalize_uri(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _clean(value, "uri")


class IngestionReceipt(FrozenModel):
    adapter_id: str
    identity_version: int
    authority_fingerprint: str = Field(min_length=64, max_length=64)
    namespace: str
    external_key: str
    schema_fingerprint: str = Field(min_length=64, max_length=64)
    manifest_fingerprint: str = Field(min_length=64, max_length=64)
    identity_fingerprint: str = Field(min_length=64, max_length=64)
    envelope_fingerprint: str = Field(min_length=64, max_length=64)
    source_id: str
    source_version_id: str
    evidence_ids: tuple[str, ...]
    claim_ids: tuple[str, ...]
    receipt_fingerprint: str = Field(min_length=64, max_length=64)


class IngestionSourceStore(Protocol):
    def register_source(
        self,
        *,
        namespace: str,
        external_key: str,
        source_type: str,
        uri: str | None = None,
    ) -> SourceRecord: ...

    def ingest_version(
        self,
        source_id: str,
        *,
        content: bytes,
        mime_type: str,
        observed_at: datetime | None = None,
    ) -> SourceVersion: ...

    def make_evidence_span(
        self,
        source_version_id: str,
        *,
        start_offset: int | None = None,
        end_offset: int | None = None,
        excerpt: str | None = None,
        extraction_method: str = "direct",
    ) -> EvidenceSpan: ...

    def assert_evidence_span(self, span: EvidenceSpan) -> None: ...


class IngestionEvidenceGraph(Protocol):
    def add_evidence(self, span: EvidenceSpan) -> EvidenceSpan: ...

    def get_evidence(self, evidence_id: str) -> EvidenceSpan: ...

    def add_claim(self, claim: Claim) -> Claim: ...

    def get_claim(self, claim_id: str) -> Claim: ...


class AdapterIdentityRegistry:
    """Immutable registry for adapter identity schemes."""

    def __init__(self) -> None:
        self._manifests: dict[tuple[str, str, int], AdapterIdentityManifest] = {}
        self._lock = RLock()

    def register(self, manifest: AdapterIdentityManifest) -> AdapterIdentityManifest:
        key = (manifest.adapter_id, manifest.authority_id, manifest.identity_version)
        with self._lock:
            existing = self._manifests.get(key)
            if existing is not None:
                if existing != manifest:
                    raise AdapterIdentityConflictError(
                        "An adapter identity version cannot be rebound to a different schema"
                    )
                return existing
            self._manifests[key] = manifest
            return manifest

    def get(
        self, adapter_id: str, authority_id: str, identity_version: int
    ) -> AdapterIdentityManifest:
        if authority_id == "":
            raise ValueError("authority_id cannot be empty")
        key = (_identifier(adapter_id, "adapter_id"), authority_id, identity_version)
        with self._lock:
            try:
                return self._manifests[key]
            except KeyError as exc:
                raise UnknownAdapterIdentityError(
                    f"Unknown adapter authority/identity scheme: {key[0]} v{identity_version}"
                ) from exc

    def resolve_identity(
        self,
        manifest: AdapterIdentityManifest,
        identity_values: tuple[AdapterIdentityValue, ...],
    ) -> tuple[tuple[str, str], ...]:
        values: dict[str, str] = {}
        for item in identity_values:
            if item.field in values:
                raise InvalidAdapterIdentityError(
                    f"Duplicate adapter identity field: {item.field}"
                )
            values[item.field] = item.value

        expected = set(manifest.identity_fields)
        supplied = set(values)
        if supplied != expected:
            missing = sorted(expected - supplied)
            extra = sorted(supplied - expected)
            details = []
            if missing:
                details.append(f"missing={','.join(missing)}")
            if extra:
                details.append(f"extra={','.join(extra)}")
            raise InvalidAdapterIdentityError(
                "Adapter identity values do not match registered schema: " + "; ".join(details)
            )
        return tuple((field, values[field]) for field in manifest.identity_fields)

    @staticmethod
    def schema_fingerprint(manifest: AdapterIdentityManifest) -> str:
        return _sha256_json(
            {
                "adapter_id": manifest.adapter_id,
                "identity_version": manifest.identity_version,
                "identity_fields": manifest.identity_fields,
            }
        )

    @staticmethod
    def authority_fingerprint(manifest: AdapterIdentityManifest) -> str:
        return _sha256_json(
            {
                "adapter_id": manifest.adapter_id,
                "authority_id": manifest.authority_id,
            }
        )

    @staticmethod
    def manifest_fingerprint(manifest: AdapterIdentityManifest) -> str:
        return _sha256_json(manifest.model_dump(mode="json"))

    @classmethod
    def namespace(cls, manifest: AdapterIdentityManifest) -> str:
        schema = cls.schema_fingerprint(manifest)
        authority = cls.authority_fingerprint(manifest)
        return (
            f"adapter:{manifest.adapter_id}:authority-{authority[:12]}:"
            f"identity-v{manifest.identity_version}:schema-{schema[:12]}"
        )

    @staticmethod
    def external_key(identity: tuple[tuple[str, str], ...]) -> str:
        return json.dumps(dict(identity), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def identity_fingerprint(
        cls,
        manifest: AdapterIdentityManifest,
        identity: tuple[tuple[str, str], ...],
    ) -> str:
        payload = {
            "authority_fingerprint": cls.authority_fingerprint(manifest),
            "schema_fingerprint": cls.schema_fingerprint(manifest),
            "identity": identity,
        }
        return _sha256_json(payload)


class IngestionPipeline:
    """Deterministic adapter-envelope → SourceStore → EvidenceGraph pipeline."""

    def __init__(
        self,
        *,
        registry: AdapterIdentityRegistry,
        source_store: IngestionSourceStore,
        evidence_graph: IngestionEvidenceGraph,
    ) -> None:
        self.registry = registry
        self.source_store = source_store
        self.evidence_graph = evidence_graph
        self._lock = RLock()

    def ingest(self, envelope: IngestionEnvelope) -> IngestionReceipt:
        with self._lock:
            manifest = self.registry.get(
                envelope.adapter_id, envelope.authority_id, envelope.identity_version
            )
            if manifest.allowed_source_types and envelope.source_type not in manifest.allowed_source_types:
                raise InvalidAdapterIdentityError(
                    f"source_type {envelope.source_type!r} is not allowed by adapter schema"
                )

            identity = self.registry.resolve_identity(manifest, envelope.identity_values)
            schema_fingerprint = self.registry.schema_fingerprint(manifest)
            authority_fingerprint = self.registry.authority_fingerprint(manifest)
            manifest_fingerprint = self.registry.manifest_fingerprint(manifest)
            namespace = self.registry.namespace(manifest)
            external_key = self.registry.external_key(identity)
            identity_fingerprint = self.registry.identity_fingerprint(manifest, identity)
            envelope_fingerprint = self._envelope_fingerprint(
                envelope=envelope,
                identity=identity,
                authority_fingerprint=authority_fingerprint,
                manifest_fingerprint=manifest_fingerprint,
            )
            self._reject_duplicate_observations(envelope.observations)

            source = self.source_store.register_source(
                namespace=namespace,
                external_key=external_key,
                source_type=envelope.source_type,
                uri=envelope.uri,
            )
            version = self.source_store.ingest_version(
                source.source_id,
                content=envelope.content,
                mime_type=envelope.mime_type,
                observed_at=envelope.observed_at,
            )

            evidence_ids: set[str] = set()
            claim_ids: set[str] = set()
            producer = (
                f"adapter:{manifest.adapter_id}@authority-{authority_fingerprint[:12]}:"
                f"identity-v{manifest.identity_version}"
            )
            for observation in sorted(
                envelope.observations,
                key=self._observation_sort_key,
            ):
                span = self.source_store.make_evidence_span(
                    version.source_version_id,
                    start_offset=observation.start_offset,
                    end_offset=observation.end_offset,
                    excerpt=observation.excerpt,
                    extraction_method="direct",
                )
                self.source_store.assert_evidence_span(span)
                span = self._ensure_evidence(span)
                evidence_ids.add(span.evidence_id)

                claim = Claim(
                    claim_id=self._claim_id(
                        evidence_id=span.evidence_id,
                        statement=observation.excerpt,
                        claim_type="source_text",
                        domain_tags=(),
                        producer=producer,
                    ),
                    statement=observation.excerpt,
                    claim_type="source_text",
                    producer=producer,
                    evidence_refs=(span.evidence_id,),
                    domain_tags=(),
                    epistemic_level=EpistemicLevel.OBSERVED,
                    validation_state=ValidationState.SUPPORTED,
                    created_at=version.created_at,
                )
                claim = self._ensure_claim(claim)
                claim_ids.add(claim.claim_id)

            evidence_tuple = tuple(sorted(evidence_ids))
            claim_tuple = tuple(sorted(claim_ids))
            receipt_fingerprint = self._receipt_fingerprint(
                adapter_id=manifest.adapter_id,
                identity_version=manifest.identity_version,
                authority_fingerprint=authority_fingerprint,
                schema_fingerprint=schema_fingerprint,
                manifest_fingerprint=manifest_fingerprint,
                identity_fingerprint=identity_fingerprint,
                envelope_fingerprint=envelope_fingerprint,
                source_id=source.source_id,
                source_version_id=version.source_version_id,
                evidence_ids=evidence_tuple,
                claim_ids=claim_tuple,
            )
            return IngestionReceipt(
                adapter_id=manifest.adapter_id,
                identity_version=manifest.identity_version,
                authority_fingerprint=authority_fingerprint,
                namespace=namespace,
                external_key=external_key,
                schema_fingerprint=schema_fingerprint,
                manifest_fingerprint=manifest_fingerprint,
                identity_fingerprint=identity_fingerprint,
                envelope_fingerprint=envelope_fingerprint,
                source_id=source.source_id,
                source_version_id=version.source_version_id,
                evidence_ids=evidence_tuple,
                claim_ids=claim_tuple,
                receipt_fingerprint=receipt_fingerprint,
            )

    def _ensure_evidence(self, span: EvidenceSpan) -> EvidenceSpan:
        try:
            existing = self.evidence_graph.get_evidence(span.evidence_id)
        except UnknownGraphNodeError:
            try:
                return self.evidence_graph.add_evidence(span)
            except DuplicateGraphNodeError:
                try:
                    raced = self.evidence_graph.get_evidence(span.evidence_id)
                except UnknownGraphNodeError as exc:
                    raise IngestionReplayConflictError(
                        f"Evidence identity raced or conflicted: {span.evidence_id}"
                    ) from exc
                if raced != span:
                    raise IngestionReplayConflictError(
                        f"Canonical evidence ID resolved to different data: {span.evidence_id}"
                    )
                return raced
        if existing != span:
            raise IngestionReplayConflictError(
                f"Canonical evidence ID resolved to different data: {span.evidence_id}"
            )
        return existing

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
                    raise IngestionReplayConflictError(
                        f"Claim identity raced or conflicted: {claim.claim_id}"
                    ) from exc
                if raced != claim:
                    raise IngestionReplayConflictError(
                        f"Canonical claim ID resolved to different data: {claim.claim_id}"
                    )
                return raced
        if existing != claim:
            raise IngestionReplayConflictError(
                f"Canonical claim ID resolved to different data: {claim.claim_id}"
            )
        return existing

    @staticmethod
    def _reject_duplicate_observations(observations: tuple[DirectObservation, ...]) -> None:
        fingerprints = [_sha256_json(item.model_dump(mode="json")) for item in observations]
        if len(set(fingerprints)) != len(fingerprints):
            raise DuplicateObservationError("Duplicate observation specs are not allowed")

    @staticmethod
    def _observation_sort_key(observation: DirectObservation) -> str:
        return _sha256_json(observation.model_dump(mode="json"))

    @staticmethod
    def _claim_id(
        *,
        evidence_id: str,
        statement: str,
        claim_type: str,
        domain_tags: tuple[str, ...],
        producer: str,
    ) -> str:
        return "cl_" + _sha256_json(
            {
                "evidence_id": evidence_id,
                "statement": statement,
                "claim_type": claim_type,
                "domain_tags": domain_tags,
                "producer": producer,
                "epistemic_level": EpistemicLevel.OBSERVED.value,
            }
        )[:40]

    @staticmethod
    def _envelope_fingerprint(
        *,
        envelope: IngestionEnvelope,
        identity: tuple[tuple[str, str], ...],
        authority_fingerprint: str,
        manifest_fingerprint: str,
    ) -> str:
        payload = {
            "manifest_fingerprint": manifest_fingerprint,
            "authority_fingerprint": authority_fingerprint,
            "adapter_id": envelope.adapter_id,
            "identity_version": envelope.identity_version,
            "identity": identity,
            "source_type": envelope.source_type,
            "content_hash": hashlib.sha256(envelope.content).hexdigest(),
            "byte_length": len(envelope.content),
            "mime_type": envelope.mime_type,
            "observed_at": envelope.observed_at.isoformat() if envelope.observed_at else None,
            "uri": envelope.uri,
            "observations": sorted(
                (item.model_dump(mode="json") for item in envelope.observations),
                key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
            ),
        }
        return _sha256_json(payload)

    @staticmethod
    def _receipt_fingerprint(
        *,
        adapter_id: str,
        identity_version: int,
        authority_fingerprint: str,
        schema_fingerprint: str,
        manifest_fingerprint: str,
        identity_fingerprint: str,
        envelope_fingerprint: str,
        source_id: str,
        source_version_id: str,
        evidence_ids: tuple[str, ...],
        claim_ids: tuple[str, ...],
    ) -> str:
        return _sha256_json(
            {
                "adapter_id": adapter_id,
                "identity_version": identity_version,
                "authority_fingerprint": authority_fingerprint,
                "schema_fingerprint": schema_fingerprint,
                "manifest_fingerprint": manifest_fingerprint,
                "identity_fingerprint": identity_fingerprint,
                "envelope_fingerprint": envelope_fingerprint,
                "source_id": source_id,
                "source_version_id": source_version_id,
                "evidence_ids": evidence_ids,
                "claim_ids": claim_ids,
            }
        )


def _sha256_json(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
