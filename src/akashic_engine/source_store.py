"""Immutable source identity, versioning, and provenance binding.

The SourceStore owns canonical source/version identifiers and content hashes.
Callers provide adapter-scoped external identity plus raw bytes; downstream
EvidenceSpan objects can then be verified against immutable stored versions.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from datetime import datetime, timezone
from threading import RLock
from pydantic import Field, field_validator

from .domain import EvidenceSpan, FrozenModel, utc_now


class SourceStoreError(RuntimeError):
    pass


class UnknownSourceError(SourceStoreError):
    pass


class UnknownSourceVersionError(SourceStoreError):
    pass


class SourceIdentityConflictError(SourceStoreError):
    pass


class SourceIntegrityError(SourceStoreError):
    pass


class InvalidSourceEvidenceError(SourceStoreError):
    pass


class SourceRecord(FrozenModel):
    """Immutable canonical identity for an upstream source lineage."""

    source_id: str
    namespace: str
    external_key: str
    source_type: str
    uri: str | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("namespace", "source_type")
    @classmethod
    def nonempty_identity_labels(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("source identity labels cannot be empty")
        return cleaned

    @field_validator("external_key")
    @classmethod
    def preserve_opaque_external_key(cls, value: str) -> str:
        if value == "":
            raise ValueError("external_key cannot be empty")
        return value


class SourceVersion(FrozenModel):
    """Immutable metadata for one captured source state.

    Raw bytes live in the store adapter. content_hash and source_version_id are
    adapter-computed and therefore are never accepted from ingestion callers.
    """

    source_version_id: str
    source_id: str
    version_index: int = Field(ge=1)
    content_hash: str = Field(min_length=64, max_length=64)
    byte_length: int = Field(ge=0)
    mime_type: str
    observed_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("mime_type")
    @classmethod
    def nonempty_mime_type(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if not cleaned:
            raise ValueError("mime_type cannot be empty")
        return cleaned

    @field_validator("observed_at")
    @classmethod
    def observed_at_must_be_timezone_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("observed_at must be timezone-aware")
        return value.astimezone(timezone.utc) if value is not None else None


class SourceArtifact(FrozenModel):
    """Rebuildable derivative linked to, but never replacing, a SourceVersion."""

    artifact_id: str
    source_version_id: str
    artifact_type: str
    processor: str
    processor_version: str
    content_hash: str = Field(min_length=64, max_length=64)
    byte_length: int = Field(ge=0)
    mime_type: str
    created_at: datetime = Field(default_factory=utc_now)


class InMemorySourceStore:
    """Reference immutable SourceStore.

    Source identity is deterministic from ``namespace`` + ``external_key``.
    Source version identity is deterministic from source identity, content hash,
    canonical MIME type, and the upstream observation timestamp when available.
    """

    def __init__(self) -> None:
        self._sources: dict[str, SourceRecord] = {}
        self._source_identity: dict[tuple[str, str], str] = {}
        self._versions: dict[str, SourceVersion] = {}
        self._version_ids_by_source: dict[str, list[str]] = {}
        self._content_by_version: dict[str, bytes] = {}
        self._artifacts: dict[str, SourceArtifact] = {}
        self._artifact_content: dict[str, bytes] = {}
        self._lock = RLock()

    def register_source(
        self,
        *,
        namespace: str,
        external_key: str,
        source_type: str,
        uri: str | None = None,
    ) -> SourceRecord:
        namespace = self._clean(namespace, "namespace").lower()
        external_key = self._opaque_key(external_key)
        source_type = self._clean(source_type, "source_type").lower()
        identity = (namespace, external_key)
        source_id = self._source_id(namespace, external_key)

        with self._lock:
            existing_id = self._source_identity.get(identity)
            if existing_id is not None:
                existing = self._sources[existing_id]
                if existing.source_type != source_type:
                    raise SourceIdentityConflictError(
                        "Existing source identity cannot be rebound to a different source_type"
                    )
                return existing

            collision = self._sources.get(source_id)
            if collision is not None:
                raise SourceIntegrityError(
                    "Deterministic source_id collision across different source identities"
                )

            record = SourceRecord(
                source_id=source_id,
                namespace=namespace,
                external_key=external_key,
                source_type=source_type,
                uri=uri,
            )
            self._sources[source_id] = record
            self._source_identity[identity] = source_id
            self._version_ids_by_source[source_id] = []
            return record

    def ingest_version(
        self,
        source_id: str,
        *,
        content: bytes,
        mime_type: str,
        observed_at: datetime | None = None,
    ) -> SourceVersion:
        if not isinstance(content, bytes):
            raise TypeError("content must be bytes")
        mime_type = self._clean(mime_type, "mime_type").lower()
        observed_at = self._canonical_timestamp(observed_at)

        with self._lock:
            self._require_source(source_id)
            content_hash = self._sha256(content)
            source_version_id = self._source_version_id(
                source_id=source_id,
                content_hash=content_hash,
                mime_type=mime_type,
                observed_at=observed_at,
            )
            existing = self._versions.get(source_version_id)
            if existing is not None:
                stored = self._content_by_version[source_version_id]
                if (
                    stored != content
                    or existing.source_id != source_id
                    or existing.content_hash != content_hash
                    or existing.byte_length != len(content)
                    or existing.mime_type != mime_type
                    or existing.observed_at != observed_at
                ):
                    raise SourceIntegrityError(
                        "Deterministic source_version_id resolved to conflicting content"
                    )
                return existing

            version_ids = self._version_ids_by_source[source_id]
            version = SourceVersion(
                source_version_id=source_version_id,
                source_id=source_id,
                version_index=len(version_ids) + 1,
                content_hash=content_hash,
                byte_length=len(content),
                mime_type=mime_type,
                observed_at=observed_at,
            )
            self._versions[source_version_id] = version
            self._content_by_version[source_version_id] = bytes(content)
            version_ids.append(source_version_id)
            return version

    def add_artifact(
        self,
        source_version_id: str,
        *,
        artifact_type: str,
        processor: str,
        processor_version: str,
        content: bytes,
        mime_type: str,
    ) -> SourceArtifact:
        if not isinstance(content, bytes):
            raise TypeError("content must be bytes")
        artifact_type = self._clean(artifact_type, "artifact_type")
        processor = self._clean(processor, "processor")
        processor_version = self._clean(processor_version, "processor_version")
        mime_type = self._clean(mime_type, "mime_type").lower()

        with self._lock:
            self._require_version(source_version_id)
            content_hash = self._sha256(content)
            artifact_id = self._artifact_id(
                source_version_id=source_version_id,
                artifact_type=artifact_type,
                processor=processor,
                processor_version=processor_version,
                content_hash=content_hash,
                mime_type=mime_type,
            )
            existing = self._artifacts.get(artifact_id)
            if existing is not None:
                if self._artifact_content[artifact_id] != content:
                    raise SourceIntegrityError("Artifact identity collision")
                return existing

            artifact = SourceArtifact(
                artifact_id=artifact_id,
                source_version_id=source_version_id,
                artifact_type=artifact_type,
                processor=processor,
                processor_version=processor_version,
                content_hash=content_hash,
                byte_length=len(content),
                mime_type=mime_type,
            )
            self._artifacts[artifact_id] = artifact
            self._artifact_content[artifact_id] = bytes(content)
            return artifact

    def get_source(self, source_id: str) -> SourceRecord:
        with self._lock:
            return self._require_source(source_id)

    def get_version(self, source_version_id: str) -> SourceVersion:
        with self._lock:
            return self._require_version(source_version_id)

    def versions_for_source(self, source_id: str) -> tuple[SourceVersion, ...]:
        with self._lock:
            self._require_source(source_id)
            return tuple(
                self._versions[version_id]
                for version_id in self._version_ids_by_source[source_id]
            )

    def read_content(self, source_version_id: str) -> bytes:
        with self._lock:
            self.verify_version(source_version_id)
            return bytes(self._content_by_version[source_version_id])

    def read_artifact_content(self, artifact_id: str) -> bytes:
        with self._lock:
            try:
                return bytes(self._artifact_content[artifact_id])
            except KeyError as exc:
                raise SourceStoreError(f"Unknown artifact: {artifact_id}") from exc

    def verify_version(self, source_version_id: str) -> SourceVersion:
        with self._lock:
            version = self._require_version(source_version_id)
            content = self._content_by_version[source_version_id]
            actual_hash = self._sha256(content)
            if actual_hash != version.content_hash or len(content) != version.byte_length:
                raise SourceIntegrityError(
                    f"Stored source content failed integrity check: {source_version_id}"
                )
            expected_id = self._source_version_id(
                source_id=version.source_id,
                content_hash=version.content_hash,
                mime_type=version.mime_type,
                observed_at=version.observed_at,
            )
            if expected_id != version.source_version_id:
                raise SourceIntegrityError(
                    f"Source version identity failed integrity check: {source_version_id}"
                )
            return version

    def make_evidence_span(
        self,
        source_version_id: str,
        *,
        start_offset: int | None = None,
        end_offset: int | None = None,
        excerpt: str | None = None,
        extraction_method: str = "direct",
    ) -> EvidenceSpan:
        """Create evidence bound to exact immutable source bytes.

        Offsets are byte offsets. When ``excerpt`` is provided without offsets,
        the store locates its unique UTF-8 byte sequence and records the exact
        byte range. Ambiguous excerpts require explicit offsets.
        """

        with self._lock:
            version = self.verify_version(source_version_id)
            content = self._content_by_version[source_version_id]
            start_offset, end_offset = self._bind_excerpt(
                content=content,
                start_offset=start_offset,
                end_offset=end_offset,
                excerpt=excerpt,
            )
            self._validate_offsets(
                byte_length=version.byte_length,
                start_offset=start_offset,
                end_offset=end_offset,
            )
            extraction_method = self._clean(extraction_method, "extraction_method").lower()
            canonical_evidence_id = self._evidence_id(
                source_version_id=version.source_version_id,
                start_offset=start_offset,
                end_offset=end_offset,
                excerpt=excerpt,
                extraction_method=extraction_method,
            )
            return EvidenceSpan(
                evidence_id=canonical_evidence_id,
                source_id=version.source_id,
                source_version_id=version.source_version_id,
                content_hash=version.content_hash,
                start_offset=start_offset,
                end_offset=end_offset,
                excerpt=excerpt,
                extraction_method=extraction_method,
                observed_at=version.observed_at,
                created_at=version.created_at,
            )

    def assert_evidence_span(self, span: EvidenceSpan) -> None:
        with self._lock:
            try:
                version = self.verify_version(span.source_version_id)
            except UnknownSourceVersionError as exc:
                raise InvalidSourceEvidenceError(
                    f"Evidence references unknown source version: {span.source_version_id}"
                ) from exc
            if span.source_id != version.source_id:
                raise InvalidSourceEvidenceError("Evidence source_id does not match SourceVersion")
            if span.content_hash != version.content_hash:
                raise InvalidSourceEvidenceError("Evidence content_hash does not match SourceVersion")
            if span.observed_at != version.observed_at:
                raise InvalidSourceEvidenceError("Evidence observed_at does not match SourceVersion")
            self._validate_offsets(
                byte_length=version.byte_length,
                start_offset=span.start_offset,
                end_offset=span.end_offset,
            )
            self._assert_excerpt_binding(
                content=self._content_by_version[span.source_version_id],
                start_offset=span.start_offset,
                end_offset=span.end_offset,
                excerpt=span.excerpt,
            )
            if span.created_at != version.created_at:
                raise InvalidSourceEvidenceError("Evidence created_at does not match SourceVersion")
            canonical_method = self._clean(span.extraction_method, "extraction_method").lower()
            if span.extraction_method != canonical_method:
                raise InvalidSourceEvidenceError("Evidence extraction_method is not canonical")
            expected_evidence_id = self._evidence_id(
                source_version_id=version.source_version_id,
                start_offset=span.start_offset,
                end_offset=span.end_offset,
                excerpt=span.excerpt,
                extraction_method=canonical_method,
            )
            if span.evidence_id != expected_evidence_id:
                raise InvalidSourceEvidenceError("Evidence evidence_id is not canonical")

    def _require_source(self, source_id: str) -> SourceRecord:
        try:
            return self._sources[source_id]
        except KeyError as exc:
            raise UnknownSourceError(source_id) from exc

    def _require_version(self, source_version_id: str) -> SourceVersion:
        try:
            return self._versions[source_version_id]
        except KeyError as exc:
            raise UnknownSourceVersionError(source_version_id) from exc

    @staticmethod
    def _clean(value: str, field_name: str) -> str:
        cleaned = unicodedata.normalize("NFC", value.strip())
        if not cleaned:
            raise ValueError(f"{field_name} cannot be empty")
        return cleaned

    @staticmethod
    def _opaque_key(value: str) -> str:
        if value == "":
            raise ValueError("external_key cannot be empty")
        return value

    @staticmethod
    def _canonical_timestamp(value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware")
        return value.astimezone(timezone.utc)

    @staticmethod
    def _validate_offsets(
        *,
        byte_length: int,
        start_offset: int | None,
        end_offset: int | None,
    ) -> None:
        if (start_offset is None) != (end_offset is None):
            raise InvalidSourceEvidenceError(
                "start_offset and end_offset must be supplied together"
            )
        if start_offset is None and end_offset is None:
            if byte_length == 0:
                raise InvalidSourceEvidenceError(
                    "source-bound evidence cannot reference an empty source"
                )
            return
        assert start_offset is not None and end_offset is not None
        if start_offset < 0:
            raise InvalidSourceEvidenceError("start_offset cannot be negative")
        if end_offset < 0:
            raise InvalidSourceEvidenceError("end_offset cannot be negative")
        if start_offset > byte_length:
            raise InvalidSourceEvidenceError("start_offset exceeds source byte length")
        if end_offset > byte_length:
            raise InvalidSourceEvidenceError("end_offset exceeds source byte length")
        if end_offset <= start_offset:
            raise InvalidSourceEvidenceError("source-bound evidence must select at least one byte")

    @staticmethod
    def _bind_excerpt(
        *,
        content: bytes,
        start_offset: int | None,
        end_offset: int | None,
        excerpt: str | None,
    ) -> tuple[int | None, int | None]:
        if (start_offset is None) != (end_offset is None):
            raise InvalidSourceEvidenceError(
                "start_offset and end_offset must be supplied together"
            )
        if excerpt is None:
            return start_offset, end_offset
        if not excerpt:
            raise InvalidSourceEvidenceError("Evidence excerpt cannot be empty")

        excerpt_bytes = excerpt.encode("utf-8")
        if start_offset is not None and end_offset is not None:
            if content[start_offset:end_offset] != excerpt_bytes:
                raise InvalidSourceEvidenceError(
                    "Evidence excerpt does not match source bytes at the supplied offsets"
                )
            return start_offset, end_offset

        first = content.find(excerpt_bytes)
        if first < 0:
            raise InvalidSourceEvidenceError("Evidence excerpt is not present in source bytes")
        second = content.find(excerpt_bytes, first + 1)
        if second >= 0:
            raise InvalidSourceEvidenceError(
                "Evidence excerpt is ambiguous; explicit byte offsets are required"
            )
        return first, first + len(excerpt_bytes)

    @staticmethod
    def _assert_excerpt_binding(
        *,
        content: bytes,
        start_offset: int | None,
        end_offset: int | None,
        excerpt: str | None,
    ) -> None:
        if excerpt is None:
            return
        if start_offset is None or end_offset is None:
            raise InvalidSourceEvidenceError(
                "Source-bound evidence with an excerpt requires exact byte offsets"
            )
        if content[start_offset:end_offset] != excerpt.encode("utf-8"):
            raise InvalidSourceEvidenceError(
                "Evidence excerpt does not match immutable source bytes"
            )

    @staticmethod
    def _source_id(namespace: str, external_key: str) -> str:
        payload = json.dumps(
            {"namespace": namespace, "external_key": external_key},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return "src_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def _source_version_id(
        *,
        source_id: str,
        content_hash: str,
        mime_type: str,
        observed_at: datetime | None,
    ) -> str:
        payload = json.dumps(
            {
                "source_id": source_id,
                "content_hash": content_hash,
                "mime_type": mime_type,
                "observed_at": observed_at.isoformat() if observed_at is not None else None,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return "sv_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:40]

    @staticmethod
    def _evidence_id(
        *,
        source_version_id: str,
        start_offset: int | None,
        end_offset: int | None,
        excerpt: str | None,
        extraction_method: str,
    ) -> str:
        payload = json.dumps(
            {
                "source_version_id": source_version_id,
                "start_offset": start_offset,
                "end_offset": end_offset,
                "excerpt": excerpt,
                "extraction_method": extraction_method,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return "ev_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:40]

    @staticmethod
    def _artifact_id(
        *,
        source_version_id: str,
        artifact_type: str,
        processor: str,
        processor_version: str,
        content_hash: str,
        mime_type: str,
    ) -> str:
        payload = json.dumps(
            {
                "source_version_id": source_version_id,
                "artifact_type": artifact_type,
                "processor": processor,
                "processor_version": processor_version,
                "content_hash": content_hash,
                "mime_type": mime_type,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return "art_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:40]

    @staticmethod
    def _sha256(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()
