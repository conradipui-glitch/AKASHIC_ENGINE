from datetime import datetime, timezone

import pytest

from akashic_engine import (
    EvidenceSpan,
    InMemoryEvidenceGraph,
    InMemorySourceStore,
    InvalidSourceEvidenceError,
    SourceBoundEvidenceGraph,
    SourceIdentityConflictError,
    SourceIntegrityError,
)


OBSERVED = datetime(2026, 8, 19, 12, 0, tzinfo=timezone.utc)


def make_store_with_version(content: bytes = b"hello world"):
    store = InMemorySourceStore()
    source = store.register_source(
        namespace="telegram",
        external_key="chat:42:message:7",
        source_type="message",
        uri="tg://chat/42/message/7",
    )
    version = store.ingest_version(
        source.source_id,
        content=content,
        mime_type="text/plain",
        observed_at=OBSERVED,
    )
    return store, source, version


def test_source_identity_is_deterministic_and_idempotent():
    first = InMemorySourceStore()
    second = InMemorySourceStore()

    a = first.register_source(
        namespace="telegram",
        external_key="chat:42:message:7",
        source_type="message",
    )
    b = second.register_source(
        namespace="telegram",
        external_key="chat:42:message:7",
        source_type="message",
    )
    again = first.register_source(
        namespace="telegram",
        external_key="chat:42:message:7",
        source_type="message",
    )

    assert a.source_id == b.source_id
    assert again == a
    assert a.source_id.startswith("src_")


def test_same_external_key_cannot_be_rebound_to_conflicting_metadata():
    store = InMemorySourceStore()
    store.register_source(
        namespace="drive",
        external_key="file-1",
        source_type="document",
        uri="drive://file-1",
    )

    with pytest.raises(SourceIdentityConflictError):
        store.register_source(
            namespace="drive",
            external_key="file-1",
            source_type="spreadsheet",
            uri="drive://file-1",
        )


def test_uri_drift_does_not_rebind_or_conflict_with_source_identity():
    store = InMemorySourceStore()
    first = store.register_source(
        namespace="drive",
        external_key="file-1",
        source_type="document",
        uri="drive://old-link",
    )
    repeated = store.register_source(
        namespace="drive",
        external_key="file-1",
        source_type="document",
        uri="drive://new-link",
    )

    assert repeated == first
    assert repeated.source_id == first.source_id
    assert repeated.uri == "drive://old-link"


def test_version_ids_and_hashes_are_store_computed_and_replayable():
    first, source, version = make_store_with_version(b"alpha")
    repeated = first.ingest_version(
        source.source_id,
        content=b"alpha",
        mime_type="text/plain",
        observed_at=OBSERVED,
    )

    second = InMemorySourceStore()
    same_source = second.register_source(
        namespace="telegram",
        external_key="chat:42:message:7",
        source_type="message",
        uri="tg://chat/42/message/7",
    )
    same_version = second.ingest_version(
        same_source.source_id,
        content=b"alpha",
        mime_type="text/plain",
        observed_at=OBSERVED,
    )

    assert repeated == version
    assert same_version.source_version_id == version.source_version_id
    assert same_version.content_hash == version.content_hash
    assert first.read_content(version.source_version_id) == b"alpha"


def test_changed_content_creates_a_new_immutable_version():
    store, source, first = make_store_with_version(b"v1")
    second = store.ingest_version(
        source.source_id,
        content=b"v2",
        mime_type="text/plain",
        observed_at=OBSERVED,
    )

    assert second.source_version_id != first.source_version_id
    assert second.version_index == 2
    assert store.read_content(first.source_version_id) == b"v1"
    assert store.read_content(second.source_version_id) == b"v2"


def test_same_bytes_captured_at_a_new_time_can_be_a_new_version():
    store, source, first = make_store_with_version(b"same")
    later = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)
    second = store.ingest_version(
        source.source_id,
        content=b"same",
        mime_type="text/plain",
        observed_at=later,
    )

    assert second.source_version_id != first.source_version_id
    assert second.content_hash == first.content_hash
    assert second.version_index == 2


def test_integrity_verification_detects_storage_corruption():
    store, _, version = make_store_with_version(b"trusted")

    # Simulate adapter/disk corruption behind the public immutable API.
    store._content_by_version[version.source_version_id] = b"tampered"  # noqa: SLF001

    with pytest.raises(SourceIntegrityError):
        store.verify_version(version.source_version_id)


def test_derived_artifact_never_replaces_raw_source_bytes():
    store, _, version = make_store_with_version(b"Raw  SOURCE\n")
    artifact = store.add_artifact(
        version.source_version_id,
        artifact_type="normalized_text",
        processor="normalizer",
        processor_version="1",
        content=b"raw source",
        mime_type="text/plain",
    )

    assert artifact.source_version_id == version.source_version_id
    assert store.read_content(version.source_version_id) == b"Raw  SOURCE\n"
    assert store.read_artifact_content(artifact.artifact_id) == b"raw source"


def test_source_store_creates_evidence_with_bound_provenance():
    store, source, version = make_store_with_version(b"hello world")
    span = store.make_evidence_span(
        version.source_version_id,
        start_offset=0,
        end_offset=5,
        excerpt="hello",
    )

    assert span.source_id == source.source_id
    assert span.source_version_id == version.source_version_id
    assert span.content_hash == version.content_hash
    assert span.observed_at == version.observed_at
    store.assert_evidence_span(span)


def test_forged_evidence_hash_or_source_identity_is_rejected():
    store, source, version = make_store_with_version()
    valid = store.make_evidence_span(version.source_version_id)

    forged_hash = valid.model_copy(update={"content_hash": "0" * 64})
    with pytest.raises(InvalidSourceEvidenceError, match="content_hash"):
        store.assert_evidence_span(forged_hash)

    forged_source = valid.model_copy(update={"source_id": source.source_id + "-alias"})
    with pytest.raises(InvalidSourceEvidenceError, match="source_id"):
        store.assert_evidence_span(forged_source)


def test_evidence_offsets_cannot_exceed_immutable_source_bytes():
    store, _, version = make_store_with_version(b"12345")

    with pytest.raises(InvalidSourceEvidenceError, match="byte length"):
        store.make_evidence_span(
            version.source_version_id,
            start_offset=0,
            end_offset=6,
        )


def test_source_bound_evidence_graph_rejects_forged_manual_spans():
    store, _, version = make_store_with_version()
    graph = SourceBoundEvidenceGraph(source_verifier=store)
    valid = store.make_evidence_span(version.source_version_id)

    assert graph.add_evidence(valid) == valid

    forged = EvidenceSpan(
        evidence_id="e2",
        source_id=valid.source_id,
        source_version_id=valid.source_version_id,
        content_hash="f" * 64,
        observed_at=valid.observed_at,
    )
    with pytest.raises(InvalidSourceEvidenceError):
        graph.add_evidence(forged)


def test_legacy_in_memory_graph_remains_a_low_level_unverified_adapter():
    graph = InMemoryEvidenceGraph()
    span = EvidenceSpan(
        evidence_id="legacy",
        source_id="trusted-import",
        source_version_id="v1",
        content_hash="a" * 64,
    )

    assert graph.add_evidence(span) == span


def test_source_namespace_is_canonicalized_for_stable_identity():
    first = InMemorySourceStore()
    second = InMemorySourceStore()
    a = first.register_source(
        namespace=" Telegram ",
        external_key="chat:1",
        source_type="Message",
    )
    b = second.register_source(
        namespace="telegram",
        external_key="chat:1",
        source_type="message",
    )

    assert a.source_id == b.source_id
    assert a.namespace == "telegram"
    assert a.source_type == "message"


def test_observation_time_is_canonicalized_to_utc_for_version_identity():
    from datetime import timedelta

    first = InMemorySourceStore()
    second = InMemorySourceStore()
    source_a = first.register_source(
        namespace="file", external_key="x", source_type="document"
    )
    source_b = second.register_source(
        namespace="file", external_key="x", source_type="document"
    )
    utc_time = datetime(2026, 8, 19, 6, 0, tzinfo=timezone.utc)
    plus_six = datetime(2026, 8, 19, 12, 0, tzinfo=timezone(timedelta(hours=6)))

    a = first.ingest_version(
        source_a.source_id,
        content=b"same",
        mime_type="text/plain",
        observed_at=utc_time,
    )
    b = second.ingest_version(
        source_b.source_id,
        content=b"same",
        mime_type="text/plain",
        observed_at=plus_six,
    )

    assert a.source_version_id == b.source_version_id
    assert a.observed_at == b.observed_at == utc_time


def test_excerpt_is_bound_to_exact_immutable_source_bytes():
    store, _, version = make_store_with_version(b"alpha beta gamma")
    span = store.make_evidence_span(
        version.source_version_id,
        excerpt="beta",
    )

    assert (span.start_offset, span.end_offset) == (6, 10)
    store.assert_evidence_span(span)

    forged = span.model_copy(update={"excerpt": "fake"})
    with pytest.raises(InvalidSourceEvidenceError, match="excerpt"):
        store.assert_evidence_span(forged)


def test_ambiguous_excerpt_requires_explicit_byte_offsets():
    store, _, version = make_store_with_version(b"same and same")

    with pytest.raises(InvalidSourceEvidenceError, match="ambiguous"):
        store.make_evidence_span(version.source_version_id, excerpt="same")

    span = store.make_evidence_span(
        version.source_version_id,
        excerpt="same",
        start_offset=9,
        end_offset=13,
    )
    assert span.start_offset == 9
    assert span.end_offset == 13


def test_source_bound_graph_rejects_manual_excerpt_not_present_in_source():
    store, _, version = make_store_with_version(b"real text")
    graph = SourceBoundEvidenceGraph(source_verifier=store)
    forged = EvidenceSpan(
        evidence_id="forged-excerpt",
        source_id=version.source_id,
        source_version_id=version.source_version_id,
        content_hash=version.content_hash,
        start_offset=0,
        end_offset=4,
        excerpt="fake",
        observed_at=version.observed_at,
    )

    with pytest.raises(InvalidSourceEvidenceError, match="excerpt"):
        graph.add_evidence(forged)


def test_default_evidence_identity_is_deterministic_for_same_source_span():
    store, _, version = make_store_with_version(b"alpha beta")
    first = store.make_evidence_span(version.source_version_id, excerpt="beta")
    second = store.make_evidence_span(version.source_version_id, excerpt="beta")

    assert first.evidence_id == second.evidence_id
    assert first.created_at == second.created_at == version.created_at
    assert first.model_dump(mode="json") == second.model_dump(mode="json")



def test_source_bound_graph_rejects_noncanonical_evidence_identity_and_created_at():
    from datetime import timedelta

    store, _, version = make_store_with_version(b"canonical evidence")
    graph = SourceBoundEvidenceGraph(source_verifier=store)
    valid = store.make_evidence_span(version.source_version_id, excerpt="evidence")

    forged_id = valid.model_copy(update={"evidence_id": "ev_forged"})
    with pytest.raises(InvalidSourceEvidenceError, match="evidence_id"):
        graph.add_evidence(forged_id)

    forged_created = valid.model_copy(update={"created_at": valid.created_at + timedelta(seconds=1)})
    with pytest.raises(InvalidSourceEvidenceError, match="created_at"):
        graph.add_evidence(forged_created)


def test_source_bound_graph_rejects_noncanonical_extraction_method_metadata():
    store, _, version = make_store_with_version(b"canonical method")
    graph = SourceBoundEvidenceGraph(source_verifier=store)
    valid = store.make_evidence_span(version.source_version_id, excerpt="method")

    forged = valid.model_copy(update={"extraction_method": " DIRECT "})
    with pytest.raises(InvalidSourceEvidenceError, match="extraction_method"):
        graph.add_evidence(forged)


def test_manual_half_open_offset_pair_is_rejected_by_source_verifier():
    store, _, version = make_store_with_version(b"abcdef")
    canonical = store.make_evidence_span(version.source_version_id)
    forged = canonical.model_copy(
        update={
            "start_offset": 1,
            "end_offset": None,
            "excerpt": None,
        }
    )

    with pytest.raises(InvalidSourceEvidenceError, match="supplied together"):
        store.assert_evidence_span(forged)


def test_zero_length_and_empty_source_evidence_are_rejected():
    store, _, version = make_store_with_version(b"abcdef")

    with pytest.raises(InvalidSourceEvidenceError, match="at least one byte"):
        store.make_evidence_span(version.source_version_id, start_offset=2, end_offset=2)

    with pytest.raises(InvalidSourceEvidenceError, match="excerpt cannot be empty"):
        store.make_evidence_span(version.source_version_id, excerpt="")

    empty_store, _, empty_version = make_store_with_version(b"")
    with pytest.raises(InvalidSourceEvidenceError, match="empty source"):
        empty_store.make_evidence_span(empty_version.source_version_id)


def test_read_content_detects_corruption_before_returning_bytes():
    store, _, version = make_store_with_version(b"trusted")
    store._content_by_version[version.source_version_id] = b"tampered"  # noqa: SLF001

    with pytest.raises(SourceIntegrityError):
        store.read_content(version.source_version_id)


def test_evidence_identity_includes_extraction_method_and_is_store_owned():
    store, _, version = make_store_with_version(b"alpha beta")
    direct = store.make_evidence_span(
        version.source_version_id,
        excerpt="beta",
        extraction_method=" DIRECT ",
    )
    parsed = store.make_evidence_span(
        version.source_version_id,
        excerpt="beta",
        extraction_method="parser",
    )

    assert direct.extraction_method == "direct"
    assert direct.evidence_id != parsed.evidence_id
    store.assert_evidence_span(direct)
    store.assert_evidence_span(parsed)



def test_source_version_identity_includes_canonical_mime_type():
    store = InMemorySourceStore()
    source = store.register_source(
        namespace="file",
        external_key="mime-sensitive",
        source_type="document",
    )
    text = store.ingest_version(
        source.source_id,
        content=b"{}",
        mime_type=" TEXT/PLAIN ",
        observed_at=OBSERVED,
    )
    json_version = store.ingest_version(
        source.source_id,
        content=b"{}",
        mime_type="application/json",
        observed_at=OBSERVED,
    )

    assert text.mime_type == "text/plain"
    assert json_version.mime_type == "application/json"
    assert text.source_version_id != json_version.source_version_id
    assert text.content_hash == json_version.content_hash


def test_external_key_is_opaque_and_not_unicode_normalized_or_trimmed():
    first = InMemorySourceStore()
    second = InMemorySourceStore()
    third = InMemorySourceStore()
    composed = "caf\u00e9"
    decomposed = "cafe\u0301"

    a = first.register_source(
        namespace="adapter", external_key=composed, source_type="message"
    )
    b = second.register_source(
        namespace="adapter", external_key=decomposed, source_type="message"
    )
    c = third.register_source(
        namespace="adapter", external_key=composed + " ", source_type="message"
    )

    assert a.source_id != b.source_id
    assert a.source_id != c.source_id
    assert a.external_key == composed
    assert b.external_key == decomposed
    assert c.external_key == composed + " "
