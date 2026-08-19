from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from akashic_engine import (
    AdapterIdentityManifest,
    AdapterIdentityRegistry,
    AdapterIdentityValue,
    DirectObservation,
    InMemorySourceStore,
    IngestionEnvelope,
    IngestionPipeline,
    SourceBoundEvidenceGraph,
)
from akashic_engine.ingestion import (
    AdapterIdentityConflictError,
    DuplicateObservationError,
    InvalidAdapterIdentityError,
    UnknownAdapterIdentityError,
)


OBSERVED = datetime(2026, 8, 19, 12, 0, tzinfo=timezone.utc)


def registry() -> AdapterIdentityRegistry:
    result = AdapterIdentityRegistry()
    result.register(
        AdapterIdentityManifest(
            adapter_id="telegram",
            authority_id="bot-primary",
            identity_version=1,
            identity_fields=("chat_id", "message_id"),
            allowed_source_types=("message",),
        )
    )
    return result


def pipeline():
    store = InMemorySourceStore()
    graph = SourceBoundEvidenceGraph(source_verifier=store)
    return IngestionPipeline(registry=registry(), source_store=store, evidence_graph=graph), store, graph


def envelope(
    *,
    content: bytes = b"hello world",
    values: tuple[AdapterIdentityValue, ...] | None = None,
    observations: tuple[DirectObservation, ...] = (),
    observed_at=OBSERVED,
    identity_version: int = 1,
    authority_id: str = "bot-primary",
    source_type: str = "message",
) -> IngestionEnvelope:
    return IngestionEnvelope(
        adapter_id="telegram",
        authority_id=authority_id,
        identity_version=identity_version,
        identity_values=values
        or (
            AdapterIdentityValue(field="chat_id", value="42"),
            AdapterIdentityValue(field="message_id", value="7"),
        ),
        source_type=source_type,
        content=content,
        mime_type="text/plain",
        observed_at=observed_at,
        uri="tg://chat/42/message/7",
        observations=observations,
    )


def test_identity_field_order_does_not_change_source_identity():
    pipe_a, _, _ = pipeline()
    pipe_b, _, _ = pipeline()
    first = pipe_a.ingest(envelope())
    second = pipe_b.ingest(
        envelope(
            values=(
                AdapterIdentityValue(field="message_id", value="7"),
                AdapterIdentityValue(field="chat_id", value="42"),
            )
        )
    )

    assert first.source_id == second.source_id
    assert first.identity_fingerprint == second.identity_fingerprint
    assert first.external_key == second.external_key


def test_missing_or_extra_identity_fields_are_rejected():
    pipe, _, _ = pipeline()
    with pytest.raises(InvalidAdapterIdentityError, match="missing=message_id"):
        pipe.ingest(
            envelope(values=(AdapterIdentityValue(field="chat_id", value="42"),))
        )

    with pytest.raises(InvalidAdapterIdentityError, match="extra=thread_id"):
        pipe.ingest(
            envelope(
                values=(
                    AdapterIdentityValue(field="chat_id", value="42"),
                    AdapterIdentityValue(field="message_id", value="7"),
                    AdapterIdentityValue(field="thread_id", value="9"),
                )
            )
        )


def test_duplicate_identity_field_is_rejected():
    pipe, _, _ = pipeline()
    with pytest.raises(InvalidAdapterIdentityError, match="Duplicate"):
        pipe.ingest(
            envelope(
                values=(
                    AdapterIdentityValue(field="chat_id", value="42"),
                    AdapterIdentityValue(field="chat_id", value="43"),
                    AdapterIdentityValue(field="message_id", value="7"),
                )
            )
        )


def test_registered_identity_version_cannot_be_rebound():
    reg = registry()
    with pytest.raises(AdapterIdentityConflictError):
        reg.register(
            AdapterIdentityManifest(
                adapter_id="telegram",
                authority_id="bot-primary",
                identity_version=1,
                identity_fields=("chat_id", "message_id", "thread_id"),
                allowed_source_types=("message",),
            )
        )


def test_unknown_identity_version_is_rejected_before_source_creation():
    pipe, store, _ = pipeline()
    with pytest.raises(UnknownAdapterIdentityError):
        pipe.ingest(envelope(identity_version=2))
    assert store._sources == {}  # noqa: SLF001


def test_identity_schema_version_creates_a_distinct_source_namespace():
    reg = registry()
    reg.register(
        AdapterIdentityManifest(
            adapter_id="telegram",
            authority_id="bot-primary",
            identity_version=2,
            identity_fields=("chat_id", "message_id"),
            allowed_source_types=("message",),
        )
    )
    store = InMemorySourceStore()
    graph = SourceBoundEvidenceGraph(source_verifier=store)
    pipe = IngestionPipeline(registry=reg, source_store=store, evidence_graph=graph)

    first = pipe.ingest(envelope(identity_version=1))
    second = pipe.ingest(envelope(identity_version=2))

    assert first.namespace != second.namespace
    assert first.source_id != second.source_id
    assert first.external_key == second.external_key


def test_same_upstream_identity_is_scoped_by_authority():
    reg = AdapterIdentityRegistry()
    for authority_id in ("account-a", "account-b"):
        reg.register(
            AdapterIdentityManifest(
                adapter_id="telegram",
                authority_id=authority_id,
                identity_version=1,
                identity_fields=("chat_id", "message_id"),
                allowed_source_types=("message",),
            )
        )
    store = InMemorySourceStore()
    graph = SourceBoundEvidenceGraph(source_verifier=store)
    pipe = IngestionPipeline(registry=reg, source_store=store, evidence_graph=graph)

    first = pipe.ingest(envelope(authority_id="account-a"))
    second = pipe.ingest(envelope(authority_id="account-b"))
    manifest_a = reg.get("telegram", "account-a", 1)
    manifest_b = reg.get("telegram", "account-b", 1)

    assert reg.schema_fingerprint(manifest_a) == reg.schema_fingerprint(manifest_b)
    assert first.authority_fingerprint != second.authority_fingerprint
    assert first.namespace != second.namespace
    assert first.source_id != second.source_id
    assert first.identity_fingerprint != second.identity_fingerprint


def test_unregistered_authority_is_rejected_before_source_creation():
    pipe, store, _ = pipeline()
    with pytest.raises(UnknownAdapterIdentityError):
        pipe.ingest(envelope(authority_id="other-account"))
    assert store._sources == {}  # noqa: SLF001


def test_adapter_schema_restricts_source_type():
    pipe, _, _ = pipeline()
    with pytest.raises(InvalidAdapterIdentityError, match="source_type"):
        pipe.ingest(envelope(source_type="document"))


def test_repeated_ingestion_is_idempotent_end_to_end():
    pipe, _, graph = pipeline()
    item = envelope(
        observations=(
            DirectObservation(excerpt="hello", start_offset=0, end_offset=5),
        )
    )

    first = pipe.ingest(item)
    second = pipe.ingest(item)

    assert first == second
    assert len(first.evidence_ids) == 1
    assert len(first.claim_ids) == 1
    explanation = graph.explain_claim(first.claim_ids[0])
    assert explanation.claim.statement == "hello"
    assert explanation.claim.evidence_refs == first.evidence_ids


def test_changed_content_same_adapter_identity_creates_new_version_same_source():
    pipe, _, _ = pipeline()
    first = pipe.ingest(envelope(content=b"hello world"))
    second = pipe.ingest(
        envelope(content=b"hello again", observed_at=OBSERVED + timedelta(minutes=1))
    )

    assert first.source_id == second.source_id
    assert first.source_version_id != second.source_version_id
    assert first.envelope_fingerprint != second.envelope_fingerprint


def test_direct_observed_claim_statement_is_exact_source_excerpt():
    pipe, _, graph = pipeline()
    receipt = pipe.ingest(
        envelope(
            content=b"alpha beta gamma",
            observations=(DirectObservation(excerpt="beta"),),
        )
    )

    claim = graph.get_claim(receipt.claim_ids[0])
    evidence = graph.get_evidence(receipt.evidence_ids[0])
    assert claim.statement == "beta"
    assert evidence.excerpt == "beta"
    assert (evidence.start_offset, evidence.end_offset) == (6, 10)


def test_duplicate_observation_specs_are_rejected():
    pipe, _, _ = pipeline()
    observation = DirectObservation(excerpt="hello")
    with pytest.raises(DuplicateObservationError):
        pipe.ingest(envelope(observations=(observation, observation)))


def test_direct_ingestion_cannot_inject_semantic_domain_or_reserved_claim_type():
    with pytest.raises(ValidationError):
        DirectObservation(excerpt="hello", domain_tags=("health",))
    with pytest.raises(ValidationError):
        DirectObservation(excerpt="hello", claim_type="user_confirmation")


def test_direct_ingestion_claim_type_is_fixed_source_text_without_domains():
    pipe, _, graph = pipeline()
    receipt = pipe.ingest(envelope(observations=(DirectObservation(excerpt="hello"),)))
    claim = graph.get_claim(receipt.claim_ids[0])
    assert claim.claim_type == "source_text"
    assert claim.domain_tags == ()


def test_observation_order_does_not_change_receipt_or_fingerprint():
    first_pipe, _, _ = pipeline()
    second_pipe, _, _ = pipeline()
    a = DirectObservation(excerpt="hello")
    b = DirectObservation(excerpt="world")

    first = first_pipe.ingest(envelope(observations=(a, b)))
    second = second_pipe.ingest(envelope(observations=(b, a)))

    assert first.envelope_fingerprint == second.envelope_fingerprint
    assert first.receipt_fingerprint == second.receipt_fingerprint
    assert first.evidence_ids == second.evidence_ids
    assert first.claim_ids == second.claim_ids


def test_identity_values_are_opaque_and_not_silently_normalized_by_core():
    first_pipe, _, _ = pipeline()
    second_pipe, _, _ = pipeline()
    composed = "caf\u00e9"
    decomposed = "cafe\u0301"
    first = first_pipe.ingest(
        envelope(
            values=(
                AdapterIdentityValue(field="chat_id", value=composed),
                AdapterIdentityValue(field="message_id", value="7"),
            )
        )
    )
    second = second_pipe.ingest(
        envelope(
            values=(
                AdapterIdentityValue(field="chat_id", value=decomposed),
                AdapterIdentityValue(field="message_id", value="7"),
            )
        )
    )

    assert first.source_id != second.source_id
    assert first.identity_fingerprint != second.identity_fingerprint


def test_identity_values_preserve_meaningful_whitespace():
    first_pipe, _, _ = pipeline()
    second_pipe, _, _ = pipeline()
    first = first_pipe.ingest(
        envelope(
            values=(
                AdapterIdentityValue(field="chat_id", value="42"),
                AdapterIdentityValue(field="message_id", value="7"),
            )
        )
    )
    second = second_pipe.ingest(
        envelope(
            values=(
                AdapterIdentityValue(field="chat_id", value="42 "),
                AdapterIdentityValue(field="message_id", value="7"),
            )
        )
    )
    assert first.source_id != second.source_id


def test_direct_ingestion_extraction_method_is_engine_owned():
    with pytest.raises(ValidationError):
        DirectObservation(excerpt="hello", extraction_method="parser")

    pipe, _, graph = pipeline()
    receipt = pipe.ingest(envelope(observations=(DirectObservation(excerpt="hello"),)))
    evidence = graph.get_evidence(receipt.evidence_ids[0])
    assert evidence.extraction_method == "direct"


def test_identity_schema_drift_without_version_bump_changes_namespace_across_registries():
    first = AdapterIdentityRegistry()
    second = AdapterIdentityRegistry()
    a = AdapterIdentityManifest(
        adapter_id="adapter", authority_id="account-1", identity_version=1, identity_fields=("object_id",)
    )
    b = AdapterIdentityManifest(
        adapter_id="adapter", authority_id="account-1", identity_version=1, identity_fields=("account_id", "object_id")
    )
    first.register(a)
    second.register(b)

    assert first.schema_fingerprint(a) != second.schema_fingerprint(b)
    assert first.namespace(a) != second.namespace(b)


def test_manifest_policy_fingerprint_is_recorded_in_receipt():
    pipe, _, _ = pipeline()
    receipt = pipe.ingest(envelope())
    manifest = pipe.registry.get("telegram", "bot-primary", 1)

    assert receipt.schema_fingerprint == pipe.registry.schema_fingerprint(manifest)
    assert receipt.manifest_fingerprint == pipe.registry.manifest_fingerprint(manifest)


def test_concurrent_duplicate_delivery_is_idempotent():
    from concurrent.futures import ThreadPoolExecutor

    reg = registry()
    store = InMemorySourceStore()
    graph = SourceBoundEvidenceGraph(source_verifier=store)
    # Separate pipeline locks simulate two independent runtime workers.
    first = IngestionPipeline(registry=reg, source_store=store, evidence_graph=graph)
    second = IngestionPipeline(registry=reg, source_store=store, evidence_graph=graph)
    item = envelope(observations=(DirectObservation(excerpt="hello"),))

    with ThreadPoolExecutor(max_workers=2) as executor:
        left = executor.submit(first.ingest, item)
        right = executor.submit(second.ingest, item)
        a = left.result()
        b = right.result()

    assert a == b
    assert graph.get_claim(a.claim_ids[0]) == graph.get_claim(b.claim_ids[0])


def test_adapter_and_identity_field_names_require_safe_canonical_identifiers():
    with pytest.raises(ValidationError):
        AdapterIdentityManifest(
            adapter_id="bad adapter", authority_id="account-1", identity_version=1, identity_fields=("id",)
        )
    with pytest.raises(ValidationError):
        AdapterIdentityManifest(
            adapter_id="good", authority_id="account-1", identity_version=1, identity_fields=("bad field",)
        )
