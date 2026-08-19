from datetime import datetime, timedelta, timezone
from threading import Barrier

import pytest
from pydantic import ValidationError

from akashic_engine import (
    AdapterIdentityManifest,
    AdapterIdentityRegistry,
    AdapterIdentityValue,
    Claim,
    ClaimExtractionProposal,
    DirectObservation,
    EpistemicLevel,
    EvidenceBundleBuilder,
    ExtractorManifest,
    InMemoryEvidenceGraph,
    InMemorySourceStore,
    IngestionEnvelope,
    IngestionPipeline,
    LongitudinalPatternEngine,
    SemanticExtractionEngine,
    SemanticExtractionPolicy,
    SourceBoundEvidenceGraph,
    ValidationState,
)
from akashic_engine.extraction import (
    DuplicateClaimProposalError,
    ExtractionReplayConflictError,
    InvalidClaimProposalError,
    InvalidExtractionInputError,
)


OBSERVED = datetime(2026, 8, 19, 12, 0, tzinfo=timezone.utc)
CONFIG = "a" * 64


class StaticExtractor:
    def __init__(self, proposals, *, config=CONFIG, barrier: Barrier | None = None):
        self.manifest = ExtractorManifest(
            extractor_id="test.semantic",
            extractor_version="1",
            configuration_fingerprint=config,
        )
        self.proposals = tuple(proposals)
        self.barrier = barrier

    def extract(self, context):
        if self.barrier is not None:
            self.barrier.wait()
        return self.proposals


class BadListExtractor(StaticExtractor):
    def extract(self, context):
        return list(self.proposals)


def make_runtime():
    registry = AdapterIdentityRegistry()
    registry.register(
        AdapterIdentityManifest(
            adapter_id="telegram",
            authority_id="bot-main",
            identity_version=1,
            identity_fields=("chat_id", "message_id"),
            allowed_source_types=("message",),
        )
    )
    store = InMemorySourceStore()
    graph = SourceBoundEvidenceGraph(source_verifier=store)
    ingestion = IngestionPipeline(
        registry=registry,
        source_store=store,
        evidence_graph=graph,
    )
    return ingestion, store, graph


def ingest_text(ingestion, *, message_id="1", text="I start projects and later lose interest", at=OBSERVED):
    content = text.encode("utf-8")
    return ingestion.ingest(
        IngestionEnvelope(
            adapter_id="telegram",
            authority_id="bot-main",
            identity_version=1,
            identity_values=(
                AdapterIdentityValue(field="chat_id", value="42"),
                AdapterIdentityValue(field="message_id", value=message_id),
            ),
            source_type="message",
            content=content,
            mime_type="text/plain",
            observed_at=at,
            observations=(DirectObservation(excerpt=text),),
        )
    )


def proposal(source_claim_id, statement="Interest drops after an intense project start", pattern_key="interest-cycle"):
    return ClaimExtractionProposal(
        source_claim_ids=(source_claim_id,),
        statement=statement,
        pattern_key=pattern_key,
    )


def test_extraction_creates_unverified_interpretation_with_derived_from_edge():
    ingestion, _, graph = make_runtime()
    receipt = ingest_text(ingestion)
    p = proposal(receipt.claim_ids[0])
    result = SemanticExtractionEngine(
        evidence_graph=graph,
        extractor=StaticExtractor((p,)),
    ).extract(receipt)

    assert len(result.derived_claim_ids) == 1
    derived = graph.get_claim(result.derived_claim_ids[0])
    assert derived.statement == p.statement
    assert derived.claim_type == "semantic_interpretation"
    assert derived.epistemic_level is EpistemicLevel.INTERPRETATION
    assert derived.validation_state is ValidationState.UNVERIFIED
    assert derived.domain_tags == ()
    assert derived.evidence_refs == receipt.evidence_ids

    source = graph.explain_claim(receipt.claim_ids[0])
    assert any(
        edge.target_claim_id == derived.claim_id and edge.relation.value == "derived_from"
        for edge in source.outgoing_relations
    )


def test_extractor_cannot_inject_epistemic_level_claim_type_domains_or_evidence():
    source_id = "source"
    with pytest.raises(ValidationError):
        ClaimExtractionProposal(
            source_claim_ids=(source_id,),
            statement="x",
            epistemic_level="observed",
        )
    with pytest.raises(ValidationError):
        ClaimExtractionProposal(
            source_claim_ids=(source_id,),
            statement="x",
            claim_type="user_confirmation",
        )
    with pytest.raises(ValidationError):
        ClaimExtractionProposal(
            source_claim_ids=(source_id,),
            statement="x",
            domain_tags=("health",),
        )
    with pytest.raises(ValidationError):
        ClaimExtractionProposal(
            source_claim_ids=(source_id,),
            statement="x",
            evidence_refs=("forged",),
        )


def test_proposal_must_reference_claims_from_exact_ingestion_input():
    ingestion, _, graph = make_runtime()
    receipt = ingest_text(ingestion)
    engine = SemanticExtractionEngine(
        evidence_graph=graph,
        extractor=StaticExtractor((proposal("other-claim"),)),
    )
    with pytest.raises(InvalidClaimProposalError, match="outside extraction input"):
        engine.extract(receipt)


def test_input_claim_must_be_literal_supported_observed_source_text():
    ingestion, _, graph = make_runtime()
    receipt = ingest_text(ingestion)
    source = graph.get_claim(receipt.claim_ids[0])
    graph._claims[source.claim_id] = source.model_copy(  # noqa: SLF001 - adversarial state corruption
        update={"epistemic_level": EpistemicLevel.INTERPRETATION}
    )
    engine = SemanticExtractionEngine(
        evidence_graph=graph,
        extractor=StaticExtractor(()),
    )
    with pytest.raises(InvalidExtractionInputError, match="OBSERVED"):
        engine.extract(receipt)


def test_receipt_without_observed_claims_cannot_be_extracted():
    ingestion, _, graph = make_runtime()
    receipt = ingestion.ingest(
        IngestionEnvelope(
            adapter_id="telegram",
            authority_id="bot-main",
            identity_version=1,
            identity_values=(
                AdapterIdentityValue(field="chat_id", value="42"),
                AdapterIdentityValue(field="message_id", value="empty"),
            ),
            source_type="message",
            content=b"text",
            mime_type="text/plain",
            observed_at=OBSERVED,
            observations=(),
        )
    )
    with pytest.raises(InvalidExtractionInputError, match="no observed claims"):
        SemanticExtractionEngine(
            evidence_graph=graph,
            extractor=StaticExtractor(()),
        ).extract(receipt)


def test_occurrence_candidate_is_built_from_derived_claim_and_same_provenance():
    ingestion, _, graph = make_runtime()
    receipt = ingest_text(ingestion)
    result = SemanticExtractionEngine(
        evidence_graph=graph,
        extractor=StaticExtractor((proposal(receipt.claim_ids[0]),)),
    ).extract(receipt)

    candidate = result.occurrence_candidates[0]
    derived = graph.get_claim(candidate.extraction_claim_id)
    assert candidate.pattern_key == "interest-cycle"
    assert candidate.occurrence.claim_id == derived.claim_id
    assert candidate.occurrence.evidence_refs == derived.evidence_refs == receipt.evidence_ids
    assert candidate.occurrence.confirmation_claim_id is None


def test_pattern_key_changes_candidate_routing_but_not_semantic_claim_identity():
    ingestion_a, _, graph_a = make_runtime()
    receipt_a = ingest_text(ingestion_a)
    first = SemanticExtractionEngine(
        evidence_graph=graph_a,
        extractor=StaticExtractor((proposal(receipt_a.claim_ids[0], pattern_key="cycle-a"),)),
    ).extract(receipt_a)

    ingestion_b, _, graph_b = make_runtime()
    receipt_b = ingest_text(ingestion_b)
    second = SemanticExtractionEngine(
        evidence_graph=graph_b,
        extractor=StaticExtractor((proposal(receipt_b.claim_ids[0], pattern_key="cycle-b"),)),
    ).extract(receipt_b)

    assert first.derived_claim_ids == second.derived_claim_ids
    assert first.occurrence_candidates[0].candidate_id != second.occurrence_candidates[0].candidate_id


def test_no_pattern_key_creates_claim_without_occurrence_candidate():
    ingestion, _, graph = make_runtime()
    receipt = ingest_text(ingestion)
    p = proposal(receipt.claim_ids[0], pattern_key=None)
    result = SemanticExtractionEngine(
        evidence_graph=graph,
        extractor=StaticExtractor((p,)),
    ).extract(receipt)

    assert len(result.derived_claim_ids) == 1
    assert result.occurrence_candidates == ()


def test_duplicate_proposals_are_rejected_instead_of_inflating_graph():
    ingestion, _, graph = make_runtime()
    receipt = ingest_text(ingestion)
    p = proposal(receipt.claim_ids[0])
    with pytest.raises(DuplicateClaimProposalError):
        SemanticExtractionEngine(
            evidence_graph=graph,
            extractor=StaticExtractor((p, p)),
        ).extract(receipt)


def test_policy_caps_proposal_volume():
    ingestion, _, graph = make_runtime()
    receipt = ingest_text(ingestion)
    proposals = tuple(
        proposal(receipt.claim_ids[0], statement=f"interpretation {index}")
        for index in range(3)
    )
    engine = SemanticExtractionEngine(
        evidence_graph=graph,
        extractor=StaticExtractor(proposals),
        policy=SemanticExtractionPolicy(max_proposals_per_input=2),
    )
    with pytest.raises(InvalidClaimProposalError, match="policy limit"):
        engine.extract(receipt)


def test_extractor_output_must_use_typed_tuple_contract():
    ingestion, _, graph = make_runtime()
    receipt = ingest_text(ingestion)
    p = proposal(receipt.claim_ids[0])
    with pytest.raises(InvalidClaimProposalError, match="tuple"):
        SemanticExtractionEngine(
            evidence_graph=graph,
            extractor=BadListExtractor((p,)),
        ).extract(receipt)


def test_proposal_order_does_not_change_receipt():
    first_ingestion, _, first_graph = make_runtime()
    first_receipt = ingest_text(first_ingestion)
    a = proposal(first_receipt.claim_ids[0], statement="A", pattern_key="a")
    b = proposal(first_receipt.claim_ids[0], statement="B", pattern_key="b")
    first = SemanticExtractionEngine(
        evidence_graph=first_graph,
        extractor=StaticExtractor((a, b)),
    ).extract(first_receipt)

    second_ingestion, _, second_graph = make_runtime()
    second_receipt = ingest_text(second_ingestion)
    second = SemanticExtractionEngine(
        evidence_graph=second_graph,
        extractor=StaticExtractor((b, a)),
    ).extract(second_receipt)

    assert first.input_fingerprint == second.input_fingerprint
    assert first.proposal_fingerprint == second.proposal_fingerprint
    assert first.derived_claim_ids == second.derived_claim_ids
    assert first.occurrence_candidates == second.occurrence_candidates
    assert first.receipt_fingerprint == second.receipt_fingerprint


def test_repeated_extraction_is_idempotent():
    ingestion, _, graph = make_runtime()
    receipt = ingest_text(ingestion)
    engine = SemanticExtractionEngine(
        evidence_graph=graph,
        extractor=StaticExtractor((proposal(receipt.claim_ids[0]),)),
    )
    first = engine.extract(receipt)
    second = engine.extract(receipt)

    assert first == second
    assert graph.get_claim(first.derived_claim_ids[0]) == graph.get_claim(second.derived_claim_ids[0])


def test_concurrent_extraction_workers_converge_on_same_claim_identity():
    from concurrent.futures import ThreadPoolExecutor

    ingestion, _, graph = make_runtime()
    receipt = ingest_text(ingestion)
    p = proposal(receipt.claim_ids[0])
    barrier = Barrier(2)
    first = SemanticExtractionEngine(
        evidence_graph=graph,
        extractor=StaticExtractor((p,), barrier=barrier),
    )
    second = SemanticExtractionEngine(
        evidence_graph=graph,
        extractor=StaticExtractor((p,), barrier=barrier),
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        left = executor.submit(first.extract, receipt)
        right = executor.submit(second.extract, receipt)
        a = left.result()
        b = right.result()
    assert a == b
    assert a.derived_claim_ids == b.derived_claim_ids


def test_extractor_configuration_change_creates_distinct_interpretation_provenance():
    first_ingestion, _, first_graph = make_runtime()
    first_receipt = ingest_text(first_ingestion)
    p = proposal(first_receipt.claim_ids[0])
    first = SemanticExtractionEngine(
        evidence_graph=first_graph,
        extractor=StaticExtractor((p,), config="a" * 64),
    ).extract(first_receipt)

    second_ingestion, _, second_graph = make_runtime()
    second_receipt = ingest_text(second_ingestion)
    second = SemanticExtractionEngine(
        evidence_graph=second_graph,
        extractor=StaticExtractor((p,), config="b" * 64),
    ).extract(second_receipt)

    assert first.extractor_manifest_fingerprint != second.extractor_manifest_fingerprint
    assert first.derived_claim_ids != second.derived_claim_ids


def test_one_proposal_can_be_grounded_in_multiple_observed_claims_without_model_chosen_evidence():
    ingestion, _, graph = make_runtime()
    content = b"alpha beta"
    receipt = ingestion.ingest(
        IngestionEnvelope(
            adapter_id="telegram",
            authority_id="bot-main",
            identity_version=1,
            identity_values=(
                AdapterIdentityValue(field="chat_id", value="42"),
                AdapterIdentityValue(field="message_id", value="multi"),
            ),
            source_type="message",
            content=content,
            mime_type="text/plain",
            observed_at=OBSERVED,
            observations=(
                DirectObservation(excerpt="alpha", start_offset=0, end_offset=5),
                DirectObservation(excerpt="beta", start_offset=6, end_offset=10),
            ),
        )
    )
    p = ClaimExtractionProposal(
        source_claim_ids=receipt.claim_ids,
        statement="The message contains two related elements",
        pattern_key="paired-elements",
    )
    result = SemanticExtractionEngine(
        evidence_graph=graph,
        extractor=StaticExtractor((p,)),
    ).extract(receipt)
    derived = graph.get_claim(result.derived_claim_ids[0])
    assert set(derived.evidence_refs) == set(receipt.evidence_ids)
    assert set(result.occurrence_candidates[0].source_claim_ids) == set(receipt.claim_ids)


def test_complete_ingestion_extraction_occurrence_pattern_chain_reaches_recurrence_two():
    ingestion, _, graph = make_runtime()
    first_receipt = ingest_text(
        ingestion,
        message_id="1",
        text="I start projects intensely and later lose interest",
        at=OBSERVED,
    )
    second_receipt = ingest_text(
        ingestion,
        message_id="2",
        text="Again I went deep into a project and then dropped it",
        at=OBSERVED + timedelta(days=90),
    )

    first_result = SemanticExtractionEngine(
        evidence_graph=graph,
        extractor=StaticExtractor((proposal(first_receipt.claim_ids[0]),)),
    ).extract(first_receipt)
    second_result = SemanticExtractionEngine(
        evidence_graph=graph,
        extractor=StaticExtractor((proposal(second_receipt.claim_ids[0]),)),
    ).extract(second_receipt)

    candidates = first_result.occurrence_candidates + second_result.occurrence_candidates
    assert {item.pattern_key for item in candidates} == {"interest-cycle"}
    derived_ids = tuple(item.extraction_claim_id for item in candidates)
    bundle = EvidenceBundleBuilder(graph).build(derived_ids)
    analysis = LongitudinalPatternEngine().build(
        pattern_type="interest-cycle",
        description="Interest drops after intense project starts",
        evidence_bundle=bundle,
        occurrences=tuple(item.occurrence for item in candidates),
    )

    assert analysis.projection.recurrence_count == 2
    assert analysis.projection.confidence_components.recurrence == 0.25
    assert analysis.projection.confidence_components.evidence_quality == 0.2625
    assert analysis.projection.first_observed == OBSERVED
    assert analysis.projection.last_observed == OBSERVED + timedelta(days=90)


def test_delivery_uri_drift_does_not_change_semantic_extraction_identity():
    registry = AdapterIdentityRegistry()
    registry.register(
        AdapterIdentityManifest(
            adapter_id="telegram",
            authority_id="bot-main",
            identity_version=1,
            identity_fields=("chat_id", "message_id"),
            allowed_source_types=("message",),
        )
    )
    store = InMemorySourceStore()
    graph = SourceBoundEvidenceGraph(source_verifier=store)
    ingestion = IngestionPipeline(registry=registry, source_store=store, evidence_graph=graph)
    text = "same source state"
    common = dict(
        adapter_id="telegram",
        authority_id="bot-main",
        identity_version=1,
        identity_values=(
            AdapterIdentityValue(field="chat_id", value="42"),
            AdapterIdentityValue(field="message_id", value="uri-drift"),
        ),
        source_type="message",
        content=text.encode(),
        mime_type="text/plain",
        observed_at=OBSERVED,
        observations=(DirectObservation(excerpt=text),),
    )
    first_ingestion = ingestion.ingest(IngestionEnvelope(**common, uri="tg://old"))
    second_ingestion = ingestion.ingest(IngestionEnvelope(**common, uri="tg://new"))
    assert first_ingestion.source_version_id == second_ingestion.source_version_id
    assert first_ingestion.receipt_fingerprint != second_ingestion.receipt_fingerprint

    p = proposal(first_ingestion.claim_ids[0])
    engine = SemanticExtractionEngine(
        evidence_graph=graph,
        extractor=StaticExtractor((p,)),
    )
    first = engine.extract(first_ingestion)
    second = engine.extract(second_ingestion)
    assert first.input_fingerprint == second.input_fingerprint
    assert first.derived_claim_ids == second.derived_claim_ids
    assert first.proposal_fingerprint == second.proposal_fingerprint
    assert first.receipt_fingerprint != second.receipt_fingerprint


def test_invalid_later_proposal_cannot_leave_partial_graph_mutation():
    ingestion, _, graph = make_runtime()
    receipt = ingest_text(ingestion)
    valid = proposal(receipt.claim_ids[0], statement="valid first")
    invalid = proposal("claim-outside-input", statement="invalid second")
    before = set(graph._claims)  # noqa: SLF001 - assert transactional firewall behavior

    with pytest.raises(InvalidClaimProposalError, match="outside extraction input"):
        SemanticExtractionEngine(
            evidence_graph=graph,
            extractor=StaticExtractor((valid, invalid)),
        ).extract(receipt)

    assert set(graph._claims) == before  # noqa: SLF001
    assert all(claim.claim_type != "semantic_interpretation" for claim in graph._claims.values())  # noqa: SLF001


def test_policy_caps_source_claims_per_input_before_extractor_call():
    ingestion, _, graph = make_runtime()
    receipt = ingestion.ingest(
        IngestionEnvelope(
            adapter_id="telegram",
            authority_id="bot-main",
            identity_version=1,
            identity_values=(
                AdapterIdentityValue(field="chat_id", value="42"),
                AdapterIdentityValue(field="message_id", value="input-cap"),
            ),
            source_type="message",
            content=b"alpha beta",
            mime_type="text/plain",
            observed_at=OBSERVED,
            observations=(
                DirectObservation(excerpt="alpha", start_offset=0, end_offset=5),
                DirectObservation(excerpt="beta", start_offset=6, end_offset=10),
            ),
        )
    )
    engine = SemanticExtractionEngine(
        evidence_graph=graph,
        extractor=StaticExtractor(()),
        policy=SemanticExtractionPolicy(max_source_claims_per_input=1),
    )
    with pytest.raises(InvalidExtractionInputError, match="policy limit"):
        engine.extract(receipt)


def test_policy_caps_source_claims_per_proposal_without_partial_mutation():
    ingestion, _, graph = make_runtime()
    receipt = ingestion.ingest(
        IngestionEnvelope(
            adapter_id="telegram",
            authority_id="bot-main",
            identity_version=1,
            identity_values=(
                AdapterIdentityValue(field="chat_id", value="42"),
                AdapterIdentityValue(field="message_id", value="proposal-cap"),
            ),
            source_type="message",
            content=b"alpha beta",
            mime_type="text/plain",
            observed_at=OBSERVED,
            observations=(
                DirectObservation(excerpt="alpha", start_offset=0, end_offset=5),
                DirectObservation(excerpt="beta", start_offset=6, end_offset=10),
            ),
        )
    )
    p = ClaimExtractionProposal(
        source_claim_ids=receipt.claim_ids,
        statement="combined interpretation",
        pattern_key="combined",
    )
    before = set(graph._claims)  # noqa: SLF001
    with pytest.raises(InvalidClaimProposalError, match="policy limit"):
        SemanticExtractionEngine(
            evidence_graph=graph,
            extractor=StaticExtractor((p,)),
            policy=SemanticExtractionPolicy(max_source_claims_per_proposal=1),
        ).extract(receipt)
    assert set(graph._claims) == before  # noqa: SLF001


def test_proposal_payload_has_bounded_statement_and_pattern_key():
    with pytest.raises(ValidationError):
        ClaimExtractionProposal(
            source_claim_ids=("c1",),
            statement="x" * 4097,
        )
    with pytest.raises(ValidationError):
        ClaimExtractionProposal(
            source_claim_ids=("c1",),
            statement="valid",
            pattern_key="a" * 129,
        )


class FlappingManifestExtractor:
    def __init__(self, proposals):
        self.proposals = tuple(proposals)
        self.manifest_reads = 0
        self._manifests = (
            ExtractorManifest(
                extractor_id="flapping",
                extractor_version="1",
                configuration_fingerprint="1" * 64,
            ),
            ExtractorManifest(
                extractor_id="flapping",
                extractor_version="1",
                configuration_fingerprint="2" * 64,
            ),
        )

    @property
    def manifest(self):
        index = min(self.manifest_reads, len(self._manifests) - 1)
        self.manifest_reads += 1
        return self._manifests[index]

    def extract(self, context):
        return self.proposals


def test_extractor_manifest_is_snapshotted_once_per_run():
    ingestion, _, graph = make_runtime()
    receipt = ingest_text(ingestion)
    extractor = FlappingManifestExtractor((proposal(receipt.claim_ids[0]),))
    result = SemanticExtractionEngine(
        evidence_graph=graph,
        extractor=extractor,
    ).extract(receipt)

    assert extractor.manifest_reads == 1
    assert result.extractor_manifest_fingerprint == extractor._manifests[0].fingerprint
    derived = graph.get_claim(result.derived_claim_ids[0])
    assert "config-111111111111" in derived.producer


def test_existing_claim_conflict_is_preflighted_before_any_new_batch_claim_is_written():
    ingestion, _, graph = make_runtime()
    receipt = ingest_text(ingestion)
    first = proposal(receipt.claim_ids[0], statement="first-new", pattern_key="first")
    second = proposal(receipt.claim_ids[0], statement="second-existing", pattern_key="second")

    existing_result = SemanticExtractionEngine(
        evidence_graph=graph,
        extractor=StaticExtractor((second,)),
    ).extract(receipt)
    conflict_id = existing_result.derived_claim_ids[0]
    existing = graph.get_claim(conflict_id)
    graph._claims[conflict_id] = existing.model_copy(  # noqa: SLF001 - adversarial corruption
        update={"statement": "corrupted-existing-claim"}
    )

    with pytest.raises(ExtractionReplayConflictError, match="Canonical extracted claim ID"):
        SemanticExtractionEngine(
            evidence_graph=graph,
            extractor=StaticExtractor((first, second)),
        ).extract(receipt)

    assert all(claim.statement != "first-new" for claim in graph._claims.values())  # noqa: SLF001


def test_duplicate_ids_in_forged_ingestion_receipt_are_rejected():
    ingestion, _, graph = make_runtime()
    receipt = ingest_text(ingestion)
    forged = receipt.model_copy(
        update={
            "claim_ids": receipt.claim_ids + receipt.claim_ids,
            "evidence_ids": receipt.evidence_ids + receipt.evidence_ids,
        }
    )
    with pytest.raises(InvalidExtractionInputError, match="duplicate claim IDs"):
        SemanticExtractionEngine(
            evidence_graph=graph,
            extractor=StaticExtractor(()),
        ).extract(forged)
