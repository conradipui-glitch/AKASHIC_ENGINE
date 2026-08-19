from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from akashic_engine import (
    AdapterIdentityManifest,
    AdapterIdentityRegistry,
    AdapterIdentityValue,
    ClaimExtractionProposal,
    DirectObservation,
    EpistemicLevel,
    EvidenceBundleBuilder,
    ExactHintPatternMatcher,
    ExtractorManifest,
    InMemorySourceStore,
    IngestionEnvelope,
    IngestionPipeline,
    LongitudinalPatternEngine,
    PatternMatchProposal,
    PatternMatcherManifest,
    PatternPromotionEngine,
    PatternPromotionPolicy,
    SemanticExtractionEngine,
    SourceBoundEvidenceGraph,
    ValidationState,
)
from akashic_engine.evidence_graph import ClaimRelationType
from akashic_engine.pattern_matcher import (
    InvalidPatternMatchInputError,
    InvalidPatternMatchProposalError,
    PatternPromotionConflictError,
)


OBSERVED = datetime(2026, 8, 19, 12, 0, tzinfo=timezone.utc)
EXTRACTOR_CONFIG = "a" * 64
MATCHER_CONFIG = "b" * 64


def _sha256_json(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


class StaticExtractor:
    def __init__(self, proposals):
        self.proposals = tuple(proposals)

    def extract(self, context):
        return self.proposals


class StaticMatcher:
    def __init__(self, proposals):
        self.proposals = tuple(proposals)

    def match(self, context):
        return self.proposals


class BadListMatcher(StaticMatcher):
    def match(self, context):
        return list(self.proposals)


def runtime():
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


def ingest(ingestion, *, message_id: str, text: str, at: datetime, uri: str | None = None):
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
            content=text.encode(),
            mime_type="text/plain",
            observed_at=at,
            uri=uri,
            observations=(DirectObservation(excerpt=text),),
        )
    )


def extraction_receipt(graph, ingestion_receipt, *, statement="Interest drops after intense starts", hint="interest-cycle"):
    source_claim_id = ingestion_receipt.claim_ids[0]
    proposal = ClaimExtractionProposal(
        source_claim_ids=(source_claim_id,),
        statement=statement,
        pattern_hint=hint,
    )
    return SemanticExtractionEngine(
        evidence_graph=graph,
        extractor=StaticExtractor((proposal,)),
        extractor_manifest=ExtractorManifest(
            extractor_id="test.semantic",
            extractor_version="1",
            configuration_fingerprint=EXTRACTOR_CONFIG,
        ),
    ).extract(ingestion_receipt)


def matcher_manifest():
    return PatternMatcherManifest(
        matcher_id="exact-hint",
        matcher_version="1",
        configuration_fingerprint=MATCHER_CONFIG,
    )


def promotion_engine(graph, matcher=None, policy=None):
    return PatternPromotionEngine(
        evidence_graph=graph,
        matcher=matcher or ExactHintPatternMatcher(),
        matcher_manifest=matcher_manifest(),
        policy=policy,
    )


def two_receipts(*, hint="interest-cycle"):
    ingestion, _, graph = runtime()
    first_ingestion = ingest(
        ingestion,
        message_id="1",
        text="I start projects intensely and later lose interest",
        at=OBSERVED,
    )
    second_ingestion = ingest(
        ingestion,
        message_id="2",
        text="Again I went deep into a project and then dropped it",
        at=OBSERVED + timedelta(days=90),
    )
    first = extraction_receipt(graph, first_ingestion, hint=hint)
    second = extraction_receipt(graph, second_ingestion, hint=hint)
    return graph, first, second


def test_exact_hint_matcher_promotes_two_independent_candidates():
    graph, first, second = two_receipts()
    result = promotion_engine(graph).promote((first, second))

    grouped = result.occurrences_by_pattern_key()
    assert set(grouped) == {"interest-cycle"}
    assert len(grouped["interest-cycle"]) == 2
    assert len({item.dependency_group_id for item in grouped["interest-cycle"]}) == 2
    assert all(item.confirmation_claim_id is None for item in grouped["interest-cycle"])


def test_complete_safe_chain_reaches_pattern_engine_only_after_promotion():
    graph, first, second = two_receipts()
    promoted = promotion_engine(graph).promote((first, second))
    occurrences = promoted.occurrences_by_pattern_key()["interest-cycle"]
    claim_ids = promoted.claim_ids_by_pattern_key()["interest-cycle"]
    bundle = EvidenceBundleBuilder(graph).build(claim_ids)

    analysis = LongitudinalPatternEngine().build(
        pattern_type="interest-cycle",
        description="Interest drops after intense project starts",
        evidence_bundle=bundle,
        occurrences=occurrences,
    )

    assert analysis.projection.recurrence_count == 2
    assert analysis.projection.confidence_components.recurrence == 0.25
    assert analysis.projection.confidence_components.evidence_quality == 0.2625
    assert analysis.projection.first_observed == OBSERVED
    assert analysis.projection.last_observed == OBSERVED + timedelta(days=90)


def test_single_candidate_cannot_be_promoted_into_recurrence():
    ingestion, _, graph = runtime()
    ingested = ingest(ingestion, message_id="1", text="one episode", at=OBSERVED)
    extracted = extraction_receipt(graph, ingested)

    with pytest.raises(InvalidPatternMatchInputError, match="at least 2 candidates"):
        promotion_engine(graph).promote((extracted,))


def test_two_interpretations_from_same_source_do_not_become_two_occurrences():
    ingestion, _, graph = runtime()
    ingested = ingest(ingestion, message_id="1", text="one source episode", at=OBSERVED)
    source_claim = ingested.claim_ids[0]
    proposals = (
        ClaimExtractionProposal(
            source_claim_ids=(source_claim,),
            statement="first interpretation",
            pattern_hint="cycle",
        ),
        ClaimExtractionProposal(
            source_claim_ids=(source_claim,),
            statement="second interpretation",
            pattern_hint="cycle",
        ),
    )
    extracted = SemanticExtractionEngine(
        evidence_graph=graph,
        extractor=StaticExtractor(proposals),
        extractor_manifest=ExtractorManifest(
            extractor_id="test.semantic",
            extractor_version="1",
            configuration_fingerprint=EXTRACTOR_CONFIG,
        ),
    ).extract(ingested)

    with pytest.raises(InvalidPatternMatchProposalError, match="independent.*groups|dependency group"):
        promotion_engine(graph).promote((extracted,))


def test_multiple_hint_variants_for_one_semantic_claim_are_rejected_as_ambiguous():
    ingestion, _, graph = runtime()
    ingested = ingest(ingestion, message_id="1", text="one source episode", at=OBSERVED)
    source_claim = ingested.claim_ids[0]
    proposals = (
        ClaimExtractionProposal(
            source_claim_ids=(source_claim,),
            statement="same interpretation",
            pattern_hint="cycle-a",
        ),
        ClaimExtractionProposal(
            source_claim_ids=(source_claim,),
            statement="same interpretation",
            pattern_hint="cycle-b",
        ),
    )
    extracted = SemanticExtractionEngine(
        evidence_graph=graph,
        extractor=StaticExtractor(proposals),
        extractor_manifest=ExtractorManifest(
            extractor_id="test.semantic",
            extractor_version="1",
            configuration_fingerprint=EXTRACTOR_CONFIG,
        ),
    ).extract(ingested)

    with pytest.raises(InvalidPatternMatchInputError, match="ambiguous"):
        promotion_engine(graph).promote((extracted,))


def test_malicious_matcher_cannot_group_different_hints_under_v1_policy():
    ingestion, _, graph = runtime()
    a_ing = ingest(ingestion, message_id="1", text="alpha episode", at=OBSERVED)
    b_ing = ingest(
        ingestion,
        message_id="2",
        text="beta episode",
        at=OBSERVED + timedelta(days=30),
    )
    a = extraction_receipt(graph, a_ing, hint="cycle-a")
    b = extraction_receipt(graph, b_ing, hint="cycle-b")
    candidate_ids = (
        a.occurrence_candidates[0].candidate_id,
        b.occurrence_candidates[0].candidate_id,
    )
    matcher = StaticMatcher(
        (PatternMatchProposal(candidate_ids=candidate_ids, pattern_key="cycle-a"),)
    )

    with pytest.raises(InvalidPatternMatchProposalError, match="exactly match"):
        promotion_engine(graph, matcher=matcher).promote((a, b))

def test_matcher_cannot_reference_candidate_outside_validated_input():
    graph, first, second = two_receipts()
    matcher = StaticMatcher((PatternMatchProposal(candidate_ids=(
        first.occurrence_candidates[0].candidate_id,
        "ocx_missing",
    ), pattern_key="interest-cycle"),))

    with pytest.raises(InvalidPatternMatchProposalError, match="outside validated input"):
        promotion_engine(graph, matcher=matcher).promote((first, second))


def test_matcher_output_is_canonically_revalidated_after_callback():
    graph, first, second = two_receipts()
    forged = PatternMatchProposal.model_construct(candidate_ids=(), pattern_key="BAD KEY")

    with pytest.raises(InvalidPatternMatchProposalError, match="canonical validation"):
        promotion_engine(graph, matcher=StaticMatcher((forged,))).promote((first, second))


def test_matcher_must_return_typed_tuple():
    graph, first, second = two_receipts()
    proposal = PatternMatchProposal(
        candidate_ids=(first.occurrence_candidates[0].candidate_id, second.occurrence_candidates[0].candidate_id),
        pattern_key="interest-cycle",
    )

    with pytest.raises(InvalidPatternMatchProposalError, match="tuple"):
        promotion_engine(graph, matcher=BadListMatcher((proposal,))).promote((first, second))


def test_forged_extraction_receipt_fingerprint_is_rejected():
    graph, first, second = two_receipts()
    forged = first.model_copy(update={"receipt_fingerprint": "0" * 64})

    with pytest.raises(InvalidPatternMatchInputError, match="receipt fingerprint mismatch"):
        promotion_engine(graph).promote((forged, second))


def test_forged_candidate_fingerprint_is_rejected_even_if_receipt_is_rehashed():
    graph, first, second = two_receipts()
    candidate = first.occurrence_candidates[0].model_copy(
        update={"candidate_fingerprint": "1" * 64}
    )
    forged = first.model_copy(update={"occurrence_candidates": (candidate,)})
    forged_fp = _sha256_json(
        {
            "ingestion_receipt_fingerprint": forged.ingestion_receipt_fingerprint,
            "extractor_manifest_fingerprint": forged.extractor_manifest_fingerprint,
            "policy_fingerprint": forged.policy_fingerprint,
            "input_fingerprint": forged.input_fingerprint,
            "proposal_fingerprint": forged.proposal_fingerprint,
            "derived_claim_ids": forged.derived_claim_ids,
            "occurrence_candidates": [item.model_dump(mode="json") for item in forged.occurrence_candidates],
        }
    )
    forged = forged.model_copy(update={"receipt_fingerprint": forged_fp})

    with pytest.raises(InvalidPatternMatchInputError, match="candidate fingerprint mismatch"):
        promotion_engine(graph).promote((forged, second))


def test_candidate_dependency_group_is_recomputed_from_evidence_sources():
    graph, first, second = two_receipts()
    candidate = first.occurrence_candidates[0]
    changed = candidate.model_copy(update={"dependency_group_id": "dep_forged"})
    fp = _sha256_json(
        {
            "pattern_hint": changed.pattern_hint,
            "extraction_claim_id": changed.extraction_claim_id,
            "source_claim_ids": changed.source_claim_ids,
            "evidence_refs": changed.evidence_refs,
            "dependency_group_id": changed.dependency_group_id,
        }
    )
    changed = changed.model_copy(
        update={"candidate_fingerprint": fp, "candidate_id": "ocx_" + fp[:40]}
    )
    forged = first.model_copy(update={"occurrence_candidates": (changed,)})
    receipt_fp = _sha256_json(
        {
            "ingestion_receipt_fingerprint": forged.ingestion_receipt_fingerprint,
            "extractor_manifest_fingerprint": forged.extractor_manifest_fingerprint,
            "policy_fingerprint": forged.policy_fingerprint,
            "input_fingerprint": forged.input_fingerprint,
            "proposal_fingerprint": forged.proposal_fingerprint,
            "derived_claim_ids": forged.derived_claim_ids,
            "occurrence_candidates": [item.model_dump(mode="json") for item in forged.occurrence_candidates],
        }
    )
    forged = forged.model_copy(update={"receipt_fingerprint": receipt_fp})

    with pytest.raises(InvalidPatternMatchInputError, match="dependency group"):
        promotion_engine(graph).promote((forged, second))


def test_missing_derived_from_provenance_blocks_promotion():
    graph, first, second = two_receipts()
    claim_id = first.occurrence_candidates[0].extraction_claim_id
    graph._relations = {  # noqa: SLF001 - adversarial graph corruption
        edge for edge in graph._relations  # noqa: SLF001
        if not (edge.source_claim_id == claim_id and edge.relation is ClaimRelationType.DERIVED_FROM)
    }

    with pytest.raises(InvalidPatternMatchInputError, match="DERIVED_FROM"):
        promotion_engine(graph).promote((first, second))


def test_contradicted_interpretation_cannot_be_promoted():
    graph, first, second = two_receipts()
    claim_id = first.occurrence_candidates[0].extraction_claim_id
    claim = graph.get_claim(claim_id)
    graph._claims[claim_id] = claim.model_copy(  # noqa: SLF001
        update={"validation_state": ValidationState.CONTRADICTED}
    )

    with pytest.raises(InvalidPatternMatchInputError, match="contradicted"):
        promotion_engine(graph).promote((first, second))


def test_uri_drift_cannot_be_counted_as_a_second_occurrence():
    ingestion, _, graph = runtime()
    first_ingestion = ingest(
        ingestion,
        message_id="same",
        text="same semantic episode",
        at=OBSERVED,
        uri="tg://old",
    )
    second_ingestion = ingest(
        ingestion,
        message_id="same",
        text="same semantic episode",
        at=OBSERVED,
        uri="tg://new",
    )
    first = extraction_receipt(graph, first_ingestion)
    second = extraction_receipt(graph, second_ingestion)
    assert first.receipt_fingerprint != second.receipt_fingerprint
    assert first.occurrence_candidates == second.occurrence_candidates

    with pytest.raises(InvalidPatternMatchInputError, match="at least 2 candidates"):
        promotion_engine(graph).promote((first, second))


def test_receipt_order_does_not_change_promotion_result():
    graph_a, first_a, second_a = two_receipts()
    a = promotion_engine(graph_a).promote((first_a, second_a))

    graph_b, first_b, second_b = two_receipts()
    b = promotion_engine(graph_b).promote((second_b, first_b))

    assert a.input_fingerprint == b.input_fingerprint
    assert a.proposal_fingerprint == b.proposal_fingerprint
    assert a.promotions == b.promotions
    assert a.receipt_fingerprint == b.receipt_fingerprint


def test_duplicate_extraction_receipt_is_rejected():
    graph, first, second = two_receipts()
    with pytest.raises(InvalidPatternMatchInputError, match="duplicate semantic extraction receipts"):
        promotion_engine(graph).promote((first, first, second))


def test_exact_hint_matcher_returns_no_match_for_distinct_hints():
    ingestion, _, graph = runtime()
    a_ing = ingest(ingestion, message_id="1", text="alpha", at=OBSERVED)
    b_ing = ingest(ingestion, message_id="2", text="beta", at=OBSERVED + timedelta(days=1))
    a = extraction_receipt(graph, a_ing, hint="alpha-cycle")
    b = extraction_receipt(graph, b_ing, hint="beta-cycle")

    result = promotion_engine(graph).promote((a, b))
    assert result.promotions == ()
    assert result.occurrences_by_pattern_key() == {}


def test_matcher_manifest_is_engine_owned_not_matcher_asserted():
    class SpoofingMatcher(ExactHintPatternMatcher):
        manifest_reads = 0

        @property
        def manifest(self):
            self.manifest_reads += 1
            return PatternMatcherManifest(
                matcher_id="spoof",
                matcher_version="999",
                configuration_fingerprint="f" * 64,
            )

    graph, first, second = two_receipts()
    matcher = SpoofingMatcher()
    trusted = matcher_manifest()
    result = PatternPromotionEngine(
        evidence_graph=graph,
        matcher=matcher,
        matcher_manifest=trusted,
    ).promote((first, second))

    assert matcher.manifest_reads == 0
    assert result.matcher_manifest_fingerprint == trusted.fingerprint


def test_promotion_policy_is_snapshotted_before_matcher_callback():
    graph, first, second = two_receipts()

    class PolicyMutatingMatcher(ExactHintPatternMatcher):
        engine = None

        def match(self, context):
            assert self.engine is not None
            self.engine.policy = PatternPromotionPolicy(min_independent_groups=3)
            return super().match(context)

    matcher = PolicyMutatingMatcher()
    engine = PatternPromotionEngine(
        evidence_graph=graph,
        matcher=matcher,
        matcher_manifest=matcher_manifest(),
        policy=PatternPromotionPolicy(min_independent_groups=2),
    )
    matcher.engine = engine
    result = engine.promote((first, second))

    assert len(result.promotions) == 2
    assert result.policy_fingerprint == PatternPromotionPolicy(min_independent_groups=2).fingerprint


def test_pattern_match_proposal_validates_identifier_and_unique_candidates():
    with pytest.raises(ValidationError):
        PatternMatchProposal(candidate_ids=("a", "b"), pattern_key="BAD KEY")
    with pytest.raises(ValidationError):
        PatternMatchProposal(candidate_ids=("a", "a"), pattern_key="good")


def test_duplicate_content_across_distinct_sources_is_not_independent_recurrence():
    ingestion, _, graph = runtime()
    first_ingestion = ingest(
        ingestion,
        message_id="1",
        text="identical captured event",
        at=OBSERVED,
    )
    second_ingestion = ingest(
        ingestion,
        message_id="2",
        text="identical captured event",
        at=OBSERVED + timedelta(days=10),
    )
    assert first_ingestion.source_id != second_ingestion.source_id
    first = extraction_receipt(graph, first_ingestion, hint="same-event")
    second = extraction_receipt(graph, second_ingestion, hint="same-event")

    with pytest.raises(InvalidPatternMatchProposalError, match="independent lineage"):
        promotion_engine(graph).promote((first, second))


def test_repeated_promotion_is_deterministic_and_idempotent():
    graph, first, second = two_receipts()
    engine = promotion_engine(graph)
    a = engine.promote((first, second))
    b = engine.promote((first, second))

    assert a == b
    assert a.occurrences_by_pattern_key() == b.occurrences_by_pattern_key()


def test_model_construct_cannot_bypass_candidate_validation_at_promotion_boundary():
    graph, first, second = two_receipts()
    original = first.occurrence_candidates[0]
    forged_candidate = original.__class__.model_construct(
        candidate_id="ocx_forged",
        pattern_hint="BAD HINT",
        extraction_claim_id=original.extraction_claim_id,
        source_claim_ids=(),
        evidence_refs=(),
        dependency_group_id="dep_forged",
        candidate_fingerprint="a" * 64,
    )
    forged = first.model_copy(update={"occurrence_candidates": (forged_candidate,)})
    receipt_fp = _sha256_json(
        {
            "ingestion_receipt_fingerprint": forged.ingestion_receipt_fingerprint,
            "extractor_manifest_fingerprint": forged.extractor_manifest_fingerprint,
            "policy_fingerprint": forged.policy_fingerprint,
            "input_fingerprint": forged.input_fingerprint,
            "proposal_fingerprint": forged.proposal_fingerprint,
            "derived_claim_ids": forged.derived_claim_ids,
            "occurrence_candidates": [
                item.model_dump(mode="json") for item in forged.occurrence_candidates
            ],
        }
    )
    forged = forged.model_copy(update={"receipt_fingerprint": receipt_fp})

    with pytest.raises(InvalidPatternMatchInputError, match="canonical validation"):
        promotion_engine(graph).promote((forged, second))


def test_matcher_cannot_mutate_graph_provenance_during_callback():
    graph, first, second = two_receipts()
    target_claim_id = first.occurrence_candidates[0].extraction_claim_id

    class GraphMutatingMatcher(ExactHintPatternMatcher):
        def match(self, context):
            result = super().match(context)
            current = graph.get_claim(target_claim_id)
            graph._claims[target_claim_id] = current.model_copy(  # noqa: SLF001
                update={"validation_state": ValidationState.CONTRADICTED}
            )
            return result

    with pytest.raises(PatternPromotionConflictError, match="context changed during matcher callback"):
        promotion_engine(graph, matcher=GraphMutatingMatcher()).promote((first, second))
