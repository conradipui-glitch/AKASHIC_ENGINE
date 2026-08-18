import pytest

from akashic_engine import (
    EngineEvent,
    EpistemicBoundaryError,
    EpistemicLevel,
    EventType,
    InMemoryEventStore,
    InvalidEventTransitionError,
)


def event(
    level: EpistemicLevel,
    value: str,
    *,
    entity_id: str = "user-1",
    payload: dict | None = None,
) -> EngineEvent:
    return EngineEvent(
        entity_type="memory",
        entity_id=entity_id,
        payload=payload if payload is not None else {"value": value},
        epistemic_level=level,
        producer="test",
        intent="test invariant",
    )


def test_append_assigns_monotonic_sequence_and_replay_is_complete():
    store = InMemoryEventStore()
    first = store.append(event(EpistemicLevel.OBSERVED, "A"))
    second = store.append(event(EpistemicLevel.INTERPRETATION, "B"))

    assert (first.seq, second.seq) == (1, 2)
    assert store.replay(entity_type="memory", entity_id="user-1") == (first, second)


def test_supersede_preserves_history_but_replaces_current_projection():
    store = InMemoryEventStore()
    first = store.append(event(EpistemicLevel.OBSERVED, "old"))
    replacement = event(EpistemicLevel.OBSERVED, "corrected")

    superseding = store.supersede(first.event_id, replacement)

    assert superseding.event_type is EventType.SUPERSEDE
    assert superseding.target_event_id == first.event_id
    assert store.replay(entity_type="memory", entity_id="user-1") == (first, superseding)
    assert store.current_records(entity_id="user-1") == (superseding,)


def test_symbolic_hypothesis_cannot_supersede_observed_fact():
    store = InMemoryEventStore()
    observed = store.append(event(EpistemicLevel.OBSERVED, "born in Omsk"))

    symbolic = event(EpistemicLevel.SYMBOLIC_HYPOTHESIS, "past-life city resonance")
    try:
        store.supersede(observed.event_id, symbolic)
    except EpistemicBoundaryError:
        pass
    else:
        raise AssertionError("symbolic hypothesis must not supersede observed truth")

    assert store.current_records(
        entity_id="user-1",
        epistemic_levels={EpistemicLevel.OBSERVED},
    ) == (observed,)


def test_symbolic_record_is_never_recalled_as_observed():
    store = InMemoryEventStore()
    observed = store.append(event(EpistemicLevel.OBSERVED, "user statement"))
    store.append(event(EpistemicLevel.SYMBOLIC_HYPOTHESIS, "archetypal echo"))

    assert store.current_records(
        epistemic_levels={EpistemicLevel.OBSERVED}
    ) == (observed,)


def test_retract_removes_from_current_projection_not_replay():
    store = InMemoryEventStore()
    original = store.append(event(EpistemicLevel.INTERPRETATION, "candidate pattern"))
    retract = store.retract(
        original.event_id,
        producer="reviewer",
        intent="correct unsupported interpretation",
        reason="insufficient evidence",
    )

    assert retract.event_type is EventType.RETRACT
    assert store.current_records(entity_id="user-1") == ()
    assert store.replay(entity_type="memory", entity_id="user-1") == (original, retract)


def test_event_payload_is_deeply_immutable_after_append():
    store = InMemoryEventStore()
    stored = store.append(
        event(
            EpistemicLevel.OBSERVED,
            "nested",
            payload={"nested": {"values": [1, 2]}},
        )
    )

    with pytest.raises(TypeError):
        stored.payload["new"] = "mutation"

    with pytest.raises(TypeError):
        stored.payload["nested"]["values"] += (3,)

    assert stored.model_dump(mode="json")["payload"] == {
        "nested": {"values": [1, 2]}
    }


def test_same_event_cannot_be_superseded_twice():
    store = InMemoryEventStore()
    original = store.append(event(EpistemicLevel.OBSERVED, "v1"))
    first_replacement = store.supersede(
        original.event_id, event(EpistemicLevel.OBSERVED, "v2")
    )

    with pytest.raises(InvalidEventTransitionError, match="no longer active"):
        store.supersede(
            original.event_id, event(EpistemicLevel.OBSERVED, "parallel-v2")
        )

    assert store.current_records(entity_id="user-1") == (first_replacement,)


def test_inactive_event_cannot_be_retracted_again():
    store = InMemoryEventStore()
    original = store.append(event(EpistemicLevel.INTERPRETATION, "candidate"))
    store.retract(
        original.event_id,
        producer="reviewer",
        intent="retract",
        reason="unsupported",
    )

    with pytest.raises(InvalidEventTransitionError, match="no longer active"):
        store.retract(
            original.event_id,
            producer="reviewer",
            intent="retract again",
            reason="duplicate",
        )
