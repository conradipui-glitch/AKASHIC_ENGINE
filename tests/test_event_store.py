from akashic_engine import (
    EngineEvent,
    EpistemicBoundaryError,
    EpistemicLevel,
    EventType,
    InMemoryEventStore,
)


def event(level: EpistemicLevel, value: str, *, entity_id: str = "user-1") -> EngineEvent:
    return EngineEvent(
        entity_type="memory",
        entity_id=entity_id,
        payload={"value": value},
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
