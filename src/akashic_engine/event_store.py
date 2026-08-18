"""Reference in-memory implementation of the append-only EventStore contract."""

from __future__ import annotations

from threading import RLock

from .domain import EngineEvent, EpistemicLevel, EventType


class EventStoreError(RuntimeError):
    pass


class EventNotFoundError(EventStoreError):
    pass


class InvalidEventTransitionError(EventStoreError):
    pass


class EpistemicBoundaryError(InvalidEventTransitionError):
    """Raised when a weaker epistemic layer attempts to rewrite observed truth."""


class InMemoryEventStore:
    """Small reference adapter used to define and test EventStore invariants.

    It is intentionally simple: PostgreSQL durability comes later. The important
    behavior here is append-only ordering, replay, non-destructive correction,
    and epistemic-boundary enforcement.
    """

    def __init__(self) -> None:
        self._events: list[EngineEvent] = []
        self._by_id: dict[str, EngineEvent] = {}
        self._lock = RLock()

    def append(self, event: EngineEvent) -> EngineEvent:
        with self._lock:
            if event.event_id in self._by_id:
                raise InvalidEventTransitionError(f"Duplicate event_id: {event.event_id}")
            if event.seq is not None:
                raise InvalidEventTransitionError("seq is adapter-assigned and must be None on append")
            stored = event.model_copy(update={"seq": len(self._events) + 1})
            self._events.append(stored)
            self._by_id[stored.event_id] = stored
            return stored

    def read(self, *, after_seq: int = 0) -> tuple[EngineEvent, ...]:
        return tuple(e for e in self._events if (e.seq or 0) > after_seq)

    def replay(self, *, entity_type: str, entity_id: str) -> tuple[EngineEvent, ...]:
        return tuple(
            e
            for e in self._events
            if e.entity_type == entity_type and e.entity_id == entity_id
        )

    def _get(self, event_id: str) -> EngineEvent:
        try:
            return self._by_id[event_id]
        except KeyError as exc:
            raise EventNotFoundError(event_id) from exc

    def supersede(self, target_event_id: str, replacement: EngineEvent) -> EngineEvent:
        with self._lock:
            target = self._get(target_event_id)
            if target.event_type is EventType.RETRACT:
                raise InvalidEventTransitionError("Cannot supersede a retract event")
            if replacement.seq is not None:
                raise InvalidEventTransitionError("replacement seq must be None")
            if target.epistemic_level is EpistemicLevel.OBSERVED and replacement.epistemic_level is not EpistemicLevel.OBSERVED:
                raise EpistemicBoundaryError(
                    "Interpretation/symbolic hypothesis cannot supersede observed truth"
                )
            if replacement.entity_type != target.entity_type or replacement.entity_id != target.entity_id:
                raise InvalidEventTransitionError("replacement must target the same entity")

            superseding = replacement.model_copy(
                update={
                    "event_type": EventType.SUPERSEDE,
                    "target_event_id": target_event_id,
                }
            )
            return self.append(superseding)

    def retract(
        self,
        target_event_id: str,
        *,
        producer: str,
        intent: str,
        reason: str,
    ) -> EngineEvent:
        with self._lock:
            target = self._get(target_event_id)
            event = EngineEvent(
                event_type=EventType.RETRACT,
                entity_type=target.entity_type,
                entity_id=target.entity_id,
                payload={"reason": reason},
                epistemic_level=target.epistemic_level,
                producer=producer,
                intent=intent,
                target_event_id=target_event_id,
            )
            return self.append(event)

    def current_records(
        self,
        *,
        entity_type: str | None = None,
        entity_id: str | None = None,
        epistemic_levels: set[EpistemicLevel] | None = None,
    ) -> tuple[EngineEvent, ...]:
        """Project currently active record/supersede events without deleting history."""

        superseded_or_retracted = {
            e.target_event_id
            for e in self._events
            if e.event_type in {EventType.SUPERSEDE, EventType.RETRACT} and e.target_event_id
        }
        result: list[EngineEvent] = []
        for event in self._events:
            if event.event_type is EventType.RETRACT:
                continue
            if event.event_id in superseded_or_retracted:
                continue
            if entity_type is not None and event.entity_type != entity_type:
                continue
            if entity_id is not None and event.entity_id != entity_id:
                continue
            if epistemic_levels is not None and event.epistemic_level not in epistemic_levels:
                continue
            result.append(event)
        return tuple(result)
