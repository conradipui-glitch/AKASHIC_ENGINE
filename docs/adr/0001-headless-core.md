# ADR-0001: Akashic Engine is a headless standalone core

**Status:** Accepted

## Decision

Akashic Engine must not depend on IVA, Telegram, Hermes, or any individual LLM provider.

Clients and agent harnesses connect through explicit adapters/interfaces.

## Consequences

- The same engine can run standalone or behind an agent host.
- Telegram is a first-party client, not part of reasoning core.
- IVA integration can be replaced without migrating Akashic state.
- Core acceptance tests must run without Telegram or IVA.
