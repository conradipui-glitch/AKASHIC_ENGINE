# Roadmap

## Milestone 0 — Definition

Project brief, source inventory, licensing rules, architecture boundaries, MVP acceptance test.

## Milestone 1 — Repository teardown

Deep code review of Open-Sable, Akashik Protocol/Core, Wikimind, Brahmanda, Loom/MultiLoom, and Destiny Council. Produce component selection matrix.

## Milestone 2 — Core contracts

Define schemas and interfaces for MemoryStore, AttunementEngine, Reader, ModelProvider, Evidence, Pattern, Reading, Replay, Plugin, and EntitlementProvider.

## Milestone 3 — v0.1 Memory + Attunement

User/profile, memory ingestion, deterministic persistence, ATTUNE scoring, pattern extraction, single reader, evidence links, saved readings, replay. Add CLI tests and an end-to-end acceptance scenario.

## Milestone 4 — Multi-reader

Analytical Reader, Symbolic Reader, Akashic Reader, Skeptic, Synthesizer. Add disagreement/conflict handling and confidence calibration.

## Milestone 5 — Portability

REST API, Python SDK, MCP server. Prove standalone mode and at least one external agent integration.

## Milestone 6 — IVA integration

Connect Akashic Engine as an external module/tool without coupling its internal state to IVA.

## Milestone 7 — Telegram

Standalone bot first; then Mini App with current themes, patterns, timeline, readings, and evidence graph.

## Milestone 8 — Plugins

Astrology, Human Design, Tarot, Dreams and future manifest-based integrations.

## Milestone 9 — Commercial layer

Entitlements, Telegram Stars, subscriptions, recurring daily/weekly/monthly/yearly insight jobs. Other payment providers remain separate adapters.

## Immediate backlog

1. Deep teardown of Akashik Protocol/Core.
2. Deep teardown of Open-Sable.
3. Draft canonical data schemas.
4. Create component selection matrix.
5. Decide initial Python/PostgreSQL/pgvector stack after teardown evidence.
