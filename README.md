# AKASHIC ENGINE

> Provider-agnostic longitudinal intelligence engine for AI agents.

Akashic Engine is an experimental headless engine that combines long-term memory, pattern mining, attunement-based retrieval, multi-reader reasoning, evidence/replay, and an optional symbolic / Akashic interpretation layer.

The project is intentionally independent from any single agent host, UI, or LLM provider. IVA, Telegram, Hermes, OpenAI-based agents, and future harnesses should connect as clients or adapters.

## Product forms

- **Headless engine** exposed through REST API, MCP, and Python SDK.
- **Standalone Telegram Bot + Mini App** as the first first-party client.
- **Agent-hosted mode** for IVA and other agent harnesses.

## Core principles

1. The engine must not depend on IVA, Telegram, or any single LLM vendor.
2. Observed evidence, analytical inference, and symbolic/esoteric interpretation remain separate.
3. Memory is longitudinal: the system should learn recurring themes across months and years.
4. Every meaningful reading should be replayable and traceable to supporting evidence.
5. Astrology, Human Design, Tarot, Dreams, Journaling, and other systems connect through adapters/plugins.
6. Billing and subscriptions are optional outer layers, never hard-wired into core reasoning.

## v0.1 boundary

The first working version is deliberately small:

- user/profile
- memory ingestion
- attunement retrieval
- pattern extraction
- one reader
- evidence links
- saved readings
- replay

Not part of v0.1: Telegram Mini App, payments, crypto, full spiritual corpus, Astrology/Human Design/Tarot integrations, or large multi-agent orchestration.

### First acceptance test

A user can add several memories/events, ask **“What patterns are repeating?”**, receive a structured reading with supporting evidence and confidence, save it, and later replay how the conclusion was produced.

## Research sources

Initial reference projects:

- Open-Sable
- Akashik Protocol / Core
- Wikimind
- Brahmanda
- Command Line Loom / MultiLoom
- Destiny Council

Every reuse decision will be classified as **KEEP / ADAPT / REWRITE / IGNORE** and tracked with source and license provenance.

## Repository status

**Phase:** Milestone 1 — repository teardown.

No third-party source code has been imported yet.

## License status

A project license has intentionally **not yet been selected**. We are auditing the licenses and provenance of all reference implementations before deciding the final licensing model.

See `docs/` and `research/` for architecture decisions and teardown notes.
