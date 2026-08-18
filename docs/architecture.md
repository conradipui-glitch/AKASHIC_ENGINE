# Architecture & Interfaces

## System boundary

Akashic Engine is headless. IVA, Telegram, Hermes, OpenAI-based agents, and future hosts are clients/adapters, not dependencies of the core.

## Core contracts

The following contracts are targets for Milestone 2. Their exact schemas must be validated against repository teardown evidence before implementation.

- `MemoryStore`: remember, recall, supersede, retract, replay.
- `AttunementEngine`: score, retrieve, explain relevance.
- `ModelProvider`: generate, reason, embed, rerank.
- `Reader`: consume structured context and return hypotheses, evidence, and confidence.
- `KnowledgeProvider`: search, contextualize, provenance.
- `Plugin`: manifest, capabilities, invoke.
- `EntitlementProvider`: `can(user, capability)`.
- `Scheduler`: register and execute recurring insight jobs.

## Planned modules

- `core` — sessions, readings, patterns, evidence, replay.
- `memory` — episodic, semantic, longitudinal, symbolic.
- `attunement` — retrieval, recurrence, weighting, conflicts.
- `readers` — analytical, symbolic, Akashic, skeptic, synthesizer.
- `knowledge` — ingest, graph, corpora, sources.
- `providers` — OpenAI-compatible and local/cloud model adapters.
- `interfaces` — REST, MCP, Python SDK.
- `clients` — Telegram bot and Mini App.
- `integrations` — Astrology, Human Design, Tarot, Dreams.
- `billing` and `scheduler` — optional outer layers.

## Data separation rule

Observed facts and user memories are stored separately from derived patterns and symbolic interpretations. Every derived conclusion must retain links to evidence and the reader/version that produced it.

## Portability rule

A clean deployment must be able to run without Telegram, IVA, payments, or any spiritual plugin.
