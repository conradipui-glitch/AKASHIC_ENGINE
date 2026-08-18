# Research workspace

This directory tracks source-level teardown work before implementation decisions are locked.

## Initial sources

| Source | Primary question |
|---|---|
| Akashik Protocol/Core | Which memory/event/replay contracts are reusable? |
| Open-Sable | Which memory, ledger, RAG, pattern and cognition modules are reusable? |
| Wikimind | Which ingestion/provenance/knowledge-graph patterns are reusable? |
| Brahmanda | Which soul continuity / karma / past-life abstractions are useful as symbolic models? |
| Loom/MultiLoom | Which branching and alternative-reading UX patterns are reusable? |
| Destiny Council | Which method registry and input-validation patterns are useful? |

## Teardown output contract

Each source teardown should record:

- repository + exact ref/commit
- license
- relevant files/classes/functions
- what the code actually does
- dependencies
- portability concerns
- KEEP / ADAPT / REWRITE / IGNORE decisions
- proposed Akashic Engine destination
- legal/provenance notes
