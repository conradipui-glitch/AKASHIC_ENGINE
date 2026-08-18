# ADR-0002: Separate evidence, inference, and symbolic interpretation

**Status:** Accepted

## Decision

The data model must distinguish:

1. observed/user-provided evidence,
2. derived analytical patterns and hypotheses,
3. symbolic or esoteric interpretations.

A derived reading must preserve provenance links to the evidence and reader/version that produced it.

## Consequences

- The system can explain why a conclusion was reached.
- A skeptical reader can challenge conclusions without deleting source evidence.
- Symbolic readings can be disabled without damaging longitudinal memory.
- Replay can reconstruct the state and reasoning context of past readings.
