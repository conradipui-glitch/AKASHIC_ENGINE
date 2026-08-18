# ADR-0003: Audit provenance before importing third-party code

**Status:** Accepted

## Decision

No third-party source code is copied into Akashic Engine until its repository, commit/ref, license, relevant files, and intended reuse mode are recorded.

Each candidate component receives one disposition:

- KEEP
- ADAPT
- REWRITE
- IGNORE

## Consequences

- Research may reference external implementations immediately.
- Code imports wait for license/provenance review.
- Architectural ideas can be rewritten independently when that is cleaner than dependency reuse.
