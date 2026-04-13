# Stroma: Architecture

## Role

Stroma is the shared substrate under higher-order products that need a neutral corpus and retrieval layer.

Its job is narrow:

1. normalize records into a stable internal shape
2. chunk record bodies into retrievable sections
3. delegate vector generation to an embedder
4. persist records, chunks, vectors, and index metadata in SQLite
5. open immutable local snapshots for typed reads and semantic search

Everything above that line belongs elsewhere.

## Product Boundary

Stroma owns:

- canonical corpus records
- content hashing and corpus fingerprints
- Markdown chunking
- embedding abstractions
- SQLite plus `sqlite-vec` storage
- semantic section retrieval

Stroma does not own:

- specification semantics
- governance graphs
- source adapters such as GitHub, filesystem discovery, or MCP
- policy analysis such as overlap, compliance, drift, or impact
- product-facing workflows, CLIs, or review/reporting layers

## Public Packages

- `corpus`
  - `Record` is the neutral artifact model.
  - `Record.Normalized` fills safe defaults and computes a content hash when absent.
  - `Fingerprint` summarizes the full indexed corpus deterministically.

- `chunk`
  - `Markdown` splits Markdown into heading-aware sections.
  - Nested headings are flattened into a path-like heading string.

- `embed`
  - `Embedder` is the only embedding contract the index depends on.
  - `Fixture` provides deterministic embeddings for tests and offline use.

- `store`
  - validates that SQLite and `sqlite-vec` are usable
  - opens read-only and read-write handles consistently
  - translates embeddings to and from `sqlite-vec` blob format

- `index`
  - rebuilds an index atomically
  - reuses unchanged embeddings from a compatible previous snapshot when available
  - persists index metadata such as schema version and embedder fingerprint
  - exposes opened snapshots, stats, section/record readers, and semantic search

## On-Disk Schema

The current schema is intentionally small:

- `records`
  - canonical normalized records
- `chunks`
  - retrievable sections keyed to a record
- `chunks_vec`
  - `vec0` virtual table storing chunk embeddings
- `metadata`
  - schema version, embedder fingerprint, embedder dimension, content fingerprint, creation time

The schema is product-neutral. There are no spec/document distinctions, no governance edges, and no transport-specific tables. Higher-order products should treat the snapshot as a Stroma-owned artifact and go through the library API instead of joining these tables directly.

## Extraction Principle

Stroma should stay boring.

If a feature introduces domain language such as spec status, applies-to rules, drift classes, review flows, or product transport contracts, it belongs in the consuming product, not in Stroma.
