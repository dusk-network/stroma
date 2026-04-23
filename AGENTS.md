# AGENTS.md

> Canonical AI policy for this repo.

## Project

Stroma is a neutral corpus and index substrate. It owns content normalization, chunking, embedding and chat completion interfaces, and SQLite-backed retrieval primitives that other products can build on. It does not own governance semantics, spec workflows, prompt templates, JSON-shape enforcement, or product-specific adapters.

## Coding Standards

- Read [README.md](README.md) and [ARCHITECTURE.md](ARCHITECTURE.md) before deep implementation work.
- Keep the public API generic. Do not leak downstream or domain-specific product semantics into the kernel.
- Prefer deterministic behavior, explicit metadata, and small reversible changes.
- Keep the repo library-first. Do not introduce a CLI or server unless explicitly requested.

## Architecture Rules

- Stroma owns only substrate concerns: records, chunking, embeddings, chat completion HTTP transport, index storage, and retrieval.
- Product layers such as governance analysis, issue workflows, CI integrations, and vendor transports stay out of scope.
- Persist machine-readable state. The SQLite index is a contract surface, not an incidental cache.
- Public packages should be usable independently; avoid hidden globals and repo-local assumptions.

## Workflow

- When planning or implementing, name the files you expect to touch and why.
- Run `git status` and `git diff --minimal` before any commit.
- Stage explicit paths. Never use `git add .`.
- Show diffs and wait for approval before committing or pushing.

## Testing

- Run the smallest check that proves the change.
- Prefer `go test ./...` for repo-wide validation.
- If you skip validation, state the reason explicitly.

## Safety

- No secrets in tracked files, docs, or fixtures.
- Do not reintroduce product-specific workflow or transport layers unless explicitly requested.
- Do not overwrite user changes or reset the worktree without explicit approval.
