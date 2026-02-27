# Code Intelligence RAG

Quality-aware code retrieval pipeline using [trueno-rag](https://crates.io/crates/trueno-rag) for RAG infrastructure and [pmat](https://crates.io/crates/pmat) for code quality signals.

## What it does

1. **Ingest** — Reads `.rs` source files, chunks them with `RecursiveChunker` (512-char windows, 50-char overlap)
2. **Enrich** — Annotates each chunk with TDG grade, cyclomatic complexity, and fault patterns via `pmat query`
3. **Index** — Builds a hybrid vector + BM25 index using `trueno-rag` pipeline (RRF fusion, k=60)
4. **Query** — Retrieves code chunks ranked by `retrieval_score * (0.6 + 0.4 * quality_score)`
5. **Evaluate** — Computes Recall@k, MRR, NDCG against built-in ground-truth Q&A pairs

## Build and run

```bash
cargo build
cargo test

# Index a codebase
cargo run -- index --dir /path/to/rust/src

# Query with natural language
cargo run -- query --text "error handling patterns" --k 5

# Run retrieval evaluation
cargo run -- eval
```

## Architecture

Single `main.rs` with five logical sections:

| Section | Purpose |
|---------|---------|
| `chunk` | Read `.rs` files, convert to `Document` |
| `enrich` | `pmat query` for quality metadata, heuristic fallback |
| `pipeline` | `RagPipelineBuilder` + quality-fused scoring |
| `eval` | Ground-truth pairs, `RetrievalMetrics::compute` |
| `main` | CLI via `clap`: `index`, `query`, `eval` subcommands |

## Dependencies

- `trueno-rag` — Chunking, embedding, BM25, vector store, RRF fusion, retrieval metrics
- `pmat` — Code quality signals (TDG grades, complexity, fault patterns) — used at runtime via CLI
- `MockEmbedder` by default (no ONNX). Enable real embeddings: `cargo build --features embeddings`
