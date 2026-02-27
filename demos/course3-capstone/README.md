# C3 MLOps Capstone: Serve Qwen2.5-Coder-1.5B

One-command demo: pull, inspect, serve, and query a 1.5B code model.

## Prerequisites
- `apr` CLI installed (`cargo install apr-cli` or build from ~/src/aprender)

## Run
```bash
cargo run --release
```

## What it does
1. Downloads Qwen2.5-Coder-1.5B-Instruct (1.5GB, cached after first run)
2. Inspects model architecture
3. Starts OpenAI-compatible API server on port 18090
4. Sends a code generation request
5. Prints the AI-generated Rust code
