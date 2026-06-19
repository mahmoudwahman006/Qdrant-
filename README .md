# Qdrant Vector Store Examples

Two standalone scripts demonstrating Qdrant vector database usage, from basic vector search to a full retrieval pipeline with Cohere embeddings.

## Files

| File | Description |
|---|---|
| `Qdrant_Simple_Form.py` | Intro to Qdrant: collections, distance metrics, manual vectors, and similarity search. No embedding model required. |
| `Qdrant_more.py` | End-to-end retrieval example: embeds text with Cohere, stores vectors + metadata in Qdrant, and queries with metadata filtering. |

## Requirements

- Python 3.8/3.9
- A Cohere API key (only needed for `Qdrant_more.py`)

### Install dependencies

```bash
pip install qdrant-client cohere python-dotenv
```

> Note: `Qdrant_more.py` was built against **Cohere SDK v5.0.0**. The `cohere.Client(...)` and `.embed()` call signatures differ from the current Cohere documentation, so don't expect newer SDK examples to match 1:1.

## Setup

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Add your Cohere API key to `.env`:
   ```
   COHERE_API_KEY="your-key-here"
   ```
3. `.env` is already excluded from version control via `.gitignore`.

## Running

Both scripts run in **local (embedded) mode** — Qdrant data is stored on disk at `./qdrant_data` rather than connecting to a Docker/server instance. Each script wipes and recreates this directory on every run, so don't run both scripts at the same time (only one process can access local-mode storage at once).

```bash
python Qdrant_Simple_Form.py
python Qdrant_more.py
```

---

### `Qdrant_Simple_Form.py`

- Creates a collection (`first_collection`) with 2-dimensional vectors and **Euclidean** distance.
- Inserts 5 points representing cities, each with a hand-written 2D vector and a `city`/`country` payload.
- Queries with a manual vector `[1.5, 1.5]` and returns the 3 nearest points.
- Includes inline notes comparing **Euclidean**, **Cosine**, and **Dot Product** distance metrics and when to use each.

### `Qdrant_more.py`

- Creates a collection (`my_collection`) with 384-dimensional vectors (matching Cohere's `embed-english-light-v3.0`) and **Cosine** distance.
- Embeds two lists of programming languages — interpreted and compiled — using Cohere's `embed()` API with `input_type="classification"`.
- Upserts each group into Qdrant with a `language` and `type` payload, using UUIDs as point IDs.
- Embeds a test query (`"c#"`) and searches with a **metadata filter** (`type == "interpreted"`) to demonstrate combining vector similarity with hard filtering.

## Key Concepts Covered

- **Local vs. server mode** — `QdrantClient(path=...)` for embedded/local storage vs. `QdrantClient(host=..., port=...)` for a running Qdrant server.
- **Distance metrics** — Euclidean, Cosine, Dot Product, and their typical use cases.
- **PointStruct** — combining vectors with payload metadata, using integer or UUID IDs.
- **Cohere `input_type`** — `search_document`, `search_query`, `classification`, `clustering`, `image`, and why it matters for v3+ embedding models.
- **Metadata filtering** — using `Filter` / `FieldCondition` / `MatchValue` to combine semantic search with exact-match filters.

## Notes / Gotchas

- Cohere embedding dimensions vary by model: light variants are 384-dim, standard v3.0 models are 1024-dim. The Qdrant collection's vector `size` **must match** the embedding model's output dimension exactly, or upserts/queries will fail.
- Both scripts call `shutil.rmtree('./qdrant_data', ...)` at startup — re-running will destroy any previously stored data.
