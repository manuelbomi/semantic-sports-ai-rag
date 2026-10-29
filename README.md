# Semantic Sports AI — Embeddings, RAG & Semantic Layer

A follow-on to `sport-ontology-knowledge-graph`. Once you have a knowledge graph
and resolved entities, the next step is making that data work with modern AI systems:
turn it into vectors, search it semantically, and plug it into an LLM pipeline that
actually stays grounded.

This repo covers:

| File | What it builds |
|------|----------------|
| `embeddings.py` | Embedding pipeline + FAISS vector search |
| `rag_pipeline.py` | Graph + vector → LLM context assembly (RAG) |
| `semantic_layer.py` | Consistent metric and feature definitions |
| `evaluation.py` | Measuring retrieval quality and LLM output relevance |

The domain is still NBA fan data, same entities as the first repo.

---

## Embeddings — the long version

An embedding is a function that maps something (a sentence, a document, a fan
profile, a player's stats) into a fixed-length vector of floats:

```
  f("LeBron wins title") → [ 0.12, -0.87, 0.43, 0.61, ..., 0.09 ]   (384 numbers)
```

The useful property: **similar inputs produce nearby vectors**. "Similar" here means
*semantically*, not just lexically — "NBA champion" and "basketball title winner"
end up close even though they share no words.

```
  Cosine distance between:
  ─────────────────────────────────────────────────────────
  "LeBron wins championship" / "NBA Finals MVP ceremony"  → 0.04  (very close)
  "Lakers home game"         / "Crypto.com Arena event"  → 0.11  (close)
  "Lakers home game"         / "quarterly earnings Q3"   → 0.73  (far)
```

This lets you build a search that finds the *right* document even when the user
phrased the query differently from how the document was written.

### Why normalize vectors?

When you L2-normalize all vectors to unit length, **cosine similarity = dot product**.
FAISS `IndexFlatIP` (inner product) then gives you cosine similarity directly,
which is faster than computing full cosine similarity from scratch.

---

## RAG — Retrieval-Augmented Generation

The problem with asking an LLM a question directly:

```
  Q: "Does Marcus qualify for our loyalty rewards?"
  LLM: [no idea — Marcus isn't in its training data]
```

RAG fixes this by retrieving relevant context *before* the LLM generates an answer:

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  RETRIEVAL PHASE                                                │
  │                                                                 │
  │  1. Graph lookup    → Marcus follows Lakers, lives in LA        │
  │  2. Graph expansion → Lakers play at Crypto.com Arena           │
  │  3. Vector search   → "fan loyalty policy" docs retrieved       │
  └─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  CONTEXT ASSEMBLY                                               │
  │                                                                 │
  │  "Fan: Marcus. City: LA. Follows: Lakers.                       │
  │   Policy: 10+ events/season = Platinum tier loyalty reward."   │
  └─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  GENERATION                                                     │
  │  LLM answers using ONLY the provided context → no hallucination │
  └─────────────────────────────────────────────────────────────────┘
```

See `rag_pipeline.py` for the full implementation.

---

## Semantic Layer

Different teams define the same concept differently:

```
  "Engaged fan" per team:
    analytics   → viewed ≥ 5 games
    revenue     → purchased ≥ 2 tickets in 90 days
    product     → opened app at least once/week
    ML team     → engagement_score model output ≥ 0.7
```

When different systems use different definitions, dashboards disagree, ML training
data is inconsistent, and LLM-generated reports are unreliable.

A semantic layer defines metrics once:

```python
ENGAGED_FAN = Metric(
    name="engaged_fan",
    definition=(
        (event_views >= 5)
        | (ticket_purchases_90d >= 2)
        | (weekly_app_opens >= 1)
    )
)
```

And every downstream consumer reads from that single definition.

See `semantic_layer.py` for a Python implementation of metric definitions,
feature computation, and a simple feature store.

---

## Evaluation

RAG without evaluation is flying blind. Two things to measure:

**Retrieval quality**: did the right documents come back?
- Precision@k: of the top-k results, how many were relevant?
- Recall@k: of all relevant docs, how many were in the top-k?
- MRR: how highly ranked was the first relevant result?

**Answer quality**: did the LLM use the context correctly?
- Faithfulness: is every claim in the answer grounded in the retrieved context?
- Answer relevance: does the answer actually address the question?

See `evaluation.py` for runnable implementations of these metrics.

---

## Setup

```bash
pip install -r requirements.txt

python embeddings.py        # build the vector index, run semantic searches
python rag_pipeline.py      # full RAG pipeline with mock LLM
python semantic_layer.py    # metric definitions + feature store
python evaluation.py        # retrieval quality metrics
```

---

## Toolchain notes

| What | Why |
|------|-----|
| `sentence-transformers` | CPU-friendly embedding model, good quality/speed tradeoff |
| `faiss-cpu` | Fast nearest-neighbour search, scales to millions of vectors |
| `networkx` | Knowledge graph traversal (used to build context) |

Production alternatives: Pinecone / Weaviate / pgvector for the vector store;
OpenAI or Cohere embeddings for higher quality; vLLM / llama.cpp for on-prem LLM serving.
