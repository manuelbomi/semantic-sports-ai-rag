"""
embeddings.py
─────────────
Embedding pipeline for NBA fan data + FAISS vector search.

Concepts:
  Embedding      → text (or structured data) converted to a dense float vector
  Similarity     → closeness in vector space ≈ semantic relatedness
  Normalization  → L2-normalize vectors so inner product = cosine similarity
  FAISS index    → fast approximate nearest-neighbour (ANN) search at scale
  Hybrid search  → combine vector similarity with graph structure (end of file)

Everything here runs on CPU. For production you'd use a larger model
(e.g. text-embedding-3-large from OpenAI, or E5-large) and a managed
vector DB like Pinecone or pgvector.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


# ─────────────────────────────────────────────────────────────────────────────
# 1. THE CORPUS
#    In a real system this comes from your graph (describe_node) + document store.
#    Here it's hand-written to make the examples readable.
# ─────────────────────────────────────────────────────────────────────────────

DOCUMENTS = [
    # --- Player facts ---
    "LeBron James is a forward for the Los Angeles Lakers wearing jersey #23. "
    "He is considered one of the greatest players in NBA history.",

    "Stephen Curry plays guard for the Golden State Warriors, number 30. "
    "He is the all-time leader in three-point field goals made.",

    "Jayson Tatum is a forward for the Boston Celtics, jersey number 0. "
    "He was the 2024 NBA Finals MVP.",

    "Giannis Antetokounmpo, nicknamed the Greek Freak, plays forward for the "
    "Milwaukee Bucks wearing number 34.",

    "Anthony Davis is a center for the Los Angeles Lakers, number 3. "
    "He is known for his elite defensive anchor presence.",

    # --- Team facts ---
    "The Los Angeles Lakers play their home games at Crypto.com Arena. "
    "They have 17 NBA championships.",

    "The Boston Celtics play at TD Garden and have the most NBA titles of any franchise.",

    "The Golden State Warriors play at Chase Center in San Francisco. "
    "They won multiple championships in the 2010s dynasty era.",

    "The Milwaukee Bucks play at Fiserv Forum. "
    "Giannis won the 2021 championship and Finals MVP with the Bucks.",

    # --- Fan profiles ---
    "Marcus is a Lakers fan based in Los Angeles. "
    "LeBron James is his favorite player. He attended the Christmas Day game.",

    "Aisha follows the Boston Celtics and lives in Boston. "
    "Jayson Tatum is her favourite player.",

    "Jordan is a Golden State Warriors fan from San Francisco. "
    "Stephen Curry is his favourite player. He attended the NBA Finals and All-Star game.",

    "Wei is also a Lakers fan living in Los Angeles.",

    "Priya supports the Milwaukee Bucks and lives in Milwaukee.",

    # --- Policies / unstructured docs ---
    "Loyalty program: fans who attend 10 or more games per season qualify for "
    "Platinum tier and receive priority access to playoff tickets plus 20% off merchandise.",

    "Season ticket members get complimentary parking at Crypto.com Arena for all "
    "regular season home games.",

    "The All-Star Game 2024 was hosted at Chase Center in San Francisco. "
    "Tickets sold out within 30 minutes of going on sale.",

    "Christmas Day NBA games are a league tradition. The 2024 Lakers vs Celtics game "
    "was held at Crypto.com Arena and attracted peak viewership.",

    "Fan behaviour policy: accounts flagged for scalping will have purchase limits "
    "applied for the following 12 months.",
]

# Keep an index → document map so we can trace which doc returned
DOC_IDS = list(range(len(DOCUMENTS)))


# ─────────────────────────────────────────────────────────────────────────────
# 2. BUILD THE FAISS INDEX
# ─────────────────────────────────────────────────────────────────────────────

print("Loading embedding model (all-MiniLM-L6-v2)...")
# 384-dimensional, fast on CPU, decent quality for retrieval
MODEL = SentenceTransformer("all-MiniLM-L6-v2")

print(f"Embedding {len(DOCUMENTS)} documents...")
# normalize_embeddings=True → each vector has L2 norm = 1.0
# after this, dot product between any two vectors equals cosine similarity
embeddings = MODEL.encode(DOCUMENTS, convert_to_numpy=True, normalize_embeddings=True)
print(f"Embedding matrix shape: {embeddings.shape}  (docs × dims)")

DIM = embeddings.shape[1]
# IndexFlatIP = brute-force inner product (= cosine sim on normalized vectors)
# For >100k docs switch to IndexIVFFlat or HNSW for approximate search
INDEX = faiss.IndexFlatIP(DIM)
INDEX.add(embeddings.astype("float32"))
print(f"FAISS index built: {INDEX.ntotal} vectors\n")


# ─────────────────────────────────────────────────────────────────────────────
# 3. SEARCH FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def semantic_search(query: str, k: int = 4) -> list[dict]:
    """
    Embed the query and return the top-k most similar documents.

    Returns a list of dicts: {rank, score, doc_id, text}
    Score is cosine similarity (1.0 = identical, 0.0 = orthogonal, -1.0 = opposite).
    """
    q_vec = MODEL.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = INDEX.search(q_vec.astype("float32"), k)
    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        results.append({
            "rank":   rank,
            "score":  round(float(score), 4),
            "doc_id": int(idx),
            "text":   DOCUMENTS[idx],
        })
    return results


def print_results(query: str, results: list[dict]):
    print(f"\nQuery: {query!r}")
    print("─" * 70)
    for r in results:
        print(f"  [{r['rank']}] score={r['score']:.4f}  id={r['doc_id']}")
        print(f"       {r['text'][:100]}...")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 4. COSINE SIMILARITY — worked example
#    Shows WHY normalized vectors make this simple.
# ─────────────────────────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Manual cosine similarity for illustration. On normalized vectors: just dot(a,b)."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def similarity_demo():
    sentences = [
        "LeBron James wins NBA championship",    # anchor
        "Lakers win the NBA Finals",             # very similar meaning
        "basketball title awarded to LA team",   # similar, different words
        "quarterly financial earnings report",   # completely unrelated
    ]
    vecs = MODEL.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
    anchor = vecs[0]
    print("Cosine similarity to 'LeBron James wins NBA championship':")
    for sent, vec in zip(sentences[1:], vecs[1:]):
        sim = cosine_similarity(anchor, vec)
        bar = "█" * int(sim * 30)
        print(f"  {sim:.3f}  {bar}  {sent!r}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. HYBRID SEARCH — combine vector score with graph structure
#    Pure vector search can miss graph-structured facts.
#    Boosting documents that mention graph-matched entities helps.
# ─────────────────────────────────────────────────────────────────────────────

ENTITY_NAMES = {
    "lebron": ["LeBron James", "LeBron", "LBJ"],
    "curry":  ["Stephen Curry", "Steph", "Curry"],
    "tatum":  ["Jayson Tatum", "JT", "Tatum"],
    "lakers": ["Lakers", "Los Angeles Lakers", "LAL"],
    "marcus": ["Marcus"],
}

def hybrid_search(query: str, k: int = 4, entity_boost: float = 0.15) -> list[dict]:
    """
    Hybrid: vector score + bonus for documents that mention query entities.

    In production this is where the knowledge graph feeds in: matched entity IDs
    are expanded to their canonical names, then those names are used to boost
    documents that contain the entity's information.
    """
    base = semantic_search(query, k=k * 2)   # over-retrieve, then re-rank

    # naive entity mention check
    query_lower = query.lower()
    matched_entities = [
        eid for eid, aliases in ENTITY_NAMES.items()
        if any(a.lower() in query_lower for a in aliases)
    ]

    for result in base:
        doc_lower = result["text"].lower()
        boost = sum(
            entity_boost
            for eid in matched_entities
            for alias in ENTITY_NAMES[eid]
            if alias.lower() in doc_lower
        )
        result["hybrid_score"] = result["score"] + boost

    base.sort(key=lambda r: r["hybrid_score"], reverse=True)
    return base[:k]


# ─────────────────────────────────────────────────────────────────────────────
# 6. RUN DEMOS
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("SIMILARITY DEMO")
    print("=" * 70)
    similarity_demo()

    print("\n" + "=" * 70)
    print("SEMANTIC SEARCH EXAMPLES")
    print("=" * 70)

    queries = [
        "Who is the best three-point shooter in basketball?",
        "Where do the Lakers play their home games?",
        "What rewards do loyal fans get?",
        "Which fans attended events in San Francisco?",
        "Tell me about the Greek Freak",
    ]
    for q in queries:
        results = semantic_search(q, k=3)
        print_results(q, results)

    print("=" * 70)
    print("HYBRID SEARCH (vector + entity boost)")
    print("=" * 70)
    hybrid_q = "What events did Marcus attend?"
    print(f"\nQuery: {hybrid_q!r}")
    print("─" * 70)
    for r in hybrid_search(hybrid_q, k=3):
        print(f"  score={r['score']:.4f}  hybrid={r['hybrid_score']:.4f}")
        print(f"  {r['text'][:110]}...")
    print()
