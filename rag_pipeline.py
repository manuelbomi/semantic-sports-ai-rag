"""
rag_pipeline.py
───────────────
A Retrieval-Augmented Generation (RAG) pipeline grounded in an NBA fan
knowledge graph.

The pipeline has four stages:

  1. Entity linking    — find graph nodes mentioned in the question
  2. Graph expansion   — BFS-expand those nodes to pull in related facts
  3. Vector search     — retrieve semantically similar documents
  4. Context assembly  — combine graph facts + retrieved docs into an LLM prompt

The LLM call is mocked so the file runs without an API key.
Swap mock_llm() for any real model (see comments inside it).

Architecture:

  Query
    │
    ├─► graph_lookup()     ─► matched entities
    │       │
    │       └─► graph_context()  ─► fact sentences (structured)
    │
    └─► vector_search()    ─► top-k documents (unstructured)

           structured + unstructured
                      │
                      ▼
               assemble_context()
                      │
                      ▼
               LLM prompt → answer
                      │
                      ▼
               citations + traceability
"""

import networkx as nx
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


# ─────────────────────────────────────────────────────────────────────────────
# 1. KNOWLEDGE GRAPH  (same domain as the first repo)
# ─────────────────────────────────────────────────────────────────────────────

G = nx.MultiDiGraph()

_nodes = {
    "marcus":  {"type": "Fan",    "name": "Marcus"},
    "aisha":   {"type": "Fan",    "name": "Aisha"},
    "jordan":  {"type": "Fan",    "name": "Jordan"},
    "wei":     {"type": "Fan",    "name": "Wei"},
    "lebron":  {"type": "Player", "name": "LeBron James",  "position": "Forward"},
    "curry":   {"type": "Player", "name": "Stephen Curry", "position": "Guard"},
    "tatum":   {"type": "Player", "name": "Jayson Tatum",  "position": "Forward"},
    "giannis": {"type": "Player", "name": "Giannis Antetokounmpo", "position": "Forward"},
    "lakers":   {"type": "Team",  "name": "Los Angeles Lakers"},
    "celtics":  {"type": "Team",  "name": "Boston Celtics"},
    "warriors": {"type": "Team",  "name": "Golden State Warriors"},
    "bucks":    {"type": "Team",  "name": "Milwaukee Bucks"},
    "la":       {"type": "City",  "name": "Los Angeles"},
    "boston":   {"type": "City",  "name": "Boston"},
    "sf":       {"type": "City",  "name": "San Francisco"},
    "crypto":   {"type": "Venue", "name": "Crypto.com Arena"},
    "td":       {"type": "Venue", "name": "TD Garden"},
    "chase":    {"type": "Venue", "name": "Chase Center"},
    "fiserv":   {"type": "Venue", "name": "Fiserv Forum"},
    "finals_g2":    {"type": "Event", "name": "NBA Finals Game 2, 2024"},
    "xmas_game":    {"type": "Event", "name": "Lakers vs Celtics, Christmas 2024"},
    "allstar_2024": {"type": "Event", "name": "All-Star Game 2024"},
}
for nid, attrs in _nodes.items():
    G.add_node(nid, **attrs)

_edges = [
    ("marcus", "la",        "LIVES_IN"),
    ("aisha",  "boston",    "LIVES_IN"),
    ("jordan", "sf",        "LIVES_IN"),
    ("wei",    "la",        "LIVES_IN"),
    ("marcus", "lakers",    "FOLLOWS"),
    ("aisha",  "celtics",   "FOLLOWS"),
    ("jordan", "warriors",  "FOLLOWS"),
    ("wei",    "lakers",    "FOLLOWS"),
    ("marcus", "lebron",    "FAVORITE_PLAYER"),
    ("aisha",  "tatum",     "FAVORITE_PLAYER"),
    ("jordan", "curry",     "FAVORITE_PLAYER"),
    ("lebron",  "lakers",   "PLAYS_FOR"),
    ("curry",   "warriors", "PLAYS_FOR"),
    ("tatum",   "celtics",  "PLAYS_FOR"),
    ("giannis", "bucks",    "PLAYS_FOR"),
    ("lakers",   "la",      "PLAYS_IN"),
    ("celtics",  "boston",  "PLAYS_IN"),
    ("warriors", "sf",      "PLAYS_IN"),
    ("lakers",   "crypto",  "HOME_COURT"),
    ("celtics",  "td",      "HOME_COURT"),
    ("warriors", "chase",   "HOME_COURT"),
    ("bucks",    "fiserv",  "HOME_COURT"),
    ("finals_g2",    "chase",  "HOSTED_AT"),
    ("xmas_game",    "crypto", "HOSTED_AT"),
    ("allstar_2024", "chase",  "HOSTED_AT"),
    ("marcus", "xmas_game",    "ATTENDED"),
    ("aisha",  "xmas_game",    "ATTENDED"),
    ("jordan", "finals_g2",    "ATTENDED"),
    ("jordan", "allstar_2024", "ATTENDED"),
]
for u, v, rel in _edges:
    G.add_edge(u, v, rel=rel)


def describe_node(node_id: str) -> str:
    """Convert a graph node + its edges into a natural-language fact sentence."""
    attrs = G.nodes[node_id]
    facts = []
    for _, tgt, data in G.out_edges(node_id, data=True):
        tgt_name = G.nodes[tgt]["name"]
        rel = data["rel"].lower().replace("_", " ")
        facts.append(f"{attrs['name']} {rel} {tgt_name}.")
    base = f"{attrs['name']} is a {attrs['type']}."
    return base + (" " + " ".join(facts) if facts else "")


# ─────────────────────────────────────────────────────────────────────────────
# 2. VECTOR INDEX  (structured graph nodes + unstructured policy docs)
# ─────────────────────────────────────────────────────────────────────────────

print("Loading embedding model...")
EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")

corpus: list[str] = []
corpus_ids: list[tuple] = []   # (source_type, id) for traceability

for nid in G.nodes():
    corpus.append(describe_node(nid))
    corpus_ids.append(("graph_node", nid))

# Policy and account documents — the "unstructured" half of the knowledge base
_docs = [
    ("policy", "loyalty",
     "Loyalty tier: fans who attend 10+ events per season reach Platinum status "
     "and receive 20% off merchandise plus priority playoff ticket access."),
    ("policy", "parking",
     "Season ticket members get complimentary parking at Crypto.com Arena "
     "for all regular season home games."),
    ("policy", "scalping",
     "Accounts flagged for ticket scalping face purchase limits for 12 months."),
    ("account_note", "marcus_note",
     "Account note — Marcus: called support about parking passes for Crypto.com Arena. "
     "Mentioned he's a lifelong Lakers fan and tries to attend most home games."),
    ("account_note", "jordan_note",
     "Account note — Jordan: purchased Finals tickets through verified resale. "
     "Frequent attender — 14 events last season, qualifies for Platinum tier."),
    ("account_note", "aisha_note",
     "Account note — Aisha: opted into Celtics newsletter, not SMS. "
     "Interested in family section seating."),
]
for kind, doc_id, text in _docs:
    corpus.append(text)
    corpus_ids.append((kind, doc_id))

print(f"Embedding {len(corpus)} documents...")
_embeddings = EMBEDDER.encode(corpus, convert_to_numpy=True, normalize_embeddings=True)
_dim = _embeddings.shape[1]
VECTOR_INDEX = faiss.IndexFlatIP(_dim)
VECTOR_INDEX.add(_embeddings.astype("float32"))
print(f"Index ready: {VECTOR_INDEX.ntotal} vectors\n")


# ─────────────────────────────────────────────────────────────────────────────
# 3. RETRIEVAL FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def graph_lookup(query: str) -> list[str]:
    """
    Toy entity linker: find nodes whose canonical name appears in the query.

    Production: replace with a proper NER model + entity resolution (see entity_resolution.py).
    """
    q = query.lower()
    return [nid for nid, attrs in G.nodes(data=True) if attrs["name"].lower() in q]


def graph_context(node_ids: list[str], hops: int = 1) -> tuple[set, list[str]]:
    """
    BFS-expand matched nodes by `hops` and return fact sentences for all reached nodes.

    hops=1 grabs immediate neighbours; hops=2 grabs neighbours-of-neighbours.
    More hops = richer context but also more noise.
    """
    visited = set(node_ids)
    frontier = set(node_ids)
    for _ in range(hops):
        next_f: set[str] = set()
        for n in frontier:
            for _, tgt, _ in G.out_edges(n, data=True):
                if tgt not in visited:
                    next_f.add(tgt)
            for src, _, _ in G.in_edges(n, data=True):
                if src not in visited:
                    next_f.add(src)
        visited |= next_f
        frontier = next_f

    facts = [describe_node(n) for n in visited]
    return visited, facts


def vector_search(query: str, k: int = 4) -> list[dict]:
    """Top-k documents by cosine similarity to the query embedding."""
    q_vec = EMBEDDER.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idxs = VECTOR_INDEX.search(q_vec.astype("float32"), k)
    return [
        {"score": float(s), "text": corpus[i], "source": corpus_ids[i]}
        for s, i in zip(scores[0], idxs[0])
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 4. MOCK LLM
#    Prints the prompt and returns a placeholder answer.
#    Swap for a real model — see comments below.
# ─────────────────────────────────────────────────────────────────────────────

def mock_llm(prompt: str) -> str:
    """
    To use a real model, replace this function body:

    # Option A — OpenAI
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content

    # Option B — Local open-weight model via llama-cpp-python (on-prem GPU)
        from llama_cpp import Llama
        llm = Llama(model_path="./models/mistral-7b-instruct.gguf", n_gpu_layers=-1)
        return llm(prompt, max_tokens=512)["choices"][0]["text"]

    # Option C — Anthropic Claude
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )
        return msg.content[0].text
    """
    print("\n── PROMPT SENT TO LLM ──────────────────────────────────────────")
    print(prompt[:1400] + ("\n... (truncated)" if len(prompt) > 1400 else ""))
    print("────────────────────────────────────────────────────────────────\n")
    return "[mock] A real LLM would answer here, grounded in the context above."


# ─────────────────────────────────────────────────────────────────────────────
# 5. THE RAG PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def ask(question: str, hops: int = 1, k: int = 4):
    print("\n" + "=" * 65)
    print(f"Q: {question}")
    print("=" * 65)

    # Stage 1 + 2: graph retrieval
    entities = graph_lookup(question)
    expanded_nodes, graph_facts = graph_context(entities, hops=hops)

    # Stage 3: vector retrieval
    vec_results = vector_search(question, k=k)

    # Stage 4: context assembly — merge graph + vector results
    context_parts = []
    if graph_facts:
        context_parts.append("GRAPH FACTS:\n" + "\n".join(f"• {f}" for f in graph_facts))
    if vec_results:
        context_parts.append(
            "RETRIEVED DOCUMENTS:\n"
            + "\n".join(
                f"• [{r['source'][0]}:{r['source'][1]}] {r['text']}"
                for r in vec_results
            )
        )
    context = "\n\n".join(context_parts) or "No relevant context found."

    prompt = f"""You are an NBA fan data assistant.
Answer the question using ONLY the provided context.
If the answer is not in the context, say so — do not guess.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    answer = mock_llm(prompt)

    # Traceability — what grounded this answer?
    print("Graph entities matched:", [G.nodes[e]["name"] for e in entities] or "none")
    print("Graph expansion size:  ", len(expanded_nodes), "nodes")
    print("Top vector results:")
    for r in vec_results[:3]:
        print(f"  [{r['score']:.3f}] ({r['source'][0]}:{r['source'][1]}) "
              f"{r['text'][:80]}...")
    print(f"\nAnswer: {answer}")
    return answer


# ─────────────────────────────────────────────────────────────────────────────
# 6. DEMO QUERIES
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ask("What team does Marcus follow and where does he live?")
    ask("Does Jordan qualify for the loyalty rewards program?")
    ask("What is Marcus's favourite player and where do they play?")
    ask("Which fans attended events at Chase Center?")
    ask("Tell me about Aisha's notification preferences.")
