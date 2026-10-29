"""
evaluation.py
─────────────
Evaluation frameworks for a RAG pipeline.

Running AI systems without measuring them is risky — you don't know if retrieval
is actually finding the right documents, and you don't know if the LLM is staying
faithful to the context or hallucinating.

Two things to evaluate:

  1. RETRIEVAL QUALITY   — did the right documents come back?
  2. ANSWER FAITHFULNESS — is every claim in the answer grounded in the context?

Metrics:
  Precision@k  → fraction of top-k results that are relevant
  Recall@k     → fraction of all relevant docs found in top-k
  MRR          → Mean Reciprocal Rank (how highly ranked was the first hit?)
  NDCG@k       → accounts for ranking order (higher is better than lower)
  Faithfulness → does the answer contain only claims supported by the context?
"""

import math


# ─────────────────────────────────────────────────────────────────────────────
# 1. RETRIEVAL METRICS
#    These need a "relevance label" — which documents are truly relevant for
#    a given query. In practice, you build an evaluation set with human labels.
# ─────────────────────────────────────────────────────────────────────────────

def precision_at_k(retrieved: list, relevant: set, k: int) -> float:
    """
    Of the top-k retrieved documents, what fraction are relevant?

    Example:
      retrieved = [doc_A, doc_B, doc_C, doc_D]   (ranked)
      relevant  = {doc_A, doc_C, doc_E}
      precision@2 = 1/2 = 0.50  (only doc_A is relevant in top 2)
      precision@4 = 2/4 = 0.50  (doc_A and doc_C in top 4)
    """
    top_k = retrieved[:k]
    hits = sum(1 for doc in top_k if doc in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved: list, relevant: set, k: int) -> float:
    """
    Of all relevant documents, what fraction appeared in the top-k results?

    High recall means you're not missing important information.
    """
    if not relevant:
        return 1.0
    top_k = set(retrieved[:k])
    hits = len(top_k & relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: list, relevant: set) -> float:
    """
    Reciprocal Rank: 1 / rank_of_first_relevant_result.

    1.0  → first result is relevant
    0.5  → second result is first relevant
    0.33 → third result is first relevant
    0.0  → no relevant result found
    """
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list, relevant: set, k: int) -> float:
    """
    Normalised Discounted Cumulative Gain.

    Rewards relevant results that appear higher in the ranking.
    A relevant doc at rank 1 is worth more than at rank 5.

    Simplified binary relevance version (0 or 1).
    """
    def dcg(results: list, rel: set, n: int) -> float:
        total = 0.0
        for i, doc in enumerate(results[:n], start=1):
            total += (1.0 if doc in rel else 0.0) / math.log2(i + 1)
        return total

    actual = dcg(retrieved, relevant, k)
    # ideal: all relevant docs come first
    ideal_order = list(relevant) + [d for d in retrieved if d not in relevant]
    ideal = dcg(ideal_order, relevant, k)
    return actual / ideal if ideal > 0 else 0.0


def evaluate_retrieval(
    queries: list[dict],   # each: {"query": str, "retrieved": [ids], "relevant": {ids}}
    k: int = 5,
) -> dict:
    """
    Run all retrieval metrics over a set of labelled queries.
    Returns mean scores across the set.
    """
    p_scores, r_scores, rr_scores, ndcg_scores = [], [], [], []
    for item in queries:
        ret = item["retrieved"]
        rel = item["relevant"]
        p_scores.append(precision_at_k(ret, rel, k))
        r_scores.append(recall_at_k(ret, rel, k))
        rr_scores.append(reciprocal_rank(ret, rel))
        ndcg_scores.append(ndcg_at_k(ret, rel, k))

    n = len(queries)
    return {
        f"precision@{k}": round(sum(p_scores) / n, 4),
        f"recall@{k}":    round(sum(r_scores) / n, 4),
        "MRR":             round(sum(rr_scores) / n, 4),
        f"NDCG@{k}":       round(sum(ndcg_scores) / n, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. ANSWER FAITHFULNESS
#    Checks whether every sentence in the answer is supported by the context.
#    This is a simplified keyword-overlap version; production uses an LLM judge
#    (ask a second LLM: "is claim X supported by the context below?")
# ─────────────────────────────────────────────────────────────────────────────

def faithfulness_score(answer: str, context: str) -> dict:
    """
    Simple faithfulness check: for each sentence in the answer, compute
    what fraction of its content words appear in the context.

    Scores:
      1.0  → every word in the answer sentence appears in the context
      0.5  → half the content words are covered
      0.0  → nothing in this sentence is grounded

    Production alternative: use an LLM-as-judge prompt:
      "Does the following context support this claim? Answer Yes/No."
    """
    import re

    def content_words(text: str) -> set[str]:
        # lowercase, strip punctuation, remove stop words
        stop = {"the", "a", "an", "is", "are", "was", "were", "in", "at",
                "of", "and", "or", "to", "for", "on", "he", "she", "they",
                "it", "his", "her", "their", "its", "be", "by", "with"}
        words = re.findall(r'\b[a-z]+\b', text.lower())
        return {w for w in words if w not in stop and len(w) > 2}

    context_words = content_words(context)
    sentences = [s.strip() for s in re.split(r'[.!?]', answer) if s.strip()]

    scores = []
    for sentence in sentences:
        s_words = content_words(sentence)
        if not s_words:
            continue
        overlap = len(s_words & context_words) / len(s_words)
        scores.append({"sentence": sentence, "score": round(overlap, 3)})

    overall = sum(s["score"] for s in scores) / len(scores) if scores else 0.0
    return {
        "overall_faithfulness": round(overall, 4),
        "per_sentence": scores,
        "verdict": "FAITHFUL" if overall >= 0.75 else
                   "PARTIALLY FAITHFUL" if overall >= 0.5 else "UNFAITHFUL",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("RETRIEVAL METRICS DEMO")
    print("=" * 60)

    # Simulated retrieval results — normally these come from running your
    # actual pipeline over a labelled evaluation set.
    eval_set = [
        {
            "query": "Where do the Lakers play?",
            "retrieved": ["doc_lakers_venue", "doc_lakers_city", "doc_marcus_fan",
                          "doc_curry_stats", "doc_celtics_venue"],
            "relevant": {"doc_lakers_venue", "doc_lakers_city"},
        },
        {
            "query": "What loyalty perks do Platinum members get?",
            "retrieved": ["doc_loyalty_policy", "doc_marcus_fan", "doc_parking_policy",
                          "doc_jordan_fan", "doc_lakers_venue"],
            "relevant": {"doc_loyalty_policy"},
        },
        {
            "query": "Which fans attended the All-Star game?",
            "retrieved": ["doc_allstar_event", "doc_jordan_fan", "doc_chase_center",
                          "doc_sf_city", "doc_warriors_team"],
            "relevant": {"doc_jordan_fan", "doc_allstar_event"},
        },
        {
            "query": "Who is the Golden State Warriors point guard?",
            "retrieved": ["doc_curry_player", "doc_warriors_team", "doc_chase_center",
                          "doc_jordan_fan", "doc_lakers_venue"],
            "relevant": {"doc_curry_player"},
        },
    ]

    results = evaluate_retrieval(eval_set, k=3)
    print(f"\n  Evaluation over {len(eval_set)} queries (k=3):\n")
    for metric, score in results.items():
        bar = "█" * int(score * 20)
        print(f"    {metric:15s}  {score:.4f}  {bar}")

    print("\n" + "=" * 60)
    print("FAITHFULNESS CHECK DEMO")
    print("=" * 60)

    test_cases = [
        {
            "label": "Grounded answer",
            "context": (
                "Marcus is a Lakers fan who lives in Los Angeles. "
                "LeBron James plays for the Los Angeles Lakers. "
                "The Lakers play at Crypto.com Arena."
            ),
            "answer": (
                "Marcus lives in Los Angeles and follows the Lakers. "
                "His favorite player LeBron James plays for the Lakers at Crypto.com Arena."
            ),
        },
        {
            "label": "Partially hallucinated",
            "context": (
                "Marcus is a Lakers fan who lives in Los Angeles."
            ),
            "answer": (
                "Marcus lives in Los Angeles. "
                "He has been a Lakers fan for over 15 years and holds courtside season tickets."
            ),
        },
        {
            "label": "Fully hallucinated",
            "context": (
                "Stephen Curry plays for the Golden State Warriors."
            ),
            "answer": (
                "LeBron James recently signed a new five-year deal with the Knicks "
                "and will debut at Madison Square Garden next season."
            ),
        },
    ]

    for case in test_cases:
        print(f"\n  [{case['label']}]")
        result = faithfulness_score(case["answer"], case["context"])
        print(f"  Verdict: {result['verdict']}  "
              f"(overall score: {result['overall_faithfulness']:.4f})")
        for s in result["per_sentence"]:
            bar = "█" * int(s["score"] * 20)
            print(f"    {s['score']:.3f}  {bar}  \"{s['sentence'][:70]}\"")
