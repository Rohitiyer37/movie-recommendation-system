"""llm_rerank.py - OPTIONAL extension: let an LLM re-order your top-20 list.

Setup:  pip install anthropic
        set ANTHROPIC_API_KEY (get a key at console.anthropic.com)
Run:    python llm_rerank.py        (tests on 20 users, prints NDCG before/after)

The LLM only re-orders candidates we give it, and every ID it returns is
checked against the candidate list, so it can never invent a movie.
"""
import json
import re
import time

import anthropic
import numpy as np

from recsys import Models, get_embeddings, load_data, top_k

MODEL = "claude-haiku-5-5"   # cheap and fast; check docs.claude.com for model names
client = anthropic.Anthropic()


def llm_rerank(history_titles, candidates):
    """candidates: list of (movie_index, title). Returns reordered list of indices."""
    cand_text = "\n".join(f"{i}: {t}" for i, t in candidates)
    prompt = (
        "A user liked these movies:\n- " + "\n- ".join(history_titles) +
        "\n\nCandidate movies (id: title):\n" + cand_text +
        "\n\nRe-rank ALL candidates from most to least likely to be enjoyed by "
        "this user. Reply with ONLY a JSON list of the ids, nothing else.")
    resp = client.messages.create(model=MODEL, max_tokens=400,
                                  messages=[{"role": "user", "content": prompt}])
    valid = [i for i, _ in candidates]
    try:
        ids = json.loads(re.search(r"\[.*\]", resp.content[0].text, re.S).group())
        ids = [int(i) for i in ids if int(i) in valid]      # drop made-up ids
    except Exception:
        ids = []
    # keep original order for anything the LLM skipped or repeated
    out, used = [], set()
    for i in ids + valid:
        if i not in used:
            out.append(i)
            used.add(i)
    return out


def ndcg(top, rel, k=10):
    hits = [1 if i in rel else 0 for i in top[:k]]
    dcg = sum(h / np.log2(r + 2) for r, h in enumerate(hits))
    idcg = sum(1 / np.log2(r + 2) for r in range(min(len(rel), k)))
    return dcg / idcg


if __name__ == "__main__":
    ratings, movies, n_users = load_data()
    emb = get_embeddings(movies)
    ratings = ratings.sort_values("timestamp").reset_index(drop=True)
    cut = int(0.8 * len(ratings))
    train, test = ratings.iloc[:cut], ratings.iloc[cut:]
    M = Models(train, n_users, len(movies), emb)

    rel = {}
    for u, m in test[test["rating"] >= 4][["uidx", "midx"]].itertuples(index=False):
        rel.setdefault(int(u), set()).add(int(m))
    users = [u for u in rel if len(M.seen(u)) > 0 and len(rel[u]) >= 3][:20]

    before, after, secs = [], [], []
    for u in users:
        cands = top_k(M.score_hybrid(u), M.seen(u), 20).tolist()
        history = movies["clean_title"].iloc[M.liked_items(u)[:15]].tolist()
        t0 = time.time()
        new_order = llm_rerank(history, [(i, movies["clean_title"].iloc[i]) for i in cands])
        secs.append(time.time() - t0)
        before.append(ndcg(cands, rel[u]))
        after.append(ndcg(new_order, rel[u]))

    print(f"NDCG@10 without LLM: {np.mean(before):.4f}")
    print(f"NDCG@10 with LLM:    {np.mean(after):.4f}")
    print(f"Average latency per user: {np.mean(secs):.1f} s")
