"""run_experiments.py - trains everything and prints the 4 tables for your report.

Run:  python run_experiments.py
Tables are also saved as CSV files in the results/ folder.
"""
import os

import numpy as np
import pandas as pd

from recsys import Models, evaluate, get_embeddings, load_data, zs

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
os.makedirs("results", exist_ok=True)

ratings, movies, n_users = load_data()
n_items = len(movies)
emb = get_embeddings(movies)
print(f"{len(ratings)} ratings, {n_users} users, {n_items} movies\n")


def make_rel(df):
    """{user: set of movies rated >= 4}"""
    rel = {}
    for u, m in df[df["rating"] >= 4][["uidx", "midx"]].itertuples(index=False):
        rel.setdefault(int(u), set()).add(int(m))
    return rel


def show(title, df, name):
    print("=" * 80)
    print(title)
    print("=" * 80)
    print(df.round(4).to_string())
    print()
    df.round(4).to_csv(os.path.join("results", name))


# ---------------------------------------------------------------------------
# TABLE 1: all models, time-based split (train on old ratings, test on new)
# ---------------------------------------------------------------------------
ratings = ratings.sort_values("timestamp").reset_index(drop=True)
cut = int(0.8 * len(ratings))
train, test = ratings.iloc[:cut], ratings.iloc[cut:]
M = Models(train, n_users, n_items, emb)
ucount = np.diff(M.R.indptr)
test_rel = make_rel(test)
users = [u for u in test_rel if ucount[u] > 0]
print(f"Evaluating on {len(users)} users\n")

warm_models = {
    "Popularity": M.score_popularity,
    "Item-kNN": M.score_knn,
    "SVD": M.score_svd,
    "SBERT content": M.score_content,
    "Hybrid (SVD+SBERT)": M.score_hybrid,
}
rows = {name: evaluate(fn, users, test_rel, M) for name, fn in warm_models.items()}
table1 = pd.DataFrame(rows).T

# RMSE for rating prediction (SVD vs a simple baseline)
t = test[ucount[test["uidx"].values] > 0]
us, ms, y = t["uidx"].values, t["midx"].values, t["rating"].values
base = M.user_mean[us]
svd_pred = np.clip(base + np.einsum("nk,kn->n", M.U[us], M.V[:, ms]), 0.5, 5)
table1["RMSE"] = np.nan
table1.loc["SVD", "RMSE"] = np.sqrt(np.mean((svd_pred - y) ** 2))
table1.loc["Popularity", "RMSE"] = np.sqrt(np.mean((M.global_mean - y) ** 2))
show("TABLE 1: model comparison (time-based split). RMSE only for rating "
     "predictors; 'Popularity' RMSE = predicting the global average", table1,
     "table1_models.csv")

# ---------------------------------------------------------------------------
# TABLE 2: COLD-START USERS - how good are recommendations after k ratings?
# ---------------------------------------------------------------------------
rng = np.random.RandomState(0)
cold_users = [u for u in users if len(M.liked_items(u)) >= 10]
rows = []
for k in [1, 3, 5, 10]:
    picks = {u: rng.choice(M.liked_items(u), k, replace=False) for u in cold_users}
    fns = {
        "Popularity": lambda u: M.pop_score,
        "SBERT profile": lambda u: M.profile_score(picks[u]),
        "SBERT + popularity": lambda u: M.cold_user_score(picks[u]),
    }
    for name, fn in fns.items():
        r = evaluate(fn, cold_users, test_rel, M)
        rows.append({"Known ratings": k, "Model": name,
                     "Recall@10": r["Recall@10"], "NDCG@10": r["NDCG@10"]})
show(f"TABLE 2: cold-start USERS ({len(cold_users)} users, only k ratings known)",
     pd.DataFrame(rows).set_index(["Known ratings", "Model"]), "table2_cold_users.csv")

# ---------------------------------------------------------------------------
# TABLE 3: COLD-START ITEMS - movies that have NO ratings at all
# ---------------------------------------------------------------------------
rs = np.random.RandomState(42)
cold_items = rs.choice(n_items, int(0.1 * n_items), replace=False)
is_cold = np.zeros(n_items, dtype=bool)
is_cold[cold_items] = True
mask = is_cold[ratings["midx"].values]
train_c, test_c = ratings[~mask], ratings[mask]
Mc = Models(train_c, n_users, n_items, emb)
rel_c = make_rel(test_c)
ucount_c = np.diff(Mc.R.indptr)
users_c = [u for u in rel_c if ucount_c[u] > 0]
rows = {
    "Random": evaluate(lambda u: rs.rand(n_items), users_c, rel_c, Mc, allowed=is_cold),
    "SBERT content": evaluate(Mc.score_content, users_c, rel_c, Mc, allowed=is_cold),
}
table3 = pd.DataFrame(rows).T[["Precision@10", "Recall@10", "NDCG@10"]]
show(f"TABLE 3: cold-start ITEMS ({len(cold_items)} movies with zero ratings; "
     "popularity, kNN and SVD cannot score them at all)", table3,
     "table3_cold_items.csv")

# ---------------------------------------------------------------------------
# TABLE 4: popularity bias and the fix (re-rank with a popularity penalty)
# ---------------------------------------------------------------------------
rows = {}
for lam in [0.0, 0.05, 0.1, 0.2, 0.4]:
    r = evaluate(M.score_hybrid, users, test_rel, M, lam=lam)
    rows[f"lambda={lam}"] = {k: r[k] for k in
                             ["NDCG@10", "Recall@10", "AvgPopularity", "Coverage", "Gini"]}
show("TABLE 4: popularity bias for the Hybrid model. lambda = strength of the "
     "popularity penalty (0 = no fix). Lower AvgPopularity/Gini and higher "
     "Coverage = fairer to niche movies", pd.DataFrame(rows).T, "table4_popularity_bias.csv")

print("Done. CSV files are in the results/ folder.")
