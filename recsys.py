"""recsys.py - all the recommender logic lives here.

Used by run_experiments.py (evaluation) and app.py (website).
"""
import io
import os
import urllib.request
import zipfile
import ssl
import certifi

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

DATA_DIR = "data"
FOLDER = os.path.join(DATA_DIR, "ml-latest-small")
URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
EMB_PATH = os.path.join("models", "embeddings.npy")


# ----------------------------------------------------------------------------
# STEP 2: data loading
# ----------------------------------------------------------------------------
def download_data():
    """Download MovieLens (about 1 MB) if we don't have it yet."""
    if os.path.exists(os.path.join(FOLDER, "ratings.csv")):
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Downloading MovieLens ml-latest-small ...")
    context = ssl.create_default_context(cafile=certifi.where())
    raw = urllib.request.urlopen(URL, context=context).read()
    zipfile.ZipFile(io.BytesIO(raw)).extractall(DATA_DIR)


def load_data():
    """Returns (ratings, movies, n_users).

    movies has one row per movie; its row number is the movie's index ("midx").
    ratings gets two extra columns: uidx (user index) and midx (movie index).
    """
    download_data()
    ratings = pd.read_csv(os.path.join(FOLDER, "ratings.csv"))
    movies = pd.read_csv(os.path.join(FOLDER, "movies.csv"))
    tags = pd.read_csv(os.path.join(FOLDER, "tags.csv"))

    movies = movies.sort_values("movieId").reset_index(drop=True)
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)\s*$")[0]
    movies["clean_title"] = movies["title"].str.replace(
        r"\s*\(\d{4}\)\s*$", "", regex=True)

    # most common user tags per movie (extra text for the SBERT model)
    tags["tag"] = tags["tag"].astype(str).str.lower()
    tag_counts = tags.groupby(["movieId", "tag"]).size().reset_index(name="n")
    tag_counts = tag_counts.sort_values("n", ascending=False)
    top_tags = (tag_counts.groupby("movieId").head(8)
                .groupby("movieId")["tag"].apply(", ".join))
    movies["tags"] = movies["movieId"].map(top_tags).fillna("")

    # the text that SBERT will read for each movie
    movies["text"] = (movies["clean_title"] + ". Genres: "
                      + movies["genres"].str.replace("|", ", ", regex=False)
                      + ". Tags: " + movies["tags"])

    movies["midx"] = np.arange(len(movies))
    uids = np.sort(ratings["userId"].unique())
    umap = {u: i for i, u in enumerate(uids)}
    mmap = dict(zip(movies["movieId"], movies["midx"]))
    ratings["uidx"] = ratings["userId"].map(umap)
    ratings["midx"] = ratings["movieId"].map(mmap)
    return ratings, movies, len(uids)


# ----------------------------------------------------------------------------
# STEP 6: SBERT embeddings
# ----------------------------------------------------------------------------
def get_embeddings(movies):
    """Turn each movie's text into a 384-number vector (cached on disk)."""
    if os.path.exists(EMB_PATH):
        return np.load(EMB_PATH)
    from sentence_transformers import SentenceTransformer
    os.makedirs("models", exist_ok=True)
    print("Computing SBERT embeddings (1-3 minutes, only done once) ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(movies["text"].tolist(), batch_size=64,
                       show_progress_bar=True, normalize_embeddings=True)
    emb = emb.astype(np.float32)
    np.save(EMB_PATH, emb)
    return emb


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def zs(x):
    """z-score: puts different score scales on the same footing."""
    x = np.asarray(x, dtype=np.float64)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else x * 0.0


def top_k(scores, exclude, k=10, allowed=None):
    """Indices of the k best items, skipping `exclude` (and not in `allowed`)."""
    s = np.asarray(scores, dtype=np.float64).copy()
    s[exclude] = -np.inf
    if allowed is not None:
        s[~allowed] = -np.inf
    k = min(k, len(s) - 1)
    idx = np.argpartition(-s, k)[:k]
    return idx[np.argsort(-s[idx])]


def gini(counts):
    """0 = every item recommended equally, 1 = one item gets everything."""
    x = np.sort(np.asarray(counts, dtype=np.float64))
    n, total = len(x), x.sum()
    if total == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * total))


# ----------------------------------------------------------------------------
# STEPS 3-4, 7, 8, 9: the models
# ----------------------------------------------------------------------------
class Models:
    """Fits every model on a ratings table and gives each one a score_xxx(u)."""

    def __init__(self, train, n_users, n_items, emb, k=50, m=20):
        self.n_users, self.n_items, self.emb = n_users, n_items, emb
        self.R = csr_matrix(
            (train["rating"].values.astype(np.float32),
             (train["uidx"].values, train["midx"].values)),
            shape=(n_users, n_items))

        # --- popularity: IMDB-style weighted rating ---
        self.pop = np.diff(self.R.tocsc().indptr)             # votes per item
        sums = np.asarray(self.R.sum(axis=0)).ravel()
        self.global_mean = float(train["rating"].mean())
        item_mean = np.where(self.pop > 0, sums / np.maximum(self.pop, 1), 0)
        wr = (self.pop / (self.pop + m) * item_mean
              + m / (self.pop + m) * self.global_mean)
        self.pop_score = np.where(self.pop > 0, wr, -1.0)

        # --- user-mean-centred ratings ---
        ucount = np.diff(self.R.indptr)
        usum = np.asarray(self.R.sum(axis=1)).ravel()
        self.user_mean = np.where(ucount > 0, usum / np.maximum(ucount, 1),
                                  self.global_mean)
        C = self.R.copy()
        rows = np.repeat(np.arange(n_users), np.diff(C.indptr))
        C.data = (C.data - self.user_mean[rows]).astype(np.float32)
        self.C = C
        self.CT = C.T.tocsr()
        self.norm = np.sqrt(np.asarray(C.multiply(C).sum(axis=0)).ravel()) + 1e-9

        # --- SVD (matrix factorization) ---
        k = min(k, n_users - 1, n_items - 1)
        svd = TruncatedSVD(n_components=k, random_state=0)
        self.U = svd.fit_transform(C)       # users x k
        self.V = svd.components_            # k x items

    # ---- small utilities ----
    def _row(self, u):
        s, e = self.R.indptr[u], self.R.indptr[u + 1]
        return self.R.indices[s:e], self.R.data[s:e]

    def seen(self, u):
        return self._row(u)[0]

    def liked_items(self, u):
        """Movies the user rated >= 4 (or everything they rated if none)."""
        idx, r = self._row(u)
        liked = idx[r >= 4]
        return liked if len(liked) > 0 else idx

    # ---- the models ----
    def score_popularity(self, u=None):
        return self.pop_score

    def score_knn(self, u):
        """Item-item collaborative filtering (cosine similarity)."""
        s, e = self.C.indptr[u], self.C.indptr[u + 1]
        v = np.zeros(self.n_items)
        v[self.C.indices[s:e]] = self.C.data[s:e]
        y = self.C @ (v / self.norm)
        return (self.CT @ y) / self.norm

    def score_svd(self, u):
        return self.U[u] @ self.V

    def profile_score(self, liked_idx):
        """Content model: average the SBERT vectors of liked movies, then
        score every movie by similarity to that average."""
        if len(liked_idx) == 0:
            return np.zeros(self.n_items)
        profile = self.emb[liked_idx].mean(axis=0)
        return self.emb @ profile

    def score_content(self, u):
        return self.profile_score(self.liked_items(u))

    def score_hybrid(self, u, w=1.0):
        return zs(self.score_svd(u)) + w * zs(self.score_content(u))

    def cold_user_score(self, liked_idx, w_pop=0.3):
        """NEW USER: no history, only a few picked movies."""
        return zs(self.profile_score(liked_idx)) + w_pop * zs(self.pop_score)

    # ---- popularity-bias fix ----
    def penalize(self, scores, lam):
        """Subtract lam x (z-scored log popularity). Higher lam = more niche."""
        return zs(scores) - lam * zs(np.log1p(self.pop))

    # ---- STEP 8: explainability ----
    def explain(self, item, liked_idx, movies):
        if len(liked_idx) == 0:
            return "Popular with other users"
        sims = self.emb[liked_idx] @ self.emb[item]
        best = int(liked_idx[int(np.argmax(sims))])
        g1 = set(movies["genres"].iloc[item].split("|"))
        g2 = set(movies["genres"].iloc[best].split("|"))
        shared = sorted((g1 & g2) - {"(no genres listed)"})
        text = f"Because you liked **{movies['clean_title'].iloc[best]}**"
        if shared:
            text += f" (shared: {', '.join(shared)})"
        return text


# ----------------------------------------------------------------------------
# STEP 5: evaluation
# ----------------------------------------------------------------------------
def evaluate(score_fn, users, test_rel, M, k=10, allowed=None, lam=0.0):
    """Average Precision@k, Recall@k, NDCG@k over users + popularity-bias stats.

    score_fn(u) -> scores for every movie.  test_rel[u] -> set of movies the
    user actually liked in the test period.
    """
    prec, rec, ndcg, all_rec = [], [], [], []
    for u in users:
        s = score_fn(u)
        if lam > 0:
            s = M.penalize(s, lam)
        top = top_k(s, M.seen(u), k, allowed)
        rel = test_rel[u]
        top_list = top.tolist()
        hits = [1 if i in rel else 0 for i in top_list]
        prec.append(sum(hits) / k)
        rec.append(sum(hits) / len(rel))
        dcg = sum(h / np.log2(r + 2) for r, h in enumerate(hits))
        idcg = sum(1 / np.log2(r + 2) for r in range(min(len(rel), k)))
        ndcg.append(dcg / idcg)
        all_rec.extend(top_list)
    all_rec = np.array(all_rec)
    return {
        "Precision@10": np.mean(prec),
        "Recall@10": np.mean(rec),
        "NDCG@10": np.mean(ndcg),
        "AvgPopularity": float(M.pop[all_rec].mean()),
        "Coverage": len(set(all_rec.tolist())) / M.n_items,
        "Gini": gini(np.bincount(all_rec, minlength=M.n_items)),
    }
