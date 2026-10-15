"""app.py - the website.   Run:  streamlit run app.py"""
import glob
import os

import numpy as np
import pandas as pd
import streamlit as st

from recsys import Models, get_embeddings, load_data, top_k

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")


@st.cache_resource(show_spinner="Loading data and models (first run takes a minute)...")
def load_all():
    ratings, movies, n_users = load_data()
    emb = get_embeddings(movies)
    M = Models(ratings, n_users, len(movies), emb)
    return movies, M, n_users


movies, M, n_users = load_all()
titles = movies["clean_title"] + " (" + movies["year"].fillna("?") + ")"


def show_cards(idx_list, liked):
    """Draw one card per recommended movie, with the explanation."""
    for i in idx_list:
        with st.container(border=True):
            st.markdown(f"**{titles.iloc[i]}**  \n"
                        f"{movies['genres'].iloc[i].replace('|', ' · ')}")
            st.caption(M.explain(i, liked, movies))


st.title("🎬 Movie Recommender")
st.caption("MovieLens data · SVD, item-kNN and SBERT models · cold start · "
           "explanations · popularity-bias control")

tab1, tab2, tab3 = st.tabs(["Recommend", "Similar movies (new-movie cold start)",
                            "Results tables"])

# ----------------------------------------------------------------------------
with tab1:
    with st.sidebar:
        st.header("Settings")
        mode = st.radio("Who are you?", ["Existing user", "I'm a new user"])
        n_rec = st.slider("How many recommendations", 5, 20, 10)
        lam = st.slider("Long-tail boost (popularity penalty)", 0.0, 0.5, 0.0, 0.05,
                        help="0 = normal. Higher = more niche / less popular movies.")
        if mode == "Existing user":
            model = st.selectbox("Model", ["Hybrid (SVD + SBERT)", "SVD", "Item-kNN",
                                           "SBERT content", "Popularity"])

    if mode == "Existing user":
        uid = st.number_input("User ID", 1, n_users, 1)
        u = int(uid) - 1
        liked = M.liked_items(u)
        with st.expander("Movies this user liked"):
            st.write(titles.iloc[liked[:20]].tolist())
        fns = {"Hybrid (SVD + SBERT)": M.score_hybrid, "SVD": M.score_svd,
               "Item-kNN": M.score_knn, "SBERT content": M.score_content,
               "Popularity": M.score_popularity}
        scores = fns[model](u)
        if lam > 0:
            scores = M.penalize(scores, lam)
        top = top_k(scores, M.seen(u), n_rec)
        st.subheader(f"Top {n_rec} for user {int(uid)} ({model})")
        show_cards(top, liked)
    else:
        st.subheader("Tell us 5 or more movies you like")
        popular = np.argsort(-M.pop)[:2000]
        options = {titles.iloc[i]: int(i) for i in popular}
        picks = st.multiselect("Search and pick movies", list(options.keys()))
        if len(picks) == 0:
            st.info("Pick at least one movie. 5 or more gives better results.")
        else:
            liked = np.array([options[p] for p in picks])
            scores = M.cold_user_score(liked)
            if lam > 0:
                scores = M.penalize(scores, lam)
            top = top_k(scores, liked, n_rec)
            st.subheader("Recommended for you")
            show_cards(top, liked)

# ----------------------------------------------------------------------------
with tab2:
    st.write("This uses only the movie's text (title, genres, tags), so it works "
             "even for a movie that nobody has rated yet.")
    q = st.text_input("Search for a movie", "Toy Story")
    matches = titles[titles.str.contains(q, case=False, regex=False)].head(50)
    if len(matches) == 0:
        st.warning("No match.")
    else:
        i = st.selectbox("Pick one", matches.index.tolist(),
                         format_func=lambda x: titles.iloc[x])
        st.write(f"Ratings this movie has: **{int(M.pop[i])}**")
        sims = M.emb @ M.emb[i]
        for j in top_k(sims, np.array([i]), 10):
            with st.container(border=True):
                st.markdown(f"**{titles.iloc[j]}**  \n"
                            f"{movies['genres'].iloc[j].replace('|', ' · ')}")
                st.caption(f"Similarity: {sims[j]:.2f}")

# ----------------------------------------------------------------------------
with tab3:
    files = sorted(glob.glob(os.path.join("results", "*.csv")))
    if not files:
        st.info("Run `python run_experiments.py` first to create the tables.")
    for f in files:
        st.markdown(f"**{os.path.basename(f)}**")
        st.dataframe(pd.read_csv(f), use_container_width=True)
