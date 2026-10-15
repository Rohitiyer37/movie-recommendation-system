"""app.py - the website.   Run:  streamlit run app.py"""
import glob
import html
import os
import re

import numpy as np
import pandas as pd
import streamlit as st

from recsys import Models, get_embeddings, load_data, top_k

st.set_page_config(page_title="Movie Recommender", page_icon="🎞️", layout="wide")

# ----------------------------------------------------------------------------
# Look and feel: quiet, white, one blue accent, Inter Tight + Open Sans.
# ----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter+Tight:wght@300;500;600;700&family=Open+Sans:wght@400;500;600&display=swap');

:root {
  --paper: #FBFBFD;
  --card: #FFFFFF;
  --mist: #F2F2F5;
  --ink: #1D1D1F;
  --graphite: #6E6E73;
  --hairline: #E5E5EA;
  --blue: #0071E3;
  --display: 'Inter Tight', 'Open Sans', -apple-system, sans-serif;
  --body: 'Open Sans', -apple-system, 'Segoe UI', sans-serif;
}

.stApp, .stApp p, .stApp label, .stApp input, .stApp textarea, .stApp button {
  font-family: var(--body);
}
.stApp { background: var(--paper); color: var(--ink); }
[data-testid="stDecoration"], footer, #MainMenu,
[data-testid="stAppDeployButton"] { display: none !important; }
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1120px; padding-top: 3.5rem; padding-bottom: 5rem; }

h1, h2, h3, h4 { font-family: var(--display) !important; color: var(--ink); letter-spacing: -0.02em; }

/* ---- hero ---- */
.hero { padding: 0.5rem 0 2.25rem; }
.hero-eyebrow { font-size: 0.8rem; font-weight: 600; color: var(--graphite);
  letter-spacing: 0.02em; margin-bottom: 0.6rem; }
.hero-title { font-family: var(--display); font-weight: 600 !important; font-size: clamp(2.4rem, 5vw, 3.6rem);
  line-height: 1.04; letter-spacing: -0.035em; margin: 0; color: var(--ink); }
.hero-sub { font-size: 1.1rem; color: var(--graphite); margin-top: 0.9rem; max-width: 36rem; line-height: 1.5; }

/* ---- tabs as a segmented control ---- */
.stTabs [role="tablist"] { gap: 2px; background: var(--mist); padding: 3px; border: none;
  box-shadow: none; border-radius: 10px; width: fit-content; max-width: 100%; overflow-x: auto; }
.stTabs [role="tablist"]::before, .stTabs [role="tablist"]::after { display: none; }
.stTabs [role="tab"] { height: 34px; padding: 0 16px; border-radius: 8px; border: none;
  background: transparent; color: var(--graphite); display: flex; align-items: center;
  transition: background .2s ease, color .2s ease; }
.stTabs [role="tab"] p { font-size: 0.88rem; font-weight: 500; color: var(--graphite); }
.stTabs [role="tab"][aria-selected="true"] p { color: var(--ink); }
.stTabs [role="tab"][aria-selected="true"] { background: var(--card); color: var(--ink);
  box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 0 0 0.5px rgba(0,0,0,0.04); }
.stTabs [role="tab"]:focus-visible { outline: 2px solid var(--blue); outline-offset: 2px; }
.stTabs .react-aria-SelectionIndicator { display: none; }
.stTabs [role="tabpanel"] { padding-top: 2rem; }

/* ---- sidebar ---- */
[data-testid="stSidebar"] { background: var(--mist); border-right: 1px solid var(--hairline); }
[data-testid="stSidebar"] h2 { font-size: 1.05rem; font-weight: 600; }

/* ---- section heading ---- */
.section-head { display: flex; align-items: baseline; justify-content: space-between;
  gap: 1rem; margin: 2rem 0 1rem; flex-wrap: wrap; }
.section-head h3 { font-family: var(--display); color: var(--ink) !important; font-size: 1.6rem !important; font-weight: 600 !important;
  letter-spacing: -0.025em; margin: 0; padding: 0; }
.section-head span { color: var(--graphite); font-size: 0.88rem; }

/* ---- movie cards ---- */
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 14px; }
.card { background: var(--card); border: 1px solid var(--hairline); border-radius: 18px;
  padding: 18px 20px 20px; display: flex; flex-direction: column; min-height: 210px;
  transition: transform .25s ease, box-shadow .25s ease; }
.card:hover { transform: translateY(-2px); box-shadow: 0 8px 28px rgba(0,0,0,0.06); }
.card-top { display: flex; justify-content: space-between; align-items: flex-start; }
.card-year { font-family: var(--display); font-weight: 300; font-size: 2.6rem;
  line-height: 1; letter-spacing: -0.04em; color: var(--ink); font-variant-numeric: tabular-nums; }
.card-rank { font-size: 0.75rem; font-weight: 600; color: var(--graphite);
  font-variant-numeric: tabular-nums; padding-top: 4px; }
.card-title { font-family: var(--display); font-weight: 600; font-size: 1.08rem;
  line-height: 1.3; letter-spacing: -0.01em; margin: 14px 0 8px; color: var(--ink); }
.card-genres { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 14px; }
.chip { font-size: 0.72rem; color: var(--graphite); background: var(--mist);
  border-radius: 999px; padding: 2px 9px; }
.card-why { margin-top: auto; padding-top: 12px; border-top: 1px solid var(--hairline);
  font-size: 0.8rem; color: var(--graphite); line-height: 1.45; }
.card-why strong { color: var(--ink); font-weight: 600; }
.meter { height: 4px; background: var(--mist); border-radius: 2px; margin-top: 8px; overflow: hidden; }
.meter > div { height: 100%; background: var(--blue); border-radius: 2px; }

/* ---- liked list ---- */
.liked { display: flex; flex-wrap: wrap; gap: 6px; }
.liked .chip { font-size: 0.8rem; color: var(--ink); background: var(--card);
  border: 1px solid var(--hairline); padding: 4px 11px; }

/* ---- widgets ---- */
[data-testid="stExpander"] details { border: 1px solid var(--hairline); border-radius: 14px; background: var(--card); }
[data-testid="stAlert"] { border-radius: 14px; }
.stTextInput input, .stNumberInput input { border-radius: 10px; }
.table-title { font-family: var(--display); font-weight: 600; font-size: 1.15rem;
  letter-spacing: -0.015em; margin: 1.75rem 0 0.6rem; }
.muted { color: var(--graphite); font-size: 0.95rem; line-height: 1.55; max-width: 40rem; }

@media (prefers-reduced-motion: reduce) { .card { transition: none; } .card:hover { transform: none; } }
@media (max-width: 640px) { .block-container { padding-top: 2rem; } .grid { grid-template-columns: 1fr; } }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading data and models (first run takes a minute)...")
def load_all():
    ratings, movies, n_users = load_data()
    # "Terminator, The" -> "The Terminator"
    movies["clean_title"] = movies["clean_title"].str.replace(
        r"^(.*), (The|A|An)$", r"\2 \1", regex=True)
    emb = get_embeddings(movies)
    M = Models(ratings, n_users, len(movies), emb)
    return movies, M, n_users


movies, M, n_users = load_all()
titles = movies["clean_title"] + " (" + movies["year"].fillna("?") + ")"


def esc(s):
    return html.escape(str(s))


def section(title, note=""):
    st.markdown(f'<div class="section-head"><h3>{esc(title)}</h3>'
                f'<span>{esc(note)}</span></div>', unsafe_allow_html=True)


def card(rank, i, footer):
    """One movie card: year, title, genre chips, and a footer line."""
    year = movies["year"].iloc[i]
    year = year if isinstance(year, str) else "—"
    genres = [g for g in movies["genres"].iloc[i].split("|") if g != "(no genres listed)"]
    chips = "".join(f'<span class="chip">{esc(g)}</span>' for g in genres)
    return (f'<div class="card"><div class="card-top"><div class="card-year">{esc(year)}</div>'
            f'<div class="card-rank">{rank}</div></div>'
            f'<div class="card-title">{esc(movies["clean_title"].iloc[i])}</div>'
            f'<div class="card-genres">{chips}</div>'
            f'<div class="card-why">{footer}</div></div>')


def show_cards(idx_list, liked):
    """Draw one card per recommended movie, with the explanation."""
    out = []
    for r, i in enumerate(idx_list, 1):
        why = esc(M.explain(i, liked, movies))
        why = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", why)
        out.append(card(r, i, why))
    st.markdown(f'<div class="grid">{"".join(out)}</div>', unsafe_allow_html=True)


st.markdown(f"""
<div class="hero">
  <div class="hero-eyebrow">MovieLens · {len(movies):,} films · {n_users:,} viewers</div>
  <h1 class="hero-title">Find your next film.</h1>
  <p class="hero-sub">Recommendations from five models, each one explained by a movie
  you already liked. Pick a viewer, or tell us your own favourites.</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["For you", "Similar movies", "Results"])

# ----------------------------------------------------------------------------
with tab1:
    with st.sidebar:
        st.header("Settings")
        mode = st.radio("Who are you?", ["Existing user", "I'm a new user"])
        n_rec = st.slider("How many recommendations", 5, 20, 10)
        lam = st.slider("Long-tail boost", 0.0, 0.5, 0.0, 0.05,
                        help="0 = normal. Higher = more niche, less popular movies.")
        if mode == "Existing user":
            model = st.selectbox("Model", ["Hybrid (SVD + SBERT)", "SVD", "Item-kNN",
                                           "SBERT content", "Popularity"])

    if mode == "Existing user":
        c1, _ = st.columns([1, 3])
        uid = c1.number_input("User ID", 1, n_users, 1)
        u = int(uid) - 1
        liked = M.liked_items(u)
        with st.expander(f"Movies user {int(uid)} liked · {len(liked)}"):
            chips = "".join(f'<span class="chip">{esc(t)}</span>'
                            for t in titles.iloc[liked[:20]])
            st.markdown(f'<div class="liked">{chips}</div>', unsafe_allow_html=True)
        fns = {"Hybrid (SVD + SBERT)": M.score_hybrid, "SVD": M.score_svd,
               "Item-kNN": M.score_knn, "SBERT content": M.score_content,
               "Popularity": M.score_popularity}
        scores = fns[model](u)
        if lam > 0:
            scores = M.penalize(scores, lam)
        top = top_k(scores, M.seen(u), n_rec)
        section(f"Top {n_rec} for user {int(uid)}", model)
        show_cards(top, liked)
    else:
        section("Tell us what you like", "5 or more works best")
        popular = np.argsort(-M.pop)[:2000]
        options = {titles.iloc[i]: int(i) for i in popular}
        picks = st.multiselect("Search and pick movies", list(options.keys()),
                               placeholder="Start typing a title")
        if len(picks) == 0:
            st.info("Pick at least one movie to see recommendations.")
        else:
            liked = np.array([options[p] for p in picks])
            scores = M.cold_user_score(liked)
            if lam > 0:
                scores = M.penalize(scores, lam)
            top = top_k(scores, liked, n_rec)
            section("Recommended for you", f"Based on {len(picks)} picks")
            show_cards(top, liked)

# ----------------------------------------------------------------------------
with tab2:
    st.markdown('<p class="muted">Matches use only a movie\'s title, genres and tags, '
                'so this works even for a film nobody has rated yet.</p>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    q = c1.text_input("Search for a movie", "Toy Story")
    matches = titles[titles.str.contains(q, case=False, regex=False)].head(50)
    if len(matches) == 0:
        st.warning(f"No movie title contains “{q}”. Try a shorter search.")
    else:
        i = c2.selectbox("Pick one", matches.index.tolist(),
                         format_func=lambda x: titles.iloc[x])
        sims = M.emb @ M.emb[i]
        section(f"Similar to {movies['clean_title'].iloc[i]}",
                f"{int(M.pop[i])} ratings")
        out = []
        for r, j in enumerate(top_k(sims, np.array([i]), 10), 1):
            pct = max(0.0, min(1.0, float(sims[j]))) * 100
            footer = (f"Similarity <strong>{sims[j]:.2f}</strong>"
                      f'<div class="meter"><div style="width:{pct:.0f}%"></div></div>')
            out.append(card(r, j, footer))
        st.markdown(f'<div class="grid">{"".join(out)}</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
with tab3:
    files = sorted(glob.glob(os.path.join("results", "*.csv")))
    if not files:
        st.info("Run `python run_experiments.py` to create the results tables.")
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        num, _, rest = name.partition("_")
        label = f"{num.replace('table', 'Table ')} · {rest.replace('_', ' ').capitalize()}"
        st.markdown(f'<div class="table-title">{esc(label)}</div>', unsafe_allow_html=True)
        st.dataframe(pd.read_csv(f, index_col=0), width="stretch")
