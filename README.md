# Movie Recommender (MovieLens + SBERT)

Models: Popularity, Item-kNN, SVD, SBERT content, Hybrid.
Extensions: cold-start users and items, "Because you liked X" explanations,
popularity-bias study with a long-tail re-ranker, optional LLM re-ranker.

## Run
```
pip install -r requirements.txt
python run_experiments.py     # builds embeddings, prints 4 result tables
streamlit run app.py          # opens the website
```

## Files
- `recsys.py`          data loading, all models, explanations, metrics
- `run_experiments.py` Tables 1-4 (saved to `results/`)
- `app.py`             Streamlit website
- `llm_rerank.py`      optional LLM re-ranker (needs an Anthropic API key)

## Results
Paste the four tables from `results/` here after you run the experiments.

## References
Add the papers you read (see your reading list).
