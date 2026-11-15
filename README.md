# 🛒 BigBasket Product Assistant — RAG Chatbot

A Retrieval-Augmented Generation chatbot that answers questions about BigBasket
products (prices, brands, ingredients, comparisons) using only the BigBasket
catalogue dataset, and says so when the answer isn't in the data.

**Pipeline:** data → cleaning → one document per product → chunking →
embeddings → vector store → hybrid retrieval → LLM with context → grounded answer

| Step | Choice | Why |
|---|---|---|
| Data | [Kaggle: BigBasket Entire Product List](https://www.kaggle.com/datasets/surajjha101/bigbasket-entire-product-list-28k-datapoints) | 27k products with a real description column |
| Cleaning | drop empty/short descriptions and duplicates → 21k rows; 5,000-row sample | faster embedding, still meaningful retrieval |
| Chunking | 1 product = 1 chunk; long descriptions split at 800 chars / 100 overlap, product header repeated on each chunk | keeps price/brand attached to every chunk |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) | free, fast, runs on CPU |
| Vector store | ChromaDB (cosine) | persistent, simple |
| Retrieval | Hybrid: vector + BM25, merged with Reciprocal Rank Fusion | meaning + exact brand names |
| LLM | Gemini (temperature 0.2), answers only from context with [n] citations | grounded answers |
| Fallback | refuse if best similarity < 0.30, plus prompt rule | no invented answers |

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Paste a Gemini API key (free at aistudio.google.com) in the sidebar, or put
`GEMINI_API_KEY = "..."` in `.streamlit/secrets.toml`.

Data: historical catalogue snapshot (Kaggle, 2022) — not live stock or prices.
