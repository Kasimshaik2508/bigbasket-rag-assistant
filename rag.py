"""
rag.py — the RAG pipeline for the BigBasket Product Assistant.

Pipeline (this is what you explain in the demo):
    CSV  ->  clean  ->  one text document per product  ->  chunk long ones
         ->  embeddings (sentence-transformers)  ->  vector store (ChromaDB)
         ->  retrieval (vector + BM25 keyword = Hybrid RAG)
         ->  LLM (Gemini) gets question + retrieved chunks  ->  grounded answer
         ->  fallback if retrieved context is not relevant enough

Everything the Streamlit UI needs is in the RAGPipeline class at the bottom.
"""

import os
import re

import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG — every "magic number" lives here so you can justify/change it.
# ---------------------------------------------------------------------------
DATA_PATH = os.getenv("BB_DATA_PATH", "data/BigBasket Products.csv")
POLICY_PATH = os.getenv("BB_POLICY_PATH", "data/bigbasket_policies.txt")  # optional
CHROMA_DIR = "chroma_db"          # where the vector store is saved on disk
COLLECTION = "bigbasket_products"

# Small, fast, free embedding model (384-dim). Runs on CPU, no API key needed.
EMBED_MODEL = "all-MiniLM-L6-v2"

# WHY a cap: embedding all ~27k rows on a laptop CPU takes ~10+ min.
# 5,000 rows embeds in ~1-2 min and is still plenty for meaningful retrieval.
# Set to None to use the full dataset (delete chroma_db/ after changing it).
MAX_ROWS = 5000

# WHY these chunk numbers: most BigBasket descriptions are short (1 product =
# 1 chunk). The few very long ones are split so a single chunk doesn't dilute
# its embedding. 800 chars ≈ 150-200 words; 100-char overlap keeps a sentence
# that crosses the boundary present in both chunks.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

TOP_K = 5                 # chunks given to the LLM
RRF_K = 60                # standard constant for Reciprocal Rank Fusion

# Fallback threshold: cosine similarity of the BEST vector match.
# For MiniLM, < ~0.30 usually means "nothing in the data is about this".
# Tune it with your out-of-dataset test questions.
SIM_THRESHOLD = 0.30

# Google retires older model names over time (gemini-2.5-flash was retired for
# new users in Nov 2026). We try these in order and skip any that return
# "not found", so one retirement can't break the app. Set GEMINI_MODEL to
# force a specific model first.
LLM_MODELS = [m for m in [os.getenv("GEMINI_MODEL"),
                          "gemini-3.8-flash",
                          "gemini-flash-latest"] if m]

FALLBACK_MSG = (
    "I couldn't find enough information about that in the BigBasket product "
    "data I have, so I don't want to guess. Try asking about a specific "
    "product, brand, category, price range or BigBasket policy."
)


# ---------------------------------------------------------------------------
# STEP 1 — LOAD & CLEAN
# ---------------------------------------------------------------------------
def load_and_clean(path: str = DATA_PATH, max_rows=MAX_ROWS) -> pd.DataFrame:
    """Read the Kaggle CSV and remove rows that would hurt retrieval."""
    df = pd.read_csv(path)

    # Normalise column names ("Sale Price " -> "sale_price") so code is robust.
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.drop(columns=[c for c in ["index", "unnamed:_0"] if c in df.columns])

    # A product with no name or no description gives the retriever nothing to
    # match on -> drop it (the brief explicitly asks to remove empty records).
    df = df.dropna(subset=["product", "description"])

    for col in ["product", "brand", "category", "sub_category", "type", "description"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()
            # collapse repeated whitespace/newlines from scraping
            df[col] = df[col].str.replace(r"\s+", " ", regex=True)

    # Very short descriptions ("Good product.") carry no retrievable detail.
    df = df[df["description"].str.len() >= 30]

    # Duplicates: same product+brand listed twice would waste 2 of our top-5 slots.
    df = df.drop_duplicates(subset=["product", "brand"])
    df = df.drop_duplicates(subset=["description"])

    df["brand"] = df["brand"].replace("", "Unknown")

    # Random sample keeps category proportions roughly the same as the full data.
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)  # fixed seed = reproducible

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# STEP 2 — ROW -> DOCUMENT TEXT
# ---------------------------------------------------------------------------
def _fmt_price(x):
    try:
        return f"₹{float(x):,.0f}"
    except (TypeError, ValueError):
        return "not listed"


def row_to_header(row) -> str:
    """Structured fields that must stay attached to every chunk of a product.

    WHY: if a long description is split, chunk #2 would otherwise lose the
    product name and price, and the LLM couldn't tell which product it's about.
    """
    discount = ""
    try:
        sp, mp = float(row["sale_price"]), float(row["market_price"])
        if mp > sp > 0:
            discount = f", {round((mp - sp) / mp * 100)}% off"
    except (TypeError, ValueError, KeyError):
        pass

    rating = row.get("rating")
    rating_txt = f"{rating}/5" if pd.notna(rating) else "no rating"

    return (
        f"Product: {row['product']}\n"
        f"Brand: {row['brand']}\n"
        f"Category: {row.get('category', '')} > {row.get('sub_category', '')}\n"
        f"Type: {row.get('type', '')}\n"
        f"Price: {_fmt_price(row.get('sale_price'))} "
        f"(MRP {_fmt_price(row.get('market_price'))}{discount})\n"
        f"Rating: {rating_txt}\n"
    )


# ---------------------------------------------------------------------------
# STEP 3 — CHUNKING
# ---------------------------------------------------------------------------
def split_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Sliding-window split on word boundaries."""
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):                       # don't cut a word in half
            space = text.rfind(" ", start, end)
            if space > start:
                end = space
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
        nxt = text.find(" ", start, end)          # start next chunk on a word
        if nxt != -1:
            start = nxt + 1
    return chunks


def build_documents(df: pd.DataFrame, policy_path: str = POLICY_PATH):
    """Return a list of {id, text, metadata} ready to embed."""
    docs = []
    for i, row in df.iterrows():
        header = row_to_header(row)
        pieces = split_text(row["description"])
        for j, piece in enumerate(pieces):
            docs.append({
                "id": f"prod-{i}-{j}",
                "text": header + f"Description: {piece}",
                # Metadata is stored next to the vector -> shown as sources in UI
                # and usable later for filters (e.g. price < 200).
                "metadata": {
                    "source": "product",
                    "product": row["product"],
                    "brand": row["brand"],
                    "category": str(row.get("category", "")),
                    "price": float(row["sale_price"]) if pd.notna(row.get("sale_price")) else -1.0,
                    "chunk": j,
                },
            })

    # Optional: BigBasket policy/FAQ text you copied from bigbasket.com.
    # Paragraphs separated by a blank line -> each paragraph is one chunk,
    # because one paragraph = one policy topic (returns, delivery, refunds...).
    if policy_path and os.path.exists(policy_path):
        with open(policy_path, encoding="utf-8") as f:
            paras = [p.strip() for p in f.read().split("\n\n") if p.strip()]
        for k, para in enumerate(paras):
            for j, piece in enumerate(split_text(para)):
                docs.append({
                    "id": f"policy-{k}-{j}",
                    "text": f"BigBasket policy / FAQ:\n{piece}",
                    "metadata": {"source": "policy", "product": "BigBasket policy",
                                 "brand": "BigBasket", "category": "Policy",
                                 "price": -1.0, "chunk": j},
                })
    return docs


def tokenize(text: str):
    """Lowercase word tokens for BM25 keyword search."""
    return re.findall(r"[a-z0-9]+", text.lower())


# ---------------------------------------------------------------------------
# STEPS 4-7 — EMBED, STORE, RETRIEVE, GENERATE
# ---------------------------------------------------------------------------
class RAGPipeline:
    def __init__(self, data_path: str = DATA_PATH):
        # Heavy imports here so the cleaning functions above can be tested
        # without installing everything.
        import chromadb
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer

        self.embedder = SentenceTransformer(EMBED_MODEL)

        # PersistentClient saves vectors to disk -> we embed ONCE, and later
        # app restarts load in seconds instead of re-embedding.
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        # cosine space: we compare direction of meaning, not vector length.
        self.collection = client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}
        )

        if self.collection.count() == 0:
            docs = build_documents(load_and_clean(data_path))
            self._index(docs)

        # BM25 lives in memory; rebuild it from what's stored in Chroma.
        stored = self.collection.get(include=["documents", "metadatas"])
        self.ids = stored["ids"]
        self.texts = stored["documents"]
        self.metas = stored["metadatas"]
        self.id_to_idx = {d: i for i, d in enumerate(self.ids)}
        self.bm25 = BM25Okapi([tokenize(t) for t in self.texts])

    # ---- STEP 4+5: embeddings -> vector store --------------------------------
    def _index(self, docs, batch=256):
        texts = [d["text"] for d in docs]
        # normalize_embeddings=True -> unit vectors, so cosine == dot product
        embs = self.embedder.encode(texts, batch_size=batch,
                                    normalize_embeddings=True,
                                    show_progress_bar=True)
        # Chroma limits how many items one add() call accepts -> add in batches.
        for s in range(0, len(docs), 2000):
            part = docs[s:s + 2000]
            self.collection.add(
                ids=[d["id"] for d in part],
                documents=[d["text"] for d in part],
                metadatas=[d["metadata"] for d in part],
                embeddings=embs[s:s + 2000].tolist(),
            )

    # ---- STEP 6: retrieval ---------------------------------------------------
    def _vector_search(self, query, n):
        q = self.embedder.encode([query], normalize_embeddings=True).tolist()
        res = self.collection.query(query_embeddings=q, n_results=n)
        ids = res["ids"][0]
        sims = [1 - d for d in res["distances"][0]]   # cosine distance -> similarity
        return ids, sims

    def _keyword_search(self, query, n):
        scores = self.bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [self.ids[i] for i in top if scores[i] > 0]

    def retrieve(self, query, k=TOP_K, mode="hybrid"):
        """mode: 'vector' | 'keyword' | 'hybrid'.

        Hybrid = Reciprocal Rank Fusion: each list votes 1/(60 + rank) for its
        results; sum the votes. WHY: embeddings understand meaning
        ("something for a sore throat" -> honey/ginger), while BM25 nails exact
        words like brand names ("Amul", "Tata Sampann") that embeddings blur.
        RRF merges them without needing to compare their different score scales.
        """
        pool = k * 4   # fetch more candidates than we need, then fuse
        v_ids, v_sims = self._vector_search(query, pool)
        best_sim = max(v_sims) if v_sims else 0.0
        sim_of = dict(zip(v_ids, v_sims))

        if mode == "vector":
            ranked = v_ids
        else:
            k_ids = self._keyword_search(query, pool)
            if mode == "keyword":
                ranked = k_ids
            else:
                fused = {}
                for rank, d in enumerate(v_ids):
                    fused[d] = fused.get(d, 0) + 1 / (RRF_K + rank + 1)
                for rank, d in enumerate(k_ids):
                    fused[d] = fused.get(d, 0) + 1 / (RRF_K + rank + 1)
                ranked = sorted(fused, key=fused.get, reverse=True)

        results = []
        for d in ranked[:k]:
            i = self.id_to_idx[d]
            results.append({"id": d, "text": self.texts[i], "metadata": self.metas[i],
                            "similarity": sim_of.get(d)})
        return results, best_sim

    # ---- STEP 7: generation --------------------------------------------------
    @staticmethod
    def build_prompt(question, chunks):
        context = "\n\n".join(f"[{n}] {c['text']}" for n, c in enumerate(chunks, 1))
        # The rules below are what make answers GROUNDED (15% of the rubric).
        return f"""You are BigBasket's product assistant. Answer the customer's question
using ONLY the numbered context below.

Rules:
- Use only facts from the context. Do not use outside knowledge.
- Cite the context numbers you used, like [1] or [2][4].
- Prices are in Indian Rupees (₹). Mention brand and price when recommending.
- For comparisons, use a short bullet list or table.
- If the context does not contain the answer (or only part of it), say so
  clearly for the missing part instead of guessing.
- If none of the context is relevant, reply exactly: "{FALLBACK_MSG}"
- This data is a historical catalogue snapshot; don't claim live stock or delivery times.

Context:
{context}

Customer question: {question}

Answer:"""

    def generate(self, prompt, api_key):
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        last_err = None
        for model in LLM_MODELS:
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    # Low temperature = less creative = sticks to the context.
                    config=types.GenerateContentConfig(temperature=0.2),
                )
                return resp.text
            except Exception as e:
                # Only move on for "this model doesn't exist / was retired".
                # Anything else (bad key, quota) is a real error -> raise it.
                if "NOT_FOUND" in str(e) or "404" in str(e):
                    last_err = e
                    continue
                raise
        raise RuntimeError(f"None of the Gemini models {LLM_MODELS} are available: {last_err}")

    def answer(self, question, api_key, k=TOP_K, mode="hybrid"):
        chunks, best_sim = self.retrieve(question, k=k, mode=mode)

        # FALLBACK layer 1 (before the LLM): if even the best semantic match is
        # weak, the data has nothing on this -> refuse without calling the LLM.
        # FALLBACK layer 2 is the prompt rule above, for partly-relevant context.
        if best_sim < SIM_THRESHOLD:
            return {"answer": FALLBACK_MSG, "chunks": chunks,
                    "best_sim": best_sim, "fallback": True}

        text = self.generate(self.build_prompt(question, chunks), api_key)
        return {"answer": text, "chunks": chunks, "best_sim": best_sim,
                "fallback": FALLBACK_MSG[:40] in text}
