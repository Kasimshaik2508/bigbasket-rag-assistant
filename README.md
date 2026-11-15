# 🛒 BigBasket Product Assistant — Advanced RAG Chatbot

An ultra-fast, production-grade Retrieval-Augmented Generation (RAG) assistant for BigBasket products, prices, comparisons, and store policies. Grounded strictly in real catalogue data with zero hallucinations, sub-2ms caching, cross-encoder re-ranking, query decomposition, and complete audit telemetry.

---

## 📌 Architecture Overview

```
User Query
    │
    ▼
[Input Safety Guardrails] ──► (Blocked if prompt injection, profanity, or out-of-scope)
    │ (Safe)
    ▼
[In-Memory Exact Hash Cache] ──► (Cache Hit: SHA-256 match returns in < 2ms)
    │ (Cache Miss)
    ▼
[Conversational Query Contextualizer] ──► (Rephrases follow-up queries like "Give me a cheaper alternative")
    │
    ▼
[Query Decomposition Engine] ──► (Splits complex/multi-intent questions into 1-3 atomic sub-queries)
    │
    ├───► Dense Vector Search (ChromaDB + all-MiniLM-L6-v2) ──► Top Candidates
    │                                                                   │
    └───► Sparse Keyword Search (BM25Okapi) ───────────────────► Top Candidates
                                                                        │
                                                                        ▼
                                                [Reciprocal Rank Fusion (RRF, k=60)]
                                                                        │
                                                                        ▼
                                                    [Cross-Encoder Re-Ranking]
                                               (cross-encoder/ms-marco-MiniLM-L-6-v2)
                                                                        │
                                                                        ▼
                                                [Similarity Relevance Threshold]
                                            (If best match < 0.30 -> Graceful Refusal)
                                                                        │
                                                                        ▼
                                                   [Multi-Provider LLM Engine]
                                               Primary: Groq (openai/gpt-oss-20b)
                                               Fallback: Google Gemini (2.0/2.5 Flash)
                                                                        │
                                                                        ▼
                                                      [Grounded Response + Sources]
                                                                        │
                                                                        ├──► [Local Audit Logger (logs/*.jsonl)]
                                                                        └──► [Streamlit UI with Dark Mode]
```

---

## ⚙️ Key Technical Features

| Feature | Implementation | Purpose & Benefit |
|---|---|---|
| **Dataset** | [Kaggle: BigBasket Entire Product List](https://www.kaggle.com/datasets/surajjha101/bigbasket-entire-product-list-28k-datapoints) | 27k product rows with natural-language descriptions; sampled to 5,000 clean records for sub-second CPU operations. |
| **Data Cleaning** | Normalization, HTML stripping, deduplication by `(product, brand)` & description hash, drop short rows (<30 chars). | Eliminates duplicate entries and low-information chunks from poisoning retrieval. |
| **Policy Ingestion** | Synthetic & official BigBasket policy dataset (`data/bigbasket_policies.txt`). | Enables verified answers for returns, cancellations, perishable goods rules, and refunds. |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions). | Free, runs 100% locally on CPU in ~20ms, zero external API latency or quota cost. |
| **Vector Store** | `ChromaDB` (Persistent, HNSW with cosine distance metric). | Fast vector similarity search with persistent disk storage in `chroma_db/`. |
| **Hybrid Retrieval** | Dense Vector (ChromaDB) + Sparse Keywords (BM25Okapi) via **Reciprocal Rank Fusion** ($k=60$). | Balances conceptual semantic search ("healthy breakfast for diabetics") with exact brand token matches ("Tata Sampann", "Saffola"). |
| **Re-Ranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2`. | Evaluates full token cross-attention on top candidate pairs to boost high-precision products into context. |
| **Caching Layer** | Exact Hash Cache (`SHA-256` keyed in memory). | Repeated questions return instantly (< 2ms) with zero LLM API cost. |
| **Query Decomposition** | Heuristic detection + atomic sub-query generation. | Decomposes compound queries (e.g. *"Suggest a cooking oil under ₹200 and explain BigBasket's return policy"*) into discrete searches, then synthesizes unified answers. |
| **Conversational Memory** | Contextual query re-writing. | Seamlessly resolves ambiguous follow-up questions (e.g., *"Give me a cheaper alternative"* after looking at face wash). |
| **Guardrails & Fallback** | Multi-layer input regex filter + similarity threshold (< 0.30) + prompt grounding rules. | Rejects prompt injections, refuses out-of-domain queries, and prevents hallucinations. |
| **LLM Orchestration** | Primary: **Groq** (`openai/gpt-oss-20b`, `gpt-oss-120b`).<br>Fallback: **Google Gemini** (`gemini-2.0-flash`, `gemini-2.5-flash`). | Ultra-fast token generation (~800 tokens/sec on Groq) with multi-model failover redundancy. |
| **Audit Logging** | JSON-Lines logger (`logs/audit_YYYYMMDD.jsonl`). | Records timestamp, latency, flow branch, cache status, confidence score, and exact chunks sent. |
| **Streamlit UI** | Modern design with **Dark Mode Toggle**, price tags, category aisle navigation, and live telemetry badges. | Fast, clean interface. |

---

## 🚀 Quickstart & Setup

### 1. Prerequisites & Virtual Environment
Ensure you have Python 3.10+ installed:

```bash
# Clone the repository
git clone https://github.com/your-username/bigbasket-rag-assistant.git
cd bigbasket-rag-assistant

# Create and activate virtual environment
uv venv
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### 2. Configure API Keys
Create a `.env` file in the root directory (or use `.streamlit/secrets.toml`):

```env
# Primary LLM (Free key from console.groq.com)
GROQ_API_KEY=your_groq_api_key_here

# Optional Fallback LLM (Free key from aistudio.google.com)
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Run the Streamlit Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧪 Automated Evaluation Suite

The project includes a standalone evaluation suite (`evaluate.py`) that benchmarks 8 representative test cases across 5 categories:
1. **Factual Lookups** (Atta price, Toor Dal specifications)
2. **Product Comparisons** (Saffola Gold vs. Fortune Sunflower Oil)
3. **Multi-Part / Decomposed Queries** (Breakfast cereal + sugar-free snack)
4. **Out-of-Dataset Refusals** (Flight bookings, loan interest rates)
5. **Security & Guardrails** (Prompt injection attempts)

To run the evaluations:

```bash
python evaluate.py
```

The script outputs pass/fail status, keyword recall, citation presence, and latency metrics, saving a complete report to `evaluation_report.json`.

---

## 📁 Repository Structure

```
├── app.py                      # Streamlit frontend with Dark Mode and telemetry
├── rag.py                      # Complete RAG pipeline (retrieval, re-ranking, LLM, cache)
├── evaluate.py                 # Automated testing and evaluation benchmark suite
├── requirements.txt            # Python package dependencies
├── pyproject.toml              # Project configuration and metadata
├── data/
│   ├── BigBasket Products.csv  # Kaggle product catalog snapshot (2022)
│   └── bigbasket_policies.txt  # Ingested return/refund policy documentation
├── chroma_db/                  # Persistent ChromaDB vector store
├── logs/                       # Local audit trails (audit_YYYYMMDD.jsonl)
└── .streamlit/
    └── config.toml             # Theme configuration
```

---

## 📝 Error & Failure Case Analysis

During benchmarking and evaluation, two primary failure modes were analyzed:

1. **Truncated Dataset Descriptions**:
   - *Problem*: Some scraped products feature single-sentence descriptions lacking explicit allergen or dietary tags (e.g. "organic" or "gluten-free").
   - *Mitigation*: Category and product header metadata are prepended directly into each chunk (`Category: Foodgrains, Oil & Masala > Atta`), allowing hybrid BM25 search to capture parent category context even if descriptions are brief.

2. **Conversational Ambiguity in Follow-ups**:
   - *Problem*: Follow-up queries such as *"Give me a cheaper alternative"* contain zero product entities, causing stateless vector retrievers to return random inexpensive items.
   - *Mitigation*: Implemented `contextualize_query()`, which leverages prior chat turns to rewrite the query into an explicit search string (*"What is a cheaper alternative face wash for oily skin under ₹406?"*) before vector retrieval executes.
