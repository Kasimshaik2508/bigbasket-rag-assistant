"""
app.py — Streamlit chat UI for the BigBasket RAG assistant.

Run:  streamlit run app.py   (or double-click run.command)

Streamlit re-runs this whole script top-to-bottom on every interaction.
So:
  - st.cache_resource keeps the heavy pipeline (model + vector store) alive
    across re-runs instead of reloading it every click.
  - st.session_state keeps the chat history across re-runs.

Look & feel:
  - Theme colours (BigBasket green) live in .streamlit/config.toml
  - The small CSS block below styles the header banner, suggestion cards and
    source cards. Streamlit has no built-in "card" widget, so a little HTML/CSS
    is the standard way to get a polished look.
"""

import html
import os

import streamlit as st

from rag import FALLBACK_MSG, RAGPipeline, SIM_THRESHOLD, TOP_K

st.set_page_config(page_title="BigBasket Product Assistant", page_icon="🛒",
                   layout="centered")

GREEN = "#689F38"
LIGHT = "#F3F8EC"

# ---------------------------------------------------------------------------
# CSS — kept in one place so it's easy to tweak
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
/* hide Streamlit's developer chrome (Deploy button, menu, footer) */
.stDeployButton, #MainMenu, footer {{ display: none !important; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2rem; padding-bottom: 6rem; max-width: 820px; }}

/* header banner */
.hero {{
  background: linear-gradient(120deg, #84C225 0%, #4E7D2A 100%);
  border-radius: 18px; padding: 26px 30px; color: white; margin-bottom: 18px;
  box-shadow: 0 6px 20px rgba(78,125,42,.25);
}}
.hero h1 {{ color: white; font-size: 1.9rem; margin: 0 0 6px 0; padding: 0; }}
.hero p  {{ margin: 0; opacity: .92; font-size: .98rem; }}
.pills {{ margin-top: 14px; display: flex; gap: 8px; flex-wrap: wrap; }}
.pill {{ background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.35);
        padding: 4px 12px; border-radius: 999px; font-size: .8rem; }}

/* welcome text above suggestion buttons */
.welcome {{ color: #4a5a40; font-size: 1.05rem; font-weight: 600; margin: 8px 0 10px; }}

/* suggestion buttons in the main area look like cards */
div[data-testid="stMainBlockContainer"] .stButton > button {{
  background: white; border: 1px solid #DDE8CF; border-radius: 14px;
  padding: 14px 16px; min-height: 78px; text-align: left; color: #2d3a24;
  justify-content: flex-start; transition: all .15s ease;
}}
div[data-testid="stMainBlockContainer"] .stButton > button p,
div[data-testid="stMainBlockContainer"] .stButton > button .stMarkdownEllipsis {{
  white-space: normal !important; overflow: visible !important;
  text-overflow: clip !important; text-align: left; line-height: 1.35;
}}
div[data-testid="stMainBlockContainer"] .stButton > button:hover {{
  border-color: {GREEN}; background: {LIGHT}; transform: translateY(-1px);
}}

/* source cards inside the "Retrieved context" expander */
.src {{ border: 1px solid #E3ECD8; border-left: 4px solid {GREEN}; border-radius: 10px;
       padding: 10px 14px; margin-bottom: 10px; background: #FBFDF8; }}
.src-title {{ font-weight: 700; color: #26331d; }}
.src-meta {{ font-size: .8rem; color: #667060; margin: 2px 0 6px; }}
.badge {{ display: inline-block; background: {LIGHT}; color: #3e6b1f; border-radius: 6px;
         padding: 1px 8px; font-size: .75rem; font-weight: 600; margin-right: 6px; }}
.src-desc {{ font-size: .85rem; color: #3b4634; }}

/* sidebar */
section[data-testid="stSidebar"] {{ background: {LIGHT}; }}
.key-ok  {{ color: #3e6b1f; font-size: .85rem; }}
.key-bad {{ color: #b45309; font-size: .85rem; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load the pipeline ONCE per server process.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading products and building the search index...")
def load_pipeline():
    return RAGPipeline()


def get_api_key():
    """Key priority: sidebar input > .streamlit/secrets.toml > env variable."""
    typed = st.session_state.get("api_key_input", "")
    if typed:
        return typed
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.getenv("GEMINI_API_KEY", "")


pipeline = load_pipeline()
n_products = len({m["product"] for m in pipeline.metas})
n_chunks = len(pipeline.ids)

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# SIDEBAR — settings only (kept short so the demo looks clean)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛒 BigBasket Assistant")
    st.text_input("Gemini API key", type="password", key="api_key_input",
                  placeholder="Paste your key here",
                  help="Free key from aistudio.google.com, or set GEMINI_API_KEY in .streamlit/secrets.toml")
    if get_api_key():
        st.markdown('<div class="key-ok">● API key connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="key-bad">● No API key yet</div>', unsafe_allow_html=True)

    st.divider()
    # Letting evaluators switch modes live SHOWS why hybrid retrieval helps
    # (try a brand name in 'vector' vs 'hybrid').
    mode = st.radio("Retrieval mode", ["hybrid", "vector", "keyword"],
                    horizontal=True,
                    help="hybrid = semantic vectors + BM25 keywords, merged with Reciprocal Rank Fusion")
    top_k = st.slider("Chunks to retrieve (k)", 2, 10, TOP_K)
    show_ctx = st.toggle("Show retrieved sources", value=True)

    st.divider()
    if st.button("🗑️  New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    with st.expander("How it works"):
        st.markdown(
            "**Data → Chunking → Embeddings → Vector store → Retrieval → LLM → Answer**\n\n"
            "1. Kaggle BigBasket catalogue, cleaned (empty/duplicate rows removed)\n"
            "2. One document per product; long descriptions split (800 chars, 100 overlap), "
            "product header repeated on every chunk\n"
            "3. `all-MiniLM-L6-v2` embeddings\n"
            "4. Stored in ChromaDB (cosine)\n"
            "5. Hybrid retrieval: vector + BM25, fused with RRF\n"
            "6. Gemini answers only from retrieved chunks, with [n] citations\n"
            f"7. Fallback if best similarity < {SIM_THRESHOLD}"
        )

# ---------------------------------------------------------------------------
# HEADER BANNER
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="hero">
  <h1>🛒 BigBasket Product Assistant</h1>
  <p>Find groceries, compare brands and check prices — every answer is grounded in the BigBasket catalogue.</p>
  <div class="pills">
    <span class="pill">📦 {n_products:,} products</span>
    <span class="pill">🧩 {n_chunks:,} chunks</span>
    <span class="pill">🔎 {mode.capitalize()} retrieval</span>
    <span class="pill">🕒 Catalogue snapshot, not live stock</span>
  </div>
</div>
""", unsafe_allow_html=True)


def render_sources(chunks, best_sim):
    """Show what the retriever fed the LLM, as readable cards.
    WHY: the brief asks you to check that retrieved chunks contain the evidence;
    this makes that visible live in the demo."""
    with st.expander(f"📚 Sources used · {len(chunks)} chunks · best match {best_sim:.0%}"):
        for n, c in enumerate(chunks, 1):
            m = c["metadata"]
            desc = c["text"].split("Description:", 1)[-1].strip()
            desc = html.escape(desc[:260] + ("…" if len(desc) > 260 else ""))
            price = f"₹{m['price']:,.0f}" if m.get("price", -1) >= 0 else ""
            sim = (f"{c['similarity']:.0%} match" if c["similarity"] is not None
                   else "keyword match")
            st.markdown(f"""
<div class="src">
  <div class="src-title">[{n}] {html.escape(m['product'])}</div>
  <div class="src-meta">
    <span class="badge">{html.escape(m['brand'])}</span>
    <span class="badge">{html.escape(m['category'])}</span>
    {f'<span class="badge">{price}</span>' if price else ''}
    <span>{sim}</span>
  </div>
  <div class="src-desc">{desc}</div>
</div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# WELCOME SCREEN — suggestion cards when the chat is empty
# ---------------------------------------------------------------------------
SUGGESTIONS = [
    ("🥣", "Suggest a healthy breakfast cereal under ₹300"),
    ("⚖️", "Compare Tata Sampann and Fortune toor dal"),
    ("🧴", "Which face wash is good for oily skin and what does it cost?"),
    ("🚚", "What is BigBasket's delivery time in Gurgaon?"),   # fallback test
]

if not st.session_state.messages:
    st.markdown('<div class="welcome">What can I help you find today?</div>',
                unsafe_allow_html=True)
    cols = st.columns(2)
    for i, (icon, q) in enumerate(SUGGESTIONS):
        if cols[i % 2].button(f"{icon}  {q}", key=f"sugg{i}", use_container_width=True):
            st.session_state.pending_question = q
            st.rerun()

# ---------------------------------------------------------------------------
# CHAT HISTORY (Streamlit forgets the screen on each re-run, so redraw it)
# ---------------------------------------------------------------------------
AVATARS = {"user": "🧑", "assistant": "🛒"}

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
        if msg.get("fallback"):
            st.info(msg["content"], icon="🤔")
        else:
            st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_ctx and msg.get("chunks"):
            render_sources(msg["chunks"], msg["best_sim"])

# ---------------------------------------------------------------------------
# NEW QUESTION: typed, or from a suggestion card
# ---------------------------------------------------------------------------
question = st.chat_input("Ask about any BigBasket product, brand or price...")
if not question:
    question = st.session_state.pop("pending_question", None)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(question)

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        api_key = get_api_key()
        if not api_key:
            st.warning("Add your Gemini API key in the sidebar first.", icon="🔑")
            st.session_state.messages.pop()   # don't keep an unanswered question
            st.stop()

        with st.spinner("Searching the catalogue..."):
            try:
                result = pipeline.answer(question, api_key, k=top_k, mode=mode)
            except Exception as e:  # e.g. bad API key / no internet
                st.error(f"Something went wrong: {e}")
                st.stop()

        content = FALLBACK_MSG if result["fallback"] else result["answer"]
        if result["fallback"]:
            st.info(content, icon="🤔")
        else:
            st.markdown(content)
        if show_ctx:
            render_sources(result["chunks"], result["best_sim"])

    st.session_state.messages.append({
        "role": "assistant", "content": content, "fallback": result["fallback"],
        "chunks": result["chunks"], "best_sim": result["best_sim"],
    })
