"""
app.py — Streamlit chat UI for the BigBasket RAG assistant.

Run:  streamlit run app.py   (or double-click run.command)

Streamlit re-runs this whole script top-to-bottom on every interaction.
So:
  - st.cache_resource keeps the heavy pipeline (model + vector store) alive
    across re-runs instead of reloading it every click.
  - st.session_state keeps the chat history across re-runs.

Design idea: a grocery shopping list + shelf price tags.
  - Theme colours live in .streamlit/config.toml
  - The CSS below adds the fonts, the shopping-list suggestions and the
    price-tag source cards (Streamlit has no built-in widget for these, so a
    small HTML/CSS layer is the standard approach).
"""

import html
import os

import streamlit as st

from rag import FALLBACK_MSG, RAGPipeline, SIM_THRESHOLD, TOP_K

st.set_page_config(page_title="BigBasket Assistant", page_icon="🧺", layout="centered")

# Design tokens — one place to change the look
LEAF, HALDI, INK, MINT, CHILLI = "#2F5D2A", "#E2A31B", "#1E2A1B", "#EEF4E8", "#B4462B"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Instrument+Sans:wght@400;500;600&display=swap');

/* body font everywhere text lives (not on icon elements, which use their own font) */
.stApp, .stMarkdown, .stMarkdown p, .stMarkdown li, button, input, textarea, label {{
  font-family: 'Instrument Sans', system-ui, sans-serif;
}}
h1, h2, h3, .display {{ font-family: 'Bricolage Grotesque', system-ui, sans-serif; }}

/* hide Streamlit's developer chrome */
.stDeployButton, #MainMenu, footer {{ display: none !important; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 3rem; padding-bottom: 7rem; max-width: 760px; }}

/* ---------- top of page ---------- */
.brand {{ display: flex; align-items: center; gap: 10px; color: {LEAF};
         font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 1.05rem; }}
.brand .mark {{ width: 34px; height: 34px; border-radius: 10px; background: {LEAF}; color: white;
               display: grid; place-items: center; font-size: 1.1rem; }}
.ask {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; color: {INK};
       font-size: clamp(2.2rem, 6vw, 3.4rem); line-height: 1.02; letter-spacing: -0.03em;
       margin: 28px 0 12px; }}
.lede {{ color: #4d5a48; font-size: 1.05rem; max-width: 34em; margin-bottom: 26px; }}

/* ---------- suggestions as a shopping list ---------- */
.list-head {{ color: #6b7766; font-size: .9rem; margin: 6px 0 2px; }}
div[data-testid="stMainBlockContainer"] .stButton > button {{
  width: 100%; background: transparent; border: none; border-bottom: 1.5px dashed #CFDCC5;
  border-radius: 0; padding: 14px 4px 14px 40px; justify-content: flex-start; color: {INK};
  position: relative; box-shadow: none;
}}
div[data-testid="stMainBlockContainer"] .stButton > button::before {{
  content: ""; position: absolute; left: 6px; top: 50%; width: 20px; height: 20px;
  transform: translateY(-50%); border: 2px solid {LEAF}; border-radius: 50%;
}}
div[data-testid="stMainBlockContainer"] .stButton > button:hover {{ background: {MINT}; color: {LEAF}; }}
div[data-testid="stMainBlockContainer"] .stButton > button:hover::before {{ background: {LEAF};
  box-shadow: inset 0 0 0 3px white; }}
div[data-testid="stMainBlockContainer"] .stButton > button:focus-visible {{
  outline: 2px solid {LEAF}; outline-offset: 2px; }}
div[data-testid="stMainBlockContainer"] .stButton > button > div,
div[data-testid="stMainBlockContainer"] .stButton > button [data-testid="stMarkdownContainer"] {{
  justify-content: flex-start !important; text-align: left !important; width: 100%;
}}
div[data-testid="stMainBlockContainer"] .stButton > button p,
div[data-testid="stMainBlockContainer"] .stButton > button .stMarkdownEllipsis {{
  white-space: normal !important; overflow: visible !important; text-overflow: clip !important;
  text-align: left; font-size: 1.02rem; line-height: 1.35;
}}
.hint {{ color: #8a9585; font-size: .82rem; margin-left: 40px; margin-top: -6px; }}

/* ---------- chat ---------- */
[data-testid="stChatMessage"] {{ background: transparent; padding: 6px 0; }}
[data-testid="stChatMessageContent"] p {{ font-size: 1.02rem; line-height: 1.6; }}
[data-testid="stChatInput"] {{ border: 2px solid {LEAF}; border-radius: 16px; }}

/* "not in our data" reply */
.nodata {{ border-left: 4px solid {CHILLI}; background: #FBF1EE; padding: 12px 16px;
          border-radius: 4px 12px 12px 4px; color: #5a2a1d; }}
.nodata b {{ color: {CHILLI}; }}

/* ---------- sources as shelf price tags (the signature element) ---------- */
.tags {{ display: flex; flex-wrap: wrap; gap: 14px 12px; padding: 6px 2px 4px; }}
.tag {{
  position: relative; flex: 1 1 210px; max-width: 100%; background: white;
  border: 1.5px solid #D8E3CF; border-radius: 6px 18px 18px 6px;
  padding: 12px 14px 12px 34px;
}}
.tag::before {{  /* the punched hole */
  content: ""; position: absolute; left: 11px; top: 16px; width: 10px; height: 10px;
  border-radius: 50%; background: {MINT}; border: 1.5px solid #B9CBAA;
}}
.tag .n {{ position: absolute; left: 10px; top: 34px; font-size: .72rem; color: #8a9585; }}
.tag .name {{ font-weight: 600; color: {INK}; font-size: .92rem; line-height: 1.3; }}
.tag .brand-l {{ color: #66715f; font-size: .8rem; margin-top: 2px; }}
.tag .row {{ display: flex; align-items: baseline; justify-content: space-between; margin-top: 8px; }}
.tag .price {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 1.25rem;
              color: {INK}; background: {HALDI}; padding: 0 8px; border-radius: 6px; }}
.tag .match {{ font-size: .75rem; color: {LEAF}; }}
.tag details {{ margin-top: 8px; font-size: .82rem; color: #4d5a48; }}
.tag summary {{ cursor: pointer; color: {LEAF}; }}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {{ background: {MINT}; }}
section[data-testid="stSidebar"] h3 {{ font-family: 'Bricolage Grotesque', sans-serif; color: {LEAF}; }}
.key-ok  {{ color: {LEAF}; font-size: .85rem; }}
.key-bad {{ color: {CHILLI}; font-size: .85rem; }}
.facts {{ font-size: .85rem; color: #4d5a48; line-height: 1.7; }}

@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; }} }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load the pipeline ONCE per server process.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Getting the product catalogue ready (first start takes a minute or two)...")
def load_pipeline():
    return RAGPipeline()


def get_api_key():
    """Key priority: sidebar input > Streamlit secrets > env variable."""
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
# SIDEBAR — settings and the facts evaluators ask about
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Settings")
    has_secret_key = False
    try:
        has_secret_key = bool(st.secrets.get("GEMINI_API_KEY"))
    except Exception:
        pass
    if not has_secret_key:   # hosted app with a secret: no need to show the box
        st.text_input("Gemini API key", type="password", key="api_key_input",
                      placeholder="Paste your key",
                      help="Free key from aistudio.google.com")
    if get_api_key():
        st.markdown('<div class="key-ok">● Connected to Gemini</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="key-bad">● Add a Gemini key to start chatting</div>',
                    unsafe_allow_html=True)

    st.divider()
    # Switching modes live SHOWS why hybrid helps (try a brand name in each).
    mode = st.radio("Search method", ["hybrid", "vector", "keyword"], horizontal=True,
                    help="hybrid = meaning-based (vector) + exact-word (BM25) search, merged with Reciprocal Rank Fusion")
    top_k = st.slider("Products to look at per question", 2, 10, TOP_K)
    show_ctx = st.toggle("Show the products used", value=True)

    st.divider()
    st.markdown(f"""<div class="facts">
{n_products:,} products, {n_chunks:,} searchable chunks<br>
Source: Kaggle BigBasket catalogue (2022 snapshot)<br>
Prices are from the dataset, not live
</div>""", unsafe_allow_html=True)

    with st.expander("How it works"):
        st.markdown(
            "**Data → Chunking → Embeddings → Vector store → Retrieval → LLM → Answer**\n\n"
            "1. Kaggle BigBasket catalogue, cleaned (empty and duplicate rows removed)\n"
            "2. One document per product; long descriptions split (800 chars, 100 overlap), "
            "product details repeated on every chunk\n"
            "3. `all-MiniLM-L6-v2` embeddings\n"
            "4. Stored in ChromaDB (cosine)\n"
            "5. Hybrid retrieval: vector + BM25, merged with RRF\n"
            "6. Gemini answers only from retrieved products, citing them as [1], [2]\n"
            f"7. If the best match is below {SIM_THRESHOLD:.0%} similarity, it says the data doesn't cover it"
        )

    if st.session_state.messages and st.button("Start a new list", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def render_tags(chunks, best_sim):
    """Retrieved products as shelf price tags.
    WHY show them: the brief asks you to check that retrieved chunks contain the
    evidence; this makes retrieval visible during the demo."""
    tags = []
    for n, c in enumerate(chunks, 1):
        m = c["metadata"]
        desc = c["text"].split("Description:", 1)[-1].strip()
        price = f"₹{m['price']:,.0f}" if m.get("price", -1) >= 0 else "Policy"
        match = (f"{c['similarity']:.0%} match" if c["similarity"] is not None
                 else "word match")
        tags.append(f"""
<div class="tag">
  <span class="n">{n}</span>
  <div class="name">{html.escape(m['product'])}</div>
  <div class="brand-l">{html.escape(m['brand'])}, {html.escape(m['category'])}</div>
  <div class="row"><span class="price">{price}</span><span class="match">{match}</span></div>
  <details><summary>Description</summary>{html.escape(desc[:420])}{'…' if len(desc) > 420 else ''}</details>
</div>""")
    with st.expander(f"Products used for this answer ({len(chunks)}), closest match {best_sim:.0%}"):
        st.markdown(f'<div class="tags">{"".join(tags)}</div>', unsafe_allow_html=True)


def render_nodata():
    st.markdown(f'<div class="nodata"><b>Not in the catalogue.</b> {html.escape(FALLBACK_MSG)}</div>',
                unsafe_allow_html=True)


AVATARS = {"user": "🙂", "assistant": "🧺"}

# ---------------------------------------------------------------------------
# TOP OF PAGE: big question on an empty chat, small brand line after
# ---------------------------------------------------------------------------
st.markdown('<div class="brand"><span class="mark">🧺</span>BigBasket Assistant</div>',
            unsafe_allow_html=True)

SUGGESTIONS = [
    ("Suggest a healthy breakfast cereal under ₹300", None),
    ("Compare Tata Sampann and Fortune toor dal", None),
    ("Which face wash suits oily skin, and what does it cost?", None),
    ("What is BigBasket's delivery time in Gurgaon?",
     "Not in the catalogue, so the assistant should say it doesn't know"),
]

if not st.session_state.messages:
    st.markdown('<div class="ask">What’s on your list today?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="lede">Ask about groceries, brands, prices or ingredients. '
        'Answers come only from the BigBasket product catalogue, with the products shown '
        'so you can check them.</div>', unsafe_allow_html=True)
    st.markdown('<div class="list-head">Try one of these</div>', unsafe_allow_html=True)
    for i, (q, hint) in enumerate(SUGGESTIONS):
        if st.button(q, key=f"sugg{i}", use_container_width=True):
            st.session_state.pending_question = q
            st.rerun()
        if hint:
            st.markdown(f'<div class="hint">{hint}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# CHAT HISTORY (Streamlit forgets the screen on each re-run, so redraw it)
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
        if msg.get("fallback"):
            render_nodata()
        else:
            st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_ctx and msg.get("chunks"):
            render_tags(msg["chunks"], msg["best_sim"])

# ---------------------------------------------------------------------------
# NEW QUESTION: typed, or from the shopping list
# ---------------------------------------------------------------------------
question = st.chat_input("Ask about a product, brand or price")
if not question:
    question = st.session_state.pop("pending_question", None)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(question)

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        api_key = get_api_key()
        if not api_key:
            st.warning("Add a Gemini API key in the sidebar to get answers.", icon="🔑")
            st.session_state.messages.pop()   # don't keep an unanswered question
            st.stop()

        with st.spinner("Checking the catalogue..."):
            try:
                result = pipeline.answer(question, api_key, k=top_k, mode=mode)
            except Exception as e:  # e.g. bad API key / no internet
                st.error(f"Couldn't get an answer: {e}. Check the API key and try again.")
                st.stop()

        content = FALLBACK_MSG if result["fallback"] else result["answer"]
        if result["fallback"]:
            render_nodata()
        else:
            st.markdown(content)
        if show_ctx:
            render_tags(result["chunks"], result["best_sim"])

    st.session_state.messages.append({
        "role": "assistant", "content": content, "fallback": result["fallback"],
        "chunks": result["chunks"], "best_sim": result["best_sim"],
    })
    st.rerun()   # redraw so the big heading collapses to the brand line
