"""
app.py — Streamlit chat UI for the BigBasket RAG assistant.

Run:  streamlit run app.py   (or double-click run.command)

Streamlit re-runs this whole script top-to-bottom on every interaction.
So:
  - st.cache_resource keeps the heavy pipeline (model + vector store) alive
    across re-runs instead of reloading it every click.
  - st.session_state keeps the chat history across re-runs.

Design idea: a grocery shopping list + a shelf of price tags.
  - Opening screen: the question + a shopping list of example questions (left),
    a "shelf" with three REAL products from the dataset hanging as price tags
    (right), and a row of real aisles (categories) with product counts.
  - Answers show their retrieved products as the same price tags.
  - Theme colours live in .streamlit/config.toml; the CSS below adds fonts,
    the list, the shelf and the tags. st.container(key=...) gives each block
    a CSS class (.st-key-<key>) so styles don't leak between sections.
"""

import html
import os
import random
from collections import Counter

import streamlit as st

from rag import FALLBACK_MSG, RAGPipeline, SIM_THRESHOLD, TOP_K

st.set_page_config(page_title="BigBasket Assistant", page_icon="🧺", layout="wide")

# Design tokens — one place to change the look
LEAF, HALDI, INK, MINT, CHILLI = "#2F5D2A", "#E2A31B", "#1E2A1B", "#EEF4E8", "#B4462B"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Instrument+Sans:wght@400;500;600&display=swap');

.stApp, .stMarkdown, .stMarkdown p, .stMarkdown li, button, input, textarea, label {{
  font-family: 'Instrument Sans', system-ui, sans-serif;
}}

/* hide Streamlit's developer chrome */
.stDeployButton, #MainMenu, footer {{ display: none !important; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2.4rem; padding-bottom: 7rem; max-width: 1120px; }}

/* ---------- brand line ---------- */
.brand {{ display: flex; align-items: center; gap: 10px; color: {LEAF};
         font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 1.05rem; }}
.brand .mark {{ width: 34px; height: 34px; border-radius: 10px; background: {LEAF};
               display: grid; place-items: center; font-size: 1.05rem; }}

/* ---------- opening screen ---------- */
.ask {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; color: {INK};
       font-size: clamp(2.4rem, 5vw, 3.9rem); line-height: 1; letter-spacing: -0.035em;
       margin: 30px 0 14px; max-width: 9.5em; }}
.lede {{ color: #4d5a48; font-size: 1.08rem; line-height: 1.55; max-width: 30em; margin-bottom: 22px; }}
.small-head {{ color: #6b7766; font-size: .9rem; margin: 4px 0 2px; }}

/* shopping-list suggestions */
.st-key-shoplist .stButton > button {{
  width: 100%; background: transparent; border: none; border-bottom: 1.5px dashed #CFDCC5;
  border-radius: 0; padding: 13px 4px 13px 40px; color: {INK}; position: relative; box-shadow: none;
}}
.st-key-shoplist .stButton > button::before {{
  content: ""; position: absolute; left: 6px; top: 50%; width: 20px; height: 20px;
  transform: translateY(-50%); border: 2px solid {LEAF}; border-radius: 50%; background: white;
}}
.st-key-shoplist .stButton > button:hover {{ background: {MINT}; color: {LEAF}; }}
.st-key-shoplist .stButton > button:hover::before {{ background: {LEAF}; box-shadow: inset 0 0 0 3px white; }}
.st-key-shoplist .stButton > button > div,
.st-key-shoplist .stButton > button [data-testid="stMarkdownContainer"] {{
  justify-content: flex-start !important; text-align: left !important; width: 100%;
}}
.st-key-shoplist .stButton > button p,
.st-key-shoplist .stButton > button .stMarkdownEllipsis {{
  white-space: normal !important; overflow: visible !important; text-overflow: clip !important;
  text-align: left; font-size: 1.02rem; line-height: 1.35;
}}
.hint {{ color: #8a9585; font-size: .82rem; margin: -4px 0 0 40px; }}

/* the shelf (right column) */
.shelf {{ position: relative; background: {MINT}; border-radius: 22px; padding: 30px 26px 26px;
         margin-top: 34px; overflow: hidden; }}
.shelf .rail {{ height: 10px; border-radius: 6px; background: {LEAF}; margin: 0 -6px 0; }}
.shelf .hang {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }}
.shelf .slot {{ display: flex; flex-direction: column; align-items: center; }}
.shelf .string {{ width: 2px; height: 40px; background: #9DB58C; }}
.shelf .slot:nth-child(2) .string {{ height: 84px; }}
.shelf .slot:nth-child(3) .string {{ height: 56px; }}
.shelf .slot:nth-child(1) .tag {{ transform: rotate(-3deg); }}
.shelf .slot:nth-child(3) .tag {{ transform: rotate(2.5deg); }}
.shelf .tag {{ width: 100%; transform-origin: top center; padding: 30px 14px 16px; }}
.shelf .tag .name {{ font-size: 1rem; }}
.shelf .tag .price {{ font-size: 1.45rem; }}
.shelf .caption {{ color: #5c6a55; font-size: .88rem; margin-top: 28px; line-height: 1.5; }}

/* price tag (used on the shelf AND under answers) */
.tag {{ position: relative; background: white; border: 1.5px solid #D2DFC7;
       border-radius: 8px 8px 16px 16px; padding: 26px 12px 12px; box-shadow: 0 2px 0 #D2DFC7; }}
.tag::before {{ content: ""; position: absolute; left: 50%; top: 9px; width: 10px; height: 10px;
               transform: translateX(-50%); border-radius: 50%; background: {MINT};
               border: 1.5px solid #B9CBAA; }}
.tag .name {{ font-weight: 600; color: {INK}; font-size: .9rem; line-height: 1.3;
             display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }}
.tag .brand-l {{ color: #66715f; font-size: .78rem; margin-top: 3px; }}
.tag .row {{ display: flex; align-items: baseline; justify-content: space-between; gap: 6px;
            margin-top: 10px; flex-wrap: wrap; }}
.tag .price {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 1.25rem;
              color: {INK}; background: {HALDI}; padding: 0 8px; border-radius: 6px; }}
.tag .mrp {{ font-size: .75rem; color: #8a9585; text-decoration: line-through; }}
.tag .match {{ font-size: .75rem; color: {LEAF}; }}
.tag .n {{ position: absolute; top: 6px; left: 10px; font-size: .72rem; color: #8a9585; }}
.tag details {{ margin-top: 8px; font-size: .82rem; color: #4d5a48; }}
.tag summary {{ cursor: pointer; color: {LEAF}; }}
.tags {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 14px; padding: 4px 2px; }}

/* aisles row */
.st-key-aisles {{ margin-top: 30px; padding-top: 22px; border-top: 1.5px solid #E1EAD8; }}
.aisle-head {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; color: {INK};
              font-size: 1.25rem; margin-bottom: 2px; }}
.aisle-sub {{ color: #6b7766; font-size: .9rem; margin-bottom: 10px; }}
.st-key-aisles .stButton > button {{
  width: 100%; border: 1.5px solid #D2DFC7; background: white; border-radius: 14px;
  padding: 12px 14px; min-height: 64px; color: {INK}; box-shadow: none;
}}
.st-key-aisles .stButton > button:hover {{ border-color: {LEAF}; background: {MINT}; }}
.st-key-aisles .stButton > button > div,
.st-key-aisles .stButton > button [data-testid="stMarkdownContainer"] {{
  justify-content: flex-start !important; text-align: left !important; width: 100%;
}}
.st-key-aisles .stButton > button p {{ white-space: normal !important; text-align: left; line-height: 1.3; }}

/* visible keyboard focus on every custom button */
.stButton > button:focus-visible {{ outline: 2px solid {LEAF}; outline-offset: 2px; }}

/* ---------- chat ---------- */
.st-key-chat {{ max-width: 800px; margin: 0 auto; }}
[data-testid="stChatMessage"] {{ background: transparent; padding: 6px 0; }}
[data-testid="stChatMessageContent"] p {{ font-size: 1.03rem; line-height: 1.6; }}
[data-testid="stChatInput"] {{ border: 2px solid {LEAF}; border-radius: 16px; }}
[data-testid="stBottomBlockContainer"] {{ max-width: 840px; }}
.nodata {{ border-left: 4px solid {CHILLI}; background: #FBF1EE; padding: 12px 16px;
          border-radius: 4px 12px 12px 4px; color: #5a2a1d; }}
.nodata b {{ color: {CHILLI}; }}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {{ background: {MINT}; }}
section[data-testid="stSidebar"] h3 {{ font-family: 'Bricolage Grotesque', sans-serif; color: {LEAF}; }}
.key-ok  {{ color: {LEAF}; font-size: .85rem; }}
.key-bad {{ color: {CHILLI}; font-size: .85rem; }}
.facts {{ font-size: .85rem; color: #4d5a48; line-height: 1.7; }}

@media (max-width: 760px) {{
  .shelf {{ min-height: 0; margin-top: 10px; }}
  .shelf .hang {{ grid-template-columns: 1fr 1fr; }}
  .shelf .slot:nth-child(3) {{ display: none; }}
}}
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
product_metas = [m for m in pipeline.metas if m.get("source") == "product" and m.get("chunk") == 0]
n_products = len(product_metas)
n_chunks = len(pipeline.ids)
aisles = Counter(m["category"] for m in product_metas).most_common(8)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "shelf" not in st.session_state:
    # three real products for the shelf, changing per visit; prefer short names
    pool = [m for m in product_metas if 0 < m["price"] < 1500 and len(m["product"]) < 48]
    st.session_state.shelf = random.sample(pool, 3) if len(pool) >= 3 else pool[:3]

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
                      placeholder="Paste your key", help="Free key from aistudio.google.com")
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
def tag_html(m, n=None, match=None, desc=None):
    """One price tag. Used on the shelf and for retrieved products."""
    price = f"₹{m['price']:,.0f}" if m.get("price", -1) >= 0 else "Policy"
    parts = [f'<div class="tag">']
    if n is not None:
        parts.append(f'<span class="n">{n}</span>')
    parts.append(f'<div class="name">{html.escape(m["product"])}</div>')
    parts.append(f'<div class="brand-l">{html.escape(m["brand"])}</div>')
    parts.append(f'<div class="row"><span class="price">{price}</span>')
    if match:
        parts.append(f'<span class="match">{match}</span>')
    parts.append('</div>')
    if desc:
        short = html.escape(desc[:420]) + ("…" if len(desc) > 420 else "")
        parts.append(f'<details><summary>Description</summary>{short}</details>')
    parts.append('</div>')
    return "".join(parts)


def render_tags(chunks, best_sim):
    """Retrieved products as price tags.
    WHY show them: the brief asks you to check that retrieved chunks contain the
    evidence; this makes retrieval visible during the demo."""
    tags = []
    for n, c in enumerate(chunks, 1):
        match = (f"{c['similarity']:.0%} match" if c["similarity"] is not None else "word match")
        desc = c["text"].split("Description:", 1)[-1].strip()
        tags.append(tag_html(c["metadata"], n=n, match=match, desc=desc))
    with st.expander(f"Products used for this answer ({len(chunks)}), closest match {best_sim:.0%}"):
        st.markdown(f'<div class="tags">{"".join(tags)}</div>', unsafe_allow_html=True)


def render_nodata():
    st.markdown(f'<div class="nodata"><b>Not in the catalogue.</b> {html.escape(FALLBACK_MSG)}</div>',
                unsafe_allow_html=True)


def ask(q):
    st.session_state.pending_question = q
    st.rerun()


AVATARS = {"user": "🙂", "assistant": "🧺"}
AISLE_ICONS = {
    "Beauty & Hygiene": "🧴", "Gourmet & World Food": "🫒", "Snacks & Branded Foods": "🍪",
    "Foodgrains, Oil & Masala": "🌾", "Cleaning & Household": "🧽", "Kitchen, Garden & Pets": "🍳",
    "Bakery, Cakes & Dairy": "🥛", "Beverages": "☕", "Baby Care": "🍼",
    "Fruits & Vegetables": "🥕", "Eggs, Meat & Fish": "🥚",
}

st.markdown('<div class="brand"><span class="mark">🧺</span>BigBasket Assistant</div>',
            unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# OPENING SCREEN (only while the chat is empty)
# ---------------------------------------------------------------------------
SUGGESTIONS = [
    ("Suggest a healthy breakfast cereal under ₹300", None),
    ("Compare Tata Sampann and Fortune toor dal", None),
    ("Which face wash suits oily skin, and what does it cost?", None),
    ("What is BigBasket's delivery time in Gurgaon?",
     "Not in the catalogue, so the assistant should say it doesn't know"),
]

if not st.session_state.messages:
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown('<div class="ask">What’s on your list today?</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="lede">Ask about groceries, brands, prices or ingredients. '
            'Every answer comes from the BigBasket product catalogue, and shows the '
            'products it used so you can check it.</div>', unsafe_allow_html=True)
        st.markdown('<div class="small-head">Try one of these</div>', unsafe_allow_html=True)
        with st.container(key="shoplist"):
            for i, (q, hint) in enumerate(SUGGESTIONS):
                if st.button(q, key=f"sugg{i}", use_container_width=True):
                    ask(q)
                if hint:
                    st.markdown(f'<div class="hint">{hint}</div>', unsafe_allow_html=True)

    with right:
        slots = "".join(f'<div class="slot"><div class="string"></div>{tag_html(m)}</div>'
                        for m in st.session_state.shelf)
        st.markdown(f"""
<div class="shelf">
  <div class="rail"></div>
  <div class="hang">{slots}</div>
  <div class="caption">Three of the {n_products:,} products the assistant can search right now.
  Prices are from the catalogue snapshot.</div>
</div>""", unsafe_allow_html=True)

    with st.container(key="aisles"):
        st.markdown('<div class="aisle-head">Browse by aisle</div>'
                    '<div class="aisle-sub">Pick an aisle to see what the assistant finds there.</div>',
                    unsafe_allow_html=True)
        cols = st.columns(4)
        for i, (cat, count) in enumerate(aisles):
            label = f"{AISLE_ICONS.get(cat, '🛒')}  {cat}  \n{count:,} products"
            if cols[i % 4].button(label, key=f"aisle{i}", use_container_width=True):
                ask(f"What are some popular {cat} products under ₹250?")

# ---------------------------------------------------------------------------
# CHAT (history + new answer), in a centred readable column
# ---------------------------------------------------------------------------
question = st.chat_input("Ask about a product, brand or price")
if not question:
    question = st.session_state.pop("pending_question", None)

with st.container(key="chat"):
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
            if msg.get("fallback"):
                render_nodata()
            else:
                st.markdown(msg["content"])
            if msg["role"] == "assistant" and show_ctx and msg.get("chunks"):
                render_tags(msg["chunks"], msg["best_sim"])

    if question:
        api_key = get_api_key()
        if not api_key:
            st.warning("Add a Gemini API key in the sidebar to get answers.", icon="🔑")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar=AVATARS["user"]):
            st.markdown(question)

        with st.chat_message("assistant", avatar=AVATARS["assistant"]):
            with st.spinner("Checking the catalogue..."):
                try:
                    result = pipeline.answer(question, api_key, k=top_k, mode=mode)
                except Exception as e:  # e.g. bad API key / no internet
                    st.session_state.messages.pop()
                    st.error(f"Couldn't get an answer from Gemini: {e}")
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
        st.rerun()   # redraw without the opening screen
