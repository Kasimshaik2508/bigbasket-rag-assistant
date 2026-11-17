"""
app.py — Streamlit chat UI for the BigBasket RAG assistant.

Run: streamlit run app.py

Layout follows familiar online-grocery patterns (header bar, aisle bar,
product cards with discount badge and struck-through MRP) so it feels like
shopping, while keeping our own name and look: the data is BigBasket's
public Kaggle catalogue, but this app is not BigBasket.
"""

import html
import os
import random
from collections import Counter

import streamlit as st

from product_art import parse_header, pick_art

from rag import FALLBACK_MSG, GUARDRAIL_MSG, RAGPipeline, SIM_THRESHOLD, TOP_K

st.set_page_config(page_title="Basket Assistant", page_icon="🧺", layout="wide")

# Initialize theme state
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

dark = st.session_state.dark_mode

# ---------------------------------------------------------------------------
# DESIGN TOKENS
# Store-style layout (white page, green accents, product cards with discount
# badges and struck-through MRP) in our own identity — no BigBasket logo or
# trademarked branding, since this is a student project on their public data.
# ---------------------------------------------------------------------------
if dark:
    BG_PAGE, BG_CARD, BG_SOFT, BG_SIDEBAR = "#0F130E", "#171C16", "#1D241B", "#131812"
    BORDER, BORDER_SOFT = "#2A3527", "#212A1F"
    TEXT, TEXT_MUTED = "#ECF0E9", "#97A392"
    GREEN, GREEN_DEEP, GREEN_SOFT = "#7BC043", "#A6D67A", "#22301C"
    BADGE_BG, BADGE_TEXT = "#4E8C1E", "#FFFFFF"
    TILE_DARK, TILE_TEXT = "#2B4A1A", "#F1F7EA"
    NODATA_BG, NODATA_BORDER, NODATA_TEXT = "#2A1612", "#B4462B", "#F0B3A6"
else:
    BG_PAGE, BG_CARD, BG_SOFT, BG_SIDEBAR = "#F6F7F4", "#FFFFFF", "#F3F6EF", "#F3F6EF"
    BORDER, BORDER_SOFT = "#E3E8DE", "#EDF0E9"
    TEXT, TEXT_MUTED = "#1C2118", "#6B7266"
    GREEN, GREEN_DEEP, GREEN_SOFT = "#4E8C1E", "#2E5A10", "#EDF5E4"
    BADGE_BG, BADGE_TEXT = "#3F7D17", "#FFFFFF"
    TILE_DARK, TILE_TEXT = "#24420F", "#FFFFFF"
    NODATA_BG, NODATA_BORDER, NODATA_TEXT = "#FDF2EF", "#B4462B", "#602619"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

.stApp, .stMarkdown, .stMarkdown p, .stMarkdown li, button, input, textarea, label,
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{
  font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
}}
.stApp {{ background: {BG_PAGE}; color: {TEXT}; }}
.stDeployButton, #MainMenu, footer {{ display: none !important; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.2rem; padding-bottom: 7rem; max-width: 1180px; }}

/* ---------- header bar ---------- */
.hdr {{ display: flex; align-items: center; gap: 16px; background: {BG_CARD};
       border: 1px solid {BORDER}; border-top: 3px solid {GREEN}; border-radius: 0 0 16px 16px;
       padding: 14px 20px; margin: -6px 0 12px; }}
.hdr .logo {{ width: 44px; height: 44px; border-radius: 12px; background: {GREEN};
             display: grid; place-items: center; font-size: 1.45rem; flex: none; }}
.hdr .title {{ font-weight: 800; font-size: 1.25rem; color: {TEXT}; letter-spacing: -0.01em; line-height: 1.1; }}
.hdr .sub {{ font-size: .8rem; color: {TEXT_MUTED}; margin-top: 2px; }}
.hdr .askbox {{ flex: 1; margin: 0 8px; border: 1.5px solid {BORDER}; border-radius: 10px;
               padding: 10px 14px; color: {TEXT_MUTED}; font-size: .92rem; background: {BG_SOFT}; }}
.hdr .chip {{ flex: none; background: {GREEN_SOFT}; color: {GREEN_DEEP}; border-radius: 10px;
             padding: 8px 12px; font-size: .8rem; font-weight: 700; line-height: 1.25; }}
.hdr .chip span {{ display: block; font-weight: 500; color: {TEXT_MUTED}; }}
.hdr .chip.off {{ background: {NODATA_BG}; color: {NODATA_BORDER}; }}

/* ---------- aisle bar ---------- */
.st-key-aislebar {{ background: {BG_CARD}; border: 1px solid {BORDER}; border-radius: 14px;
                   padding: 8px 10px; margin-bottom: 18px; }}
.st-key-aislebar [data-testid="stHorizontalBlock"] {{ gap: 6px; align-items: center; }}
.aisle-label {{ background: {GREEN}; color: white; font-weight: 700; border-radius: 9px;
               padding: 9px 12px; font-size: .9rem; text-align: center; white-space: nowrap; }}
.st-key-aislebar .stButton > button {{ background: transparent; border: none; box-shadow: none;
  color: {TEXT}; font-weight: 600; padding: 8px 4px; min-height: 0; }}
.st-key-aislebar .stButton > button:hover {{ color: {GREEN}; background: {GREEN_SOFT}; }}
.st-key-aislebar .stButton > button p {{ font-size: .88rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}

/* ---------- quick-question tiles ---------- */
.st-key-tiles .stButton > button {{ width: 100%; min-height: 78px; border-radius: 14px;
  border: 1px solid {BORDER}; background: {BG_CARD}; color: {TEXT}; box-shadow: none; padding: 12px 14px; }}
.st-key-tiles .stButton > button p {{ font-weight: 800; font-size: .95rem; line-height: 1.25;
  white-space: normal !important; text-align: center; }}
.st-key-tiles .stButton > button:hover {{ border-color: {GREEN}; color: {GREEN_DEEP}; }}
.st-key-tile0 .stButton > button {{ background: {TILE_DARK}; color: {TILE_TEXT}; border-color: {TILE_DARK}; }}
.st-key-tile0 .stButton > button:hover {{ color: {TILE_TEXT}; filter: brightness(1.12); }}

/* ---------- section panel (like a store shelf row) ---------- */
.panel-head {{ display: flex; align-items: baseline; justify-content: space-between; margin: 22px 0 10px; }}
.panel-head h3 {{ font-size: 1.35rem; font-weight: 800; color: {TEXT}; margin: 0; padding: 0; }}
.panel-head span {{ color: {TEXT_MUTED}; font-size: .85rem; }}

/* ---------- product card ---------- */
.pcard {{ position: relative; background: {BG_CARD}; border: 1px solid {BORDER};
         border-radius: 14px; padding: 10px 10px 12px; }}
.pcard .img {{ position: relative; aspect-ratio: 4 / 3; border-radius: 10px; border: 1px solid {BORDER_SOFT};
              display: flex; align-items: center; justify-content: center; font-size: 3.6rem; line-height: 1; }}
.pcard .badge {{ position: absolute; top: 0; left: 0; background: {BADGE_BG}; color: {BADGE_TEXT};
                font-size: .72rem; font-weight: 800; padding: 4px 9px; border-radius: 10px 0 10px 0; }}
.pcard .rank {{ position: absolute; top: 8px; right: 8px; background: {BG_CARD}; color: {TEXT_MUTED};
               border: 1px solid {BORDER}; font-size: .7rem; font-weight: 700; padding: 1px 7px; border-radius: 999px; }}
.pcard .meta {{ display: flex; justify-content: space-between; align-items: center; margin-top: 10px; gap: 6px; }}
.pcard .brand {{ color: {TEXT_MUTED}; font-size: .8rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.pcard .rating {{ flex: none; font-size: .72rem; font-weight: 700; color: {GREEN_DEEP}; background: {GREEN_SOFT};
                 padding: 1px 7px; border-radius: 999px; }}
.pcard .pname {{ color: {TEXT}; font-weight: 600; font-size: .95rem; line-height: 1.3; margin-top: 3px;
                min-height: 2.6em; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }}
.pcard .size {{ margin-top: 8px; font-size: .78rem; color: {TEXT_MUTED}; background: {BG_SOFT};
               border-radius: 7px; padding: 5px 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.pcard .prow {{ display: flex; align-items: baseline; gap: 7px; margin-top: 9px; flex-wrap: wrap; }}
.pcard .price {{ font-weight: 800; font-size: 1.2rem; color: {TEXT}; }}
.pcard .mrp {{ font-size: .82rem; color: {TEXT_MUTED}; text-decoration: line-through; }}
.pcard .match {{ margin-left: auto; font-size: .72rem; font-weight: 700; color: {GREEN}; }}
.pcard details {{ margin-top: 8px; font-size: .8rem; color: {TEXT_MUTED}; line-height: 1.45; }}
.pcard summary {{ cursor: pointer; color: {GREEN}; font-weight: 700; }}
.pgrid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; margin: 8px 0 4px; }}

/* "Ask about this" button under featured cards */
.st-key-featured .stButton > button {{ width: 100%; border: 1.5px solid {GREEN}; color: {GREEN};
  background: {BG_CARD}; font-weight: 700; border-radius: 10px; margin-top: -4px; }}
.st-key-featured .stButton > button:hover {{ background: {GREEN}; color: white; }}

/* ---------- chat ---------- */
.st-key-chat {{ max-width: 960px; margin: 0 auto; }}
[data-testid="stChatMessage"] {{ background: {BG_CARD}; border: 1px solid {BORDER}; border-radius: 16px;
  padding: 14px 16px; margin-bottom: 10px; }}
[data-testid="stChatMessageContent"] p {{ font-size: 1rem; line-height: 1.6; color: {TEXT}; }}
[data-testid="stChatInput"] {{ border: 2px solid {GREEN}; border-radius: 14px; background: {BG_CARD}; }}
[data-testid="stBottomBlockContainer"] {{ max-width: 980px; background: {BG_PAGE}; }}
.answer-head {{ font-size: .8rem; font-weight: 700; color: {TEXT_MUTED}; margin: 14px 0 2px; }}
.pills {{ display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 8px; font-size: .74rem; }}
.pills span {{ background: {BG_SOFT}; color: {TEXT_MUTED}; border: 1px solid {BORDER_SOFT};
              padding: 2px 8px; border-radius: 999px; }}
.pills span.hl {{ background: {GREEN_SOFT}; color: {GREEN_DEEP}; font-weight: 700; border-color: {GREEN_SOFT}; }}
.nodata {{ border-left: 4px solid {NODATA_BORDER}; background: {NODATA_BG}; color: {NODATA_TEXT};
          padding: 12px 16px; border-radius: 4px 12px 12px 4px; }}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {{ background: {BG_SIDEBAR}; border-right: 1px solid {BORDER}; }}
section[data-testid="stSidebar"] h3 {{ color: {GREEN_DEEP}; font-weight: 800; }}
.key-ok {{ color: {GREEN}; font-size: .85rem; font-weight: 600; }}
.key-bad {{ color: {NODATA_BORDER}; font-size: .85rem; font-weight: 600; }}
.facts {{ font-size: .82rem; color: {TEXT_MUTED}; line-height: 1.7; }}

.footnote {{ text-align: center; color: {TEXT_MUTED}; font-size: .78rem; margin-top: 28px; }}
.stButton > button:focus-visible {{ outline: 2px solid {GREEN}; outline-offset: 2px; }}

@media (max-width: 760px) {{
  .hdr .askbox {{ display: none; }}
  .hdr {{ flex-wrap: wrap; }}
}}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PIPELINE INITIALIZATION
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Indexing product catalogue (first run takes ~1-2 min)...")
def load_pipeline():
    return RAGPipeline()


def get_api_keys():
    groq_key = st.session_state.get("groq_key_input", "") or os.getenv("GROQ_API_KEY", "")
    try:
        if not groq_key:
            groq_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        pass

    gemini_key = st.session_state.get("gemini_key_input", "") or os.getenv("GEMINI_API_KEY", "")
    try:
        if not gemini_key:
            gemini_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        pass

    return groq_key, gemini_key


pipeline = load_pipeline()
product_metas = [m for m in pipeline.metas if m.get("source") == "BigBasket Catalogue" and m.get("chunk") == 0]
n_products = len(product_metas)
n_chunks = len(pipeline.ids)
aisles = Counter(m["category"] for m in product_metas).most_common(8)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "featured" not in st.session_state:
    # 4 real products for the opening screen, with their chunk text (for MRP,
    # rating and type). Prefer discounted items so the cards show a badge.
    pool = [(m, t) for m, t in zip(pipeline.metas, pipeline.texts)
            if m.get("source") == "BigBasket Catalogue" and m.get("chunk") == 0
            and 0 < m.get("price", 0) < 1500 and len(m.get("product", "")) < 48]
    deals = [p for p in pool if "% off" in p[1]]
    pick_from = deals if len(deals) >= 4 else pool
    st.session_state.featured = random.sample(pick_from, 4) if len(pick_from) >= 4 else pick_from[:4]

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Settings")

    # Dark Mode Toggle
    is_dark = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode, key="theme_toggle")
    if is_dark != st.session_state.dark_mode:
        st.session_state.dark_mode = is_dark
        st.rerun()

    st.markdown("---")
    groq_k, gemini_k = get_api_keys()

    st.text_input(
        "Groq API Key (Primary)",
        type="password",
        key="groq_key_input",
        value=groq_k if groq_k else "",
        placeholder="Paste Groq key",
        help="Uses openai/gpt-oss-20b for ultra-fast generation",
    )
    st.text_input(
        "Gemini API Key (Fallback)",
        type="password",
        key="gemini_key_input",
        value=gemini_k if gemini_k else "",
        placeholder="Optional fallback key",
    )

    if groq_k:
        st.markdown('<div class="key-ok">● Connected to Groq (gpt-oss-20b)</div>', unsafe_allow_html=True)
        if gemini_k:
            st.markdown('<div class="key-ok">● Gemini Fallback Standby</div>', unsafe_allow_html=True)
    elif gemini_k:
        st.markdown('<div class="key-ok">● Connected to Gemini Engine</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="key-bad">● Add a key to start querying</div>', unsafe_allow_html=True)

    st.divider()
    mode = st.radio(
        "Search Retrieval",
        ["hybrid", "vector", "keyword"],
        horizontal=True,
        help="Hybrid merges vector semantics with BM25 keywords via Reciprocal Rank Fusion",
    )
    top_k = st.slider("Products to inspect", 2, 10, TOP_K)
    rerank = st.toggle("Cross-encoder re-ranking", value=True, help="Re-ranks with ms-marco-MiniLM-L-6-v2")
    decompose = st.toggle("Query decomposition", value=True, help="Splits multi-part compound queries")
    show_ctx = st.toggle("Show retrieved product tags", value=True)

    st.divider()
    st.markdown(
        f"""<div class="facts">
<b>Index Status:</b> {n_products:,} products ({n_chunks:,} chunks)<br>
<b>Embeddings:</b> all-MiniLM-L6-v2 (CPU)<br>
<b>Database:</b> ChromaDB (cosine space)
</div>""",
        unsafe_allow_html=True,
    )

    with st.expander("Pipeline Architecture"):
        st.markdown(
            "1. **Input Guardrail**: Blocks off-topic queries and prompt injections\n"
            "2. **Exact Hash Cache**: Returns repeated queries instantly in <2ms\n"
            "3. **Query Decomposition**: Resolves conversational follow-ups and splits multi-intent queries\n"
            "4. **Hybrid Retrieval**: Combines ChromaDB vector search + BM25Okapi using RRF ($k=60$)\n"
            "5. **Cross-Encoder Re-ranker**: ms-marco-MiniLM-L-6-v2 refines top candidates\n"
            "6. **Grounded Generation**: Groq `gpt-oss-20b` generates answers with numbered citations"
        )

    with st.expander("Cache Statistics"):
        stats = pipeline.cache_stats()
        st.markdown(f"**Hits:** {stats['hits']} | **Misses:** {stats['misses']} | **Size:** {stats['size']}")
        if st.button("Clear Cache", use_container_width=True):
            pipeline.clear_cache()
            st.rerun()

    if st.session_state.messages and st.button("Start New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# UI HELPERS
# ---------------------------------------------------------------------------
import re as _re


def parse_card_facts(text):
    """Read MRP and rating out of a chunk's header lines.
    The header (built in rag.py row_to_header) looks like:
      Price: ₹190 (MRP ₹250, 24% off)
      Rating: 4.2/5
    Stored metadata only has the sale price, so we read the rest from text."""
    mrp = _re.search(r"MRP ₹([\d,]+)", text or "")
    rating = _re.search(r"Rating:\s*([\d.]+)/5", text or "")
    mrp_val = float(mrp.group(1).replace(",", "")) if mrp else None
    return mrp_val, (float(rating.group(1)) if rating else None)


def card_html(m, text="", rank=None, match=None, show_desc=False):
    """One store-style product card (picture, badge, brand, name, price, MRP)."""
    sub, ptype = parse_header(text)
    emoji, tint = pick_art(m.get("product", ""), m.get("category", ""), sub, ptype)
    label = html.escape(ptype or sub or m.get("category", "") or "Product")
    price = m.get("price", -1)
    mrp, rating = parse_card_facts(text)

    is_policy = price is None or price < 0
    badge = ""
    if not is_policy and mrp and mrp > price > 0:
        badge = f'<span class="badge">{round((mrp - price) / mrp * 100)}% OFF</span>'

    parts = ['<div class="pcard">',
             f'<div class="img" style="background:{tint}" role="img" aria-label="{label}">{badge}{emoji}',
             (f'<span class="rank">#{rank}</span>' if rank else ''), '</div>',
             '<div class="meta">',
             f'<span class="brand">{html.escape(m.get("brand", ""))}</span>',
             (f'<span class="rating">★ {rating:.1f}</span>' if rating else ''), '</div>',
             f'<div class="pname">{html.escape(m.get("product", "Unknown"))}</div>',
             f'<div class="size">{label}</div>',
             '<div class="prow">']
    if is_policy:
        parts.append('<span class="price">Policy</span>')
    else:
        parts.append(f'<span class="price">₹{price:,.0f}</span>')
        if mrp and mrp > price:
            parts.append(f'<span class="mrp">₹{mrp:,.0f}</span>')
    if match:
        parts.append(f'<span class="match">{match}</span>')
    parts.append('</div>')
    if show_desc:
        desc = (text or "").split("Description:", 1)[-1].strip()
        if desc:
            short = html.escape(desc[:280]) + ("…" if len(desc) > 280 else "")
            parts.append(f'<details><summary>Description</summary>{short}</details>')
    parts.append('</div>')
    return "".join(parts)


def render_products(chunks, best_sim):
    """Retrieved products shown as store cards right under the answer.
    WHY visible: the brief asks to check that retrieved chunks contain the
    evidence — this makes retrieval part of the demo."""
    cards = []
    for n, c in enumerate(chunks, 1):
        sim = c.get("similarity") or c.get("similarity_score") or c.get("score")
        match = f"{sim:.0%} match" if sim is not None else "word match"
        text = c.get("text") or c.get("content", "")
        cards.append(card_html(c["metadata"], text, rank=n, match=match, show_desc=True))
    st.markdown(f'<div class="answer-head">Products used for this answer '
                f'(closest match {best_sim:.0%})</div><div class="pgrid">{"".join(cards)}</div>',
                unsafe_allow_html=True)


def render_nodata():
    st.markdown(f'<div class="nodata"><b>Not in the catalogue.</b> {html.escape(FALLBACK_MSG)}</div>',
                unsafe_allow_html=True)


def ask(q):
    st.session_state.pending_question = q
    st.rerun()


AVATARS = {"user": "🙂", "assistant": "🧺"}
AISLE_SHORT = {
    "Beauty & Hygiene": "Beauty", "Gourmet & World Food": "Gourmet",
    "Snacks & Branded Foods": "Snacks", "Foodgrains, Oil & Masala": "Staples",
    "Cleaning & Household": "Household", "Kitchen, Garden & Pets": "Kitchen",
    "Bakery, Cakes & Dairy": "Bakery & Dairy", "Beverages": "Beverages",
    "Baby Care": "Baby Care", "Fruits & Vegetables": "Fruits & Veg", "Eggs, Meat & Fish": "Eggs & Meat",
}

# ---------------------------------------------------------------------------
# HEADER BAR
# ---------------------------------------------------------------------------
groq_k, gemini_k = get_api_keys()
if groq_k:
    chip = '<div class="chip">⚡ Groq connected<span>gpt-oss-20b</span></div>'
elif gemini_k:
    chip = '<div class="chip">⚡ Gemini connected<span>fallback model</span></div>'
else:
    chip = '<div class="chip off">Add an API key<span>in the sidebar</span></div>'

st.markdown(f"""
<div class="hdr">
  <div class="logo">🧺</div>
  <div>
    <div class="title">Basket Assistant</div>
    <div class="sub">AI shopping help built on the BigBasket catalogue dataset</div>
  </div>
  <div class="askbox">Ask about any of {n_products:,} products below</div>
  {chip}
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# AISLE BAR (real categories from the data; click = ask about that aisle)
# ---------------------------------------------------------------------------
with st.container(key="aislebar"):
    cols = st.columns([1.35] + [1] * 6)
    cols[0].markdown('<div class="aisle-label">Shop by aisle</div>', unsafe_allow_html=True)
    for i, (cat, count) in enumerate(aisles[:6]):
        if cols[i + 1].button(AISLE_SHORT.get(cat, cat), key=f"aisle{i}", help=f"{cat}: {count:,} products",
                              use_container_width=True):
            ask(f"What are some popular {cat} products under ₹250?")

# ---------------------------------------------------------------------------
# OPENING SCREEN (only while the chat is empty)
# ---------------------------------------------------------------------------
TILES = [
    "Breakfast cereal under ₹300",
    "Compare Saffola Gold and Fortune oil",
    "Face wash for oily skin, with price",
    "Return policy for spoiled food",
]
TILE_QUESTIONS = [
    "Suggest a healthy breakfast cereal under ₹300",
    "Compare Saffola Gold and Fortune Sunflower oil",
    "Which face wash suits oily skin, and what does it cost?",
    "What is BigBasket's return policy for spoiled food items?",
]

if not st.session_state.messages:
    with st.container(key="tiles"):
        tcols = st.columns(4)
        for i, label in enumerate(TILES):
            with tcols[i].container(key=f"tile{i}"):
                if st.button(label, key=f"tilebtn{i}", use_container_width=True):
                    ask(TILE_QUESTIONS[i])

    st.markdown(f'<div class="panel-head"><h3>From the catalogue</h3>'
                f'<span>4 of {n_products:,} products the assistant can search</span></div>',
                unsafe_allow_html=True)
    with st.container(key="featured"):
        fcols = st.columns(4)
        for i, (m, text) in enumerate(st.session_state.featured):
            with fcols[i]:
                st.markdown(card_html(m, text), unsafe_allow_html=True)
                if st.button("Ask about this", key=f"feat{i}", use_container_width=True):
                    ask(f"Tell me about {m.get('product')} by {m.get('brand')}: what is it, "
                        f"what does it cost, and are there similar options?")

    st.markdown('<div class="footnote">Student project built on the public Kaggle BigBasket product '
                'dataset (2022 snapshot). Not affiliated with BigBasket. Prices come from the dataset, '
                'not the live store.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# CHAT CONVERSATION
# ---------------------------------------------------------------------------
question = st.chat_input("Search or ask: products, brands, prices or return policies")
if not question:
    question = st.session_state.pop("pending_question", None)

with st.container(key="chat"):
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
            if msg.get("pills"):
                st.markdown(msg["pills"], unsafe_allow_html=True)
            if msg.get("guardrail"):
                st.markdown(f'<div class="nodata"><b>Outside what I can help with.</b> '
                            f'{html.escape(msg["content"])}</div>', unsafe_allow_html=True)
            elif msg.get("fallback"):
                render_nodata()
            else:
                st.markdown(msg["content"])
            if msg["role"] == "assistant" and show_ctx and msg.get("chunks"):
                render_products(msg["chunks"], msg.get("best_sim", 0.0))

    if question:
        if not groq_k and not gemini_k:
            st.warning("Add a Groq or Gemini API key in the sidebar to get answers.", icon="🔑")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar=AVATARS["user"]):
            st.markdown(question)

        with st.chat_message("assistant", avatar=AVATARS["assistant"]):
            with st.spinner("Searching the catalogue..."):
                try:
                    result = pipeline.answer(
                        question,
                        groq_api_key=groq_k,
                        gemini_api_key=gemini_k,
                        k=top_k,
                        mode=mode,
                        rerank=rerank,
                        decompose=decompose,
                        chat_history=st.session_state.messages[:-1],
                    )
                except Exception as e:
                    st.session_state.messages.pop()
                    st.error(f"Couldn't get an answer: {e}")
                    st.stop()

            content = result["answer"]
            cached_flag = result.get("cached", False)
            pills = (
                '<div class="pills">'
                f'<span class="hl">⏱ {result.get("latency_ms", 0):.0f} ms</span>'
                f'<span>Flow: {html.escape(str(result.get("flow", "hybrid_rerank")))}</span>'
                f'<span>Cache: {"hit" if cached_flag else "miss"}</span>'
                f'<span>Confidence: {result.get("confidence_score", 0):.0%}</span>'
                '</div>'
            )
            st.markdown(pills, unsafe_allow_html=True)

            if result.get("guardrail"):
                st.markdown(f'<div class="nodata"><b>Outside what I can help with.</b> '
                            f'{html.escape(content)}</div>', unsafe_allow_html=True)
            elif result.get("fallback"):
                render_nodata()
            else:
                if result.get("decomposed") and len(result.get("sub_queries", [])) > 1:
                    st.caption("Split into: " + ", ".join(f'"{q}"' for q in result["sub_queries"]))
                st.markdown(content)

            if show_ctx and result.get("chunks"):
                render_products(result["chunks"], result.get("best_sim", 0.0))

            with st.expander("Backend audit log", expanded=False):
                st.json(result.get("audit_entry", {
                    "latency_ms": result.get("latency_ms"),
                    "cached": cached_flag,
                    "flow": result.get("flow"),
                    "confidence_score": result.get("confidence_score"),
                    "chunks": [c.get("metadata", {}).get("product") for c in result.get("chunks", [])],
                }))

        st.session_state.messages.append({
            "role": "assistant",
            "content": content,
            "pills": pills,
            "fallback": result.get("fallback", False),
            "guardrail": result.get("guardrail", False),
            "cached": cached_flag,
            "chunks": result.get("chunks", []),
            "best_sim": result.get("best_sim", 0.0),
        })
        st.rerun()
