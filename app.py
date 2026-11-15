"""
app.py — Streamlit chat UI for the BigBasket RAG assistant.

Run: streamlit run app.py
"""

import html
import os
import random
from collections import Counter

import streamlit as st

from rag import FALLBACK_MSG, GUARDRAIL_MSG, RAGPipeline, SIM_THRESHOLD, TOP_K

st.set_page_config(page_title="BigBasket Assistant", page_icon="🧺", layout="wide")

# Initialize theme state
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

dark = st.session_state.dark_mode

# Precision Design Tokens: Editorial Warmth + Dark Mode System
if dark:
    BG_PAGE = "#0F120E"
    BG_CARD = "#171D16"
    BG_CARD_SUBTLE = "#1C241B"
    BG_SIDEBAR = "#141A13"
    BORDER_MAIN = "#283526"
    BORDER_MUTED = "#1E271C"
    TEXT_MAIN = "#EDEFEA"
    TEXT_MUTED = "#8D9A89"
    TEXT_ACCENT = "#52A347"
    LEAF = "#429339"
    HALDI = "#D9971E"
    HALDI_TEXT = "#120E04"
    TAG_BG = "#1A2218"
    TAG_BORDER = "#2B3928"
    TAG_STRING = "#3F563B"
    RAIL_BG = "#385232"
    HOVER_BG = "#222D20"
    CHAT_INPUT_BORDER = "#3E5837"
    CHAT_USER_BG = "#1C241B"
    CHAT_BOT_BG = "#151B14"
    NODATA_BG = "#261310"
    NODATA_BORDER = "#8A2E1C"
    NODATA_TEXT = "#E59E90"
else:
    BG_PAGE = "#FAFCF8"
    BG_CARD = "#FFFFFF"
    BG_CARD_SUBTLE = "#F0F5EC"
    BG_SIDEBAR = "#EEF4E8"
    BORDER_MAIN = "#D5E2CC"
    BORDER_MUTED = "#E2ECE0"
    TEXT_MAIN = "#1A2617"
    TEXT_MUTED = "#556450"
    TEXT_ACCENT = "#255320"
    LEAF = "#2F5D2A"
    HALDI = "#E2A31B"
    HALDI_TEXT = "#1B1705"
    TAG_BG = "#FFFFFF"
    TAG_BORDER = "#D2DFC7"
    TAG_STRING = "#9DB58C"
    RAIL_BG = "#2F5D2A"
    HOVER_BG = "#EEF4E8"
    CHAT_INPUT_BORDER = "#2F5D2A"
    CHAT_USER_BG = "#F4F8F1"
    CHAT_BOT_BG = "#FFFFFF"
    NODATA_BG = "#FDF2EF"
    NODATA_BORDER = "#B4462B"
    NODATA_TEXT = "#602619"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

html, body, [class*="css"], .stApp, .stMarkdown, .stMarkdown p, .stMarkdown li, [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{
  font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
  color: {TEXT_MAIN} !important;
}}

.stApp {{
  background-color: {BG_PAGE} !important;
}}

/* Universal text color enforcement for all markdown and typography */
p, span, li, h1, h2, h3, h4, h5, h6, strong, b, em, label, details, summary {{
  color: {TEXT_MAIN};
}}

.stMarkdown,
[data-testid="stMarkdownContainer"],
[data-testid="stChatMessageContent"] {{
  color: {TEXT_MAIN} !important;
}}

[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] li,
[data-testid="stChatMessageContent"] span,
[data-testid="stChatMessageContent"] div,
[data-testid="stChatMessageContent"] strong,
[data-testid="stChatMessageContent"] b {{
  color: {TEXT_MAIN} !important;
}}

/* Clean header and footer */
.stDeployButton, #MainMenu, footer {{ display: none !important; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.8rem; padding-bottom: 6.5rem; max-width: 1100px; }}

/* Navigation bar */
.top-nav {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding-bottom: 16px;
  border-bottom: 1px solid {BORDER_MUTED};
  margin-bottom: 24px;
}}
.brand {{
  display: flex;
  align-items: center;
  gap: 12px;
  color: {LEAF};
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 700;
  font-size: 1.25rem;
  letter-spacing: -0.02em;
}}
.brand .mark {{
  width: 36px;
  height: 36px;
  border-radius: 10px;
  background: {LEAF};
  color: white;
  display: grid;
  place-items: center;
  font-size: 1.15rem;
  box-shadow: 0 2px 6px rgba(0,0,0,0.08);
}}
.status-pill {{
  font-size: 0.8rem;
  font-weight: 600;
  padding: 4px 10px;
  border-radius: 9999px;
  background: {BG_CARD_SUBTLE};
  color: {TEXT_MUTED};
  border: 1px solid {BORDER_MAIN};
}}

/* Typography */
.hero-title {{
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 700;
  color: {TEXT_MAIN};
  font-size: clamp(2.2rem, 4.5vw, 3.4rem);
  line-height: 1.08;
  letter-spacing: -0.035em;
  margin: 18px 0 12px;
}}
.hero-subtitle {{
  color: {TEXT_MUTED};
  font-size: 1.05rem;
  line-height: 1.55;
  max-width: 32em;
  margin-bottom: 24px;
}}
.section-label {{
  font-size: 0.82rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: {TEXT_MUTED};
  margin-bottom: 10px;
}}

/* Shopping list interactive cards */
.st-key-shoplist .stButton > button {{
  width: 100%;
  background: {BG_CARD};
  border: 1px solid {BORDER_MAIN};
  border-radius: 12px;
  padding: 13px 18px;
  margin-bottom: 8px;
  color: {TEXT_MAIN};
  text-align: left;
  transition: all 0.18s cubic-bezier(0.16, 1, 0.3, 1);
  box-shadow: 0 1px 3px rgba(0,0,0,0.02);
}}
.st-key-shoplist .stButton > button:hover {{
  border-color: {LEAF};
  background: {BG_CARD_SUBTLE};
  transform: translateY(-1px);
  box-shadow: 0 3px 10px rgba(0,0,0,0.05);
}}
.st-key-shoplist .stButton > button > div,
.st-key-shoplist .stButton > button [data-testid="stMarkdownContainer"] {{
  justify-content: flex-start !important;
  text-align: left !important;
  width: 100%;
}}
.st-key-shoplist .stButton > button p {{
  font-size: 0.98rem;
  font-weight: 500;
  line-height: 1.35;
  margin: 0;
}}

/* Shelf aesthetic */
.shelf {{
  position: relative;
  background: {BG_CARD_SUBTLE};
  border: 1px solid {BORDER_MAIN};
  border-radius: 20px;
  padding: 24px 20px 20px;
  margin-top: 14px;
  overflow: hidden;
}}
.shelf .rail {{
  height: 8px;
  border-radius: 4px;
  background: {RAIL_BG};
  margin: 0 -4px;
}}
.shelf .hang {{
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
}}
.shelf .slot {{
  display: flex;
  flex-direction: column;
  align-items: center;
}}
.shelf .string {{
  width: 2px;
  height: 34px;
  background: {TAG_STRING};
}}
.shelf .slot:nth-child(2) .string {{ height: 68px; }}
.shelf .slot:nth-child(3) .string {{ height: 48px; }}
.shelf .slot:nth-child(1) .tag {{ transform: rotate(-2.5deg); }}
.shelf .slot:nth-child(3) .tag {{ transform: rotate(2deg); }}

/* Price Tag Card Design */
.tag {{
  position: relative;
  background: {TAG_BG};
  border: 1px solid {TAG_BORDER};
  border-radius: 12px;
  padding: 22px 14px 14px;
  box-shadow: 0 3px 8px rgba(0,0,0,0.03);
  transition: transform 0.2s ease;
}}
.tag::before {{
  content: "";
  position: absolute;
  left: 50%;
  top: 8px;
  width: 8px;
  height: 8px;
  transform: translateX(-50%);
  border-radius: 50%;
  background: {BG_CARD_SUBTLE};
  border: 1px solid {BORDER_MAIN};
}}
.tag .name {{
  font-weight: 600;
  color: {TEXT_MAIN};
  font-size: 0.88rem;
  line-height: 1.3;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}}
.tag .brand-l {{
  color: {TEXT_MUTED};
  font-size: 0.76rem;
  margin-top: 3px;
}}
.tag .row {{
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 6px;
  margin-top: 10px;
  flex-wrap: wrap;
}}
.tag .price {{
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 700;
  font-size: 1.18rem;
  color: {HALDI_TEXT};
  background: {HALDI};
  padding: 1px 7px;
  border-radius: 6px;
}}
.tag .match {{
  font-size: 0.74rem;
  font-weight: 600;
  color: {LEAF};
}}
.tag .n {{
  position: absolute;
  top: 6px;
  left: 8px;
  font-size: 0.72rem;
  font-weight: 700;
  color: {TEXT_MUTED};
}}
.tag details {{
  margin-top: 8px;
  font-size: 0.8rem;
  color: {TEXT_MUTED};
}}
.tag summary {{
  cursor: pointer;
  color: {LEAF};
  font-weight: 600;
}}
.tags {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 12px;
  padding: 6px 2px;
}}

/* Aisles Navigation */
.aisles-container {{
  margin-top: 36px;
  padding-top: 24px;
  border-top: 1px solid {BORDER_MUTED};
}}
.st-key-aisles .stButton > button {{
  width: 100%;
  border: 1px solid {BORDER_MAIN};
  background: {BG_CARD};
  border-radius: 12px;
  padding: 12px 14px;
  min-height: 60px;
  color: {TEXT_MAIN};
  box-shadow: 0 1px 2px rgba(0,0,0,0.02);
  transition: all 0.16s ease;
}}
.st-key-aisles .stButton > button:hover {{
  border-color: {LEAF};
  background: {BG_CARD_SUBTLE};
  transform: translateY(-1px);
}}
.st-key-aisles .stButton > button p {{
  font-size: 0.88rem;
  font-weight: 600;
  line-height: 1.25;
  margin: 0;
}}

/* Chat Area Styling */
.st-key-chat {{
  max-width: 820px;
  margin: 0 auto;
}}
[data-testid="stChatMessage"] {{
  background: transparent;
  padding: 10px 0;
}}
[data-testid="stChatMessageContent"] {{
  border-radius: 14px;
  padding: 14px 18px;
  font-size: 1rem;
  line-height: 1.6;
  border: 1px solid {BORDER_MUTED};
  color: {TEXT_MAIN} !important;
}}
[data-testid="stChatMessageContent"] * {{
  color: {TEXT_MAIN} !important;
}}
[data-testid="stChatMessageContent"][data-testid*="user"] {{
  background-color: {CHAT_USER_BG} !important;
}}
[data-testid="stChatMessageContent"][data-testid*="assistant"] {{
  background-color: {CHAT_BOT_BG} !important;
}}

/* Streamlit Expander styling in dark mode */
[data-testid="stExpander"] {{
  background-color: {BG_CARD} !important;
  border: 1px solid {BORDER_MAIN} !important;
  border-radius: 12px !important;
}}
[data-testid="stExpander"] details {{
  background-color: {BG_CARD} !important;
}}
[data-testid="stExpander"] summary {{
  color: {TEXT_MAIN} !important;
}}
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span {{
  color: {TEXT_MAIN} !important;
}}
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {{
  color: {TEXT_MAIN} !important;
}}

/* ==========================================================================
   CHAT INPUT SYSTEM (All Streamlit container, shadow & sub-element overrides)
   ========================================================================== */
footer {{ visibility: hidden !important; display: none !important; }}

/* Bottom fixed docked area */
[data-testid="stBottom"],
[data-testid="stBottomBlockContainer"],
footer[data-testid="stBottom"] {{
  background-color: {BG_PAGE} !important;
  background: {BG_PAGE} !important;
}}

[data-testid="stBottomBlockContainer"] * {{
  box-sizing: border-box;
}}

/* Target every parent container and shadow around stChatInput */
[data-testid="stChatInput"],
[data-testid="stChatInputContainer"],
div[data-testid="stChatInput"] > div,
div[data-testid="stChatInput"] > div > div {{
  background-color: {BG_CARD} !important;
  background: {BG_CARD} !important;
  border-radius: 16px !important;
}}

[data-testid="stChatInput"] {{
  border: 1.5px solid {CHAT_INPUT_BORDER} !important;
  box-shadow: 0 4px 18px rgba(0,0,0,{'0.4' if dark else '0.04'}) !important;
}}

[data-testid="stChatInput"]:focus-within {{
  border-color: {LEAF} !important;
  box-shadow: 0 0 0 2px rgba(66, 147, 57, 0.25) !important;
}}

/* Inner textarea styling */
[data-testid="stChatInput"] textarea,
[data-testid="stChatInputContainer"] textarea,
div[data-testid="stChatInput"] textarea,
div[data-baseweb="textarea"],
div[data-baseweb="base-input"] {{
  background-color: {BG_CARD} !important;
  background: {BG_CARD} !important;
  color: {TEXT_MAIN} !important;
  -webkit-text-fill-color: {TEXT_MAIN} !important;
  caret-color: {LEAF} !important;
  font-size: 0.98rem !important;
  line-height: 1.5 !important;
}}

[data-testid="stChatInput"] textarea::placeholder,
div[data-baseweb="textarea"] textarea::placeholder {{
  color: {TEXT_MUTED} !important;
  -webkit-text-fill-color: {TEXT_MUTED} !important;
  opacity: 0.8 !important;
}}

/* Submit button styling */
[data-testid="stChatInput"] button,
[data-testid="stChatInputSubmitButton"] {{
  color: {LEAF} !important;
  fill: {LEAF} !important;
  background: transparent !important;
  border: none !important;
}}
[data-testid="stChatInput"] button svg,
[data-testid="stChatInputSubmitButton"] svg {{
  fill: {LEAF} !important;
  color: {LEAF} !important;
}}
[data-testid="stChatInput"] button:hover svg {{
  fill: {TEXT_MAIN} !important;
}}

/* Alerts */
.nodata {{
  border-left: 4px solid {NODATA_BORDER};
  background: {NODATA_BG};
  padding: 12px 16px;
  border-radius: 4px 12px 12px 4px;
  color: {NODATA_TEXT};
  font-size: 0.95rem;
}}
.nodata b {{ color: {NODATA_BORDER}; }}

/* Sidebar refinement */
section[data-testid="stSidebar"] {{
  background: {BG_SIDEBAR};
  border-right: 1px solid {BORDER_MUTED};
}}
section[data-testid="stSidebar"] h3 {{
  font-family: 'Space Grotesk', sans-serif;
  color: {LEAF};
  font-size: 1.15rem;
  margin-bottom: 12px;
}}
.key-ok {{ color: {LEAF}; font-size: 0.84rem; font-weight: 600; }}
.key-bad {{ color: #D94025; font-size: 0.84rem; font-weight: 600; }}
.facts {{ font-size: 0.84rem; color: {TEXT_MUTED}; line-height: 1.65; }}

/* Compact custom toggle bar */
.toggle-bar {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 6px 12px;
  border-radius: 10px;
  background: {BG_CARD};
  border: 1px solid {BORDER_MAIN};
  margin-bottom: 18px;
}}

@media (max-width: 760px) {{
  .shelf {{ display: none; }}
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
if "shelf" not in st.session_state:
    pool = [m for m in product_metas if 0 < m.get("price", 0) < 1500 and len(m.get("product", "")) < 48]
    st.session_state.shelf = random.sample(pool, 3) if len(pool) >= 3 else pool[:3]

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
def tag_html(m, n=None, match=None, desc=None):
    price = f"₹{m['price']:,.0f}" if m.get("price", -1) >= 0 else "Policy"
    parts = ['<div class="tag">']
    if n is not None:
        parts.append(f'<span class="n">#{n}</span>')
    parts.append(f'<div class="name">{html.escape(m.get("product", "Unknown"))}</div>')
    parts.append(f'<div class="brand-l">{html.escape(m.get("brand", "Unknown"))}</div>')
    parts.append(f'<div class="row"><span class="price">{price}</span>')
    if match:
        parts.append(f'<span class="match">{match}</span>')
    parts.append('</div>')
    if desc:
        short = html.escape(desc[:280]) + ("…" if len(desc) > 280 else "")
        parts.append(f'<details><summary>Details</summary>{short}</details>')
    parts.append('</div>')
    return "".join(parts)


def render_tags(chunks, best_sim):
    tags = []
    for n, c in enumerate(chunks, 1):
        sim = c.get("similarity") or c.get("similarity_score") or c.get("score")
        match = (f"{sim:.0%} match" if sim is not None else "word match")
        content = c.get("text") or c.get("content", "")
        desc = content.split("Description:", 1)[-1].strip()
        tags.append(tag_html(c["metadata"], n=n, match=match, desc=desc))
    with st.expander(f"Retrieved Catalogue Evidence ({len(chunks)} items, top match {best_sim:.0%})", expanded=False):
        st.markdown(f'<div class="tags">{"".join(tags)}</div>', unsafe_allow_html=True)


def render_nodata():
    st.markdown(
        f'<div class="nodata"><b>Out of Catalogue Bounds.</b> {html.escape(FALLBACK_MSG)}</div>',
        unsafe_allow_html=True,
    )


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

# Top Navigation Bar
st.markdown(f"""
<div class="top-nav">
  <div class="brand">
    <span class="mark">🧺</span>
    BigBasket Assistant
  </div>
  <div class="status-pill">
    {"🌙 Dark Mode" if dark else "☀️ Light Mode"} · RAG Active
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# OPENING SCREEN (Empty State)
# ---------------------------------------------------------------------------
SUGGESTIONS = [
    ("Which face wash suits oily skin, and what does it cost?", "Natural language beauty & hygiene query"),
    ("Compare Saffola Gold and Fortune Sunflower oil", "Multi-brand comparison query"),
    ("Suggest a healthy breakfast cereal under ₹300", "Price-filtered query"),
    ("What is BigBasket's return policy for spoiled food items?", "Policy & FAQ retrieval query"),
]

if not st.session_state.messages:
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown('<div class="hero-title">What groceries are on your list?</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="hero-subtitle">Ask questions about prices, ingredients, brands, or BigBasket policies. '
            'Every recommendation is verified against the catalogue dataset.</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-label">Try an example query</div>', unsafe_allow_html=True)
        with st.container(key="shoplist"):
            for i, (q, hint) in enumerate(SUGGESTIONS):
                if st.button(f"→  {q}", key=f"sugg{i}", use_container_width=True):
                    ask(q)

    with right:
        slots = "".join(f'<div class="slot"><div class="string"></div>{tag_html(m)}</div>' for m in st.session_state.shelf)
        st.markdown(
            f"""
        <div class="shelf">
          <div class="rail"></div>
          <div class="hang">{slots}</div>
        </div>""",
            unsafe_allow_html=True,
        )

    with st.container(key="aisles"):
        st.markdown('<div class="aisles-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Browse by category aisle</div>', unsafe_allow_html=True)
        cols = st.columns(4)
        for i, (cat, count) in enumerate(aisles):
            label = f"{AISLE_ICONS.get(cat, '🛒')} {cat} ({count:,})"
            if cols[i % 4].button(label, key=f"aisle{i}", use_container_width=True):
                ask(f"What are popular {cat} items under ₹250?")
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# CHAT CONVERSATION
# ---------------------------------------------------------------------------
question = st.chat_input("Ask about grocery products, brands, or return policies...")
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
                render_tags(msg["chunks"], msg.get("best_sim", 0.0))

    if question:
        groq_k, gemini_k = get_api_keys()
        if not groq_k and not gemini_k:
            st.warning("Please provide a Groq or Gemini API key in the sidebar.", icon="🔑")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar=AVATARS["user"]):
            st.markdown(question)

        with st.chat_message("assistant", avatar=AVATARS["assistant"]):
            with st.spinner("Searching product catalogue and grounding answer..."):
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
                    st.error(f"Search retrieval error: {e}")
                    st.stop()

            content = result["answer"]
            latency_txt = f"{result.get('latency_ms', 0):.0f}ms"
            flow_txt = result.get("flow", "hybrid_rerank")
            cached_flag = result.get("cached", False)

            # Metadata Pill / Metrics Badge
            telemetry_badge = f"""
            <div style="display:flex; gap:8px; align-items:center; margin-bottom:8px; flex-wrap:wrap; font-size:0.78rem;">
                <span style="background:{'#243322' if dark else '#E4EFE0'}; color:{LEAF}; padding:2px 8px; border-radius:6px; font-weight:600;">
                    ⏱ {latency_txt}
                </span>
                <span style="background:{'#1F281E' if dark else '#EEF4E8'}; color:{TEXT_MUTED}; padding:2px 8px; border-radius:6px; border:1px solid {BORDER_MUTED};">
                    Flow: {flow_txt}
                </span>
                <span style="background:{'#1F281E' if dark else '#EEF4E8'}; color:{TEXT_MUTED}; padding:2px 8px; border-radius:6px; border:1px solid {BORDER_MUTED};">
                    Cache: {'⚡ HIT' if cached_flag else 'MISS'}
                </span>
                <span style="background:{'#1F281E' if dark else '#EEF4E8'}; color:{TEXT_MUTED}; padding:2px 8px; border-radius:6px; border:1px solid {BORDER_MUTED};">
                    Confidence: {result.get('confidence_score', 0):.0%}
                </span>
            </div>
            """
            st.markdown(telemetry_badge, unsafe_allow_html=True)

            if result.get("guardrail"):
                st.markdown(f'<div class="nodata"><b>Policy Guardrail:</b> {html.escape(content)}</div>', unsafe_allow_html=True)
            elif result.get("fallback"):
                render_nodata()
            else:
                if result.get("decomposed") and len(result.get("sub_queries", [])) > 1:
                    sq_tags = ", ".join(f'"{q}"' for q in result["sub_queries"])
                    st.caption(f"🔍 Decomposed sub-queries: {sq_tags}")
                st.markdown(content)

            if show_ctx and result.get("chunks"):
                render_tags(result["chunks"], result.get("best_sim", 0.0))

            # Audit Inspection Expander
            with st.expander("🔍 Backend Audit Log & Chunk Telemetry", expanded=False):
                st.json(result.get("audit_entry", {
                    "latency_ms": result.get("latency_ms"),
                    "cached": cached_flag,
                    "flow": flow_txt,
                    "confidence_score": result.get("confidence_score"),
                    "chunks": [c.get("metadata", {}).get("product") for c in result.get("chunks", [])]
                }))

        st.session_state.messages.append({
            "role": "assistant",
            "content": content,
            "fallback": result.get("fallback", False),
            "guardrail": result.get("guardrail", False),
            "cached": result.get("cached", False),
            "chunks": result.get("chunks", []),
            "best_sim": result.get("best_sim", 0.0),
        })
        st.rerun()
