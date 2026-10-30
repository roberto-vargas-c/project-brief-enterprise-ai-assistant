"""Look and feel of the Streamlit app: global CSS, the hero header and the answer cards.

The colours match the theme in .streamlit/config.toml (dark, violet and cyan accents).
"""

from __future__ import annotations

import html

import streamlit as st

from ironstore_assistant.rag import page_label

GLOBAL_CSS = """
<style>
:root { --violet: #8b5cf6; --cyan: #22d3ee; --pink: #f472b6; --green: #34d399; --amber: #fbbf24;
        --muted: #8b97b5; --panel: rgba(17, 26, 46, .72); --line: rgba(139, 151, 181, .18); }

/* page background: soft moving aurora behind everything */
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(900px 500px at 12% -10%, rgba(139, 92, 246, .22), transparent 60%),
    radial-gradient(800px 500px at 100% 0%, rgba(34, 211, 238, .14), transparent 60%),
    radial-gradient(700px 600px at 50% 120%, rgba(244, 114, 182, .10), transparent 60%),
    #070b18;
  background-attachment: fixed;
}
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; max-width: 1100px; }

/* sidebar */
[data-testid="stSidebar"] { border-right: 1px solid var(--line); }
[data-testid="stSidebar"] h3 { font-size: .78rem !important; letter-spacing: .14em; text-transform: uppercase;
                                color: var(--muted) !important; margin-top: .4rem; }
.brand { display: flex; gap: .7rem; align-items: center; margin: -.6rem 0 .4rem; }
.brand .logo { width: 2.6rem; height: 2.6rem; border-radius: .8rem; display: grid; place-items: center;
               font-size: 1.35rem; background: linear-gradient(135deg, var(--violet), var(--cyan));
               box-shadow: 0 6px 24px rgba(139, 92, 246, .45); }
.brand b { font-family: "Space Grotesk", sans-serif; font-size: 1.15rem; display: block; line-height: 1.1; }
.brand span { color: var(--muted); font-size: .78rem; }
.foot { color: var(--muted); font-size: .75rem; line-height: 1.6; }
.foot code { background: rgba(139, 92, 246, .15); color: #c4b5fd; padding: .05rem .35rem; border-radius: .35rem; }

/* hero */
@keyframes shimmer { to { background-position: 200% center; } }
@keyframes float { 50% { transform: translateY(-6px); } }
@keyframes rise { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
.hero { text-align: center; padding: 1.2rem 0 1.4rem; animation: rise .7s ease both; }
@keyframes spin { to { transform: rotate(360deg); } }
.hero .orb { width: 4.2rem; height: 4.2rem; border-radius: 50%; display: inline-block; position: relative;
             background: radial-gradient(circle at 32% 30%, #f5f3ff 0 8%, #a78bfa 30%, #6d28d9 62%, #1e1b4b 100%);
             box-shadow: 0 0 40px rgba(139, 92, 246, .65), inset -6px -8px 18px rgba(0, 0, 0, .45);
             animation: float 3.5s ease-in-out infinite; }
.hero .orb::after { content: ""; position: absolute; inset: -9px; border-radius: 50%;
                    border: 2px solid transparent; border-top-color: #67e8f9; border-right-color: rgba(103, 232, 249, .3);
                    animation: spin 2.8s linear infinite; }
.hero h1 { font-family: "Space Grotesk", sans-serif; font-size: 3rem !important; line-height: 1.05 !important;
           margin: .3rem 0 .5rem; padding: 0 !important;
           background: linear-gradient(90deg, #c4b5fd, #67e8f9, #f9a8d4, #c4b5fd); background-size: 200% auto;
           -webkit-background-clip: text; background-clip: text; color: transparent;
           animation: shimmer 6s linear infinite; }
.hero p { color: var(--muted); font-size: 1.05rem; margin: 0 auto; max-width: 40rem; }
.stats { display: flex; justify-content: center; gap: .7rem; flex-wrap: wrap; margin-top: 1.2rem; }
.stat { background: var(--panel); border: 1px solid var(--line); border-radius: 999px; padding: .4rem .95rem;
        font-size: .85rem; color: var(--muted); backdrop-filter: blur(8px); animation: rise .6s ease both; }
.stat b { color: #fff; font-family: "Space Grotesk", sans-serif; font-size: 1rem; margin-right: .25rem; }
.try { text-align: center; color: var(--muted); font-size: .8rem; letter-spacing: .14em; text-transform: uppercase;
       margin: .6rem 0 .5rem; }

/* buttons: example questions and sidebar button */
.stButton button { background: var(--panel); border: 1px solid var(--line); backdrop-filter: blur(8px);
                   transition: transform .15s ease, border-color .15s ease, box-shadow .15s ease; }
.stButton button:hover { transform: translateY(-2px); border-color: var(--violet); color: #fff;
                         box-shadow: 0 8px 28px rgba(139, 92, 246, .30); }
.stButton button p { white-space: normal; }

/* chat */
[data-testid="stChatMessage"] { background: var(--panel); border: 1px solid var(--line); border-radius: 1.1rem;
                                padding: 1rem 1.1rem; backdrop-filter: blur(10px); animation: rise .4s ease both; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) {
  background: linear-gradient(135deg, rgba(139, 92, 246, .22), rgba(34, 211, 238, .10));
  border-color: rgba(139, 92, 246, .40); }
[data-testid="stBottomBlockContainer"], [data-testid="stBottom"] > div { background: transparent; }
[data-testid="stBottomBlockContainer"] { max-width: 1100px; }
[data-testid="stChatInput"] { border: 1px solid rgba(139, 92, 246, .45) !important; border-radius: 1.2rem !important;
                              box-shadow: 0 0 0 4px rgba(139, 92, 246, .08), 0 10px 40px rgba(0, 0, 0, .35); }

/* the live 3D map fills the space between the question and the chat input */
.st-key-livemap iframe { height: max(480px, calc(100vh - 250px)) !important; display: block; }
.st-key-livemap [data-testid="stElementContainer"] { height: auto !important; }
[data-testid="stChatMessage"]:has(.st-key-livemap) { padding: .5rem; }

/* "show my work" steps (st.status) */
[data-testid="stExpander"] details { background: rgba(10, 16, 32, .55); border: 1px solid var(--line);
                                     border-radius: .9rem; }
[data-testid="stExpander"] details summary:hover { color: #c4b5fd; }

/* answer extras */
.conf { display: inline-flex; align-items: center; gap: .55rem; margin: .6rem 0 .2rem; padding: .35rem .8rem;
        border-radius: 999px; font-size: .82rem; font-weight: 600; border: 1px solid var(--c);
        background: color-mix(in srgb, var(--c) 14%, transparent); color: var(--c); }
.conf .meter { display: inline-flex; gap: 3px; }
.conf .meter i { width: 14px; height: 6px; border-radius: 3px; background: color-mix(in srgb, var(--c) 25%, transparent); }
.conf .meter i.on { background: var(--c); box-shadow: 0 0 8px var(--c); }
.src-title { color: var(--muted); font-size: .75rem; letter-spacing: .14em; text-transform: uppercase; margin: .9rem 0 .4rem; }
.src-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: .55rem;
            margin-bottom: .8rem; }
.src { background: rgba(10, 16, 32, .6); border: 1px solid var(--line); border-radius: .8rem; padding: .6rem .75rem;
       font-size: .85rem; animation: rise .45s ease both; transition: border-color .15s ease; }
.src:hover { border-color: var(--cyan); }
.src summary { cursor: pointer; list-style: none; }
.src summary::-webkit-details-marker { display: none; }
.src .num { display: inline-grid; place-items: center; width: 1.45rem; height: 1.45rem; border-radius: .45rem;
            margin-right: .45rem; font-weight: 700; font-size: .75rem; color: #0b1020;
            background: linear-gradient(135deg, var(--violet), var(--cyan)); }
.src .meta { color: var(--muted); font-size: .75rem; margin-top: .3rem; }
.src .snip { margin-top: .45rem; padding-top: .45rem; border-top: 1px dashed var(--line); color: #cbd5e1;
             font-size: .8rem; white-space: pre-wrap; max-height: 14rem; overflow: auto; }
</style>
"""

LOGO = ('<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="#fff" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round"><path d="M12 2l8.5 5v10L12 22l-8.5-5V7z"/>'
        '<circle cx="12" cy="12" r="3"/></svg>')
CONFIDENCE = {"high": ("High confidence", "#34d399", 3), "medium": ("Medium confidence", "#fbbf24", 2),
              "low": ("Low confidence", "#f87171", 1)}


def inject_css() -> None:
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def brand() -> None:
    st.markdown('<div class="brand"><div class="logo">' + LOGO + '</div><div><b>IronStore Assistant</b>'
                '<span>Answers with sources</span></div></div>', unsafe_allow_html=True)


def hero(n_documents: int, n_chunks: int, n_products: int) -> None:
    stats = [(n_documents, "documents"), (n_chunks, "searchable chunks"), (n_products, "products")]
    chips = "".join(f'<span class="stat" style="animation-delay:{.15 + i * .1:.2f}s"><b>{v:,}</b>{label}</span>'
                    for i, (v, label) in enumerate(stats))
    st.markdown(f"""<div class="hero"><div class="orb"></div>
<h1>Ask IronStore anything</h1>
<p>Answers from the company's own policies and product catalogs, with the page they came from.
Turn on <b>Show my work</b> to watch how each answer is found.</p>
<div class="stats">{chips}</div></div>""", unsafe_allow_html=True)


def confidence_badge(level: str) -> str:
    label, color, bars = CONFIDENCE[level]
    meter = "".join(f'<i class="{"on" if i < bars else ""}"></i>' for i in range(3))
    return f'<div class="conf" style="--c:{color}"><span class="meter">{meter}</span>{label}</div>'


def source_cards(sources) -> str:
    cards = []
    for i, s in enumerate(sources):
        meta = [page_label(s.page, s.page_end)]
        if s.doc_id:
            meta.append(f"ID {s.doc_id}")
        if s.last_reviewed:
            meta.append(f"reviewed {s.last_reviewed}")
        title = s.title + (" · OUTDATED" if s.status == "outdated" else "")
        cards.append(f'<details class="src" style="animation-delay:{i * .08:.2f}s"><summary>'
                     f'<span class="num">{s.number}</span>{html.escape(title)}'
                     f'<div class="meta">{html.escape(" · ".join(meta))}</div></summary>'
                     f'<div class="meta">File: {html.escape(s.source)}</div>'
                     f'<div class="snip">{html.escape(s.snippet)}</div></details>')
    return '<div class="src-title">Sources · click to read</div><div class="src-grid">' + "".join(cards) + "</div>"
