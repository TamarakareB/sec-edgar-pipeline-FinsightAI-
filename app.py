# app.py
# ─────────────────────────────────────────────────────────────────────────────
# FinSightAI — Premium Dark Theme
# Run with: streamlit run app.py
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import json
from pathlib import Path

sys.path.insert(0, "src")

st.set_page_config(
    page_title="FinSightAI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium Dark Theme CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background:#0A0F1E !important; }
[data-testid="stSidebar"] { background:#111827 !important; border-right:1px solid #1E2D45 !important; }
[data-testid="stSidebar"] * { color:#94A3B8 !important; }
.block-container { padding:1.5rem 2rem !important; max-width:100% !important; }
h1,h2,h3 { color:#F1F5F9 !important; font-weight:500 !important; }
p,li,span { color:#94A3B8; }
label { color:#94A3B8 !important; }
[data-testid="stSelectbox"]>div,
[data-testid="stTextInput"]>div>div,
[data-testid="stTextArea"]>div>textarea {
    background:#111827 !important; border:1px solid #1E2D45 !important;
    border-radius:8px !important; color:#F1F5F9 !important;
}
[data-testid="stButton"]>button {
    background:linear-gradient(135deg,#3B82F6,#6366F1) !important;
    color:white !important; border:none !important;
    border-radius:8px !important; font-weight:500 !important;
    padding:0.4rem 1.2rem !important;
}
[data-testid="stButton"]>button:hover { opacity:0.9 !important; }
[data-testid="stRadio"] label { color:#94A3B8 !important; }
[data-testid="stMetric"] {
    background:#111827 !important; border:1px solid #1E2D45 !important;
    border-radius:10px !important; padding:12px 16px !important;
}
[data-testid="stMetricLabel"] { color:#475569 !important; font-size:11px !important; text-transform:uppercase; letter-spacing:0.5px; }
[data-testid="stMetricValue"] { color:#F1F5F9 !important; }
[data-testid="stExpander"] { background:#111827 !important; border:1px solid #1E2D45 !important; border-radius:10px !important; }
[data-testid="stTabs"] [data-baseweb="tab-list"] { background:#111827 !important; border-radius:8px !important; padding:3px !important; gap:2px !important; }
[data-testid="stTabs"] [data-baseweb="tab"] { background:transparent !important; color:#475569 !important; border-radius:6px !important; }
[data-testid="stTabs"] [aria-selected="true"] { background:#1C2539 !important; color:#F1F5F9 !important; }
[data-testid="stDataFrame"] { border:1px solid #1E2D45 !important; border-radius:10px !important; }
.stDataFrame { background:#111827 !important; }
[data-testid="stPlotlyChart"]>div { background:#111827 !important; border:1px solid #1E2D45 !important; border-radius:10px !important; overflow:hidden; }
[data-testid="stAlert"] { background:#111827 !important; border:1px solid #1E2D45 !important; color:#94A3B8 !important; }
hr { border-color:#1E2D45 !important; }
[data-testid="stMarkdownContainer"] p { color:#94A3B8; }
[data-testid="stDownloadButton"]>button { background:#1C2539 !important; border:1px solid #1E2D45 !important; color:#94A3B8 !important; }

.page-header { padding:0 0 16px; border-bottom:1px solid #1E2D45; margin-bottom:20px; display:flex; justify-content:space-between; align-items:flex-end; }
.page-title { font-size:22px; font-weight:500; color:#F1F5F9 !important; }
.page-sub { font-size:12px; color:#475569; margin-top:3px; }
.pill-live { font-size:10px; padding:3px 10px; border-radius:99px; color:#34D399; border:1px solid rgba(52,211,153,.3); background:rgba(52,211,153,.07); }
.pill-info { font-size:10px; padding:3px 10px; border-radius:99px; color:#93C5FD; border:1px solid rgba(147,197,253,.3); background:rgba(147,197,253,.07); }

.glow-card { background:linear-gradient(135deg,#111827,rgba(59,130,246,.06)); border:1px solid rgba(99,102,241,.25); border-radius:12px; padding:16px 18px; position:relative; margin-bottom:12px; }
.glow-card::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,#3B82F6,#6366F1,#14B8A6); border-radius:12px 12px 0 0; }
.glow-card-title { font-size:10px; color:#475569; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:10px; }
.answer-text { font-size:13px; line-height:1.8; color:#CBD5E1; }
.answer-text b, .answer-text strong { color:#93C5FD; }

.dark-card { background:#111827; border:1px solid #1E2D45; border-radius:10px; padding:14px 16px; margin-bottom:8px; }
.dark-card-label { font-size:10px; color:#475569; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px; }

.tag-risk { display:inline-block; padding:2px 8px; border-radius:99px; font-size:10px; font-weight:500; margin:2px; background:rgba(244,63,94,.15); color:#FDA4AF; border:1px solid rgba(244,63,94,.2); }
.tag-mda  { display:inline-block; padding:2px 8px; border-radius:99px; font-size:10px; font-weight:500; margin:2px; background:rgba(20,184,166,.12); color:#5EEAD4; border:1px solid rgba(20,184,166,.2); }
.tag-fin  { display:inline-block; padding:2px 8px; border-radius:99px; font-size:10px; font-weight:500; margin:2px; background:rgba(245,158,11,.12); color:#FCD34D; border:1px solid rgba(245,158,11,.2); }
.tag-biz  { display:inline-block; padding:2px 8px; border-radius:99px; font-size:10px; font-weight:500; margin:2px; background:rgba(99,102,241,.15); color:#A5B4FC; border:1px solid rgba(99,102,241,.2); }
.tag-gen  { display:inline-block; padding:2px 8px; border-radius:99px; font-size:10px; font-weight:500; margin:2px; background:rgba(148,163,184,.12); color:#94A3B8; border:1px solid rgba(148,163,184,.2); }

.citation-card { background:#1C2539; border:1px solid #1E2D45; border-radius:8px; padding:10px 13px; margin:5px 0; }
.citation-header { font-size:12px; font-weight:500; color:#F1F5F9; margin-bottom:3px; }
.citation-preview { font-size:11px; color:#475569; line-height:1.5; }

.conf-bar-bg { height:5px; background:#1C2539; border-radius:3px; margin-top:6px; overflow:hidden; }
.conf-bar-fill { height:5px; background:linear-gradient(90deg,#10B981,#3B82F6); border-radius:3px; }

.comp-card { background:#111827; border:1px solid #1E2D45; border-radius:10px; padding:14px; position:relative; overflow:hidden; }
.comp-stripe-blue::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,#3B82F6,#6366F1); }
.comp-stripe-teal::before  { content:''; position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,#14B8A6,#10B981); }
.comp-stripe-amber::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,#F59E0B,#F43F5E); }
.comp-stripe-purple::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,#8B5CF6,#EC4899); }
.comp-ticker { font-size:17px; font-weight:500; color:#F1F5F9; margin-bottom:3px; }
.comp-meta   { font-size:10px; color:#475569; margin-bottom:8px; }
.comp-bar    { height:4px; border-radius:2px; margin-bottom:10px; }
.comp-text   { font-size:12px; color:#94A3B8; line-height:1.6; }

.year-card        { background:#111827; border:1px solid #1E2D45; border-radius:8px; padding:10px; text-align:center; }
.year-card-active { background:rgba(59,130,246,.08); border:1px solid #3B82F6; border-radius:8px; padding:10px; text-align:center; }
.year-num  { font-size:13px; font-weight:500; color:#F1F5F9; }
.year-dot-on  { width:8px; height:8px; border-radius:50%; background:#10B981; margin:5px auto; box-shadow:0 0 6px #10B981; }
.year-dot-off { width:8px; height:8px; border-radius:50%; background:#1E2D45; margin:5px auto; }
.year-lbl  { font-size:10px; color:#475569; }

.risk-col   { background:#111827; border:1px solid #1E2D45; border-radius:9px; padding:13px; }
.risk-col-title { font-size:11px; font-weight:500; margin-bottom:10px; display:flex; align-items:center; gap:6px; }
.risk-item  { font-size:11px; color:#94A3B8; padding:5px 0; border-bottom:1px solid #1E2D45; }
.risk-item:last-child { border-bottom:none; }

.section-divider { font-size:10px; color:#475569; text-transform:uppercase; letter-spacing:0.5px; margin:14px 0 8px; border-bottom:1px solid #1E2D45; padding-bottom:6px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
DB_PATH = Path("data/sql/finsightai.db")
FIELDS  = [
    "total_revenue","net_income","gross_profit","operating_income",
    "eps_diluted","total_assets","total_liabilities","total_equity",
    "long_term_debt","cash_and_equivalents","operating_cash_flow",
    "capital_expenditures","research_and_development",
]
FIELD_LABELS = {f: f.replace("_"," ").title() for f in FIELDS}
ALL_TICKERS  = [
    "AAPL","MSFT","AMZN","NVDA","GOOGL","META","TSLA","JPM","V","UNH",
    "XOM","WMT","MA","JNJ","PG","HD","AVGO","MRK","CVX","ABBV",
    "COST","PEP","ADBE","KO","MCD","CSCO","CRM","ACN","TMO","AMD",
    "INTC","NEE","LIN","TXN","QCOM","WFC","BAC","MS","GS","CAT",
    "SBUX","NOW","RTX","SPGI","BLK","T","DE","BMY","LLY","BRK-B",
]
SECTORS = {
    "Technology":  ["AAPL","MSFT","NVDA","GOOGL","META","ADBE","CRM","AMD","INTC","CSCO"],
    "Financials":  ["JPM","BAC","WFC","GS","MS","V","MA","BLK","SPGI"],
    "Healthcare":  ["UNH","JNJ","LLY","MRK","ABBV","TMO","BMY"],
    "Energy":      ["XOM","CVX"],
    "Consumer":    ["WMT","COST","PG","KO","PEP","MCD","SBUX"],
    "Industrials": ["CAT","DE","RTX"],
}
SECTION_TAG_MAP = {
    "risk_factors":             '<span class="tag-risk">Risk Factors</span>',
    "mda":                      '<span class="tag-mda">MD&A</span>',
    "financial_statements":     '<span class="tag-fin">Financials</span>',
    "business_overview":        '<span class="tag-biz">Business Overview</span>',
    "quantitative_disclosures": '<span class="tag-mda">Market Risk</span>',
}
GRADIENT_COLORS  = ["#3B82F6","#6366F1","#14B8A6","#10B981","#F59E0B","#F43F5E","#8B5CF6","#EC4899"]
STRIPE_CLASSES   = ["comp-stripe-blue","comp-stripe-teal","comp-stripe-amber","comp-stripe-purple"]
BAR_GRADIENTS    = [
    "background:linear-gradient(90deg,#3B82F6,#6366F1)",
    "background:linear-gradient(90deg,#14B8A6,#10B981)",
    "background:linear-gradient(90deg,#F59E0B,#F43F5E)",
    "background:linear-gradient(90deg,#8B5CF6,#EC4899)",
]
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#111827", plot_bgcolor="#111827",
    font=dict(color="#94A3B8",family="sans-serif",size=12),
    xaxis=dict(gridcolor="#1E2D45",linecolor="#1E2D45",tickcolor="#475569"),
    yaxis=dict(gridcolor="#1E2D45",linecolor="#1E2D45",tickcolor="#475569"),
    margin=dict(l=40,r=20,t=40,b=40),
)

def section_tag(key):
    return SECTION_TAG_MAP.get(key, '<span class="tag-gen">General</span>')

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:4px 0 16px">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
        <div style="width:32px;height:32px;background:linear-gradient(135deg,#3B82F6,#6366F1);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:16px">📊</div>
        <div>
          <div style="font-size:16px;font-weight:500;color:#F1F5F9">FinSightAI</div>
          <div style="font-size:10px;color:#475569">Financial Intelligence</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    mode = st.radio("", [
        "💬 Ask a Question",
        "🔄 Compare Companies",
        "📈 5-Year Trend Analysis",
        "📊 Financial Dashboard",
        "🗄️ SQL Explorer",
    ], label_visibility="collapsed")
    st.divider()
    st.markdown('<div style="font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px">Company</div>', unsafe_allow_html=True)
    ticker = st.selectbox("", ["All"]+ALL_TICKERS, label_visibility="collapsed", key="sb_t")
    st.markdown('<div style="font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px;margin-top:8px">Year</div>', unsafe_allow_html=True)
    year = st.selectbox("", ["All",2023,2022,2021,2020,2019], label_visibility="collapsed", key="sb_y")
    st.divider()
    st.markdown('<div style="font-size:10px;color:#475569;text-align:center">SEC EDGAR · 50 companies · 2019–2023</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODE 1 — ASK A QUESTION
# ══════════════════════════════════════════════════════════════════════════════
if mode == "💬 Ask a Question":
    st.markdown("""<div class="page-header"><div><div class="page-title">Ask FinSightAI</div><div class="page-sub">Grounded answers from 250+ SEC filings with citations, confidence scores &amp; follow-ups</div></div><div style="display:flex;gap:8px"><div class="pill-live">● Live</div><div class="pill-info">50 companies</div></div></div>""", unsafe_allow_html=True)

    with st.expander("💡 Example questions", expanded=False):
        c1,c2,c3 = st.columns(3)
        examples = {
            "Risk Analysis":         ["What are Apple's main liquidity risks?","What supply chain risks does Tesla mention?","What regulatory risks does JPMorgan face?"],
            "Financial Performance": ["How did Microsoft's revenue grow from 2021 to 2023?","What is Amazon's capital expenditure trend?","What was NVIDIA's R&D spending in 2023?"],
            "Strategy & Outlook":    ["What does Apple say about AI in its filings?","How does Walmart describe its competitive strategy?","What does Goldman Sachs say about market risk?"],
        }
        for col,(cat,qs) in zip([c1,c2,c3],examples.items()):
            col.markdown(f'<div style="font-size:11px;font-weight:500;color:#94A3B8;margin-bottom:6px">{cat}</div>', unsafe_allow_html=True)
            for q in qs:
                if col.button(q, key=f"ex_{q[:25]}", use_container_width=True):
                    st.session_state["question"] = q

    question = st.text_area("", value=st.session_state.get("question",""), height=80,
        placeholder="e.g. What are Apple's main liquidity risks in 2023?", label_visibility="collapsed")

    cb1,cb2 = st.columns([3,1])
    search_clicked = cb1.button("🔍  Search & Analyze", type="primary", use_container_width=True)
    show_fu = cb2.checkbox("Follow-ups", value=True)

    if search_clicked:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching filings and generating grounded answer..."):
                try:
                    from qa_pipeline import ask
                    t = ticker if ticker != "All" else None
                    y = year   if year   != "All" else None
                    result = ask(question, ticker=t, year=y, include_followups=show_fu)

                    conf     = result.get("confidence", 0)
                    conf_pct = int(conf * 100)
                    conf_col = "#10B981" if conf > 0.7 else "#F59E0B" if conf > 0.4 else "#F43F5E"

                    col_ans, col_meta = st.columns([3,1])

                    with col_meta:
                        st.markdown(f'<div class="dark-card"><div class="dark-card-label">Confidence</div><div style="font-size:24px;font-weight:500;color:{conf_col}">{conf_pct}%</div><div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{conf_pct}%"></div></div></div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="dark-card"><div class="dark-card-label">Sources used</div><div style="font-size:24px;font-weight:500;color:#93C5FD">{result.get("chunks_used",0)}</div></div>', unsafe_allow_html=True)
                        sections = result.get("sections_used",[])
                        if sections:
                            tags = "".join(section_tag(s) for s in sections[:4])
                            st.markdown(f'<div class="dark-card"><div class="dark-card-label">Sections searched</div><div style="margin-top:6px">{tags}</div></div>', unsafe_allow_html=True)

                    with col_ans:
                        ans_html = result["answer"].replace("\n","<br>")
                        st.markdown(f'<div class="glow-card"><div class="glow-card-title">Answer</div><div class="answer-text">{ans_html}</div></div>', unsafe_allow_html=True)

                        if result.get("citations"):
                            st.markdown('<div class="section-divider">Sources</div>', unsafe_allow_html=True)
                            for cite in result["citations"]:
                                tg = section_tag(cite.get("section_key","unknown")) if "section_key" in cite else f'<span class="tag-gen">{cite.get("section","General")}</span>'
                                url_link = f' · <a href="{cite["url"]}" target="_blank" style="color:#3B82F6;font-size:10px">SEC ↗</a>' if cite.get("url") else ""
                                st.markdown(f'<div class="citation-card"><div class="citation-header">Source {cite["source"]} · {cite["ticker"]} {cite["form"]} {cite["year"]} {tg}<span style="font-size:10px;color:#475569;float:right">score:{cite["score"]}{url_link}</span></div><div class="citation-preview">{cite["preview"]}</div></div>', unsafe_allow_html=True)

                        if show_fu and result.get("followups"):
                            st.markdown('<div class="section-divider">Suggested Follow-ups</div>', unsafe_allow_html=True)
                            for fu in result["followups"]:
                                if st.button(f"→  {fu}", key=f"fu_{fu[:30]}", use_container_width=True):
                                    st.session_state["question"] = fu
                                    st.rerun()

                except Exception as e:
                    st.error(f"Error: {e}")
                    st.info("Make sure you have run all pipeline steps. See SETUP_GUIDE.md")


# ══════════════════════════════════════════════════════════════════════════════
# MODE 2 — COMPARE COMPANIES
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "🔄 Compare Companies":
    st.markdown("""<div class="page-header"><div><div class="page-title">Multi-Company Comparison</div><div class="page-sub">Analyze the same topic across multiple companies' SEC filings simultaneously</div></div><div class="pill-info">Exceptional Feature</div></div>""", unsafe_allow_html=True)

    cc1,cc2 = st.columns([3,1])
    compare_q  = cc1.text_input("", placeholder="e.g. What are the main liquidity risks?", label_visibility="collapsed")
    compare_yr = cc2.selectbox("", ["All",2023,2022,2021,2020,2019], label_visibility="collapsed", key="cyr")

    mode_sel = st.radio("", ["Pick companies","Compare by sector"], horizontal=True, label_visibility="collapsed")
    if mode_sel == "Pick companies":
        selected = st.multiselect("", ALL_TICKERS, default=["AAPL","MSFT","GOOGL"], label_visibility="collapsed")
    else:
        s_choice = st.selectbox("", list(SECTORS.keys()), label_visibility="collapsed")
        selected = SECTORS[s_choice][:5]
        st.markdown(f'<div style="font-size:12px;color:#475569">Comparing: {" · ".join(selected)}</div>', unsafe_allow_html=True)

    if st.button("🔄  Run Comparison", type="primary"):
        if not compare_q.strip():
            st.warning("Please enter a comparison question.")
        elif len(selected) < 2:
            st.warning("Please select at least 2 companies.")
        else:
            with st.spinner(f"Analyzing {len(selected)} companies..."):
                try:
                    from compare_engine import compare_companies
                    y      = compare_yr if compare_yr != "All" else None
                    result = compare_companies(compare_q, tickers=selected, year=y)

                    cols = st.columns(len(selected))
                    for i,(col,tick) in enumerate(zip(cols,selected)):
                        n   = result["chunks_per_company"].get(tick,0)
                        sc  = STRIPE_CLASSES[i % len(STRIPE_CLASSES)]
                        bg  = BAR_GRADIENTS[i % len(BAR_GRADIENTS)]
                        cites = result["citations"].get(tick,[])
                        prev  = cites[0]["preview"][:120] if cites else "No relevant filings found."
                        col.markdown(f'<div class="comp-card {sc}"><div class="comp-ticker">{tick}</div><div class="comp-meta">{n} sources found</div><div class="comp-bar" style="{bg};width:{min(100,n*25)}%"></div><div class="comp-text">{prev}</div></div>', unsafe_allow_html=True)

                    ans_html = result["answer"].replace("\n","<br>")
                    st.markdown(f'<div class="glow-card" style="margin-top:14px"><div class="glow-card-title">Comparative Analysis &amp; Analyst Takeaway</div><div class="answer-text">{ans_html}</div></div>', unsafe_allow_html=True)

                    st.markdown('<div class="section-divider">Sources by Company</div>', unsafe_allow_html=True)
                    tabs = st.tabs(selected)
                    for tab,tick in zip(tabs,selected):
                        with tab:
                            for c in result["citations"].get(tick,[]):
                                st.markdown(f'<div class="citation-card"><div class="citation-header">{c["citation"]}<span style="float:right;font-size:10px;color:#475569">score:{c["score"]}</span></div><div class="citation-preview">{c["preview"]}</div></div>', unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# MODE 3 — 5-YEAR TREND ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "📈 5-Year Trend Analysis":
    st.markdown("""<div class="page-header"><div><div class="page-title">5-Year Trend Analysis</div><div class="page-sub">Track how language, risks and financials evolved from 2019 to 2023</div></div><div class="pill-info">Exceptional Feature</div></div>""", unsafe_allow_html=True)

    analysis_type = st.radio("", ["Text Trend","Financial Metrics","Emerging Risks"], horizontal=True, label_visibility="collapsed")
    tc1,tc2 = st.columns([1,3])
    trend_ticker = tc1.selectbox("", ALL_TICKERS, key="tt", label_visibility="collapsed")

    if analysis_type == "Text Trend":
        trend_q = tc2.text_input("", placeholder="Topic to track e.g. supply chain risk, AI strategy", label_visibility="collapsed", key="tq")
        if st.button("📈  Analyze Trend", type="primary"):
            if not trend_q.strip():
                st.warning("Please enter a topic.")
            else:
                with st.spinner(f"Analyzing {trend_ticker} across 5 years..."):
                    try:
                        from trend_analyzer import analyze_trend
                        result = analyze_trend(trend_ticker, trend_q)
                        dp     = result.get("data_points",{})
                        yc     = st.columns(5)
                        for col,yr_val in zip(yc,[2019,2020,2021,2022,2023]):
                            found = dp.get(yr_val,0)
                            dot   = '<div class="year-dot-on"></div>' if found > 0 else '<div class="year-dot-off"></div>'
                            cc    = "year-card-active" if found > 0 else "year-card"
                            lbl   = f"{found} sources" if found > 0 else "no data"
                            col.markdown(f'<div class="{cc}"><div class="year-num">{yr_val}</div>{dot}<div class="year-lbl">{lbl}</div></div>', unsafe_allow_html=True)
                        ans_html = result["answer"].replace("\n","<br>")
                        st.markdown(f'<div class="glow-card" style="margin-top:14px"><div class="glow-card-title">Trend Analysis — {trend_ticker} · {trend_q}</div><div class="answer-text">{ans_html}</div></div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error: {e}")

    elif analysis_type == "Financial Metrics":
        if not DB_PATH.exists():
            st.error("Database not found. Run sql_database.py first.")
        else:
            metric = tc2.selectbox("", FIELDS, format_func=lambda x: FIELD_LABELS[x], key="tm", label_visibility="collapsed")
            conn   = sqlite3.connect(DB_PATH)
            df     = pd.read_sql(f"SELECT year, {metric} FROM financial_metrics WHERE ticker=? AND form='10-K' AND {metric} IS NOT NULL ORDER BY year", conn, params=(trend_ticker,))
            conn.close()
            if df.empty:
                st.warning(f"No financial data for {trend_ticker}.")
            else:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df["year"], y=df[metric], mode="lines+markers+text",
                    line=dict(color="#3B82F6",width=3),
                    marker=dict(size=10,color="#6366F1",line=dict(color="#3B82F6",width=2)),
                    text=[f"${v:,.0f}M" for v in df[metric]], textposition="top center",
                    textfont=dict(color="#94A3B8",size=11),
                    fill="tozeroy", fillcolor="rgba(59,130,246,0.07)",
                    name=FIELD_LABELS[metric],
                ))
                fig.update_layout(**PLOTLY_LAYOUT,
                    title=dict(text=f"{trend_ticker} — {FIELD_LABELS[metric]} (2019–2023)", font=dict(color="#F1F5F9",size=14)),
                    height=380)
                st.plotly_chart(fig, use_container_width=True)
                if len(df) > 1:
                    df["YoY Growth %"] = (df[metric].pct_change()*100).round(1)
                    st.dataframe(df.rename(columns={metric:FIELD_LABELS[metric]}), hide_index=True, use_container_width=True)

    else:
        if st.button("🔍  Detect Emerging Risks", type="primary"):
            with st.spinner(f"Comparing {trend_ticker} risk language 2019 vs 2023..."):
                try:
                    from trend_analyzer import detect_emerging_risks
                    result = detect_emerging_risks(trend_ticker)
                    rc1,rc2,rc3 = st.columns(3)
                    for col,title,dot_col,txt_col,items in [
                        (rc1,"New risks (emerged)","#F43F5E","#FDA4AF",result["emerging"]),
                        (rc2,"Consistent risks",   "#F59E0B","#FCD34D",result["stable"]),
                        (rc3,"Faded risks",         "#10B981","#34D399",result["faded"]),
                    ]:
                        rows = "".join(f'<div class="risk-item">{r}</div>' for r in (items or ["None detected"]))
                        col.markdown(f'<div class="risk-col"><div class="risk-col-title"><div style="width:7px;height:7px;border-radius:50%;background:{dot_col};flex-shrink:0"></div><span style="color:{txt_col}">{title}</span></div>{rows}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="dark-card" style="margin-top:12px"><div style="font-size:12px;color:#94A3B8">{result["summary"]}</div></div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# MODE 4 — FINANCIAL DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "📊 Financial Dashboard":
    st.markdown("""<div class="page-header"><div><div class="page-title">Financial Dashboard</div><div class="page-sub">Extracted metrics from 250+ SEC filings across 50 companies</div></div><div class="pill-live">● 4,305 data points</div></div>""", unsafe_allow_html=True)

    if not DB_PATH.exists():
        st.error("Database not found. Run sql_database.py first.")
    else:
        conn = sqlite3.connect(DB_PATH)
        total_docs = pd.read_sql("SELECT COUNT(*) as n FROM financial_metrics", conn).iloc[0]["n"]
        companies  = pd.read_sql("SELECT COUNT(DISTINCT ticker) as n FROM financial_metrics", conn).iloc[0]["n"]
        yrs        = pd.read_sql("SELECT MIN(year) as mn, MAX(year) as mx FROM financial_metrics", conn).iloc[0]

        mc1,mc2,mc3,mc4 = st.columns(4)
        mc1.metric("Total Filings",  f"{int(total_docs):,}")
        mc2.metric("Companies",      f"{int(companies)}")
        mc3.metric("Year Range",     f"{int(yrs['mn'])}–{int(yrs['mx'])}")
        mc4.metric("Data Points",    f"{int(total_docs)*len(FIELDS):,}")

        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        tab1,tab2,tab3 = st.tabs(["Cross-Company View","Single Company Deep Dive","Rankings"])

        with tab1:
            dm1,dm2 = st.columns([2,1])
            metric  = dm1.selectbox("Metric:", FIELDS, format_func=lambda x: FIELD_LABELS[x], key="dbm")
            yr_filt = dm2.selectbox("Year:", [2023,2022,2021,2020,2019], key="dby")
            tf = f"AND ticker='{ticker}'" if ticker != "All" else ""
            df = pd.read_sql(f"SELECT ticker, {metric} FROM financial_metrics WHERE form='10-K' AND year={yr_filt} AND {metric} IS NOT NULL {tf} ORDER BY {metric} DESC LIMIT 20", conn)
            if not df.empty:
                fig = px.bar(df, x="ticker", y=metric, color=metric,
                    color_continuous_scale=[[0,"#1C2539"],[0.5,"#3B82F6"],[1,"#6366F1"]],
                    labels={metric:"Value ($M)","ticker":""})
                fig.update_layout(**PLOTLY_LAYOUT, showlegend=False,
                    title=dict(text=f"{FIELD_LABELS[metric]} — {yr_filt}", font=dict(color="#F1F5F9",size=14)), height=380)
                fig.update_traces(marker_line_color="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df.rename(columns={metric:f"{FIELD_LABELS[metric]} ($M)"}), hide_index=True, use_container_width=True)

        with tab2:
            dt1,dt2 = st.columns([1,2])
            sel_t   = dt1.selectbox("Company:", ALL_TICKERS, key="dt")
            mets    = dt2.multiselect("Metrics:", FIELDS, default=["total_revenue","net_income","operating_cash_flow"], format_func=lambda x: FIELD_LABELS[x])
            if mets:
                df2 = pd.read_sql(f"SELECT year, {', '.join(mets)} FROM financial_metrics WHERE ticker=? AND form='10-K' ORDER BY year", conn, params=(sel_t,))
                if not df2.empty:
                    fig2 = go.Figure()
                    for i,m in enumerate(mets):
                        if m in df2.columns:
                            fig2.add_trace(go.Scatter(x=df2["year"], y=df2[m], name=FIELD_LABELS[m],
                                mode="lines+markers", line=dict(color=GRADIENT_COLORS[i%len(GRADIENT_COLORS)],width=2), marker=dict(size=8)))
                    fig2.update_layout(**PLOTLY_LAYOUT,
                        title=dict(text=f"{sel_t} — Multi-Metric Trend", font=dict(color="#F1F5F9",size=14)),
                        height=400, legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color="#94A3B8")))
                    st.plotly_chart(fig2, use_container_width=True)
                    st.dataframe(df2, hide_index=True, use_container_width=True)

        with tab3:
            rk1,rk2 = st.columns([2,1])
            rank_m  = rk1.selectbox("Rank by:", FIELDS, format_func=lambda x: FIELD_LABELS[x], key="rm")
            rank_y  = rk2.selectbox("Year:", [2023,2022,2021,2020,2019], key="ry")
            df_rank = pd.read_sql(f"SELECT ticker, {rank_m} FROM financial_metrics WHERE form='10-K' AND year={rank_y} AND {rank_m} IS NOT NULL ORDER BY {rank_m} DESC LIMIT 15", conn)
            if not df_rank.empty:
                df_rank.insert(0,"Rank",range(1,len(df_rank)+1))
                df_rank[f"{FIELD_LABELS[rank_m]} ($M)"] = df_rank[rank_m].apply(lambda x: f"${x:,.0f}M")
                st.dataframe(df_rank[["Rank","ticker",f"{FIELD_LABELS[rank_m]} ($M)"]], hide_index=True, use_container_width=True)

        conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# MODE 5 — SQL EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("""<div class="page-header"><div><div class="page-title">SQL Explorer</div><div class="page-sub">Query the financial database directly with full SQL support</div></div><div class="pill-live">● Database connected</div></div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="dark-card" style="margin-bottom:14px">
      <div style="font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px">Available tables</div>
      <div style="font-size:12px;color:#94A3B8;line-height:1.8">
        <b style="color:#93C5FD">financial_metrics</b> — ticker, year, form, total_revenue, net_income, gross_profit,
        operating_income, eps_diluted, total_assets, total_liabilities, total_equity,
        long_term_debt, cash_and_equivalents, operating_cash_flow, capital_expenditures, research_and_development<br>
        <b style="color:#93C5FD">rag_evaluations</b> — question, answer, ticker, year, completeness, faithfulness, hallucination
      </div>
    </div>
    """, unsafe_allow_html=True)

    default_sql = """-- Top 10 companies by revenue in 2023
SELECT ticker, year, total_revenue, net_income, eps_diluted
FROM financial_metrics
WHERE form = '10-K' AND year = 2023
ORDER BY total_revenue DESC
LIMIT 10;"""

    sql = st.text_area("", value=default_sql, height=140, label_visibility="collapsed")

    sc1,sc2,_ = st.columns([1,1,4])
    run_clicked = sc1.button("▶  Run Query", type="primary")

    if run_clicked:
        if not DB_PATH.exists():
            st.error("Database not found. Run sql_database.py first.")
        else:
            try:
                conn = sqlite3.connect(DB_PATH)
                df   = pd.read_sql(sql, conn)
                conn.close()
                st.markdown(f'<div style="font-size:11px;color:#475569;margin-bottom:8px">{len(df)} rows returned</div>', unsafe_allow_html=True)
                st.dataframe(df, use_container_width=True, hide_index=True)
                sc2.download_button("⬇  CSV", df.to_csv(index=False), "results.csv", "text/csv")
            except Exception as e:
                st.error(f"SQL error: {e}")
