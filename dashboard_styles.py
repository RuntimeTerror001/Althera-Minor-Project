"""
Althera Dashboard — CSS Styles & Theme
"""

DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Global Reset ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: #06060f !important;
    color: #e0e0f0 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1f 0%, #0a0a18 100%) !important;
    border-right: 1px solid rgba(100,100,255,0.08) !important;
}

/* ── Glassmorphic Card ── */
.glass-card {
    background: rgba(15,15,35,0.7);
    border: 1px solid rgba(100,100,255,0.12);
    border-radius: 16px;
    padding: 24px;
    backdrop-filter: blur(20px);
    margin-bottom: 16px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.glass-card-accent {
    background: linear-gradient(135deg, rgba(20,20,50,0.8), rgba(15,15,40,0.6));
    border: 1px solid rgba(80,120,255,0.2);
    border-radius: 16px;
    padding: 24px;
    backdrop-filter: blur(20px);
    margin-bottom: 16px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05);
}

/* ── Metric Cards ── */
.metric-card {
    background: rgba(15,15,40,0.6);
    border: 1px solid rgba(100,100,255,0.1);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.metric-value {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
}
.metric-label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #7a7a9a;
    margin-top: 4px;
}

/* ── Step Progress ── */
.step-active {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    color: white;
    padding: 8px 18px;
    border-radius: 10px;
    font-weight: 600;
    margin: 4px 0;
    font-size: 0.85rem;
}
.step-done {
    background: rgba(34,197,94,0.15);
    color: #4ade80;
    padding: 8px 18px;
    border-radius: 10px;
    font-weight: 500;
    margin: 4px 0;
    border: 1px solid rgba(34,197,94,0.2);
    font-size: 0.85rem;
}
.step-pending {
    background: rgba(30,30,50,0.5);
    color: #5a5a7a;
    padding: 8px 18px;
    border-radius: 10px;
    font-weight: 400;
    margin: 4px 0;
    border: 1px solid rgba(100,100,255,0.06);
    font-size: 0.85rem;
}

/* ── Score Gauge ── */
.score-excellent { color: #4ade80; }
.score-normal { color: #60a5fa; }
.score-mild { color: #fbbf24; }
.score-low { color: #f87171; }
.score-display {
    font-size: 4rem;
    font-weight: 900;
    text-align: center;
    line-height: 1;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(59,130,246,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(59,130,246,0.5) !important;
}

/* ── AI Report ── */
.ai-report {
    background: rgba(10,10,30,0.8);
    border: 1px solid rgba(139,92,246,0.2);
    border-radius: 12px;
    padding: 24px;
    line-height: 1.8;
    font-size: 0.95rem;
    color: #c0c0e0;
    white-space: pre-wrap;
}

/* ── Title ── */
.main-title {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.subtitle {
    color: #6a6a8a;
    font-size: 0.9rem;
    margin-top: 0;
}

/* ── Test Area ── */
.test-display {
    font-size: 3rem;
    font-weight: 700;
    text-align: center;
    padding: 40px;
    background: rgba(15,15,40,0.5);
    border-radius: 16px;
    border: 1px solid rgba(100,100,255,0.1);
    min-height: 120px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.test-instruction {
    text-align: center;
    color: #8a8aaa;
    font-size: 1rem;
    padding: 12px;
}
.emotion-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.emotion-happy { background: rgba(74,222,128,0.2); color: #4ade80; border: 1px solid rgba(74,222,128,0.3); }
.emotion-sad { background: rgba(96,165,250,0.2); color: #60a5fa; border: 1px solid rgba(96,165,250,0.3); }
.emotion-neutral { background: rgba(251,191,36,0.2); color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }

/* ── Plotly Charts ── */
[data-testid="stPlotlyChart"] { border-radius: 12px; overflow: hidden; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: rgba(20,20,45,0.5) !important;
    border-radius: 8px !important;
    color: #8a8aaa !important;
    border: 1px solid rgba(100,100,255,0.08) !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #3b82f6, #7c3aed) !important;
    color: white !important;
}
</style>
"""

PLOTLY_DARK_TEMPLATE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,10,30,0.5)",
    font=dict(family="Inter", color="#b0b0d0"),
    xaxis=dict(gridcolor="rgba(100,100,255,0.06)", zerolinecolor="rgba(100,100,255,0.1)"),
    yaxis=dict(gridcolor="rgba(100,100,255,0.06)", zerolinecolor="rgba(100,100,255,0.1)"),
    margin=dict(l=40, r=20, t=40, b=40),
)
