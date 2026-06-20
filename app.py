import streamlit as st
import pandas as pd
import sys, os

# ── Path setup ─────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.loader import load_dataset
from core.cleaner import clean_dataset
from core.profiler import profile_dataset
from core.semantics import classify_columns
from core.metric_engine import infer_kpis
from core.patterns import discover_patterns
def select_charts(profile, semantics, df=None):
    def _safe_cats(df, cols, max_u=50):
        return [c for c in cols if c in df.columns and 2 <= df[c].nunique() <= max_u]

    date_cols = semantics.get("date", [])
    numeric_cols = semantics.get("numeric", [])
    raw_cats = semantics.get("categorical", [])

    categorical_cols = _safe_cats(df, raw_cats) if df is not None else [
        c for c in raw_cats if not any(
            kw in c.lower() for kw in ["id","code","key","name","ref","sku"]
        )
    ]
    safe_metrics = [
        c for c in numeric_cols
        if c in profile.get("numeric_cols", [])
        and not any(kw in c.lower() for kw in ["id","code","key","zip","postal","row"])
    ]

    specs = []
    if date_cols and safe_metrics:
        specs.append({"type":"time_series","date":date_cols[0],"metric":safe_metrics[0],"label":f"{safe_metrics[0]} over time"})
    if categorical_cols and safe_metrics:
        specs.append({"type":"category_bar","category":categorical_cols[0],"metric":safe_metrics[0],"label":f"{safe_metrics[0]} by {categorical_cols[0]}"})
    if categorical_cols and safe_metrics:
        cat = categorical_cols[1] if len(categorical_cols)>1 else categorical_cols[0]
        m = safe_metrics[1] if len(safe_metrics)>1 else safe_metrics[0]
        specs.append({"type":"pie","category":cat,"metric":m,"label":f"{m} share by {cat}"})
    if safe_metrics:
        specs.append({"type":"histogram","column":safe_metrics[0],"label":f"Distribution of {safe_metrics[0]}"})
    if len(safe_metrics) >= 3:
        specs.append({"type":"correlation","columns":safe_metrics,"label":"Correlation Matrix"})
    if categorical_cols and safe_metrics:
        specs.append({"type":"box","category":categorical_cols[0],"metric":safe_metrics[0],"label":f"{safe_metrics[0]} spread by {categorical_cols[0]}"})
    if len(safe_metrics) >= 2:
        specs.append({"type":"scatter","x":safe_metrics[0],"y":safe_metrics[1],"label":f"{safe_metrics[0]} vs {safe_metrics[1]}"})
    return specs
from viz.charts import (
    time_series, category_bar, histogram,
    scatter_plot, correlation_heatmap, pie_chart, box_plot,
)
from ai.explanation_generator import generate_ai_insights

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AnalytiQ · Data Intelligence Platform",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0F1117;
    color: #E2E8F0;
}
.stApp { background-color: #0F1117; }

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 3rem 2rem; max-width: 100%; }

/* ── Force sidebar always visible — hide ALL toggle/collapse controls ── */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[data-testid="stSidebarCollapseButton"],
section[data-testid="stSidebar"] > div > div > div > button {
    display: none !important;
    visibility: hidden !important;
    pointer-events: none !important;
}

/* ── Sidebar always expanded width ── */
[data-testid="stSidebar"] {
    min-width: 260px !important;
    max-width: 260px !important;
    width: 260px !important;
    transform: none !important;
    left: 0 !important;
}

/* ── Sidebar styling ── */
[data-testid="stSidebar"] {
    background: #13151F !important;
    border-right: 1px solid #1E2132 !important;
    min-width: 260px !important;
    max-width: 260px !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }
[data-testid="stSidebarNav"] { display: none; }

/* ── Sidebar logo area ── */
.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0 0 1.5rem 0;
    border-bottom: 1px solid #1E2132;
    margin-bottom: 1.5rem;
}
.sidebar-logo-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #6366F1 0%, #22D3EE 100%);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
}
.sidebar-logo-text {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 18px;
    font-weight: 700;
    letter-spacing: -0.3px;
    color: #E2E8F0;
}
.sidebar-logo-sub {
    font-size: 10px;
    color: #475569;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: -2px;
}

/* ── Sidebar section headers ── */
.sidebar-section {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #475569;
    margin: 1.2rem 0 0.5rem 0;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: #1A1D27;
    border: 1.5px dashed #2D3148;
    border-radius: 12px;
    padding: 0.5rem;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: #6366F1; }
[data-testid="stFileUploaderDropzoneInstructions"] { color: #64748B !important; }

/* ── KPI cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.kpi-card {
    background: #1A1D27;
    border: 1px solid #1E2132;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.15s, border-color 0.15s;
}
.kpi-card:hover { transform: translateY(-2px); border-color: #6366F1; }
.kpi-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 14px 14px 0 0;
}
.kpi-card.accent1::before { background: linear-gradient(90deg, #6366F1, #818CF8); }
.kpi-card.accent2::before { background: linear-gradient(90deg, #22D3EE, #06B6D4); }
.kpi-card.accent3::before { background: linear-gradient(90deg, #F59E0B, #FBBF24); }
.kpi-card.accent4::before { background: linear-gradient(90deg, #10B981, #34D399); }
.kpi-card.accent5::before { background: linear-gradient(90deg, #EC4899, #F472B6); }
.kpi-card.accent6::before { background: linear-gradient(90deg, #8B5CF6, #A78BFA); }

.kpi-label {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: #64748B;
    margin-bottom: 0.5rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.kpi-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 26px;
    font-weight: 700;
    color: #F1F5F9;
    line-height: 1;
    letter-spacing: -0.5px;
}
.kpi-sub {
    font-size: 11px;
    color: #475569;
    margin-top: 4px;
}

/* ── Section headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid #1E2132;
}
.section-icon {
    width: 28px; height: 28px;
    background: rgba(99,102,241,0.15);
    border-radius: 7px;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px;
}
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 15px;
    font-weight: 600;
    color: #CBD5E1;
    letter-spacing: -0.2px;
}

/* ── Chart cards ── */
.chart-card {
    background: #1A1D27;
    border: 1px solid #1E2132;
    border-radius: 14px;
    padding: 0.25rem;
    margin-bottom: 1rem;
}

/* ── AI insights card ── */
.insights-card {
    background: linear-gradient(135deg, #1A1D27 0%, #161926 100%);
    border: 1px solid #2D3148;
    border-left: 3px solid #6366F1;
    border-radius: 14px;
    padding: 1.5rem 1.8rem;
    font-size: 14px;
    line-height: 1.8;
    color: #CBD5E1;
}
.insights-card ul { margin: 0; padding-left: 1rem; }
.insights-card li { margin-bottom: 0.4rem; }

/* ── Data quality badge ── */
.quality-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(16,185,129,0.12);
    border: 1px solid rgba(16,185,129,0.25);
    color: #10B981;
    font-size: 11px;
    font-weight: 500;
    padding: 3px 10px;
    border-radius: 20px;
}

/* ── Tab styling ── */
[data-testid="stTabs"] [role="tablist"] {
    background: #13151F;
    border: 1px solid #1E2132;
    border-radius: 10px;
    padding: 4px;
    gap: 2px;
}
[data-testid="stTabs"] [role="tab"] {
    background: transparent;
    color: #64748B;
    border-radius: 7px;
    font-size: 13px;
    font-weight: 500;
    padding: 6px 16px;
    border: none;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: #1E2132;
    color: #E2E8F0;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: #1A1D27;
    border: 1px solid #2D3148;
    border-radius: 8px;
    color: #E2E8F0;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #1A1D27;
    border: 1px solid #1E2132 !important;
    border-radius: 10px;
}

/* ── Top bar ── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #1E2132;
}
.topbar-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: #F1F5F9;
    letter-spacing: -0.5px;
}
.topbar-meta {
    display: flex;
    gap: 1rem;
    align-items: center;
}
.meta-chip {
    background: #1A1D27;
    border: 1px solid #1E2132;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 12px;
    color: #94A3B8;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def fmt_number(val):
    if not isinstance(val, (int, float)):
        return str(val)
    if abs(val) >= 1_000_000_000:
        return f"{val/1_000_000_000:.1f}B"
    if abs(val) >= 1_000_000:
        return f"{val/1_000_000:.1f}M"
    if abs(val) >= 1_000:
        return f"{val/1_000:.1f}K"
    if isinstance(val, float):
        return f"{val:,.2f}"
    return f"{int(val):,}"


ACCENT_CLASSES = ["accent1", "accent2", "accent3", "accent4", "accent5", "accent6"]


def render_kpi_cards(kpis):
    cards_html = '<div class="kpi-grid">'
    for i, (label, value) in enumerate(kpis):
        cls = ACCENT_CLASSES[i % len(ACCENT_CLASSES)]
        formatted = fmt_number(value)
        cards_html += f"""
        <div class="kpi-card {cls}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{formatted}</div>
        </div>"""
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)


def render_section(icon, title):
    st.markdown(f"""
    <div class="section-header">
        <div class="section-icon">{icon}</div>
        <div class="section-title">{title}</div>
    </div>""", unsafe_allow_html=True)


def render_chart(fig, key=None):
    if fig is not None:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=key)
        st.markdown('</div>', unsafe_allow_html=True)


def chart_dispatcher(spec, df, key=None):
    t = spec["type"]
    if t == "time_series":
        return time_series(df, spec["date"], spec["metric"])
    elif t == "category_bar":
        return category_bar(df, spec["category"], spec["metric"], top_n=15)
    elif t == "histogram":
        return histogram(df, spec["column"])
    elif t == "scatter":
        return scatter_plot(df, spec["x"], spec["y"])
    elif t == "correlation":
        return correlation_heatmap(df, spec["columns"])
    elif t == "pie":
        return pie_chart(df, spec["category"], spec["metric"], top_n=10)
    elif t == "box":
        return box_plot(df, spec["category"], spec["metric"], top_n=12)
    return None


# ── Session state for file persistence ────────────────────────────────────────
if "uploaded_file_bytes" not in st.session_state:
    st.session_state.uploaded_file_bytes = None
    st.session_state.uploaded_file_name = None


def handle_upload(f):
    if f is not None:
        st.session_state.uploaded_file_bytes = f.read()
        st.session_state.uploaded_file_name = f.name


# ── Sidebar (static — no collapse button) ─────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">⬡</div>
        <div>
            <div class="sidebar-logo-text">AnalytiQ</div>
            <div class="sidebar-logo-sub">Data Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Dataset</div>', unsafe_allow_html=True)
    sidebar_upload = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx"],
        label_visibility="collapsed",
        key="sidebar_uploader",
        on_change=lambda: handle_upload(st.session_state.sidebar_uploader),
    )

    has_file = st.session_state.uploaded_file_bytes is not None

    if has_file:
        st.markdown(f"""
        <div style="background:#1A2A1A;border:1px solid #1E3A1E;border-radius:8px;
                    padding:8px 12px;margin:0.5rem 0;font-size:12px;color:#4ADE80;">
            ✓ {st.session_state.uploaded_file_name}
        </div>
        """, unsafe_allow_html=True)

        if st.button("✕ Clear dataset", use_container_width=True):
            st.session_state.uploaded_file_bytes = None
            st.session_state.uploaded_file_name = None
            st.rerun()

        st.markdown('<div class="sidebar-section">Configuration</div>', unsafe_allow_html=True)
        show_profile = st.toggle("Show dataset profile", value=False)
        ai_insights = st.toggle("AI insights", value=False)
    else:
        show_profile = False
        ai_insights = False

    st.markdown("---")
    st.markdown(
        '<div style="font-size:11px;color:#334155;text-align:center">AnalytiQ v1.0 · Built with Streamlit</div>',
        unsafe_allow_html=True,
    )


# ── Main ───────────────────────────────────────────────────────────────────────

if not has_file:
    # ── Landing state ─────────────────────────────────────────────────────────
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                min-height:30vh;text-align:center;gap:1.2rem;padding-top:3rem;">
        <div style="font-size:56px;filter:drop-shadow(0 0 30px #6366F1);">⬡</div>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:36px;font-weight:700;
                    color:#F1F5F9;letter-spacing:-1px;line-height:1.2;">
            Turn raw data into<br>
            <span style="background:linear-gradient(135deg,#6366F1,#22D3EE);
                         -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                instant intelligence
            </span>
        </div>
        <div style="color:#64748B;font-size:15px;max-width:480px;line-height:1.7;">
            Upload any CSV or Excel file. AnalytiQ auto-detects your metrics,
            selects the most relevant charts, and surfaces business insights —
            powered by an LLM pipeline.
        </div>
        <div style="display:flex;gap:1rem;margin-top:0.5rem;flex-wrap:wrap;justify-content:center;">
            <span style="background:#1A1D27;border:1px solid #1E2132;border-radius:20px;
                         padding:6px 16px;font-size:12px;color:#94A3B8;">📊 Auto KPIs</span>
            <span style="background:#1A1D27;border:1px solid #1E2132;border-radius:20px;
                         padding:6px 16px;font-size:12px;color:#94A3B8;">🤖 AI Insights</span>
            <span style="background:#1A1D27;border:1px solid #1E2132;border-radius:20px;
                         padding:6px 16px;font-size:12px;color:#94A3B8;">📈 Smart Charts</span>
            <span style="background:#1A1D27;border:1px solid #1E2132;border-radius:20px;
                         padding:6px 16px;font-size:12px;color:#94A3B8;">🔍 Data Profiling</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.markdown("""
        <div style="background:#1A1D27;border:2px dashed #2D3148;border-radius:16px;
                    padding:1.5rem;text-align:center;margin-bottom:0.5rem;">
            <div style="font-size:28px;margin-bottom:0.5rem;">📂</div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:16px;
                        font-weight:600;color:#CBD5E1;margin-bottom:0.3rem;">
                Drop your file here
            </div>
            <div style="font-size:12px;color:#475569;">CSV or Excel · any size</div>
        </div>
        """, unsafe_allow_html=True)
        landing_upload = st.file_uploader(
            "Upload dataset",
            type=["csv", "xlsx"],
            label_visibility="collapsed",
            key="landing_uploader",
        )
        if landing_upload is not None:
            handle_upload(landing_upload)
            st.rerun()

    st.stop()


# ── Pipeline ───────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_pipeline(file_bytes, file_name):
    import io
    file_like = io.BytesIO(file_bytes)
    file_like.name = file_name

    raw_df = load_dataset(file_like)
    df = clean_dataset(raw_df)
    profile = profile_dataset(df)
    semantics = classify_columns(df)
    kpis = infer_kpis(df, semantics)
    patterns = discover_patterns(df, profile)
    chart_specs = select_charts(profile, semantics, df)
    return df, raw_df, profile, semantics, kpis, patterns, chart_specs


with st.spinner("Analysing your dataset…"):
    try:
        df, raw_df, profile, semantics, kpis, patterns, chart_specs = run_pipeline(
            st.session_state.uploaded_file_bytes,
            st.session_state.uploaded_file_name,
        )
    except Exception as e:
        st.error(f"**Failed to process file:** {e}")
        st.stop()


# ── Top bar ────────────────────────────────────────────────────────────────────
fname = st.session_state.uploaded_file_name.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()
null_pct = raw_df.isnull().mean().mean() * 100
dups = len(raw_df) - len(raw_df.drop_duplicates())

st.markdown(f"""
<div class="topbar">
    <div class="topbar-title">{fname}</div>
    <div class="topbar-meta">
        <span class="meta-chip">🗃 {profile['rows']:,} rows</span>
        <span class="meta-chip">⬡ {profile['columns']} cols</span>
        <span class="quality-badge">✓ {100-null_pct:.0f}% complete</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_overview, tab_charts, tab_explore, tab_data = st.tabs([
    "  Overview  ", "  Charts  ", "  Explorer  ", "  Data  "
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:

    # KPI cards
    render_section("📊", "Key Performance Indicators")
    if kpis:
        render_kpi_cards(kpis)
    else:
        st.info("No numeric metrics detected in this dataset.")

    # AI Insights
    if ai_insights:
        render_section("🤖", "AI-Generated Insights")
        with st.spinner("Generating insights…"):
            insight_text = generate_ai_insights(profile, semantics, patterns, kpis)

        # Format bullet points
        lines = [l.strip() for l in insight_text.strip().splitlines() if l.strip()]
        formatted = "".join(f"<li>{l.lstrip('•- ')}</li>" for l in lines)
        st.markdown(
            f'<div class="insights-card"><ul>{formatted}</ul></div>',
            unsafe_allow_html=True,
        )
    else:
        render_section("🤖", "AI-Generated Insights")
        st.markdown(
            '<div class="insights-card" style="color:#475569;font-style:italic;">'
            'Enable "AI insights" in the sidebar.'
            '</div>',
            unsafe_allow_html=True,
        )

    # Data quality summary
    render_section("🔍", "Dataset Profile")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Numeric cols", len(profile["numeric_cols"]))
    c2.metric("Categorical cols", len(profile["categorical_cols"]))
    c3.metric("Date cols", len(profile["date_cols"]))
    c4.metric("Duplicate rows removed", dups)

    if show_profile:
        with st.expander("Column types detected", expanded=False):
            col_data = []
            for c in df.columns:
                if c in profile["date_cols"]:
                    col_type = "📅 Date"
                elif c in profile["numeric_cols"]:
                    col_type = "🔢 Numeric"
                elif c in semantics.get("id", []):
                    col_type = "🔑 ID"
                else:
                    col_type = "🏷 Categorical"
                col_data.append({"Column": c, "Type": col_type, "Nulls (%)": f"{df[c].isnull().mean()*100:.1f}%"})
            st.dataframe(pd.DataFrame(col_data), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — AUTO CHARTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_charts:
    if not chart_specs:
        st.info("Not enough column variety to auto-generate charts. Try the Explorer tab.")
    else:
        render_section("📈", f"{len(chart_specs)} charts auto-selected for your data")

        # Render charts in 2-column grid
        pairs = [chart_specs[i:i+2] for i in range(0, len(chart_specs), 2)]
        for pair_idx, pair in enumerate(pairs):
            cols = st.columns(len(pair))
            for col_idx, (col, spec) in enumerate(zip(cols, pair)):
                with col:
                    fig = chart_dispatcher(spec, df)
                    render_chart(fig, key=f"auto_chart_{pair_idx}_{col_idx}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CUSTOM EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab_explore:
    render_section("🎛", "Custom Chart Builder")

    # ── Clean column lists from semantics (not raw profiler) ──────────────────
    numeric_cols  = semantics.get("numeric", [])
    date_cols     = semantics.get("date", [])
    id_cols       = semantics.get("id", [])

    # Only categoricals with ≤50 unique values (genuinely groupable)
    clean_cats = [
        c for c in semantics.get("categorical", [])
        if c in df.columns and 2 <= df[c].nunique() <= 50
    ]

    # Fallbacks if lists are empty
    any_col = df.columns.tolist()

    chart_type = st.selectbox(
        "Chart type",
        ["Bar chart", "Line / Time series", "Scatter plot",
         "Histogram", "Box plot", "Pie / Donut", "Correlation heatmap"],
        key="explorer_chart_type",
    )

    fig = None
    warn_msg = None

    # ── Bar chart ─────────────────────────────────────────────────────────────
    if chart_type == "Bar chart":
        cat_opts = clean_cats or any_col
        num_opts = numeric_cols or any_col
        c1, c2 = st.columns(2)
        x = c1.selectbox("Category (X axis)", cat_opts, key="bar_x")
        y = c2.selectbox("Metric (Y axis)", num_opts, key="bar_y")
        top_n = st.slider("Max bars (+ Others)", 5, 30, 15, key="bar_topn")
        fig = category_bar(df, x, y, top_n=top_n)

    # ── Line / Time series ────────────────────────────────────────────────────
    elif chart_type == "Line / Time series":
        if not date_cols:
            warn_msg = "No date columns detected in this dataset."
        else:
            num_opts = numeric_cols or any_col
            c1, c2 = st.columns(2)
            x = c1.selectbox("Date column", date_cols, key="line_x")
            y = c2.selectbox("Metric", num_opts, key="line_y")
            fig = time_series(df, x, y)

    # ── Scatter plot ──────────────────────────────────────────────────────────
    elif chart_type == "Scatter plot":
        if len(numeric_cols) < 2:
            warn_msg = "Need at least 2 numeric columns for a scatter plot."
        else:
            c1, c2 = st.columns(2)
            x = c1.selectbox("X axis", numeric_cols, key="scatter_x")
            y_opts = [c for c in numeric_cols if c != x]
            y = c2.selectbox("Y axis", y_opts, key="scatter_y")
            fig = scatter_plot(df, x, y)

    # ── Histogram ─────────────────────────────────────────────────────────────
    elif chart_type == "Histogram":
        if not numeric_cols:
            warn_msg = "No numeric columns detected."
        else:
            col = st.selectbox("Column", numeric_cols, key="hist_col")
            fig = histogram(df, col)

    # ── Box plot ──────────────────────────────────────────────────────────────
    elif chart_type == "Box plot":
        cat_opts = clean_cats or any_col
        num_opts = numeric_cols or any_col
        c1, c2 = st.columns(2)
        cat    = c1.selectbox("Category", cat_opts, key="box_cat")
        metric = c2.selectbox("Metric", num_opts, key="box_metric")
        top_n  = st.slider("Max categories (+ Others)", 5, 20, 10, key="box_topn")
        fig = box_plot(df, cat, metric, top_n=top_n)

    # ── Pie / Donut ───────────────────────────────────────────────────────────
    elif chart_type == "Pie / Donut":
        cat_opts = clean_cats or any_col
        num_opts = numeric_cols or any_col
        if not cat_opts:
            warn_msg = "No suitable categorical columns found (all have too many unique values)."
        else:
            c1, c2 = st.columns(2)
            cat    = c1.selectbox("Category", cat_opts, key="pie_cat")
            metric = c2.selectbox("Metric", num_opts, key="pie_metric")
            top_n  = st.slider("Max slices (+ Others)", 3, 15, 8, key="pie_topn")
            fig = pie_chart(df, cat, metric, top_n=top_n)
            if fig is None:
                n_unique = df[cat].nunique()
                warn_msg = (
                    f"**'{cat}'** has {n_unique} unique values — the top {top_n} "
                    f"account for less than 40% of total {metric}, making this chart uninformative. "
                    f"Try a column with fewer categories, or switch to a Bar chart."
                )

    # ── Correlation heatmap ───────────────────────────────────────────────────
    elif chart_type == "Correlation heatmap":
        if len(numeric_cols) < 2:
            warn_msg = "Need at least 2 numeric columns for a correlation heatmap."
        else:
            selected = st.multiselect(
                "Select numeric columns (min 2)",
                numeric_cols,
                default=numeric_cols[:6],
                key="corr_cols",
            )
            if len(selected) < 2:
                warn_msg = "Select at least 2 columns."
            else:
                fig = correlation_heatmap(df, selected)

    # ── Render or warn ────────────────────────────────────────────────────────
    if fig is not None:
        render_chart(fig, key=f"explorer_{chart_type.replace(' ','_').replace('/','')}")
    elif warn_msg:
        st.markdown(f"""
        <div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.25);
                    border-left:3px solid #6366F1;border-radius:10px;padding:1rem 1.4rem;
                    font-size:13px;color:#CBD5E1;line-height:1.7;">
            ℹ️ &nbsp; {warn_msg}
        </div>
        """, unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DATA TABLE
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    render_section("🗃", "Dataset Preview")

    search = st.text_input("Filter rows (contains)", placeholder="Search any value…")
    display_df = df.copy()
    if search:
        mask = display_df.apply(
            lambda col: col.astype(str).str.contains(search, case=False, na=False)
        ).any(axis=1)
        display_df = display_df[mask]

    st.dataframe(display_df, use_container_width=True, height=450)
    st.caption(f"Showing {len(display_df):,} of {len(df):,} rows · {df.columns.size} columns")

    # Download cleaned data
    csv = df.to_csv(index=False).encode()
    st.download_button(
        "⬇ Download cleaned dataset",
        data=csv,
        file_name=f"{fname.lower().replace(' ','_')}_cleaned.csv",
        mime="text/csv",
    )