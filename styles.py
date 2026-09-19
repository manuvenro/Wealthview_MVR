import streamlit as st

# ── Design tokens ────────────────────────────────────────────────────────────
BG          = "#0e1117"
SURFACE     = "#161b24"
SURFACE_2   = "#1c2333"
BORDER      = "#232b3a"
BORDER_SOFT = "#1c2333"

TEXT_PRIMARY   = "#e8e0d5"
TEXT_SECONDARY = "#b8c5d0"
TEXT_MUTED     = "#7a8799"

GOLD        = "#c9a84c"
GOLD_LIGHT  = "#e8c97a"
GOLD_DIM    = "rgba(201,168,76,0.12)"
GOLD_BORDER = "rgba(201,168,76,0.25)"

POSITIVE    = "#5a8f6e"
POSITIVE_BG = "rgba(90,143,110,0.10)"
NEGATIVE    = "#9b4d4d"
NEGATIVE_BG = "rgba(155,77,77,0.10)"

BLUE        = "#4a6fa5"
BLUE_BG     = "rgba(74,111,165,0.10)"

# ── Plotly layout ─────────────────────────────────────────────────────────────
PLOTLY_DARK = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter, sans-serif', color=TEXT_SECONDARY, size=11),
    margin=dict(l=0, r=0, t=40, b=0),
    xaxis=dict(gridcolor=BORDER, tickfont=dict(size=10, color=TEXT_SECONDARY),
               title_font=dict(size=11, color=TEXT_MUTED),
               showline=False, zeroline=False),
    yaxis=dict(gridcolor=BORDER, tickfont=dict(size=10, color=TEXT_SECONDARY),
               title_font=dict(size=11, color=TEXT_MUTED),
               showline=False, zeroline=False),
    hovermode='x unified',
    hoverlabel=dict(bgcolor=SURFACE_2, bordercolor=GOLD,
                    font=dict(size=12, color=TEXT_PRIMARY, family='Inter')),
)


def plotly_layout(**kwargs):
    """Fusiona PLOTLY_DARK con kwargs sin duplicar claves."""
    return {**PLOTLY_DARK, **kwargs}


def inject_global_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Inter:wght@300;400;500;600;700&family=DM+Mono:wght@300;400;500&display=swap');

    /* RESET */
    #MainMenu, footer, header,
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"] {{ display: none !important; }}

    .stApp {{
        background: {BG} !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
    }}
    .block-container {{
        padding-top: 28px !important;
        padding-bottom: 28px !important;
        max-width: 1440px !important;
    }}

    /* SIDEBAR */
    [data-testid="stSidebar"] {{
        background: #0b0f18 !important;
        border-right: 1px solid {BORDER} !important;
    }}
    [data-testid="stSidebar"] p {{
        color: {TEXT_MUTED} !important;
        font-size: 11px !important;
        padding: 0 12px !important;
        letter-spacing: 0.3px;
    }}
    [data-testid="stSidebar"] .stRadio > label {{ display: none !important; }}
    [data-testid="stSidebar"] .stRadio > div {{ gap: 2px !important; }}
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stRadio > div > label,
    [data-testid="stSidebar"] [role="radiogroup"] label,
    [data-testid="stSidebar"] [data-baseweb="radio"] label {{
        display: flex !important;
        align-items: center !important;
        padding: 11px 16px !important;
        border-radius: 6px !important;
        margin: 2px 8px !important;
        cursor: pointer !important;
        transition: all 0.15s ease !important;
        color: {TEXT_SECONDARY} !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        border: none !important;
        letter-spacing: 0.2px !important;
    }}
    [data-testid="stSidebar"] .stRadio label:hover,
    [data-testid="stSidebar"] .stRadio > div > label:hover {{
        background: {SURFACE} !important;
        color: {TEXT_PRIMARY} !important;
    }}
    [data-testid="stSidebar"] .stRadio label[data-checked="true"],
    [data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] {{
        background: {GOLD_DIM} !important;
        color: {GOLD_LIGHT} !important;
        font-weight: 600 !important;
        border-left: 2px solid {GOLD} !important;
    }}
    [data-testid="stSidebar"] .stRadio label > div:first-child,
    [data-testid="stSidebar"] .stRadio > div > label > div:first-child,
    [data-testid="stSidebar"] [data-baseweb="radio"] > div:first-child {{
        display: none !important;
    }}
    [data-testid="stSidebar"] .stRadio label span,
    [data-testid="stSidebar"] .stRadio label p,
    [data-testid="stSidebar"] .stRadio label div {{
        color: inherit !important;
        font-size: inherit !important;
        font-weight: inherit !important;
    }}
    [data-testid="stSidebar"] .stButton > button {{
        background: transparent !important;
        border: 1px solid {BORDER} !important;
        color: {TEXT_MUTED} !important;
        border-radius: 6px !important;
        font-size: 12px !important;
        margin: 4px 10px !important;
        width: calc(100% - 20px) !important;
        letter-spacing: 0.3px;
        transition: all 0.15s !important;
    }}
    [data-testid="stSidebar"] .stButton > button:hover {{
        border-color: {NEGATIVE} !important;
        color: #c47a7a !important;
    }}
    [data-testid="stSidebar"] hr {{
        border-color: {BORDER} !important;
        margin: 12px 10px !important;
    }}

    /* TIPOGRAFÍA */
    h1 {{
        font-family: 'Playfair Display', Georgia, serif !important;
        font-size: 26px !important;
        font-weight: 700 !important;
        color: {TEXT_PRIMARY} !important;
        letter-spacing: -0.3px !important;
        margin-bottom: 2px !important;
    }}
    h2 {{
        font-family: 'Playfair Display', Georgia, serif !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        color: {TEXT_PRIMARY} !important;
    }}
    h3 {{
        font-family: 'Inter', sans-serif !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        color: {TEXT_SECONDARY} !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }}
    p, li {{
        color: {TEXT_SECONDARY} !important;
        font-size: 13px !important;
        line-height: 1.6 !important;
    }}

    /* MÉTRICAS */
    [data-testid="stMetric"] {{
        background: {SURFACE} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 8px !important;
        padding: 18px 20px !important;
    }}
    [data-testid="stMetricLabel"] > div,
    [data-testid="stMetricLabel"] > div > div,
    [data-testid="stMetricLabel"] p {{
        font-size: 10px !important;
        font-weight: 700 !important;
        color: {TEXT_MUTED} !important;
        text-transform: uppercase !important;
        letter-spacing: 1.1px !important;
    }}
    [data-testid="stMetricValue"] > div,
    [data-testid="stMetricValue"] > div > div {{
        font-family: 'Playfair Display', serif !important;
        font-size: 26px !important;
        font-weight: 700 !important;
        color: {TEXT_PRIMARY} !important;
        letter-spacing: -0.5px !important;
        line-height: 1.15 !important;
    }}
    [data-testid="stMetricDelta"] > div {{
        font-size: 12px !important;
        font-weight: 600 !important;
    }}

    /* DATAFRAMES */
    [data-testid="stDataFrame"] {{
        border: 1px solid {BORDER} !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }}
    [data-testid="stDataFrame"] td {{
        color: {TEXT_SECONDARY} !important;
        font-size: 12px !important;
    }}
    [data-testid="stDataFrame"] th {{
        color: {TEXT_MUTED} !important;
        font-size: 10px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
        font-weight: 700 !important;
    }}

    /* BOTONES */
    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, {GOLD} 0%, #a8832a 100%) !important;
        border: none !important;
        border-radius: 6px !important;
        color: #0b0f18 !important;
        font-weight: 700 !important;
        font-size: 11px !important;
        padding: 10px 22px !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 8px rgba(201,168,76,0.25), 0 1px 3px rgba(0,0,0,0.4) !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        opacity: 0.90 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 14px rgba(201,168,76,0.35), 0 2px 6px rgba(0,0,0,0.4) !important;
    }}
    .stButton > button[kind="primary"]:active {{
        transform: translateY(0) !important;
        box-shadow: 0 1px 4px rgba(201,168,76,0.2) !important;
    }}
    .stButton > button {{
        background: {SURFACE_2} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 6px !important;
        color: {TEXT_SECONDARY} !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        letter-spacing: 0.7px !important;
        text-transform: uppercase !important;
        padding: 10px 18px !important;
        transition: all 0.18s ease !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.25) !important;
    }}
    .stButton > button:hover {{
        border-color: {GOLD}88 !important;
        color: {GOLD_LIGHT} !important;
        background: {SURFACE} !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
    }}
    .stButton > button:active {{
        transform: translateY(1px) !important;
        box-shadow: none !important;
    }}

    /* INPUTS */
    .stTextInput input, .stNumberInput input {{
        background: {SURFACE} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 6px !important;
        color: {TEXT_PRIMARY} !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        padding: 10px 14px !important;
        transition: border-color 0.2s !important;
    }}
    .stTextInput input:focus, .stNumberInput input:focus {{
        border-color: {GOLD} !important;
        box-shadow: 0 0 0 3px {GOLD_DIM} !important;
        outline: none !important;
    }}
    .stSelectbox > div > div {{
        background: {SURFACE} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 6px !important;
        color: {TEXT_PRIMARY} !important;
    }}
    [data-baseweb="select"] span,
    [data-baseweb="select"] div {{
        color: {TEXT_PRIMARY} !important;
    }}
    [data-testid="stForm"] {{
        background: {SURFACE} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 8px !important;
        padding: 20px !important;
    }}

    /* TABS */
    .stTabs [data-baseweb="tab-list"] {{
        background: transparent !important;
        border-bottom: 1px solid {BORDER} !important;
        gap: 0 !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent !important;
        color: {TEXT_SECONDARY} !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        padding: 10px 20px !important;
        border: none !important;
        letter-spacing: 0.3px !important;
        transition: color 0.15s !important;
    }}
    .stTabs [data-baseweb="tab"]:hover {{ color: {TEXT_PRIMARY} !important; }}
    .stTabs [aria-selected="true"] {{
        color: {GOLD_LIGHT} !important;
        font-weight: 600 !important;
        border-bottom: 2px solid {GOLD} !important;
    }}
    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span {{
        color: inherit !important;
        font-size: inherit !important;
    }}

    /* ALERTS */
    [data-testid="stAlert"] {{
        border-radius: 6px !important;
        border: none !important;
        font-size: 13px !important;
    }}

    /* SEPARADORES Y CAPTION */
    hr {{ border-color: {BORDER} !important; margin: 24px 0 !important; }}
    [data-testid="stCaptionContainer"] p,
    [data-testid="stCaptionContainer"] {{
        color: {TEXT_MUTED} !important;
        font-size: 11px !important;
        letter-spacing: 0.2px !important;
    }}

    /* EXPANDER */
    [data-testid="stExpander"] summary {{
        color: {TEXT_SECONDARY} !important;
        font-size: 13px !important;
    }}
    [data-testid="stExpander"] summary:hover {{
        color: {TEXT_PRIMARY} !important;
    }}

    /* TOGGLE / CHECKBOX / RADIO */
    [data-testid="stCheckbox"] p,
    .stCheckbox label p,
    .stRadio label p {{
        color: {TEXT_SECONDARY} !important;
        font-size: 13px !important;
    }}

    /* SLIDER */
    [data-testid="stSlider"] p {{
        color: {TEXT_SECONDARY} !important;
    }}

    /* SELECTBOX desplegable */
    [data-baseweb="popover"] li {{
        color: {TEXT_SECONDARY} !important;
        font-size: 13px !important;
    }}
    [data-baseweb="popover"] li:hover {{
        background: {SURFACE} !important;
        color: {TEXT_PRIMARY} !important;
    }}

    /* SCROLLBAR */
    ::-webkit-scrollbar {{ width: 4px; height: 4px; }}
    ::-webkit-scrollbar-track {{ background: {BG}; }}
    ::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 10px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: {GOLD}; }}

    /* SPINNER */
    .stSpinner > div {{ border-top-color: {GOLD} !important; }}

    /* PROGRESS */
    [data-testid="stSlider"] .st-emotion-cache-1dp5vir {{
        background: {GOLD} !important;
    }}

    /* ═══════════════════════════════
       NÚMEROS TABULARES — ESTILO FINANCIERO
    ═══════════════════════════════ */

    /* Fuente tabular para TODOS los números de la app */
    [data-testid="stMetricValue"] > div,
    [data-testid="stMetricValue"] > div > div {{
        font-family: 'DM Mono', 'Roboto Mono', 'Courier New', monospace !important;
        font-variant-numeric: tabular-nums !important;
        font-feature-settings: "tnum" 1 !important;
        font-size: 24px !important;
        font-weight: 500 !important;
        letter-spacing: -0.3px !important;
        color: {TEXT_PRIMARY} !important;
    }}

    /* Labels de métricas */
    [data-testid="stMetricLabel"] > div,
    [data-testid="stMetricLabel"] p {{
        font-family: 'Inter', sans-serif !important;
        font-size: 10px !important;
        font-weight: 700 !important;
        color: {TEXT_MUTED} !important;
        text-transform: uppercase !important;
        letter-spacing: 1.2px !important;
    }}

    /* Delta de métricas */
    [data-testid="stMetricDelta"] > div {{
        font-family: 'DM Mono', monospace !important;
        font-variant-numeric: tabular-nums !important;
        font-size: 11px !important;
        font-weight: 500 !important;
    }}

    /* Celdas de dataframe */
    [data-testid="stDataFrame"] td,
    [data-testid="stDataFrame"] [role="gridcell"] {{
        font-family: 'DM Mono', 'Roboto Mono', monospace !important;
        font-variant-numeric: tabular-nums !important;
        font-feature-settings: "tnum" 1 !important;
        font-size: 12px !important;
        font-weight: 400 !important;
        color: {TEXT_SECONDARY} !important;
        letter-spacing: 0.01em !important;
    }}

    /* Inputs numéricos */
    .stNumberInput input,
    input[type="number"] {{
        font-family: 'DM Mono', 'Roboto Mono', monospace !important;
        font-variant-numeric: tabular-nums !important;
        font-size: 14px !important;
        font-weight: 400 !important;
        color: {TEXT_PRIMARY} !important;
        letter-spacing: 0.02em !important;
    }}

    /* Selectbox con números */
    [data-baseweb="select"] span {{
        font-variant-numeric: tabular-nums !important;
    }}

    /* Números en texto general y listas */
    .stMarkdown code,
    code {{
        font-family: 'DM Mono', monospace !important;
        background: {SURFACE_2} !important;
        color: {GOLD_LIGHT} !important;
        padding: 1px 6px !important;
        border-radius: 3px !important;
        font-size: 12px !important;
    }}

    </style>
    """, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = ""):
    sub_html = f"<p style='color:{TEXT_MUTED}; font-size:12px; margin:4px 0 0 0; letter-spacing:0.3px;'>{subtitle}</p>" if subtitle else ""
    st.markdown(f"""
    <div style='margin-bottom:24px; padding-bottom:16px; border-bottom:1px solid {BORDER};'>
        <h1 style='margin:0;'>{title}</h1>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def section_label(text: str):
    st.markdown(
        f"<p style='font-size:10px; font-weight:700; color:{TEXT_MUTED}; "
        f"text-transform:uppercase; letter-spacing:1.2px; margin:0 0 10px 0;'>{text}</p>",
        unsafe_allow_html=True
    )


def kpi_card(label: str, value: str, sub: str = "", color: str = None) -> str:
    val_color = color if color else TEXT_PRIMARY
    sub_html = f"<div style='font-size:11px; color:{TEXT_MUTED}; margin-top:5px; letter-spacing:0.2px; font-family:Inter,sans-serif;'>{sub}</div>" if sub else ""
    return f"""
    <div style='background:{SURFACE}; border:1px solid {BORDER}; border-radius:8px;
                padding:18px 20px;'>
        <div style='font-size:10px; font-weight:700; color:{TEXT_MUTED};
                    text-transform:uppercase; letter-spacing:1.2px; margin-bottom:10px;
                    font-family:Inter,sans-serif;'>{label}</div>
        <div style='font-family:"DM Mono","Roboto Mono","Courier New",monospace; font-size:24px;
                    font-weight:500; color:{val_color}; letter-spacing:-0.3px; line-height:1.15;
                    font-variant-numeric:tabular-nums;'>{value}</div>
        {sub_html}
    </div>"""




def gold_divider():
    st.markdown(
        f"<div style='height:1px; background:linear-gradient(90deg, {GOLD}44, transparent); margin:20px 0;'></div>",
        unsafe_allow_html=True
    )


def positive_color(v: float) -> str:
    return POSITIVE if v >= 0 else NEGATIVE


def badge(text: str, type: str = "neutral") -> str:
    color_map = {
        "positive": (POSITIVE, "rgba(90,143,110,0.15)"),
        "negative": (NEGATIVE, "rgba(155,77,77,0.15)"),
        "gold":     (GOLD, GOLD_DIM),
        "neutral":  (TEXT_SECONDARY, SURFACE_2),
    }
    c, bg = color_map.get(type, color_map["neutral"])
    return (f"<span style='background:{bg}; color:{c}; border:1px solid {c}44; "
            f"border-radius:4px; padding:2px 8px; font-size:11px; font-weight:600; "
            f"letter-spacing:0.3px;'>{text}</span>")
