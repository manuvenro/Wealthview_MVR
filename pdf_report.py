"""
WealthView — Investment Memo PDF v2
Informe institucional premium: matplotlib charts, header/footer por pagina,
portada navy-gold, metricas con semaforo de color.
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import io
import os
from datetime import datetime

# matplotlib se importa de forma lazy dentro de las funciones
# para evitar crash si no esta instalado

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, PageBreak, Image as RLImage
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from modules.utils import ensure_portfolio_data
from modules.i18n import get_lang

# ── Fuentes ───────────────────────────────────────────────────────────────────
try:
    pdfmetrics.registerFont(TTFont('Arial',       'C:/Windows/Fonts/arial.ttf'))
    pdfmetrics.registerFont(TTFont('Arial-Bold',  'C:/Windows/Fonts/arialbd.ttf'))
    pdfmetrics.registerFont(TTFont('Arial-Italic','C:/Windows/Fonts/ariali.ttf'))
    TF, BF, IF = 'Arial-Bold', 'Arial', 'Arial-Italic'
except Exception:
    TF, BF, IF = 'Helvetica-Bold', 'Helvetica', 'Helvetica-Oblique'

# ── Colores WealthView ────────────────────────────────────────────────────────
NAVY        = colors.HexColor('#0d1b2a')
NAVY_MED    = colors.HexColor('#1a3550')
NAVY_LIGHT  = colors.HexColor('#1f4068')
GOLD        = colors.HexColor('#c9a84c')
GOLD_LIGHT  = colors.HexColor('#e8d5a3')
GOLD_PALE   = colors.HexColor('#fdf6e8')
WHITE       = colors.white
GRAY_DARK   = colors.HexColor('#3d4654')
GRAY_MED    = colors.HexColor('#6b7280')
GRAY_LIGHT  = colors.HexColor('#f6f3ee')
GREEN       = colors.HexColor('#2d6a4f')
GREEN_PALE  = colors.HexColor('#d4edda')
RED         = colors.HexColor('#9b2226')
RED_PALE    = colors.HexColor('#f8d7da')
LINE_CLR    = colors.HexColor('#d4c5a9')
GOLD_BDR    = colors.HexColor('#c9a84c')

W, H   = A4
LMAR   = 18 * mm
RMAR   = 18 * mm
HDR_H  = 13 * mm
FTR_H  =  8 * mm
TMAR   = 16 * mm + HDR_H
BMAR   = 14 * mm + FTR_H
PW     = W - LMAR - RMAR

# Colores matplotlib alineados con WealthView
MPL_COLORS = [
    '#c9a84c', '#1d4e89', '#2d6a4f', '#7b5ea7',
    '#9b2226', '#1a6b6b', '#a0522d', '#4a6fa5',
    '#6b8e23', '#c05a5a',
]

# ── Preparar logo (RGBA -> RGB con fondo navy) ────────────────────────────────
def _prepare_logo(logo_path):
    try:
        from PIL import Image as PILImage
        import tempfile
        img = PILImage.open(logo_path)
        if img.mode == 'RGBA':
            bg = PILImage.new('RGBA', img.size, (13, 27, 42, 255))
            composited = PILImage.alpha_composite(bg, img)
            result = composited.convert('RGB')
            tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            result.save(tmp.name, format='PNG')
            tmp.close()
            return tmp.name
        return logo_path
    except Exception:
        return logo_path


# ── Charts matplotlib ─────────────────────────────────────────────────────────

def _donut_png(labels, values, w_mm=110, h_mm=85):
    """Donut chart de distribucion de activos. Devuelve bytes PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    total = sum(values)
    if total == 0:
        return None
    fig, ax = plt.subplots(figsize=(w_mm / 25.4, h_mm / 25.4), dpi=150)
    fig.patch.set_facecolor('#fdf6e8')
    ax.set_facecolor('#fdf6e8')

    wedges, _ = ax.pie(
        values,
        colors=MPL_COLORS[:len(values)],
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.55, edgecolor='white', linewidth=1.8),
        radius=0.85,
    )
    # Texto central
    aum_txt = ('${:.1f}M'.format(total / 1e6) if total >= 1e6
               else '${:.0f}K'.format(total / 1e3))
    ax.text(0,  0.10, aum_txt,   ha='center', va='center',
            fontsize=12, fontweight='bold', color='#0d1b2a')
    ax.text(0, -0.14, 'AUM',     ha='center', va='center',
            fontsize=6.5, color='#6b7280', fontfamily='monospace')

    # Leyenda
    pcts = [v / total * 100 for v in values]
    handles = [
        mpatches.Patch(facecolor=MPL_COLORS[i % len(MPL_COLORS)],
                       edgecolor='white', linewidth=0.5,
                       label='{:<14s}  {:5.1f}%'.format(labels[i][:14], pcts[i]))
        for i in range(len(labels))
    ]
    ax.legend(
        handles=handles,
        loc='center left', bbox_to_anchor=(1.02, 0.5),
        fontsize=6.5, frameon=False, labelcolor='#3d4654',
        handlelength=1.0, handleheight=0.85,
        borderpad=0, labelspacing=0.45,
    )
    ax.set_aspect('equal')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='#fdf6e8', pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _montecarlo_bar_png(scenarios, values, total_aum, w_mm=120, h_mm=68):
    """Grafico de barras horizontales para Monte Carlo. Devuelve bytes PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bar_colors = ['#2d6a4f', '#4a7c59', '#c9a84c', '#c05a5a', '#9b2226']
    fig, ax = plt.subplots(figsize=(w_mm / 25.4, h_mm / 25.4), dpi=150)
    fig.patch.set_facecolor('#fdf6e8')
    ax.set_facecolor('#fdf6e8')

    bars = ax.barh(scenarios, values, color=bar_colors,
                   height=0.52, edgecolor='white', linewidth=0.6)

    # Etiquetas de valor
    x_max = max(values) * 1.32
    for bar, val in zip(bars, values):
        pct = (val / total_aum - 1) * 100
        sign = '+' if pct >= 0 else ''
        ax.text(val + x_max * 0.01,
                bar.get_y() + bar.get_height() / 2,
                '${:,.0f}  ({}{:.1f}%)'.format(val, sign, pct),
                va='center', fontsize=6.2, color='#3d4654')

    # Linea AUM actual
    ax.axvline(x=total_aum, color='#c9a84c', linewidth=1.4,
               linestyle='--', alpha=0.9, label='AUM actual')

    ax.set_xlabel('Valor proyectado', fontsize=6.5, color='#6b7280')
    ax.tick_params(axis='y', labelsize=7.0, labelcolor='#3d4654', length=0)
    ax.tick_params(axis='x', labelsize=5.8, labelcolor='#6b7280')
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: '${:,.0f}'.format(x)))
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#d4c5a9')
    ax.set_xlim(left=0, right=x_max)
    ax.grid(axis='x', color='#d4c5a9', linewidth=0.4, alpha=0.7)

    ax.legend(fontsize=6.2, frameon=False, labelcolor='#6b7280',
              loc='lower right')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='#fdf6e8', pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── Callbacks de pagina (header / footer / portada) ──────────────────────────

def _make_callbacks(username, date_str, prepared_logo):

    def _on_cover(canvas, doc):
        canvas.saveState()
        # Fondo navy completo
        canvas.setFillColor(NAVY)
        canvas.rect(0, 0, W, H, fill=1, stroke=0)
        # Barra dorada izquierda
        canvas.setFillColor(GOLD)
        canvas.rect(0, 0, 4 * mm, H, fill=1, stroke=0)
        # Franja dorada inferior
        canvas.setFillColor(GOLD)
        canvas.rect(0, 0, W, 5 * mm, fill=1, stroke=0)
        canvas.restoreState()

    def _on_later(canvas, doc):
        canvas.saveState()
        # Barra dorada izquierda (identidad de marca)
        canvas.setFillColor(GOLD)
        canvas.rect(0, 0, 4 * mm, H, fill=1, stroke=0)

        # Header bar navy
        canvas.setFillColor(NAVY)
        canvas.rect(4 * mm, H - HDR_H, W - 4 * mm, HDR_H, fill=1, stroke=0)

        # Logo en header (si disponible) — acotado a HDR_H para que no desborde
        logo_h_pts = HDR_H - 4 * mm   # margen superior/inferior de 2mm cada lado
        logo_w_pts = logo_h_pts * 3.5  # ratio aproximado del logo
        logo_x = LMAR + 2 * mm
        logo_y = H - HDR_H + (HDR_H - logo_h_pts) / 2
        if prepared_logo and os.path.exists(prepared_logo):
            try:
                canvas.drawImage(
                    prepared_logo,
                    logo_x, logo_y,
                    width=logo_w_pts, height=logo_h_pts,
                    preserveAspectRatio=True, mask='auto'
                )
            except Exception:
                pass

        # Texto WealthView en header
        text_x = logo_x + logo_w_pts + 4 * mm
        canvas.setFillColor(GOLD)
        canvas.setFont(TF, 8)
        canvas.drawString(text_x, H - HDR_H / 2 - 2.8, 'WEALTHVIEW')

        canvas.setFillColor(colors.HexColor('#888888'))
        canvas.setFont(BF, 7)
        canvas.drawString(text_x + 52, H - HDR_H / 2 - 2.8,
                          '  |  INVESTMENT MEMO')

        # Numero de pagina y usuario
        canvas.setFillColor(colors.HexColor('#aaaaaa'))
        canvas.setFont(BF, 6.5)
        canvas.drawRightString(
            W - RMAR, H - HDR_H / 2 - 2.8,
            '{}  |  Pag. {}'.format(username, doc.page)
        )

        # Linea dorada bajo header
        canvas.setStrokeColor(GOLD)
        canvas.setLineWidth(0.8)
        canvas.line(LMAR, H - HDR_H, W - RMAR, H - HDR_H)

        # Footer
        canvas.setStrokeColor(LINE_CLR)
        canvas.setLineWidth(0.4)
        canvas.line(LMAR, BMAR - FTR_H + 4 * mm, W - RMAR, BMAR - FTR_H + 4 * mm)

        canvas.setFillColor(GRAY_MED)
        canvas.setFont(IF, 5.5)
        canvas.drawCentredString(
            W / 2, BMAR - FTR_H + 1.5 * mm,
            'CONFIDENCIAL  |  Solo para uso del destinatario autorizado  '
            '|  WealthView  |  {}'.format(date_str)
        )
        canvas.restoreState()

    return _on_cover, _on_later


# ── Estilos ───────────────────────────────────────────────────────────────────

def _styles():
    s = getSampleStyleSheet()

    def add(name, **kw):
        if name not in s:
            s.add(ParagraphStyle(name=name, **kw))
        return s[name]

    # Portada
    add('CvrEye',   fontName=BF,  fontSize=8,  textColor=GOLD,
        spaceAfter=6, leading=11, alignment=TA_LEFT)
    add('CvrTitle', fontName=TF,  fontSize=34, textColor=WHITE,
        spaceAfter=6, leading=42, alignment=TA_LEFT)
    add('CvrSub',   fontName=IF,  fontSize=12, textColor=GOLD_LIGHT,
        spaceAfter=4, leading=16, alignment=TA_LEFT)
    add('CvrMeta',  fontName=BF,  fontSize=10,
        textColor=colors.HexColor('#cccccc'),
        spaceAfter=4, leading=15, alignment=TA_LEFT)
    add('CvrKpiV',  fontName=TF,  fontSize=20, textColor=GOLD,
        spaceAfter=1, leading=25, alignment=TA_CENTER)
    add('CvrKpiL',  fontName=BF,  fontSize=6.5,
        textColor=colors.HexColor('#9999aa'),
        spaceAfter=0, leading=9, alignment=TA_CENTER)
    add('CvrConf',  fontName=IF,  fontSize=7,
        textColor=colors.HexColor('#888888'),
        spaceAfter=0, leading=10, alignment=TA_LEFT)

    # Secciones
    add('SecLbl',   fontName=TF,  fontSize=7,  textColor=GOLD,
        spaceAfter=0, leading=9,  spaceBefore=12, alignment=TA_LEFT)
    add('SecTitle', fontName=TF,  fontSize=12, textColor=NAVY,
        spaceBefore=2, spaceAfter=4, leading=16)
    add('SubTitle', fontName=TF,  fontSize=9.5, textColor=NAVY_MED,
        spaceBefore=7, spaceAfter=3, leading=13)

    # Cuerpo
    add('Body',     fontName=BF,  fontSize=9,  textColor=GRAY_DARK,
        spaceAfter=5, leading=14, alignment=TA_JUSTIFY)
    add('BodySm',   fontName=BF,  fontSize=7.5, textColor=GRAY_MED,
        spaceAfter=3, leading=11)
    add('Caption',  fontName=IF,  fontSize=7,  textColor=GRAY_MED,
        spaceAfter=2, leading=10, alignment=TA_CENTER)

    # Tablas
    add('TblH',  fontName=TF, fontSize=8, textColor=WHITE,
        alignment=TA_CENTER, leading=11)
    add('TblC',  fontName=BF, fontSize=8, textColor=GRAY_DARK, leading=11)
    add('TblCR', fontName=BF, fontSize=8, textColor=GRAY_DARK,
        alignment=TA_RIGHT, leading=11)

    # Metricas
    add('MetL',  fontName=BF, fontSize=6.5, textColor=GRAY_MED,    leading=9)
    add('MetV',  fontName=TF, fontSize=15,  textColor=NAVY,         leading=19)
    add('MetVG', fontName=TF, fontSize=15,  textColor=GREEN,        leading=19)
    add('MetVR', fontName=TF, fontSize=15,  textColor=RED,          leading=19)
    add('MetN',  fontName=IF, fontSize=6.5, textColor=GRAY_MED,    leading=9)

    # Disclaimer
    add('Disc', fontName=IF, fontSize=6.5, textColor=GRAY_MED,
        alignment=TA_CENTER, leading=10)

    # Memo IA
    add('MemoHdr', fontName=TF, fontSize=9, textColor=WHITE, leading=12)
    add('MemoB',   fontName=BF, fontSize=8.5, textColor=GRAY_DARK,
        spaceAfter=4, leading=13, alignment=TA_JUSTIFY)

    return s


# ── Helpers ───────────────────────────────────────────────────────────────────
_GRID = colors.HexColor('#e5e0d8')

def _tbl_style(zebra=True, hdr=NAVY):
    base = [
        ('BACKGROUND',    (0, 0), (-1, 0),  hdr),
        ('TEXTCOLOR',     (0, 0), (-1, 0),  WHITE),
        ('FONTNAME',      (0, 0), (-1, 0),  TF),
        ('FONTSIZE',      (0, 0), (-1, -1), 8),
        ('TOPPADDING',    (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING',   (0, 0), (-1, -1), 7),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 7),
        ('ALIGN',         (1, 0), (-1, -1), 'RIGHT'),
        ('ALIGN',         (0, 0), (0, -1),  'LEFT'),
        ('GRID',          (0, 0), (-1, -1), 0.3, _GRID),
        ('FONTNAME',      (0, 1), (-1, -1), BF),
        ('LINEBELOW',     (0, 0), (-1, 0),  1.2, GOLD),
    ]
    if zebra:
        base.append(('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, GRAY_LIGHT]))
    return base

def _hr():
    return HRFlowable(width='100%', thickness=0.4,
                      color=LINE_CLR, spaceAfter=4, spaceBefore=2)

def _gold_hr():
    return HRFlowable(width='100%', thickness=1.5,
                      color=GOLD, spaceAfter=6, spaceBefore=0)

def _sec(label, s):
    return [Paragraph(label.upper(), s['SecLbl']), _gold_hr()]


# ── Calculo de metricas ───────────────────────────────────────────────────────
def _calc_metrics(portfolio_df):
    total_aum = portfolio_df['Total Value ($)'].sum()
    tickers   = portfolio_df['Ticker'].tolist()
    if not tickers or total_aum <= 0:
        return None
    try:
        all_t = list(set(tickers + ['SPY']))
        raw   = yf.download(all_t, period='1y', auto_adjust=True, progress=False, threads=False)
        close = (raw['Close'] if isinstance(raw.columns, pd.MultiIndex)
                 else raw[['Close']])
        close = close.ffill().dropna()
        rets  = close.pct_change().dropna().clip(-0.5, 0.5)

        valid = [t for t in tickers if t in rets.columns]
        if not valid:
            return None
        vw = (portfolio_df[portfolio_df['Ticker'].isin(valid)]['Total Value ($)'].values)
        vw = vw / vw.sum()

        pr     = rets[valid].dot(vw)
        mu     = pr.mean() * 252
        sig    = pr.std() * np.sqrt(252)
        rf     = st.session_state.get('risk_free_rate', 4.0) / 100
        sharpe = (mu - rf) / sig if sig > 0 else 0

        cumul  = (1 + pr).cumprod()
        mdd    = ((cumul - cumul.cummax()) / cumul.cummax()).min()
        var95  = np.percentile(pr, 5)
        var99  = np.percentile(pr, 1)

        beta, alpha, treynor = 1.0, 0.0, 0.0
        if 'SPY' in rets.columns:
            cov_m    = np.cov(pr, rets['SPY'])
            beta     = cov_m[0, 1] / np.var(rets['SPY'])
            bench_mu = rets['SPY'].mean() * 252
            alpha    = mu - (rf + beta * (bench_mu - rf))
            treynor  = (mu - rf) / beta if beta != 0 else 0

        down    = pr[pr < 0]
        sortino = ((mu - rf) / (down.std() * np.sqrt(252))
                   if len(down) > 1 else 0)

        daily_mu  = pr.mean()
        daily_sig = pr.std()
        days, n_sim = 252, 1000
        sim = np.zeros((days, n_sim))
        sim[0] = total_aum
        for i in range(1, days):
            z      = np.random.normal(0, 1, n_sim)
            gf     = np.exp((daily_mu - 0.5 * daily_sig ** 2) + daily_sig * z)
            sim[i] = sim[i - 1] * np.clip(gf, 0.3, 3.0)
        final = sim[-1]

        return dict(
            annual_ret=mu, annual_vol=sig, sharpe=sharpe,
            sortino=sortino, beta=beta, alpha=alpha, treynor=treynor,
            max_dd=mdd, var95=var95, var99=var99,
            mc_p5=np.percentile(final, 5),  mc_p25=np.percentile(final, 25),
            mc_p50=np.median(final),         mc_p75=np.percentile(final, 75),
            mc_p95=np.percentile(final, 95), total_aum=total_aum,
        )
    except Exception:
        return None


# ── Portada ───────────────────────────────────────────────────────────────────
def _build_cover(story, s, logo_path, username, total_aum, n_pos,
                 date_str, metrics):
    # El fondo navy lo pinta el callback _on_cover — aqui solo ponemos contenido

    # Logo
    prepared = _prepare_logo(logo_path) if os.path.exists(logo_path) else None
    if prepared:
        try:
            logo_img = RLImage(prepared, width=40 * mm, height=14 * mm, kind='proportional')
            story.append(Spacer(1, 14 * mm))
            story.append(logo_img)
            story.append(Spacer(1, 14 * mm))
        except Exception:
            story.append(Spacer(1, 28 * mm))
    else:
        story.append(Spacer(1, 28 * mm))

    # Eyebrow + titulo
    story.append(Paragraph('INVESTMENT MEMO', s['CvrEye']))
    story.append(Paragraph('Análisis Institucional\nde Cartera', s['CvrTitle']))
    story.append(Spacer(1, 5 * mm))
    story.append(HRFlowable(
        width='38%', thickness=2.5, color=GOLD,
        spaceBefore=0, spaceAfter=8))
    story.append(Spacer(1, 2 * mm))

    # Metadata cliente
    gold_bold = '<font color="#c9a84c"><b>{}</b></font>'
    story.append(Paragraph(
        '{} {}'.format(gold_bold.format('Cliente  '), username), s['CvrMeta']))
    story.append(Paragraph(
        '{} {}'.format(gold_bold.format('Fecha    '), date_str), s['CvrMeta']))
    story.append(Spacer(1, 10 * mm))

    # KPI bar (AUM / posiciones / retorno / volatilidad)
    def _kpi_cell(lbl, val):
        return Table(
            [[Paragraph(val, s['CvrKpiV'])],
             [Paragraph(lbl, s['CvrKpiL'])]],
            colWidths=[PW / 4 - 4 * mm]
        )

    ret_str = ('{:+.1f}%'.format(metrics['annual_ret'] * 100)
               if metrics else 'n/d')
    vol_str = ('{:.1f}%'.format(metrics['annual_vol'] * 100)
               if metrics else 'n/d')
    kpi_cells = [
        _kpi_cell('AUM TOTAL',     '${:,.0f}'.format(total_aum)),
        _kpi_cell('POSICIONES',    str(n_pos)),
        _kpi_cell('RETORNO 1Y',   ret_str),
        _kpi_cell('VOLATILIDAD',  vol_str),
    ]
    kpi_tbl = Table([kpi_cells], colWidths=[PW / 4] * 4)
    kpi_tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), NAVY_MED),
        ('INNERGRID',     (0, 0), (-1, -1), 0.5, colors.HexColor('#2a4a6a')),
        ('BOX',           (0, 0), (-1, -1), 1.5, GOLD),
        ('TOPPADDING',    (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(kpi_tbl)
    story.append(Spacer(1, 14 * mm))

    # Nota de confidencialidad
    story.append(Paragraph(
        'CONFIDENCIAL  |  Solo para uso del destinatario autorizado',
        s['CvrConf']))
    story.append(PageBreak())


# ── Executive Summary ─────────────────────────────────────────────────────────
def _build_exec_summary(story, s, portfolio_df, metrics, username):
    story += _sec('Executive Summary', s)

    total_aum = portfolio_df['Total Value ($)'].sum()
    n_pos     = len(portfolio_df)

    def _mcard(lbl, val, note='', tone=0):
        """tone: 0 neutral, 1 positive, 2 negative"""
        vs = {0: 'MetV', 1: 'MetVG', 2: 'MetVR'}[tone]
        rows = [
            [Paragraph(lbl,  s['MetL'])],
            [Paragraph(val,  s[vs])],
        ]
        if note:
            rows.append([Paragraph(note, s['MetN'])])
        return Table(rows, colWidths=[PW / 4 - 4 * mm])

    if metrics:
        m = metrics
        ret_pos  = 1 if m['annual_ret'] > 0 else 2
        shr_pos  = 1 if m['sharpe']    > 1 else 2
        alp_pos  = 1 if m['alpha']     > 0 else 2
        srt_pos  = 1 if m['sortino']   > 1 else 0
        kpis = [
            _mcard('AUM TOTAL',        '${:,.0f}'.format(total_aum)),
            _mcard('RETORNO ANUAL 1Y', '{:+.2f}%'.format(m['annual_ret'] * 100),
                   'historico 12m', ret_pos),
            _mcard('SHARPE RATIO',     '{:.2f}'.format(m['sharpe']),
                   '>1 bueno | >2 excelente', shr_pos),
            _mcard('MAX DRAWDOWN',     '{:.1f}%'.format(m['max_dd'] * 100),
                   'caida pico a valle', 2),
            _mcard('VOLATILIDAD',      '{:.1f}%'.format(m['annual_vol'] * 100),
                   'anualizada'),
            _mcard('BETA vs S&P 500',  '{:.2f}'.format(m['beta']),
                   '1.0 = mercado'),
            _mcard('ALPHA DE JENSEN',  '{:+.2f}%'.format(m['alpha'] * 100),
                   'exceso benchmark', alp_pos),
            _mcard('SORTINO RATIO',    '{:.2f}'.format(m['sortino']),
                   'downside vol', srt_pos),
        ]
        rows2 = [kpis[:4], kpis[4:]]
    else:
        rows2 = [[
            _mcard('AUM TOTAL',  '${:,.0f}'.format(total_aum)),
            _mcard('POSICIONES', str(n_pos)),
            _mcard('CLASES',
                   str(portfolio_df['Asset Type'].nunique()
                       if 'Asset Type' in portfolio_df.columns else 'n/d')),
            _mcard('DATOS', 'n/d'),
        ]]

    kpi_grid = Table(rows2, colWidths=[PW / 4] * 4)
    kpi_grid.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), GOLD_PALE),
        ('BOX',           (0, 0), (-1, -1), 1.5, GOLD_BDR),
        ('INNERGRID',     (0, 0), (-1, -1), 0.5, LINE_CLR),
        ('LINEBELOW',     (0, 0), (-1, 0),  1.0, LINE_CLR),
        ('TOPPADDING',    (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING',   (0, 0), (-1, -1), 10),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 10),
        ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(kpi_grid)
    story.append(Spacer(1, 5 * mm))

    # Parrafo narrativo
    if metrics:
        m = metrics
        sh_txt = ('superior a 1 (buena relacion riesgo/retorno)'
                  if m['sharpe'] > 1 else 'por debajo de 1 (revisar eficiencia)')
        al_txt = ('positivo: +outperformance vs benchmark'
                  if m['alpha'] > 0 else 'negativo: underperformance vs benchmark')
        beta_txt = ('superior' if m['beta'] > 1 else 'inferior')
        text = (
            'La cartera de <b>{}</b> presenta un '
            '<b>retorno anual estimado del {:+.2f}%</b> con volatilidad '
            'anualizada del {:.1f}%. El Ratio de Sharpe de <b>{:.2f}</b> es '
            '{}. La Beta de {:.2f} indica sensibilidad {} al mercado. '
            'El Alpha de Jensen es {} ({:+.2f}%). '
            'El Maximum Drawdown a 12 meses es <b>{:.1f}%</b> y el '
            'VaR diario al 95%% equivale a <b>${:,.0f}</b>.'
        ).format(
            username,
            m['annual_ret'] * 100, m['annual_vol'] * 100,
            m['sharpe'], sh_txt,
            m['beta'], beta_txt,
            al_txt, m['alpha'] * 100,
            m['max_dd'] * 100,
            abs(m['var95']) * total_aum,
        )
    else:
        text = (
            'La cartera de <b>{}</b> esta compuesta por {} posiciones '
            'con un AUM total de <b>${:,.0f}</b>. No fue posible calcular '
            'metricas cuantitativas (datos historicos insuficientes).'
        ).format(username, n_pos, total_aum)
    story.append(Paragraph(text, s['Body']))
    story.append(Spacer(1, 3 * mm))


# ── Composicion de cartera ────────────────────────────────────────────────────
def _build_portfolio(story, s, portfolio_df):
    story += _sec('Composicion de Cartera', s)

    total_aum = portfolio_df['Total Value ($)'].sum()
    TYPE_MAP = {
        'equity': 'Equity', 'bond_etf': 'Bond ETF', 'bond': 'Bono',
        'fund': 'Fondo', 'option_call': 'Call', 'option_put': 'Put',
        'future': 'Futuro', 'warrant': 'Warrant',
    }

    # Tabla de posiciones
    hdrs = ['Ticker', 'Nombre', 'Tipo', 'Titulos', 'Precio', 'Valor', 'Peso']
    rows = [hdrs]
    for _, row in portfolio_df.sort_values(
            'Total Value ($)', ascending=False).iterrows():
        atype = str(row.get('Asset Type', 'equity'))
        peso  = (row['Total Value ($)'] / total_aum * 100
                 if total_aum > 0 else 0)
        price = float(
            row.get('Current Price ($)', row.get('Underlying Price', 0)) or 0)
        name  = str(row.get('Name', row['Ticker']))[:20]
        rows.append([
            str(row['Ticker']),
            name,
            TYPE_MAP.get(atype, atype),
            '{:,.2f}'.format(float(row['Shares'])),
            '${:,.2f}'.format(price),
            '${:,.0f}'.format(float(row['Total Value ($)'])),
            '{:.1f}%'.format(peso),
        ])
    rows.append(['TOTAL', '', '', '', '',
                 '${:,.0f}'.format(total_aum), '100.0%'])

    tbl_w = PW * 0.53
    ratio = [18, 30, 16, 16, 20, 22, 14]
    s_ratio = sum(ratio)
    cw_tbl = [tbl_w * r / s_ratio for r in ratio]

    pos_tbl = Table(rows, colWidths=cw_tbl, repeatRows=1)
    sty = _tbl_style()
    sty += [
        ('BACKGROUND', (0, -1), (-1, -1), GOLD_LIGHT),
        ('FONTNAME',   (0, -1), (-1, -1), TF),
    ]
    pos_tbl.setStyle(TableStyle(sty))

    # Donut chart (matplotlib)
    chart_w = PW * 0.43
    donut_png = None
    if 'Asset Type' in portfolio_df.columns and \
            portfolio_df['Asset Type'].nunique() > 1:
        by_type = portfolio_df.groupby('Asset Type')['Total Value ($)'].sum()
        lbl = [TYPE_MAP.get(k, k) for k in by_type.index]
        val = by_type.values.tolist()
    else:
        top = portfolio_df.nlargest(9, 'Total Value ($)')
        lbl = top['Ticker'].tolist()
        val = top['Total Value ($)'].tolist()
    try:
        png = _donut_png(lbl, val,
                         w_mm=int(chart_w / mm),
                         h_mm=int(chart_w / mm * 0.9))
        donut_img = RLImage(io.BytesIO(png), width=chart_w,
                            height=chart_w * 0.82)
        chart_cell = donut_img
    except Exception:
        chart_cell = Paragraph('(grafico no disponible)', s['Caption'])

    layout = Table([[pos_tbl, chart_cell]],
                   colWidths=[tbl_w, chart_w])
    layout.setStyle(TableStyle([
        ('VALIGN',      (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (1, 0), (1, 0),    8),
    ]))
    story.append(layout)
    story.append(Spacer(1, 3 * mm))


# ── Metricas de riesgo ────────────────────────────────────────────────────────
def _build_risk(story, s, metrics, total_aum):
    story += _sec('Analisis de Riesgo Cuantitativo', s)

    if not metrics:
        story.append(Paragraph(
            'No fue posible calcular metricas cuantitativas '
            '(datos historicos insuficientes).',
            s['Body']))
        return

    m = metrics

    def _kpi(lbl, val, note='', tone=0):
        vs = {0: 'MetV', 1: 'MetVG', 2: 'MetVR'}[tone]
        items = [
            [Paragraph(lbl,  s['MetL'])],
            [Paragraph(val,  s[vs])],
        ]
        if note:
            items.append([Paragraph(note, s['MetN'])])
        return Table(items, colWidths=[PW / 3 - 4 * mm])

    grid = [
        [
            _kpi('RETORNO ANUAL ESP.',
                 '{:+.2f}%'.format(m['annual_ret'] * 100),
                 'media diaria x 252',
                 1 if m['annual_ret'] > 0 else 2),
            _kpi('VOLATILIDAD ANUAL',
                 '{:.2f}%'.format(m['annual_vol'] * 100),
                 'std diaria x sqrt(252)'),
            _kpi('RATIO DE SHARPE',
                 '{:.2f}'.format(m['sharpe']),
                 '>1 bueno  |  >2 excelente',
                 1 if m['sharpe'] > 1 else 2),
        ],
        [
            _kpi('BETA vs S&P 500',
                 '{:.2f}'.format(m['beta']),
                 '1.0 = igual al mercado'),
            _kpi('ALPHA DE JENSEN',
                 '{:+.2f}%'.format(m['alpha'] * 100),
                 'exceso sobre benchmark',
                 1 if m['alpha'] > 0 else 2),
            _kpi('SORTINO RATIO',
                 '{:.2f}'.format(m['sortino']),
                 'solo volatilidad bajista',
                 1 if m['sortino'] > 1 else 0),
        ],
        [
            _kpi('MAX DRAWDOWN',
                 '{:.1f}%'.format(m['max_dd'] * 100),
                 'caida pico a valle 12m', 2),
            _kpi('VaR 95%% (1 dia)',
                 '${:,.0f}'.format(abs(m['var95']) * m['total_aum']),
                 '{:.2f}%% diario'.format(m['var95'] * 100), 2),
            _kpi('VaR 99%% (1 dia)',
                 '${:,.0f}'.format(abs(m['var99']) * m['total_aum']),
                 '{:.2f}%% diario'.format(m['var99'] * 100), 2),
        ],
    ]
    risk_tbl = Table(grid, colWidths=[PW / 3] * 3)
    risk_tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), GOLD_PALE),
        ('BOX',           (0, 0), (-1, -1), 1.5, GOLD_BDR),
        ('INNERGRID',     (0, 0), (-1, -1), 0.5, LINE_CLR),
        ('TOPPADDING',    (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 9),
        ('LEFTPADDING',   (0, 0), (-1, -1), 10),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 10),
        ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(risk_tbl)
    story.append(Spacer(1, 4 * mm))

    # Glosario metodologico
    story.append(Paragraph('Nota Metodologica', s['SubTitle']))
    notes = [
        ('Sharpe',  '(Rp-Rf)/sp. Rf = tasa libre de riesgo configurada.'),
        ('Beta',    'Cov(p,SPY)/Var(SPY). Sensibilidad sistematica al mercado.'),
        ('Alpha',   'Rp-[Rf+b(Rm-Rf)]. Alpha>0 indica outperformance.'),
        ('VaR',     'Percentil 5/1 de ret. historicos diarios x AUM. '
                    'Perdida maxima esperada con 95/99%% de confianza.'),
        ('Max DD',  'Min((Vt-maxVs)/maxVs). Mayor caida pico-a-valle en 12m.'),
        ('Sortino', '(Rp-Rf)/s_down. Solo penaliza la vol. bajista.'),
    ]
    nd = [[Paragraph('<b>{}</b>'.format(k), s['BodySm']),
           Paragraph(v, s['BodySm'])] for k, v in notes]
    nt = Table(nd, colWidths=[22 * mm, PW - 22 * mm])
    nt.setStyle(TableStyle([
        ('TOPPADDING',    (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING',   (0, 0), (-1, -1), 4),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 4),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [WHITE, GRAY_LIGHT]),
        ('GRID',          (0, 0), (-1, -1), 0.3, _GRID),
    ]))
    story.append(nt)
    story.append(Spacer(1, 4 * mm))


# ── Monte Carlo ───────────────────────────────────────────────────────────────
def _build_montecarlo(story, s, metrics, total_aum):
    story += _sec(
        'Proyeccion Monte Carlo - 1 Ano (1.000 simulaciones)', s)

    if not metrics:
        story.append(Paragraph('Datos insuficientes.', s['Body']))
        return

    m = metrics
    sc_lbls = ['Muy Optimista', 'Optimista',
               'Caso Base (Mediana)', 'Pesimista', 'Muy Pesimista']
    sc_pcts = ['P95', 'P75', 'P50', 'P25', 'P5']
    sc_vals = [m['mc_p95'], m['mc_p75'], m['mc_p50'],
               m['mc_p25'], m['mc_p5']]

    mc_rows = [['Escenario', 'Percentil', 'Valor ($)', 'Variacion']]
    for lbl, pct, val in zip(sc_lbls, sc_pcts, sc_vals):
        chg = (val / total_aum - 1) * 100
        sign = '+' if chg >= 0 else ''
        mc_rows.append([lbl, pct,
                         '${:,.0f}'.format(val),
                         '{}{:.1f}%%'.format(sign, chg)])
    mc_rows.append(['AUM Actual', '—',
                    '${:,.0f}'.format(total_aum), '—'])

    tbl_w = PW * 0.50
    cw = [tbl_w * r for r in [0.38, 0.12, 0.28, 0.22]]
    mc_tbl = Table(mc_rows, colWidths=cw)
    sty = _tbl_style(zebra=False)
    sty += [
        ('FONTNAME',   (0, 3), (-1, 3),  TF),
        ('BACKGROUND', (0, 3), (-1, 3),  GOLD_PALE),
        ('BACKGROUND', (0, -1), (-1, -1), GRAY_LIGHT),
        ('TEXTCOLOR',  (3, 1), (3, 2),   GREEN),
        ('TEXTCOLOR',  (3, 4), (3, 5),   RED),
    ]
    mc_tbl.setStyle(TableStyle(sty))

    # Grafico de barras
    chart_w = PW * 0.46
    try:
        bar_png = _montecarlo_bar_png(
            sc_lbls[::-1], sc_vals[::-1], total_aum,
            w_mm=int(chart_w / mm), h_mm=72,
        )
        bar_img = RLImage(io.BytesIO(bar_png), width=chart_w,
                          height=66 * mm)
        chart_cell = bar_img
    except Exception:
        chart_cell = Paragraph('(grafico no disponible)', s['Caption'])

    side = Table([[mc_tbl, chart_cell]],
                 colWidths=[tbl_w, chart_w])
    side.setStyle(TableStyle([
        ('VALIGN',       (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING',  (1, 0), (1, 0),    8),
    ]))
    story.append(side)
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        'Las proyecciones se basan en el Movimiento Browniano Geometrico '
        'calibrado con los ultimos 12 meses. No predicen rendimientos futuros. '
        'Los percentiles representan la distribucion estadistica de '
        '1.000 trayectorias posibles.',
        s['BodySm']))
    story.append(Spacer(1, 4 * mm))


# ── Memo IA ───────────────────────────────────────────────────────────────────
def _build_ai_memo(story, s, memo_data):
    import re
    story.append(PageBreak())
    story += _sec('Investment Memo — Analisis IA', s)
    story.append(Paragraph(
        'Generado el {}  |  Modelo: GPT-4o  |  AUM: ${:,.0f}'.format(
            memo_data.get('date', '—'), memo_data.get('aum', 0)),
        s['BodySm']))
    story.append(Spacer(1, 3 * mm))

    memo_text = memo_data.get('text', '')
    if not memo_text:
        story.append(Paragraph(
            'Sin memo disponible. Generalo desde Investment Memo IA.',
            s['Body']))
        return

    clr_map = {
        'EXECUTIVE': NAVY,   'PORTFOLIO': NAVY_MED,
        'RISK': RED,         'MARKET': colors.HexColor('#1d4e89'),
        'POSITION': GREEN,   'SYSTEMATIC': GOLD,
    }
    sections = re.split(
        r'\*\*([A-ZA-Z][A-Z \/\-]+)\*\*', memo_text)

    if sections[0].strip():
        story.append(Paragraph(sections[0].strip(), s['MemoB']))

    for i in range(1, len(sections) - 1, 2):
        title   = sections[i].strip()
        content = sections[i + 1].strip() if i + 1 < len(sections) else ''
        clr     = next(
            (v for k, v in clr_map.items() if k in title.upper()),
            GRAY_MED)
        hdr = Table([[Paragraph(title, s['MemoHdr'])]], colWidths=[PW])
        hdr.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1, -1), clr),
            ('LEFTPADDING',   (0, 0), (-1, -1), 10),
            ('TOPPADDING',    (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(hdr)
        for line in content.split('\n'):
            line = line.strip()
            if line:
                if line.startswith(('• ', '- ')):
                    line = '&bull;&nbsp;' + line[2:]
                story.append(Paragraph(line, s['MemoB']))
        story.append(Spacer(1, 3 * mm))


# ── Apendice ──────────────────────────────────────────────────────────────────
def _build_appendix(story, s, portfolio_df, metrics):
    story.append(PageBreak())
    story += _sec('Apendice — Datos Cuantitativos', s)

    story.append(Paragraph('A.1  Detalle de Posiciones', s['SubTitle']))
    total_aum = portfolio_df['Total Value ($)'].sum()
    TYPE_MAP = {
        'equity': 'Equity', 'bond_etf': 'Bond ETF', 'bond': 'Bono',
        'fund': 'Fondo', 'option_call': 'Call', 'option_put': 'Put',
        'future': 'Futuro', 'warrant': 'Warrant',
    }
    rows = [['Ticker', 'Nombre', 'Tipo', 'Titulos',
             'Precio ($)', 'Valor ($)', 'Peso%%', 'Notas']]
    for _, row in portfolio_df.sort_values(
            'Total Value ($)', ascending=False).iterrows():
        atype = str(row.get('Asset Type', 'equity'))
        peso  = (row['Total Value ($)'] / total_aum * 100
                 if total_aum > 0 else 0)
        price = float(
            row.get('Current Price ($)', row.get('Underlying Price', 0)) or 0)
        name  = str(row.get('Name', row['Ticker']))[:18]
        note  = ''
        if atype == 'bond':
            note = 'C:{}%% V:{}'.format(
                row.get('Coupon %%', '—'),
                str(row.get('Maturity', ''))[:7])
        elif atype in ('option_call', 'option_put', 'future', 'warrant'):
            note = 'K:{} x{}'.format(
                row.get('Strike', '—'),
                int(row.get('Multiplier', 100) or 100))
        rows.append([
            str(row['Ticker']), name,
            TYPE_MAP.get(atype, atype),
            '{:,.2f}'.format(float(row['Shares'])),
            '${:,.2f}'.format(price),
            '${:,.0f}'.format(float(row['Total Value ($)'])),
            '{:.1f}%%'.format(peso), note,
        ])

    ratio = [16, 28, 16, 16, 20, 22, 14, 28]
    cw = [PW * r / sum(ratio) for r in ratio]
    tbl = Table(rows, colWidths=cw, repeatRows=1)
    tbl.setStyle(TableStyle(_tbl_style()))
    story.append(tbl)
    story.append(Spacer(1, 5 * mm))

    if metrics:
        story.append(Paragraph(
            'A.2  Resumen Completo de Metricas', s['SubTitle']))
        m = metrics
        m_rows = [['Metrica', 'Valor', 'Interpretacion'],
            ['Retorno Anual Esp.',     '{:+.4f}%%'.format(m['annual_ret'] * 100),
             'Media diaria x 252'],
            ['Volatilidad Anual',      '{:.4f}%%'.format(m['annual_vol'] * 100),
             'Std diaria x sqrt(252)'],
            ['Sharpe Ratio',           '{:.4f}'.format(m['sharpe']),
             '(Rp-Rf)/sp'],
            ['Sortino Ratio',          '{:.4f}'.format(m['sortino']),
             '(Rp-Rf)/s_down'],
            ['Beta vs SPY',            '{:.4f}'.format(m['beta']),
             'Cov(p,m)/Var(m)'],
            ['Alpha de Jensen',        '{:+.4f}%%'.format(m['alpha'] * 100),
             'Rp-[Rf+b(Rm-Rf)]'],
            ['Ratio de Treynor',       '{:+.4f}%%'.format(m['treynor'] * 100),
             '(Rp-Rf)/b'],
            ['Max Drawdown',           '{:.4f}%%'.format(m['max_dd'] * 100),
             'Min caida pico-a-valle'],
            ['VaR 95%% (1d) $',        '${:,.2f}'.format(
                 abs(m['var95']) * m['total_aum']),
             'Percentil 5 x AUM'],
            ['VaR 99%% (1d) $',        '${:,.2f}'.format(
                 abs(m['var99']) * m['total_aum']),
             'Percentil 1 x AUM'],
            ['MC Mediana P50 (1Y)',     '${:,.0f}'.format(m['mc_p50']),
             'Mediana Monte Carlo'],
            ['MC Pesimista P5 (1Y)',    '${:,.0f}'.format(m['mc_p5']),
             'Percentil 5 MC'],
            ['MC Optimista P95 (1Y)',   '${:,.0f}'.format(m['mc_p95']),
             'Percentil 95 MC'],
        ]
        mt = Table(m_rows, colWidths=[55 * mm, 38 * mm, PW - 93 * mm],
                   repeatRows=1)
        mt.setStyle(TableStyle(_tbl_style()))
        story.append(mt)


# ── Disclaimer ────────────────────────────────────────────────────────────────
def _build_disclaimer(story, s):
    story.append(Spacer(1, 8 * mm))
    story.append(_hr())
    story.append(Paragraph(
        'AVISO LEGAL: Este documento ha sido generado automaticamente por '
        'WealthView con fines exclusivamente informativos. No constituye '
        'asesoramiento financiero, de inversion, legal ni fiscal. Las '
        'proyecciones y metricas se basan en datos historicos y modelos '
        'estadisticos que no garantizan rendimientos futuros. WealthView no '
        'asume responsabilidad por decisiones tomadas sobre la base de este '
        'informe. Generado el {}.'.format(
            datetime.today().strftime('%d/%m/%Y a las %H:%M UTC')),
        s['Disc']))


# ── Generador principal ───────────────────────────────────────────────────────
def generate_pdf(username, portfolio_df):
    buffer   = io.BytesIO()
    total    = portfolio_df['Total Value ($)'].sum()
    n_pos    = len(portfolio_df)
    date_str = datetime.today().strftime('%d/%m/%Y')

    logo_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'logo.png')
    prepared_logo = (_prepare_logo(logo_path)
                     if os.path.exists(logo_path) else None)

    on_cover, on_later = _make_callbacks(username, date_str, prepared_logo)

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=RMAR, leftMargin=LMAR,
        topMargin=TMAR, bottomMargin=BMAR,
        title='WealthView Investment Memo — {} — {}'.format(username, date_str),
        author='WealthView',
        subject='Investment Memo Institucional',
    )

    s       = _styles()
    metrics = _calc_metrics(portfolio_df)
    story   = []

    _build_cover(story, s, logo_path, username, total, n_pos,
                 date_str, metrics)
    _build_exec_summary(story, s, portfolio_df, metrics, username)
    _build_portfolio(story, s, portfolio_df)
    story.append(PageBreak())
    _build_risk(story, s, metrics, total)
    _build_montecarlo(story, s, metrics, total)

    if ('last_investment_memo' in st.session_state
            and st.session_state['last_investment_memo'].get('text')):
        _build_ai_memo(story, s, st.session_state['last_investment_memo'])

    _build_appendix(story, s, portfolio_df, metrics)
    _build_disclaimer(story, s)

    doc.build(story, onFirstPage=on_cover, onLaterPages=on_later)
    buffer.seek(0)
    return buffer.getvalue()


# ── UI Streamlit ──────────────────────────────────────────────────────────────
def render_pdf_export():
    from modules.i18n import t
    from modules.styles import GOLD, SURFACE, BORDER, TEXT_MUTED

    es = get_lang() == 'es'
    st.title(
        'Exportar Investment Memo PDF' if es else 'Export Investment Memo PDF')

    portfolio_df = ensure_portfolio_data()
    if portfolio_df is None or portfolio_df.empty:
        st.warning(
            'Primero configura tu portfolio.' if es
            else 'Please set up your portfolio first.')
        if st.button(
                'Ir a Portfolio Overview' if es else 'Go to Portfolio Overview',
                type='primary'):
            st.session_state.page = 'Portfolio Overview'
            st.rerun()
        return

    total_aum = portfolio_df['Total Value ($)'].sum()
    username  = st.session_state.username
    n_pos     = len(portfolio_df)

    st.markdown(
        "<div style='background:{s}; border:1px solid {b}; border-radius:8px;"
        " padding:18px 22px; margin-bottom:20px;'>"
        "<div style='color:{g}; font-size:11px; font-weight:700;"
        " text-transform:uppercase; letter-spacing:1.2px; margin-bottom:10px;'>"
        "{lbl}</div>"
        "<div style='display:grid; grid-template-columns:1fr 1fr; gap:8px;"
        " color:#d1d5db; font-size:12px;'>"
        "<div>Portada premium navy-gold con datos del cliente</div>"
        "<div>8 KPIs con semaforo de color (verde/rojo)</div>"
        "<div>Tabla de posiciones + donut chart (matplotlib)</div>"
        "<div>9 metricas de riesgo detalladas con metodologia</div>"
        "<div>Grafico Monte Carlo con barra de escenarios</div>"
        "<div>Apendice con metricas completas a 4 decimales</div>"
        "</div></div>".format(
            s=SURFACE, b=BORDER, g=GOLD,
            lbl='Contenido del PDF' if es else 'PDF Contents'),
        unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric('Usuario' if es else 'User', username)
    c2.metric('Posiciones' if es else 'Positions', str(n_pos))
    c3.metric('AUM', '${:,.0f}'.format(total_aum))

    has_memo = ('last_investment_memo' in st.session_state
                and bool(st.session_state.get(
                    'last_investment_memo', {}).get('text')))
    if has_memo:
        memo_dt = st.session_state['last_investment_memo'].get('date', '—')
        st.success(
            'Memo IA disponible ({}) — se incluira en el PDF.'.format(memo_dt))
    else:
        st.info(
            'Sin memo IA. Generalo en Investment Memo IA para incluirlo.'
            if es else
            'No AI memo. Generate one in Investment Memo AI to include it.')

    st.markdown('---')
    if st.button(
            'Generar Investment Memo PDF' if es else 'Generate Investment Memo PDF',
            type='primary', use_container_width=True):
        with st.spinner(
                'Generando PDF...' if es else 'Generating PDF...'):
            try:
                pdf_bytes = generate_pdf(username, portfolio_df)
                filename  = 'WealthView_InvestmentMemo_{}_{}.pdf'.format(
                    username, datetime.today().strftime('%Y%m%d'))
                # Guardar en session_state para que el download_button persista
                st.session_state['_pdf_bytes']    = pdf_bytes
                st.session_state['_pdf_filename'] = filename
                st.success(
                    'PDF generado. Pulsa el boton para descargarlo.'
                    if es else
                    'PDF generated. Click the button below to download it.')
            except Exception as e:
                import traceback
                st.error('Error generando el PDF: {}'.format(e))
                st.code(traceback.format_exc())

    # Download button siempre visible si hay bytes generados
    if st.session_state.get('_pdf_bytes'):
        st.download_button(
            label=('⬇ Descargar Investment Memo PDF'
                   if es else '⬇ Download Investment Memo PDF'),
            data=st.session_state['_pdf_bytes'],
            file_name=st.session_state.get('_pdf_filename', 'WealthView_Memo.pdf'),
            mime='application/pdf',
            use_container_width=True,
            key='pdf_download_btn',
        )
