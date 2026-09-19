"""
rebalancing.py — Rebalanceo Automático de Portfolio
Modos: equal weight | manual % por ticker
Muestra: drift vs target, trades exactos, alertas de umbral
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from modules.styles import (
    inject_global_css, page_header, section_label, gold_divider,
    GOLD, GOLD_DIM, GOLD_BORDER, SURFACE, SURFACE_2, BORDER,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    POSITIVE, POSITIVE_BG, NEGATIVE, NEGATIVE_BG, PLOTLY_DARK,
    kpi_card, badge, positive_color, plotly_layout
)
from modules.utils import ensure_portfolio_data, no_portfolio_warning

# ── Constantes ────────────────────────────────────────────────────────────────
_SS_TARGETS   = "rebal_targets"
_SS_MODE      = "rebal_mode"
_SS_THRESHOLD = "rebal_threshold"
_SS_CASH      = "rebal_cash"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_current_weights(portfolio_data: pd.DataFrame) -> pd.DataFrame:
    """Devuelve DataFrame con Ticker, Value, Weight (%) actual."""
    df = portfolio_data.copy()
    value_col = "Total Value ($)" if "Total Value ($)" in df.columns else "Value"
    if value_col not in df.columns:
        return pd.DataFrame()
    df = df.groupby("Ticker", as_index=False)[value_col].sum()
    total = df[value_col].sum()
    if total <= 0:
        return pd.DataFrame()
    df["weight_pct"] = (df[value_col] / total * 100).round(2)
    df = df.rename(columns={value_col: "value"})
    return df.sort_values("value", ascending=False).reset_index(drop=True)


def _compute_trades(
    current_df: pd.DataFrame,
    targets: dict,
    total_aum: float,
    extra_cash: float = 0.0,
) -> pd.DataFrame:
    """
    Calcula los trades necesarios para llegar al target.
    targets: {ticker: target_pct}  — deben sumar 100
    """
    portfolio_total = total_aum + extra_cash
    rows = []
    for ticker, target_pct in targets.items():
        target_val = portfolio_total * target_pct / 100
        cur_row = current_df[current_df["Ticker"] == ticker]
        cur_val = float(cur_row["value"].iloc[0]) if not cur_row.empty else 0.0
        cur_pct = float(cur_row["weight_pct"].iloc[0]) if not cur_row.empty else 0.0
        diff_val = target_val - cur_val
        diff_pct = target_pct - cur_pct
        rows.append({
            "Ticker":      ticker,
            "Actual %":    round(cur_pct, 2),
            "Target %":    round(target_pct, 2),
            "Drift":       round(diff_pct, 2),
            "Valor actual": round(cur_val, 2),
            "Valor target": round(target_val, 2),
            "Trade ($)":   round(diff_val, 2),
            "Acción":      "COMPRAR" if diff_val > 0 else "VENDER",
        })
    return pd.DataFrame(rows).sort_values("Trade ($)")


def _drift_gauge(drift: float, threshold: float) -> str:
    """Color de semáforo según drift."""
    abs_d = abs(drift)
    if abs_d < threshold * 0.5:
        return POSITIVE
    elif abs_d < threshold:
        return GOLD
    else:
        return NEGATIVE


# ── Render principal ──────────────────────────────────────────────────────────

def render_rebalancing():
    inject_global_css()
    page_header(
        "Rebalanceo Automático",
        "Equal weight · Target manual · Drift alerts · Trades exactos"
    )

    portfolio_data = ensure_portfolio_data()
    if portfolio_data is None or portfolio_data.empty:
        no_portfolio_warning()
        return

    current_df = _get_current_weights(portfolio_data)
    if current_df.empty:
        st.warning("No se pudieron calcular los pesos actuales.")
        return

    tickers   = current_df["Ticker"].tolist()
    total_aum = current_df["value"].sum()

    # ── Panel de configuración ────────────────────────────────────────────────
    with st.expander("⚙️ Configuración de rebalanceo", expanded=True):
        col_mode, col_thresh, col_cash = st.columns(3)

        with col_mode:
            mode = st.radio(
                "Modo de target",
                ["Equal Weight", "Manual"],
                index=0 if st.session_state.get(_SS_MODE, "Equal Weight") == "Equal Weight" else 1,
                key="rebal_mode_radio",
                horizontal=True,
            )
            st.session_state[_SS_MODE] = mode

        with col_thresh:
            threshold = st.slider(
                "Umbral de alerta (%)",
                min_value=1.0, max_value=20.0,
                value=float(st.session_state.get(_SS_THRESHOLD, 5.0)),
                step=0.5, format="%.1f%%",
                key="rebal_thresh_slider",
                help="Alerta si cualquier posición se desvía más de este % del target"
            )
            st.session_state[_SS_THRESHOLD] = threshold

        with col_cash:
            extra_cash = st.number_input(
                "Cash adicional a invertir ($)",
                min_value=0.0, value=float(st.session_state.get(_SS_CASH, 0.0)),
                step=100.0, format="%.2f",
                key="rebal_cash_input",
                help="Si vas a inyectar capital nuevo, indícalo aquí para optimizar los trades"
            )
            st.session_state[_SS_CASH] = extra_cash

        # Targets manuales
        targets: dict[str, float] = {}

        if mode == "Equal Weight":
            eq_w = round(100.0 / len(tickers), 4)
            targets = {t: eq_w for t in tickers}
            st.info(f"Equal weight: **{eq_w:.2f}%** por posición ({len(tickers)} activos)")
        else:
            st.markdown("**Define el % objetivo para cada activo** (deben sumar 100%)")
            saved = st.session_state.get(_SS_TARGETS, {})
            # Default: distribución actual redondeada
            cols_per_row = 4
            ticker_chunks = [tickers[i:i+cols_per_row] for i in range(0, len(tickers), cols_per_row)]
            for chunk in ticker_chunks:
                cols = st.columns(len(chunk))
                for col, tk in zip(cols, chunk):
                    with col:
                        default_val = saved.get(tk, round(100.0 / len(tickers), 1))
                        targets[tk] = st.number_input(
                            tk, min_value=0.0, max_value=100.0,
                            value=float(default_val),
                            step=0.5, format="%.1f",
                            key=f"rebal_target_{tk}"
                        )
            st.session_state[_SS_TARGETS] = targets

            total_target = sum(targets.values())
            diff_100 = abs(total_target - 100.0)
            if diff_100 > 0.1:
                st.error(f"⚠️ Los targets suman **{total_target:.1f}%** — deben sumar exactamente 100%")
            else:
                st.success(f"✅ Targets correctos: {total_target:.1f}%")

    if not targets or (mode == "Manual" and abs(sum(targets.values()) - 100.0) > 0.1):
        st.stop()

    # ── Calcular trades ───────────────────────────────────────────────────────
    trades_df = _compute_trades(current_df, targets, total_aum, extra_cash)

    # ── KPIs de drift ─────────────────────────────────────────────────────────
    gold_divider()
    section_label("Estado del Portfolio")

    max_drift    = trades_df["Drift"].abs().max()
    n_alerts     = (trades_df["Drift"].abs() >= threshold).sum()
    total_trades = trades_df[trades_df["Trade ($)"].abs() > 1]["Trade ($)"].abs().sum()
    pct_drift    = (max_drift / threshold * 100) if threshold > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card(
            "AUM Total", f"${total_aum:,.0f}",
            f"{len(tickers)} posiciones"
        ), unsafe_allow_html=True)
    with c2:
        drift_color = _drift_gauge(max_drift, threshold)
        st.markdown(kpi_card(
            "Max Drift", f"{max_drift:.1f}%",
            "vs target", color=drift_color
        ), unsafe_allow_html=True)
    with c3:
        alert_color = NEGATIVE if n_alerts > 0 else POSITIVE
        st.markdown(kpi_card(
            "Posiciones fuera de rango", str(int(n_alerts)),
            f"umbral {threshold:.1f}%", color=alert_color
        ), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card(
            "Trades necesarios", f"${total_trades:,.0f}",
            "volumen total a mover"
        ), unsafe_allow_html=True)

    # ── Gráfico: Actual vs Target ─────────────────────────────────────────────
    gold_divider()
    section_label("Actual vs Target")

    sorted_tickers = trades_df.sort_values("Ticker")["Ticker"].tolist()
    actual_vals  = [float(trades_df[trades_df["Ticker"] == t]["Actual %"].iloc[0]) for t in sorted_tickers]
    target_vals  = [float(trades_df[trades_df["Ticker"] == t]["Target %"].iloc[0]) for t in sorted_tickers]
    drift_vals   = [float(trades_df[trades_df["Ticker"] == t]["Drift"].iloc[0]) for t in sorted_tickers]
    bar_colors   = [NEGATIVE if abs(d) >= threshold else GOLD if abs(d) >= threshold * 0.5 else POSITIVE
                    for d in drift_vals]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Actual", x=sorted_tickers, y=actual_vals,
        marker_color=SURFACE_2, marker_line_color=GOLD_BORDER, marker_line_width=1,
        hovertemplate="%{x}: %{y:.2f}%<extra>Actual</extra>"
    ))
    fig.add_trace(go.Scatter(
        name="Target", x=sorted_tickers, y=target_vals,
        mode="markers", marker=dict(color=GOLD, size=10, symbol="diamond"),
        hovertemplate="%{x}: %{y:.2f}%<extra>Target</extra>"
    ))
    fig.update_layout(**plotly_layout(
        height=320,
        barmode="group",
        legend=dict(orientation="h", y=1.08),
        yaxis_title="Peso (%)",
        margin=dict(l=40, r=20, t=30, b=40),
    ))
    st.plotly_chart(fig, use_container_width=True)

    # ── Gráfico de drift ──────────────────────────────────────────────────────
    fig2 = go.Figure()
    fig2.add_hline(y=threshold, line=dict(color=NEGATIVE, width=1, dash="dash"),
                   annotation_text=f"+{threshold}% umbral")
    fig2.add_hline(y=-threshold, line=dict(color=NEGATIVE, width=1, dash="dash"),
                   annotation_text=f"-{threshold}% umbral")
    fig2.add_hline(y=0, line=dict(color=BORDER, width=1))
    fig2.add_trace(go.Bar(
        x=sorted_tickers, y=drift_vals,
        marker_color=bar_colors,
        hovertemplate="%{x}: %{y:+.2f}%<extra>Drift</extra>"
    ))
    fig2.update_layout(**plotly_layout(
        height=260,
        title=dict(text="Drift (Actual − Target)", font=dict(size=13, color=TEXT_SECONDARY)),
        yaxis_title="Drift (%)",
        margin=dict(l=40, r=20, t=40, b=40),
    ))
    st.plotly_chart(fig2, use_container_width=True)

    # ── Tabla de trades ───────────────────────────────────────────────────────
    gold_divider()
    section_label("Trades para Rebalancear")

    if extra_cash > 0:
        st.info(f"Se incluyen **${extra_cash:,.0f}** de cash adicional en el cálculo.")

    display_df = trades_df.copy()
    display_df["Actual %"]    = display_df["Actual %"].map(lambda x: f"{x:.2f}%")
    display_df["Target %"]    = display_df["Target %"].map(lambda x: f"{x:.2f}%")
    display_df["Drift"]       = display_df["Drift"].map(lambda x: f"{x:+.2f}%")
    display_df["Valor actual"] = display_df["Valor actual"].map(lambda x: f"${x:,.0f}")
    display_df["Valor target"] = display_df["Valor target"].map(lambda x: f"${x:,.0f}")
    display_df["Trade ($)"]   = display_df["Trade ($)"].map(lambda x: f"${abs(x):,.0f}")

    def _style_row(row):
        if "COMPRAR" in str(row.get("Acción", "")):
            return [f"color:{POSITIVE}"] * len(row)
        elif "VENDER" in str(row.get("Acción", "")):
            return [f"color:{NEGATIVE}"] * len(row)
        return [""] * len(row)

    st.dataframe(
        display_df[["Ticker", "Actual %", "Target %", "Drift", "Valor actual", "Valor target", "Trade ($)", "Acción"]]
        .reset_index(drop=True)
        .style.apply(_style_row, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    # Resumen compras/ventas
    compras = trades_df[trades_df["Trade ($)"] > 1]["Trade ($)"].sum()
    ventas  = trades_df[trades_df["Trade ($)"] < -1]["Trade ($)"].abs().sum()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(kpi_card("Total a comprar", f"${compras:,.0f}", "", color=POSITIVE), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Total a vender",  f"${ventas:,.0f}",  "", color=NEGATIVE), unsafe_allow_html=True)
    with c3:
        net = compras - ventas
        st.markdown(kpi_card("Flujo neto", f"${net:+,.0f}", "positivo = más cash necesario"), unsafe_allow_html=True)

    # ── Alerta de umbral ──────────────────────────────────────────────────────
    if n_alerts > 0:
        gold_divider()
        st.markdown(f"""
        <div style="border:1px solid {NEGATIVE};border-radius:8px;padding:14px 18px;
                    background:rgba(155,77,77,0.08);margin-bottom:12px;">
            <b style="color:{NEGATIVE};">⚠️ {int(n_alerts)} posición(es) fuera de rango</b><br>
            <span style="color:{TEXT_SECONDARY};font-size:0.9rem;">
                Las siguientes posiciones superan el umbral de <b>{threshold:.1f}%</b>:<br>
                {', '.join(
                    f'<b>{row["Ticker"]}</b> ({row["Drift"]:+.2f}%)'
                    for _, row in trades_df[trades_df["Drift"].abs() >= threshold].iterrows()
                )}
            </span>
        </div>""", unsafe_allow_html=True)
    else:
        st.success(f"✅ Portfolio en equilibrio — ninguna posición supera el umbral de {threshold:.1f}%")

    # ── Nota sobre automatización ─────────────────────────────────────────────
    gold_divider()
    st.caption(
        "💡 Para rebalancear automáticamente de forma periódica, configura una tarea "
        "programada en **Configuración → Automatización**."
    )
