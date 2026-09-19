import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from modules.utils import ensure_portfolio_data, no_portfolio_warning


def render_time_horizon():
    st.title("Horizonte Temporal ⏳")
    st.caption("Backtesting institucional y proyecciones futuras usando retorno total (dividendos reinvertidos).")

    portfolio_df = ensure_portfolio_data()
    if portfolio_df is None or portfolio_df.empty:
        no_portfolio_warning()
        return

    total_aum = portfolio_df['Total Value ($)'].sum()
    if total_aum <= 0:
        st.error("El valor total del portfolio debe ser mayor que cero.")
        return

    tab1, tab2 = st.tabs(["📈 Pasado: Backtest Histórico", "🔮 Futuro: Proyección a 10 años"])

    with tab1:
        st.subheader("Crecimiento Histórico del Portfolio")
        st.caption("Simula cómo habría evolucionado una inversión de $10.000 en este portfolio exacto.")

        horizon_years = st.selectbox("Periodo de análisis", [1, 5, 10, 20, 30], index=2)

        if st.button("▶️ Ejecutar Backtest", type="primary", key="run_backtest"):
            with st.spinner(f"Descargando {horizon_years} años de datos históricos..."):
                start_date = datetime.today() - timedelta(days=365 * horizon_years)
                tickers = portfolio_df['Ticker'].tolist()

                try:
                    raw = yf.download(tickers, start=start_date, end=datetime.today(),
                                      auto_adjust=True, progress=False, threads=False)
                    if isinstance(raw.columns, pd.MultiIndex):
                        data = raw['Close']
                    else:
                        data = raw[['Close']].copy()
                        data.columns = tickers

                    data = data.ffill().dropna()

                    if data.empty:
                        st.error("No hay suficientes datos históricos para este periodo.")
                    else:
                        valid_tickers = [t for t in tickers if t in data.columns]
                        data = data[valid_tickers]
                        valid_mask = portfolio_df['Ticker'].isin(valid_tickers)
                        valid_values = portfolio_df[valid_mask]['Total Value ($)'].values
                        weights = valid_values / valid_values.sum()
                        returns = data.pct_change().dropna()
                        port_returns = (returns * weights).sum(axis=1)
                        cumulative = (1 + port_returns).cumprod() * 10000

                        fig = px.line(
                            cumulative,
                            title=f"Crecimiento de $10.000 en {horizon_years} años (dividendos reinvertidos)",
                            labels={'value': 'Valor del Portfolio ($)', 'Date': 'Fecha'}
                        )
                        fig.update_layout(
                            showlegend=False,
                            template="plotly_white",
                            hovermode="x"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        final_value = cumulative.iloc[-1]
                        cagr = (final_value / 10000) ** (1 / horizon_years) - 1

                        col1, col2, col3 = st.columns(3)
                        col1.metric("Inversión Inicial", "$10.000")
                        col2.metric("Valor Final", f"${final_value:,.2f}")
                        col3.metric("CAGR (Retorno Anualizado)", f"{cagr * 100:.2f}%")

                except Exception as e:
                    st.error(f"Error al obtener datos históricos: {str(e)}")

    with tab2:
        st.subheader("Cono de Incertidumbre — Próximos 10 años")
        st.caption("Simulación Monte Carlo proyectando los próximos 10 años, basada en la volatilidad histórica del portfolio.")

        if st.button("▶️ Generar Proyección a 10 años", type="primary", key="run_projection"):
            with st.spinner("Simulando miles de trayectorias futuras..."):
                try:
                    start_date = datetime.today() - timedelta(days=365 * 10)
                    tickers = portfolio_df['Ticker'].tolist()

                    raw = yf.download(tickers, start=start_date, end=datetime.today(),
                                      auto_adjust=True, progress=False, threads=False)
                    if isinstance(raw.columns, pd.MultiIndex):
                        data = raw['Close']
                    else:
                        data = raw[['Close']].copy()
                        data.columns = tickers

                    data = data.ffill().dropna()
                    returns = data.pct_change().dropna()
                    weights = portfolio_df['Total Value ($)'].values / total_aum
                    port_returns = (returns * weights).sum(axis=1)

                    mu = port_returns.mean()
                    sigma = port_returns.std()

                    days_to_simulate = 252 * 10
                    num_simulations = 1000

                    simulation_df = np.zeros((days_to_simulate, num_simulations))
                    simulation_df[0] = total_aum

                    for i in range(1, days_to_simulate):
                        shock = np.random.normal(loc=mu, scale=sigma, size=num_simulations)
                        simulation_df[i] = simulation_df[i - 1] * (1 + shock)

                    p10 = np.percentile(simulation_df, 10, axis=1)
                    p50 = np.percentile(simulation_df, 50, axis=1)
                    p90 = np.percentile(simulation_df, 90, axis=1)

                    years = np.linspace(0, 10, days_to_simulate)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=years, y=p90, mode='lines',
                        line=dict(width=0), showlegend=False, hoverinfo='skip'
                    ))
                    fig.add_trace(go.Scatter(
                        x=years, y=p10, mode='lines',
                        line=dict(width=0), fill='tonexty',
                        fillcolor='rgba(0, 168, 107, 0.2)',
                        name='Intervalo de confianza 80%'
                    ))
                    fig.add_trace(go.Scatter(
                        x=years, y=p50, mode='lines',
                        name='Trayectoria esperada (Mediana)',
                        line=dict(color='#00a86b', width=3)
                    ))
                    fig.update_layout(
                        title="Proyección de Riqueza a 10 años (Intervalo de confianza 80%)",
                        xaxis_title="Años desde hoy",
                        yaxis_title="Riqueza Proyectada ($)",
                        template="plotly_white",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Percentil 10 (Pesimista)", f"${p10[-1]:,.2f}")
                    col2.metric("Mediana (Esperado)", f"${p50[-1]:,.2f}")
                    col3.metric("Percentil 90 (Optimista)", f"${p90[-1]:,.2f}")

                except Exception as e:
                    st.error(f"Error al calcular la proyección: {str(e)}")
