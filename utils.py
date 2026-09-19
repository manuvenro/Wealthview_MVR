import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from modules.fx import get_ticker_currency, get_fx_rate, get_base_currency, convert_to_base


def safe_df_display(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte columnas object a str para evitar SEGV de pyarrow en st.dataframe().
    Las columnas numéricas (float/int) se mantienen sin tocar."""
    out = df.copy()
    for col in out.select_dtypes(include='object').columns:
        out[col] = out[col].fillna('—').astype(str)
    return out


def resolve_isin_name(isin: str) -> str:
    """
    Intenta resolver un ISIN a nombre legible.
    1. OpenFIGI API (Bloomberg, gratuita)
    2. yfinance .info['longName']
    3. Devuelve el ISIN si no encuentra nada
    """
    isin = isin.strip().upper()

    # 1. OpenFIGI
    try:
        resp = requests.post(
            "https://api.openfigi.com/v3/mapping",
            json=[{"idType": "ID_ISIN", "idValue": isin}],
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        if resp.status_code == 200:
            data = resp.json()
            if data and isinstance(data, list) and data[0].get('data'):
                name = data[0]['data'][0].get('name', '')
                if name:
                    return name
    except Exception:
        pass

    # 2. yfinance longName
    try:
        info = yf.Ticker(isin).info
        name = info.get('longName') or info.get('shortName')
        if name:
            return name
    except Exception:
        pass

    return isin  # fallback: devolver el ISIN


def ensure_portfolio_data():
    """
    Garantiza que st.session_state.portfolio_data esté siempre disponible.
    - Si ya existe con precios, lo devuelve tal cual.
    - Si sólo existe 'portfolio' (Ticker+Shares), hace fetch automático de precios.
    - Si no hay portfolio, devuelve None.
    """
    # Ya tenemos datos con precios
    if 'portfolio_data' in st.session_state and not st.session_state.portfolio_data.empty:
        return st.session_state.portfolio_data

    # Tenemos el portfolio base pero sin precios — fetch automático
    if 'portfolio' in st.session_state and not st.session_state.portfolio.empty:
        portfolio_df = st.session_state.portfolio.copy()
        results = []
        DERIV_TYPES = ('option_call', 'option_put', 'future', 'warrant')

        # ── Batch prefetch de todos los precios en UNA sola petición ────────────
        from modules.price_cache import get_prices as _get_prices
        all_tickers = [
            str(r['Ticker']).strip().upper()
            for _, r in portfolio_df.iterrows()
            if str(r.get('Ticker', '')).strip()
        ]
        prices_map = _get_prices(all_tickers) if all_tickers else {}

        for _, row in portfolio_df.iterrows():
            ticker_symbol = str(row['Ticker']).strip().upper()
            if not ticker_symbol:
                continue
            asset_type = str(row.get('Asset Type', 'equity')) if 'Asset Type' in row else 'equity'
            price = prices_map.get(ticker_symbol, 0.0) or 0.0
            shares = float(row['Shares'])
            name = str(row['Name']) if 'Name' in row and pd.notna(row.get('Name')) and str(row.get('Name', '')).strip() else ticker_symbol

            # ── Derivados: calcular valor delta-ajustado ─────────────────────
            if asset_type in DERIV_TYPES:
                multiplier  = float(row['Multiplier']) if pd.notna(row.get('Multiplier')) else 100.0
                strike      = float(row['Strike'])     if pd.notna(row.get('Strike'))     else 0.0
                premium     = float(row['Premium'])    if pd.notna(row.get('Premium'))    else 0.0
                contracts   = shares

                if asset_type == 'future':
                    # Notional completo (delta = 1)
                    total_value = contracts * multiplier * price
                    display_price = price
                elif asset_type == 'option_call':
                    intrinsic   = max(0.0, price - strike)
                    total_value = intrinsic * contracts * multiplier if intrinsic > 0 \
                                  else premium * contracts * multiplier
                    display_price = intrinsic if intrinsic > 0 else premium
                elif asset_type == 'option_put':
                    intrinsic   = max(0.0, strike - price)
                    total_value = intrinsic * contracts * multiplier if intrinsic > 0 \
                                  else premium * contracts * multiplier
                    display_price = intrinsic if intrinsic > 0 else premium
                elif asset_type == 'warrant':
                    intrinsic   = max(0.0, price - strike)
                    total_value = intrinsic * contracts * multiplier if intrinsic > 0 \
                                  else premium * contracts * multiplier
                    display_price = intrinsic if intrinsic > 0 else premium
                else:
                    total_value   = 0.0
                    display_price = 0.0

                results.append({
                    'Ticker':             ticker_symbol,
                    'Name':               name,
                    'Asset Type':         asset_type,
                    'Shares':             contracts,
                    'Current Price ($)':  round(display_price, 4),
                    'Total Value ($)':    round(total_value, 2),
                    'Underlying Price':   round(price, 2),
                    'Strike':             strike,
                    'Multiplier':         multiplier,
                    'Premium':            premium,
                })
            else:
                avg_cost_raw = row.get('Avg Cost')
                avg_cost_val = float(avg_cost_raw) if avg_cost_raw is not None and str(avg_cost_raw) not in ('', 'nan', 'None') else None

                # ── Multi-divisa ─────────────────────────────────────────────
                # Si el portfolio ya tiene Currency guardada, la usamos;
                # si no, la detectamos via yfinance (con fallback a USD)
                stored_ccy = row.get('Currency')
                if stored_ccy and isinstance(stored_ccy, str) and len(stored_ccy) == 3:
                    currency = stored_ccy.upper()
                else:
                    currency = get_ticker_currency(ticker_symbol)

                base_ccy = get_base_currency()
                fx_rate  = get_fx_rate(currency, base_ccy)
                total_native = round(price * shares, 2)
                total_base   = round(total_native * fx_rate, 2)

                results.append({
                    'Ticker':             ticker_symbol,
                    'Name':               name,
                    'Asset Type':         asset_type,
                    'Shares':             shares,
                    'Current Price ($)':  round(price, 2),
                    'Total Value ($)':    total_native,
                    'Underlying Price':   round(price, 2),
                    'Strike':             None,
                    'Multiplier':         None,
                    'Premium':            None,
                    'Avg Cost':           round(avg_cost_val, 4) if avg_cost_val else None,
                    'Currency':           currency,
                    'FX Rate':            fx_rate,
                    f'Total Value ({base_ccy})': total_base,
                })

        if results:
            df = pd.DataFrame(results)
            st.session_state.portfolio_data = df
            return df

    return None


def navigate_to(page: str):
    """Cambia la página activa en session state y hace rerun."""
    st.session_state.page = page
    st.rerun()


def no_portfolio_warning():
    """Muestra un aviso estándar con botón de navegación cuando no hay portfolio."""
    from modules.i18n import t
    st.warning(t("general.no_portfolio"))
    if st.button(t("general.go_portfolio"), type="primary"):
        navigate_to(t("nav.portfolio"))



def jarque_bera_test(returns_series) -> dict | None:
    """
    Test de normalidad Jarque-Bera.
    JB = (n/6) * (S² + K²/4)
      S = skewness de los retornos
      K = exceso de curtosis  (kurtosis - 3)  <-- pandas .kurtosis() lo devuelve directamente
    H0: los retornos siguen una distribución normal.
    Estadístico ~ χ²(2) bajo H0.
    Umbrales:   JB < 4.61 → normal (90%)
                4.61 ≤ JB < 9.21 → dudosa (rechazada al 5%)
                JB ≥ 9.21 → no normal (rechazada al 1%)
    """
    import numpy as np
    from scipy import stats as _stats

    r = returns_series.dropna()
    n = len(r)
    if n < 30:
        return None          # Muestra insuficiente

    S = float(r.skew())        # Asimetría de Fisher
    K = float(r.kurtosis())    # Exceso de curtosis (pandas ya resta 3)

    JB = (n / 6.0) * (S ** 2 + K ** 2 / 4.0)
    p_value = float(1.0 - _stats.chi2.cdf(JB, df=2))

    if JB < 4.61:
        status = "normal"
    elif JB < 9.21:
        status = "dudosa"
    else:
        status = "no_normal"

    return {
        "jb":               round(JB, 3),
        "p_value":          round(p_value, 4),
        "skewness":         round(S, 4),
        "excess_kurtosis":  round(K, 4),
        "status":           status,        # 'normal' | 'dudosa' | 'no_normal'
        "is_normal":        status == "normal",
        "n":                n,
    }


def ljung_box_test(returns_series, lags=(1, 2, 3)) -> dict | None:
    """
    Test de Ljung-Box para autocorrelación serial.
    H0: no hay autocorrelación hasta el lag k.
    Q = n(n+2) * sum(rho_k^2 / (n-k))  ~  chi2(len(lags))
    Umbrales p-valor: >= 0.05 no rechaza (independiente)
                      0.01-0.05 dudosa
                      < 0.01 rechaza (dependencia serial)
    """
    import numpy as np
    from scipy import stats as _stats

    r = returns_series.dropna()
    n = len(r)
    if n < 30:
        return None

    rho = {}
    for lag in lags:
        rho[lag] = float(r.autocorr(lag=lag))

    Q = float(n * (n + 2) * sum(rho[k] ** 2 / (n - k) for k in lags))
    p_value = float(1.0 - _stats.chi2.cdf(Q, df=len(lags)))

    if p_value >= 0.05:
        status = "normal"
    elif p_value >= 0.01:
        status = "dudosa"
    else:
        status = "no_normal"

    return {
        "Q":             round(Q, 3),
        "p_value":       round(p_value, 4),
        "rho":           {k: round(v, 4) for k, v in rho.items()},
        "lags":          list(lags),
        "status":        status,
        "is_independent": status == "normal",
        "n":             n,
    }


def arch_lm_test(returns_series, lags: int = 5) -> dict | None:
    """
    Test ARCH-LM de Engle para efectos ARCH (volatility clustering).
    Regresiona retornos^2 sobre sus propios lags.
    LM = n * R2  ~  chi2(lags)
    H0: no hay efectos ARCH (varianza constante).
    """
    import numpy as np
    from scipy import stats as _stats

    r =returns_series.dropna()
    n = len(r)
    if n < lags + 10:
        return None

    r_sq = r.values ** 2
    Y = r_sq[lags:]
    X_cols = [r_sq[lags - i - 1:n - i - 1] for i in range(lags)]
    X = np.column_stack([np.ones(len(Y))] + X_cols)

    try:
        beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        Y_hat = X @ beta
        SS_res = float(np.sum((Y - Y_hat) ** 2))
        SS_tot = float(np.sum((Y - Y.mean()) ** 2))
        R2 = max(0.0, 1.0 - SS_res / SS_tot) if SS_tot > 0 else 0.0
    except Exception:
        return None

    LM = float((n - lags) * R2)
    p_value = float(1.0 - _stats.chi2.cdf(LM, df=lags))

    if p_value >= 0.05:
        status = "normal"
    elif p_value >= 0.01:
        status = "dudosa"
    else:
        status = "no_normal"

    return {
        "LM":             round(LM, 3),
        "p_value":        round(p_value, 4),
        "R2":             round(R2, 6),
        "lags":           lags,
        "status":         status,
        "is_homoscedastic": status == "normal",
        "n":              n,
    }

