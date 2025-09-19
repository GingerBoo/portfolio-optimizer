# app.py — Streamlit frontend (interaktivní tabulky)
import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ⬇⬇⬇ N A H R A Ď  touto URL svou skutečnou backend adresu ⬇⬇⬇
API_URL = "https://portfolio-optimizer-tt81.onrender.com/optimize"

st.set_page_config(page_title="Anti-Correlation Portfolio", layout="wide")
st.title("📉 Anti-Correlation Portfolio Optimizer")

# --- Parametry od uživatele ---
with st.sidebar:
    st.header("⚙️ Parametry")
    tickers = st.text_input("Tickery (oddělit čárkou)", "AAPL,MSFT,XOM,JNJ,GLD,TLT")
    start = st.text_input("Start (YYYY-MM-DD)", "2019-01-01")
    freq = st.selectbox("Frekvence", ["ME", "D"], index=0)
    ret_target = st.number_input("Cílový výnos p.a.", value=0.07, step=0.01, format="%.2f")
    vol_max = st.number_input("Max volatilita p.a.", value=0.22, step=0.01, format="%.2f")
    run = st.button("🔮 Optimalizovat", type="primary", use_container_width=True)

# --- Akce po kliknutí ---
if run:
    body = {
        "tickers": [t.strip() for t in tickers.split(",") if t.strip()],
        "start": start,
        "freq": freq,
        "ret_target": float(ret_target),
        "vol_max": float(vol_max),
    }

    with st.spinner("Počítám…"):
        try:
            r = requests.post(API_URL, json=body, timeout=180)
        except Exception as e:
            st.error(f"Chyba volání API: {e}")
            st.stop()

    if not r.ok:
        st.error(f"Chyba API: HTTP {r.status_code}")
        st.stop()

    resp = r.json()

    if not resp.get("ok", False):
        st.warning("Nenašlo se řešení. Zkus snížit cíl výnosu (např. 0.06) nebo zvýšit max volatilitu (např. 0.25), případně přidat různorodé tickery.")
        st.stop()

    # --- Shrnutí metrik ---
    c1, c2, c3 = st.columns(3)
    exp = resp["portfolio"]["expected_return"] * 100
    vol = resp["portfolio"]["volatility"] * 100
    avgc = resp["correlation"]["avg_pairwise"]
    c1.metric("🎯 Oček. výnos p.a.", f"{exp:.2f}%")
    c2.metric("🌪️ Volatilita p.a.", f"{vol:.2f}%")
    c3.metric("🔗 Průměrná korelace", f"{avgc:.3f}")

    # --- Váhy (interaktivní tabulka) ---
    weights_df = pd.DataFrame(resp["portfolio"]["weights"])
    weights_df["weight"] = (weights_df["weight"] * 100).round(2)
    weights_df = weights_df.sort_values("weight", ascending=False).reset_index(drop=True)

    st.subheader("Váhy v portfoliu")
    st.dataframe(weights_df, use_container_width=True)

    # --- Metriky aktiv (interaktivní tabulka) ---
    assets_df = pd.DataFrame(resp["assets"])
    assets_df["ann_mean_return"] = (assets_df["ann_mean_return"] * 100).round(2)
    assets_df["ann_vol"] = (assets_df["ann_vol"] * 100).round(2)

    st.subheader("Metriky aktiv")
    st.dataframe(assets_df, use_container_width=True)

    # --- Bar chart vah ---
    st.subheader("Graf vah")
    fig, ax = plt.subplots()
    ax.bar(weights_df["ticker"], weights_df["weight"])
    ax.set_title("Váhy v portfoliu (%)")
    ax.set_xlabel("Ticker"); ax.set_ylabel("Váha (%)")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

    # --- Exporty ---
    st.download_button("⬇️ Stáhnout váhy (CSV)", data=weights_df.to_csv(index=False).encode("utf-8"),
                       file_name="vahy.csv", mime="text/csv")
    st.download_button("⬇️ Stáhnout metriky aktiv (CSV)", data=assets_df.to_csv(index=False).encode("utf-8"),
                       file_name="aktiva.csv", mime="text/csv")
