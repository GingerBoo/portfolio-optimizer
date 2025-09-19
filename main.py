# main.py
# requirements: fastapi uvicorn yfinance pandas numpy cvxpy scikit-learn pydantic scs
import yfinance as yf
import pandas as pd
import numpy as np
import cvxpy as cp
from sklearn.covariance import LedoitWolf
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Portfolio Optimizer", version="1.0")

# CORS: povolíme volání odkudkoli (ChatGPT Actions)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# Základní "zdraví" endpoint, aby nasazení prošlo
@app.get("/")
def health():
    return {"status": "ok"}

class OptimizeRequest(BaseModel):
    tickers: List[str]
    start: str = "2019-01-01"    # YYYY-MM-DD
    freq: str = "ME"             # "ME" = monthly end, "D" = daily
    ret_target: float = 0.07
    vol_max: Optional[float] = 0.22  # může být i None

def fetch_prices(tickers, start):
    data = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    return data.dropna(how="all").dropna(axis=1, how="all")

def compute_returns(prices, freq="ME"):
    rets = np.log(prices).diff().dropna()
    if freq.upper() in ("ME","M","MONTHLY"):
        rets = rets.resample("ME").sum()  # měsíční log-výnosy
    return rets

def annualize(returns, period="ME"):
    k = 12 if period.upper() in ("ME","M","MONTHLY") else 252
    mu = returns.mean().values * k
    Sigma = LedoitWolf().fit(returns.values).covariance_ * k
    vol = np.sqrt(np.diag(Sigma))
    with np.errstate(divide='ignore', invalid='ignore'):
        Rho = Sigma / np.outer(vol, vol)
        Rho[np.isnan(Rho)] = 0.0
        np.fill_diagonal(Rho, 1.0)
    return mu, Sigma, Rho, vol

def optimize(mu, Sigma, Rho, ret_target=0.07, vol_max=None):
    n = len(mu)
    w = cp.Variable(n)
    obj = cp.Minimize(cp.quad_form(w, Rho))      # proxy za nízké korelace
    cons = [cp.sum(w) == 1, w >= 0, mu @ w >= ret_target]
    if vol_max is not None:
        cons.append(cp.quad_form(w, Sigma) <= vol_max**2)
    prob = cp.Problem(obj, cons)
    # SCS je dostupný jako pip balíček "scs"
    prob.solve(solver=cp.SCS, verbose=False)
    if w.value is None:
        return None
    ww = np.clip(w.value, 0, 1)
    return ww / ww.sum()

@app.post("/optimize")
def optimize_endpoint(req: OptimizeRequest):
    prices = fetch_prices(req.tickers, req.start)
    rets = compute_returns(prices, req.freq)
    mu, Sigma, Rho, vol = annualize(rets, req.freq)
    weights = optimize(mu, Sigma, Rho, req.ret_target, req.vol_max)
    if weights is None:
        return {"ok": False, "error": "INFEASIBLE_TRY_RELAXING_CONSTRAINTS"}

    tickers = list(prices.columns)
    port_ret = float((weights * mu).sum())
    port_vol = float(np.sqrt(weights @ Sigma @ weights))

    # korelační statistiky
    rho_df = pd.DataFrame(Rho, index=tickers, columns=tickers)
    mask = ~np.eye(len(rho_df), dtype=bool)
    avg_corr = float(rho_df.where(mask).stack().mean())
    min_corr = float(rho_df.where(mask).stack().min())
    max_corr = float(rho_df.where(mask).stack().max())

    return {
        "ok": True,
        "inputs": {
            "tickers": tickers, "start": req.start, "freq": req.freq,
            "ret_target": req.ret_target, "vol_max": req.vol_max
        },
        "portfolio": {
            "expected_return": port_ret,
            "volatility": port_vol,
            "weights": [{"ticker": t, "weight": float(w)} for t, w in zip(tickers, weights)],
        },
        "assets": [
            {"ticker": t, "ann_mean_return": float(m), "ann_vol": float(v)}
            for t, m, v in zip(tickers, mu, np.sqrt(np.diag(Sigma)))
        ],
        "correlation": {
            "avg_pairwise": avg_corr, "min_pairwise": min_corr, "max_pairwise": max_corr
        }
    }

