import yfinance as yf
import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay


def fetch_data(tickers, start_date, end_date):
    tickers = list(dict.fromkeys(tickers))
    df = yf.download(tickers, start=start_date, end=end_date, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        if "Adj Close" in df.columns.levels[0]:
            df = df.xs("Adj Close", axis=1, level=0)
        else:
            df = df.xs("Close", axis=1, level=0)
    else:
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        df = df[[col]]
        df.columns = tickers
    return df.dropna(how="all").dropna(how="all", axis=1)


def compute_daily_returns(price_df):
    # Logarithmic returns
    return np.log(price_df / price_df.shift(1)).dropna(how="all")


def estimate_expected_return(returns_df, tickers, end_date):
    """Estimate annualized simple returns from log returns"""
    log_daily_mean = returns_df.loc[:end_date, tickers].mean()
    ann_log = log_daily_mean * 252
    simple_ann = np.exp(ann_log) - 1
    return simple_ann


def compute_expected_portfolio_return(exp_returns, weights, period_start, period_end):
    """Compute annual and period expected portfolio return"""
    tickers = list(exp_returns.index)
    w = np.array([weights[t] for t in tickers])
    ann_port = np.dot(exp_returns.values, w)

    days = len(pd.date_range(start=period_start, end=period_end, freq=BDay()))
    simple_period = ann_port * (days / 252)

    return ann_port, simple_period


def compute_realized_portfolio_return(price_df, weights, period_start, period_end):
    """Compute realized portfolio return for a period"""
    tickers = list(weights.keys())
    df = price_df[tickers].loc[period_start:period_end].dropna(how="all", axis=1)
    df = df.dropna(how="any", axis=0)

    if df.empty:
        return np.nan
    start_price = df.iloc[0]
    end_price = df.iloc[-1]
    ind_ret = (end_price / start_price) - 1
    w = np.array([weights[t] for t in tickers])
    return np.dot(ind_ret.values, w)


if __name__ == "__main__":
    # Parameters
    portfolio_tickers = ["FN"]
    portfolio_weights = {"FN": 1}
    benchmark = "FN"
    hist_start = "2020-01-31"
    hist_end = "2025-06-01"
    expected_end_date = "2025-01-31"
    period_start = "2025-01-31"
    period_end = "2025-06-01"

    # 1. Fetch data
    price_df = fetch_data(portfolio_tickers, hist_start, hist_end)
    bench_price_df = fetch_data([benchmark], hist_start, hist_end)

    # 2. Daily log returns
    returns_df = compute_daily_returns(price_df)
    bench_returns_df = compute_daily_returns(bench_price_df)

    # 3. Expected returns
    exp_returns = estimate_expected_return(returns_df, portfolio_tickers, expected_end_date)
    ann_exp, per_exp = compute_expected_portfolio_return(
        exp_returns, portfolio_weights, period_start, period_end
    )

    # 4. Realized returns
    real_port_ret = compute_realized_portfolio_return(
        price_df, portfolio_weights, period_start, period_end
    )
    real_bench_ret = compute_realized_portfolio_return(
        bench_price_df, {benchmark: 1}, period_start, period_end
    )

    # 5. Risk calculation
    port_daily = returns_df.loc[period_start:period_end, portfolio_tickers].dot(
        np.array([portfolio_weights[t] for t in portfolio_tickers])
    )
    bench_daily = bench_returns_df.loc[period_start:period_end, benchmark]
    std_port = port_daily.std(ddof=0)
    std_bench = bench_daily.std(ddof=0)
    risk_port = std_port * np.sqrt(252)
    risk_bench = std_bench * np.sqrt(252)

    # 6. Output
    print(f"\U0001F4C8 Yıllık Beklenen Getiri (Portföy, basit/log): {ann_exp:.2%}")
    print(f"⏳ Dönem Beklenen Getiri (Portföy): {per_exp:.2%}")
    print(f"🏆 Gerçekleşen Getiri (Portföy): {real_port_ret:.2%}")
    print(f"📊 Gerçekleşen Getiri (Benchmark): {real_bench_ret:.2%}")
    print(f"⚠️ Yıllık Risk (Portföy, log): {risk_port:.2%}")
    print(f"⚠️ Yıllık Risk (Benchmark, log): {risk_bench:.2%}")
