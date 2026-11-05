"""
Stock screener for the Laura Gao investment competition.

Purpose: pick candidate stocks for the GROWTH SLEEVE of the portfolio
(the part NOT needed for the $500k operating reserve, which should sit in
Treasuries / a bond ladder maturing 2033-2042).

Pipeline:  universe -> fetch fundamentals -> hard filters -> score -> diversify -> output

Install:   pip install yfinance pandas lxml
Run:       python stock_screener.py            (S&P 500 universe)
           python stock_screener.py tickers.csv (your own list, column "ticker")
"""

import sys
import time
import pandas as pd
import yfinance as yf

# ------------------------------------------------------------------
# 1. CONFIG: tune these. Rationale is next to each one.
# ------------------------------------------------------------------
CFG = {
    "min_market_cap": 10e9,         # large/mid cap only: liquidity + lower blow-up risk
    "min_revenue_growth": 0.03,     # revenue must be growing (>3% YoY)
    "min_fcf": 0,                   # must be FCF positive (hard rule)
    "min_fcf_margin": 0.05,         # FCF / revenue >= 5%: real cash generation
    "min_ebitda_margin": 0.15,      # "reasonable" EBITDA: >=15% margin (healthy, not just thin-margin)
    "max_debt_to_ebitda": 2.5,      # solid leverage: <=2.5x (3x is the usual danger line)
    "max_net_debt_to_ebitda": 2.0,  # net of cash
    "max_pe": 25,                   # target lower P/E; S&P avg is ~20-22
    "min_pe": 3,                    # filter out bad/negative or distorted earnings
    "max_per_sector": 3,            # diversification cap in the final list
    "top_n": 15,                    # size of final list
}

# Score weights (sum to 1). Higher score = better.
WEIGHTS = {
    "pe_score": 0.30,        # lower P/E is better (your stated priority)
    "leverage_score": 0.20,  # lower debt/EBITDA is better
    "fcf_score": 0.20,       # higher FCF margin is better
    "ebitda_score": 0.15,    # higher EBITDA margin is better
    "growth_score": 0.15,    # higher revenue growth is better
}


# ------------------------------------------------------------------
# 2. UNIVERSE
# ------------------------------------------------------------------
def load_universe(path=None):
    if path:
        return pd.read_csv(path)["ticker"].dropna().str.replace(".", "-", regex=False).tolist()
    import requests
    from io import StringIO
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).text
    table = pd.read_html(StringIO(html))[0]
    return table["Symbol"].str.replace(".", "-", regex=False).tolist()
# ------------------------------------------------------------------
# 3. DATA
# ------------------------------------------------------------------
def fetch_fundamentals(ticker):
    try:
        i = yf.Ticker(ticker).info
        return {
            "ticker": ticker,
            "name": i.get("shortName"),
            "sector": i.get("sector"),
            "market_cap": i.get("marketCap"),
            "revenue": i.get("totalRevenue"),
            "revenue_growth": i.get("revenueGrowth"),
            "fcf": i.get("freeCashflow"),
            "ebitda": i.get("ebitda"),
            "total_debt": i.get("totalDebt"),
            "cash": i.get("totalCash"),
            "pe": i.get("trailingPE"),
            "forward_pe": i.get("forwardPE"),
        }
    except Exception as e:
        print(f"  skip {ticker}: {e}")
        return None


def build_dataframe(tickers):
    rows = []
    for n, t in enumerate(tickers, 1):
        print(f"[{n}/{len(tickers)}] {t}", end="\r")
        r = fetch_fundamentals(t)
        if r:
            rows.append(r)
        time.sleep(0.1)  # be polite to the API
    df = pd.DataFrame(rows)

    # Derived metrics
    df["fcf_margin"] = df["fcf"] / df["revenue"]
    df["ebitda_margin"] = df["ebitda"] / df["revenue"]
    df["debt_to_ebitda"] = df["total_debt"] / df["ebitda"]
    df["net_debt_to_ebitda"] = (df["total_debt"] - df["cash"]) / df["ebitda"]
    # Use the blend of trailing and forward P/E when both exist (less distortion)
    df["pe_used"] = df[["pe", "forward_pe"]].mean(axis=1, skipna=True)
    return df


# ------------------------------------------------------------------
# 4. HARD FILTERS (pass/fail)
# ------------------------------------------------------------------
def apply_filters(df, c=CFG):
    d = df.dropna(subset=["market_cap", "revenue", "revenue_growth", "fcf",
                          "ebitda", "total_debt", "pe_used"]).copy()
    mask = (
        (d["market_cap"] >= c["min_market_cap"])
        & (d["revenue_growth"] >= c["min_revenue_growth"])          # revenue growing
        & (d["fcf"] > c["min_fcf"])                                 # positive FCF
        & (d["fcf_margin"] >= c["min_fcf_margin"])
        & (d["ebitda"] > 0)
        & (d["ebitda_margin"] >= c["min_ebitda_margin"])            # reasonable EBITDA
        & (d["debt_to_ebitda"] <= c["max_debt_to_ebitda"])          # solid leverage
        & (d["net_debt_to_ebitda"] <= c["max_net_debt_to_ebitda"])
        & (d["pe_used"].between(c["min_pe"], c["max_pe"]))          # lower P/E
    )
    return d[mask].copy()


# ------------------------------------------------------------------
# 5. SCORING (percentile rank within survivors, 0-1)
# ------------------------------------------------------------------
def score(d, w=WEIGHTS):
    d = d.copy()
    d["pe_score"] = d["pe_used"].rank(pct=True, ascending=False)               # low P/E -> high
    d["leverage_score"] = d["debt_to_ebitda"].rank(pct=True, ascending=False)  # low debt -> high
    d["fcf_score"] = d["fcf_margin"].rank(pct=True)
    d["ebitda_score"] = d["ebitda_margin"].rank(pct=True)
    d["growth_score"] = d["revenue_growth"].rank(pct=True)
    d["total_score"] = sum(d[k] * v for k, v in w.items())
    return d.sort_values("total_score", ascending=False)


# ------------------------------------------------------------------
# 6. DIVERSIFY (cap names per sector)
# ------------------------------------------------------------------
def diversify(d, c=CFG):
    picks, counts = [], {}
    for _, row in d.iterrows():
        s = row["sector"] or "Unknown"
        if counts.get(s, 0) < c["max_per_sector"]:
            picks.append(row)
            counts[s] = counts.get(s, 0) + 1
        if len(picks) == c["top_n"]:
            break
    return pd.DataFrame(picks)


# ------------------------------------------------------------------
# 7. MAIN
# ------------------------------------------------------------------
def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    tickers = load_universe(path)
    print(f"Universe: {len(tickers)} tickers")

    raw = build_dataframe(tickers)
    raw.to_csv("raw_fundamentals.csv", index=False)

    passed = apply_filters(raw)
    print(f"\nPassed hard filters: {len(passed)} / {len(raw)}")
    if passed.empty:
        print("Nothing passed. Loosen thresholds in CFG.")
        return

    ranked = score(passed)
    final = diversify(ranked)

    cols = ["ticker", "name", "sector", "pe_used", "revenue_growth", "fcf_margin",
            "ebitda_margin", "debt_to_ebitda", "total_score"]
    out = final[cols].round(3)
    print("\n=== FINAL CANDIDATES ===")
    print(out.to_string(index=False))
    out.to_csv("final_picks.csv", index=False)
    ranked.to_csv("all_ranked.csv", index=False)
    print("\nSaved: final_picks.csv, all_ranked.csv, raw_fundamentals.csv")


if __name__ == "__main__":
    main()
