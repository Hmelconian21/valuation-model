#!/usr/bin/env python3
"""
outcomes.py

For a given ticker: pulls live price + implied vol from the options market,
computes a probability table of where the stock could land at different
future dates (1 week / 1 month / next earnings / 3 months), and pairs it
with the fundamental + news context that explains WHY the market is
pricing that range (upcoming earnings, recent headlines, growth/margin
snapshot).

This does NOT predict the future. It shows what the OPTIONS MARKET is
currently implying, based on live IV, under a standard lognormal
(risk-neutral) assumption. Treat it as "what's priced in," not a forecast.

USAGE
-----
    python3 outcomes.py NVDA
    python3 outcomes.py NVDA --levels -30,-20,-10,-5,0,5,10,20,30
"""

import sys
import math
import argparse
from datetime import datetime, date

import yfinance as yf
from scipy.stats import norm

RISK_FREE_RATE = 0.045
DEFAULT_LEVELS = [-20, -10, -5, 0, 5, 10, 20]
HORIZONS_DAYS = {"1 Week": 7, "1 Month": 30, "3 Months": 91}


# --------------------------------------------------------------------------
# Probability math
# --------------------------------------------------------------------------

def prob_above(spot: float, target: float, iv: float, t_years: float, r: float = RISK_FREE_RATE) -> float:
    """Risk-neutral probability that price finishes ABOVE target at time t.
    Standard lognormal / Black-Scholes N(d2)."""
    if t_years <= 0:
        return 1.0 if spot > target else 0.0
    if iv <= 0:
        return 1.0 if spot > target else 0.0
    d2 = (math.log(spot / target) + (r - 0.5 * iv ** 2) * t_years) / (iv * math.sqrt(t_years))
    return norm.cdf(d2)


def expected_move(spot: float, iv: float, t_years: float) -> float:
    """1 standard deviation dollar move over the horizon (approx, common trader shorthand)."""
    return spot * iv * math.sqrt(t_years)


# --------------------------------------------------------------------------
# Data fetching
# --------------------------------------------------------------------------

def get_spot(tk: yf.Ticker) -> float:
    try:
        price = tk.fast_info["last_price"]
        if price:
            return float(price)
    except Exception:
        pass
    hist = tk.history(period="1d")
    return float(hist["Close"].iloc[-1])


def get_atm_iv(tk: yf.Ticker, spot: float, target_days: int) -> tuple[float, str] | None:
    """Finds the expiration closest to target_days out, pulls IV of the
    strike closest to at-the-money. Returns (iv, expiration_used) or None."""
    try:
        expirations = tk.options
    except Exception:
        return None
    if not expirations:
        return None

    best_exp, best_diff = None, None
    for exp in expirations:
        exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
        days_out = (exp_date - date.today()).days
        if days_out <= 0:
            continue
        diff = abs(days_out - target_days)
        if best_diff is None or diff < best_diff:
            best_exp, best_diff = exp, diff

    if best_exp is None:
        return None

    try:
        chain = tk.option_chain(best_exp)
        calls = chain.calls
        calls = calls.assign(dist=(calls["strike"] - spot).abs())
        atm_row = calls.sort_values("dist").iloc[0]
        iv = float(atm_row["impliedVolatility"])
        if iv > 0:
            return iv, best_exp
    except Exception:
        pass
    return None


def get_next_earnings(tk: yf.Ticker) -> date | None:
    try:
        cal = tk.calendar
        if isinstance(cal, dict) and "Earnings Date" in cal:
            dates = cal["Earnings Date"]
            if dates:
                d = dates[0]
                return d if isinstance(d, date) else d.date()
        if hasattr(cal, "empty") and not cal.empty and "Earnings Date" in cal.index:
            val = cal.loc["Earnings Date"][0]
            return val if isinstance(val, date) else val.date()
    except Exception:
        pass
    return None


def get_news(tk: yf.Ticker, limit: int = 4) -> list[dict]:
    try:
        news = tk.news or []
        out = []
        for item in news[:limit]:
            content = item.get("content", item)  # yfinance schema has shifted over versions
            title = content.get("title") or item.get("title")
            publisher = (content.get("provider") or {}).get("displayName") if isinstance(content.get("provider"), dict) else item.get("publisher")
            if title:
                out.append({"title": title, "publisher": publisher or "Unknown source"})
        return out
    except Exception:
        return []


def get_financial_snapshot(tk: yf.Ticker) -> dict:
    try:
        info = tk.info
    except Exception:
        info = {}
    return {
        "revenue_growth": info.get("revenueGrowth"),
        "gross_margins": info.get("grossMargins"),
        "profit_margins": info.get("profitMargins"),
        "forward_pe": info.get("forwardPE"),
        "trailing_pe": info.get("trailingPE"),
        "target_mean_price": info.get("targetMeanPrice"),
        "recommendation": info.get("recommendationKey"),
        "short_pct_float": info.get("shortPercentOfFloat"),
    }


# --------------------------------------------------------------------------
# Display
# --------------------------------------------------------------------------

def pct(x) -> str:
    if x is None:
        return "N/A"
    return f"{x*100:.1f}%"


def money(x) -> str:
    if x is None:
        return "N/A"
    return f"${x:,.2f}"


def compute_outcomes(ticker: str, levels: list[int] | None = None) -> dict:
    """Fetches + computes everything needed for the outcomes report and
    returns it as plain data, so callers other than the CLI (dashboard,
    menu bar) can reuse it without scraping printed text."""
    levels = list(DEFAULT_LEVELS) if levels is None else levels
    tk = yf.Ticker(ticker)
    spot = get_spot(tk)

    earnings_date = get_next_earnings(tk)
    days_to_earnings = (earnings_date - date.today()).days if earnings_date else None

    data = {
        "ticker": ticker.upper(),
        "spot": spot,
        "earnings_date": earnings_date.strftime("%Y-%m-%d") if earnings_date else None,
        "days_to_earnings": days_to_earnings,
        "fundamentals": get_financial_snapshot(tk),
        "news": get_news(tk),
        "levels": levels,
        "horizons": {},
        "probability_table": [],
        "error": None,
    }

    horizons = dict(HORIZONS_DAYS)
    if days_to_earnings and days_to_earnings > 0:
        horizons[f"Next Earnings ({earnings_date.strftime('%m/%d')})"] = days_to_earnings

    iv_by_horizon = {}
    for label, days in horizons.items():
        result = get_atm_iv(tk, spot, days)
        if result:
            iv_by_horizon[label] = result  # (iv, expiration_used)

    if not iv_by_horizon:
        data["error"] = "No live options data found for this ticker — can't build the probability table."
        return data

    active_labels = [h for h in horizons if h in iv_by_horizon]
    for label in active_labels:
        iv, exp_used = iv_by_horizon[label]
        days = horizons[label]
        t = days / 365
        em = expected_move(spot, iv, t)
        data["horizons"][label] = {
            "days": days,
            "iv": iv,
            "exp_used": exp_used,
            "expected_move": em,
            "expected_move_pct": em / spot if spot else None,
        }

    for lvl in sorted(levels, reverse=True):
        target = spot * (1 + lvl / 100)
        probs = {}
        for label in active_labels:
            iv, _ = iv_by_horizon[label]
            t = horizons[label] / 365
            probs[label] = prob_above(spot, target, iv, t)
        data["probability_table"].append({"level": lvl, "target": target, "probs": probs})

    return data


def print_report(ticker: str, levels: list[int]):
    data = compute_outcomes(ticker, levels)

    print("=" * 78)
    print(f"{data['ticker']} — OUTCOME PROBABILITIES".center(78))
    print("=" * 78)
    print(f"Current price: {money(data['spot'])}")

    if data["earnings_date"]:
        print(f"Next earnings: {data['earnings_date']} ({data['days_to_earnings']} days out)")
    else:
        print("Next earnings: not available")

    fin = data["fundamentals"]
    print()
    print("FUNDAMENTALS SNAPSHOT")
    print("-" * 78)
    print(f"  Revenue growth (YoY):   {pct(fin['revenue_growth'])}")
    print(f"  Gross margin:           {pct(fin['gross_margins'])}")
    print(f"  Profit margin:          {pct(fin['profit_margins'])}")
    print(f"  Forward P/E:            {fin['forward_pe'] if fin['forward_pe'] else 'N/A'}")
    print(f"  Analyst avg target:     {money(fin['target_mean_price'])}")
    print(f"  Analyst rating:         {fin['recommendation'] or 'N/A'}")
    print(f"  Short % of float:       {pct(fin['short_pct_float'])}")

    print()
    print("RECENT HEADLINES")
    print("-" * 78)
    if data["news"]:
        for n in data["news"]:
            print(f"  • {n['title']}  ({n['publisher']})")
    else:
        print("  No recent headlines found.")

    if data["error"]:
        print()
        print(data["error"])
        return

    print()
    print("IMPLIED EXPECTED MOVE (1 standard deviation, from live options IV)")
    print("-" * 78)
    for label, h in data["horizons"].items():
        print(f"  {label:28} IV: {h['iv']*100:5.1f}%   ±{money(h['expected_move'])}   "
              f"({pct(h['expected_move_pct'])})   [exp used: {h['exp_used']}]")

    print()
    print("PROBABILITY OF FINISHING ABOVE EACH LEVEL")
    print("-" * 78)
    active_labels = list(data["horizons"].keys())
    col_width = max(22, max(len(h) for h in active_labels) + 4)
    header = f"{'Level':>16}" + "".join(f"{h:>{col_width}}" for h in active_labels)
    print(header)

    for row_data in data["probability_table"]:
        lvl, target = row_data["level"], row_data["target"]
        row = f"{lvl:+d}% ({target:.2f})".rjust(16)
        for label in active_labels:
            p = row_data["probs"][label]
            marker = "*" if lvl == 0 else " "
            cell = f"{p*100:.1f}%{marker}"
            row += cell.rjust(col_width)
        print(row)

    print()
    print("* = current price row (should show ~50% — sanity check)")
    print("=" * 78)
    print("Note: probabilities reflect the options market's current implied")
    print("volatility under a standard lognormal model. This is what's priced")
    print("in, not a prediction — IV can and does change with news flow.")
    print("=" * 78)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Options-implied outcome probabilities for a ticker.")
    parser.add_argument("ticker", help="Ticker symbol, e.g. NVDA")
    parser.add_argument(
        "--levels",
        default=",".join(str(x) for x in DEFAULT_LEVELS),
        help="Comma-separated %% move levels to test, e.g. -20,-10,0,10,20",
    )
    args = parser.parse_args()

    levels = [int(x.strip()) for x in args.levels.split(",")]
    print_report(args.ticker, levels)


if __name__ == "__main__":
    main()
