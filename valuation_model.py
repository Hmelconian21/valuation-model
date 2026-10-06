#!/usr/bin/env python3
"""
valuation_model.py

Composite stock valuation: blends three independent, imperfect models into
one report instead of leaning on any single one.

1. DCF fair value       — reuses dcf.py's compute_dcf() as-is (cash flow
                           projection, discounted back at estimated WACC).
2. Comps (P/E) model     — the stock's own trailing P/E vs. its own
                           historical average P/E (mean-reversion to its own
                           multiple, not a peer-group comp), applied to
                           trailing EPS.
3. Analyst consensus     — Wall Street's mean/median ~12-month price target
                           from .info, as a third, independent cross-check.

Only the analyst target carries any timeline (the Street's conventional
~12-month horizon, not a date this data guarantees) — the DCF and comps
figures are both fair-value-as-of-today, not a projected future price.

Each method's fair value is shown side by side with a WEIGHTED COMPOSITE
price (equal-weighted by default, override with --weights) and an
agreement/disagreement read across the three — a wide spread is a signal to
dig into WHY, not something to average away. Recent headlines are printed as
context (same as dcf.py/outcomes.py).

None of these three is ground truth. This is a framework for comparing
models against each other, not a prediction.

USAGE
-----
    python3 valuation_model.py NVDA
    python3 valuation_model.py NVDA --weights 0.5,0.25,0.25   # dcf,comps,analyst
    python3 valuation_model.py NVDA --dcf-growth 0.15 --dcf-discount-rate 0.10
    python3 valuation_model.py NVDA --comps-years 8 --news-count 6
"""

import sys
import argparse
import statistics

import yfinance as yf

import dcf
from outcomes import get_news

DEFAULT_WEIGHTS = {"dcf": 1 / 3, "comps": 1 / 3, "analyst": 1 / 3}
METHOD_LABELS = {
    "dcf": "DCF Fair Value",
    "comps": "Comps (Historical P/E)",
    "analyst": "Analyst Target (~12mo)",
}
DEFAULT_COMPS_YEARS = 5
DEFAULT_NEWS_COUNT = 4
MAX_SANE_PE = 200   # historical P/E data points outside this are dropped as distorted (e.g. near-zero EPS year)
AGREEMENT_STRONG = 0.08   # spread/mean below this -> "Strong agreement"
AGREEMENT_MODERATE = 0.20  # spread/mean below this -> "Moderate agreement", above -> "Wide disagreement"


# --------------------------------------------------------------------------
# Comps (P/E vs. own history) model
# --------------------------------------------------------------------------

def compute_comps(tk: yf.Ticker, spot: float, lookback_years: int = DEFAULT_COMPS_YEARS) -> dict:
    """Comps here means the stock against ITS OWN history, not a peer group —
    yfinance doesn't give a clean peer set to compare against, but a
    multiple's own historical range is a legitimate (and simpler) comps
    read: is the market paying more or less for this company's earnings
    than it usually has?"""
    try:
        info = tk.info
    except Exception:
        info = {}

    trailing_eps = info.get("trailingEps")
    current_pe = info.get("trailingPE")
    if current_pe is None and trailing_eps and trailing_eps > 0 and spot:
        current_pe = spot / trailing_eps

    data = {
        "current_pe": current_pe, "trailing_eps": trailing_eps,
        "historical_pe": [], "avg_historical_pe": None,
        "implied_price": None, "upside": None, "error": None,
    }

    try:
        fin = tk.financials
    except Exception:
        fin = None

    if fin is None or fin.empty or "Diluted EPS" not in fin.index:
        data["error"] = "No historical EPS data available — can't build a comps model."
        return data

    eps_row = fin.loc["Diluted EPS"].dropna()

    try:
        price_hist = tk.history(period=f"{lookback_years + 1}y")
    except Exception:
        price_hist = None

    if price_hist is None or price_hist.empty:
        data["error"] = "No historical price data available — can't build a comps model."
        return data

    tz = price_hist.index.tz
    for period_end, eps in eps_row.items():
        if eps is None or eps <= 0:
            continue  # P/E is undefined for a loss year — skip rather than distort the average
        d = period_end.tz_localize(tz) if tz is not None and period_end.tzinfo is None else period_end
        try:
            idx = price_hist.index.get_indexer([d], method="nearest")[0]
        except Exception:
            continue
        if idx < 0:
            continue
        price = float(price_hist["Close"].iloc[idx])
        pe = price / eps
        if 0 < pe < MAX_SANE_PE:
            data["historical_pe"].append({
                "period_end": period_end.strftime("%Y-%m-%d"),
                "eps": float(eps), "price": price, "pe": pe,
            })

    data["historical_pe"].sort(key=lambda x: x["period_end"])

    if not data["historical_pe"]:
        data["error"] = "Couldn't compute any historical P/E data points — can't build a comps model."
        return data

    data["avg_historical_pe"] = statistics.mean(x["pe"] for x in data["historical_pe"])

    if trailing_eps and trailing_eps > 0:
        data["implied_price"] = data["avg_historical_pe"] * trailing_eps
        if spot:
            data["upside"] = (data["implied_price"] - spot) / spot
    else:
        data["error"] = "Trailing EPS is negative or unavailable — can't apply the historical multiple."

    return data


# --------------------------------------------------------------------------
# Analyst consensus
# --------------------------------------------------------------------------

def compute_analyst_target(tk: yf.Ticker, spot: float) -> dict:
    try:
        info = tk.info
    except Exception:
        info = {}

    target_mean = info.get("targetMeanPrice")
    target_median = info.get("targetMedianPrice")
    target_price = target_mean if target_mean is not None else target_median

    return {
        "target_mean": target_mean,
        "target_median": target_median,
        "target_high": info.get("targetHighPrice"),
        "target_low": info.get("targetLowPrice"),
        "num_analysts": info.get("numberOfAnalystOpinions"),
        "recommendation": info.get("recommendationKey"),
        "implied_price": target_price,
        "upside": (target_price - spot) / spot if target_price and spot else None,
        "error": None if target_price else "No analyst price targets available for this ticker.",
    }


# --------------------------------------------------------------------------
# Composite
# --------------------------------------------------------------------------

def is_usable_price(price) -> bool:
    """A negative/zero implied price (e.g. a DCF run off a negative FCF base)
    isn't a meaningful fair value — don't let it drag down a weighted
    average with methods that produced a real number."""
    return price is not None and price > 0


def normalize_weights(w: dict) -> dict:
    total = sum(w.values())
    if total <= 0:
        raise ValueError("Weights must sum to a positive number.")
    return {k: v / total for k, v in w.items()}


def agreement_assessment(values: list[float]) -> dict:
    lo, hi, mean = min(values), max(values), statistics.mean(values)
    spread_pct = (hi - lo) / mean if mean else None
    if spread_pct is None:
        label = "N/A"
    elif spread_pct < AGREEMENT_STRONG:
        label = "Strong agreement"
    elif spread_pct < AGREEMENT_MODERATE:
        label = "Moderate agreement"
    else:
        label = "Wide disagreement"
    return {"low": lo, "high": hi, "mean": mean, "spread_pct": spread_pct, "label": label}


def compute_valuation(ticker: str, weights: dict | None = None, dcf_growth: float | None = None,
                       dcf_discount_rate: float | None = None, comps_years: int = DEFAULT_COMPS_YEARS,
                       news_count: int = DEFAULT_NEWS_COUNT) -> dict:
    """Fetches + computes all three models and returns plain data — same
    decoupling as dcf.py's compute_dcf() / outcomes.py's compute_outcomes(),
    so this could feed a dashboard later without scraping printed text."""
    ticker = ticker.upper()
    weights = normalize_weights(dict(weights) if weights else dict(DEFAULT_WEIGHTS))

    tk = yf.Ticker(ticker)
    spot = dcf.get_spot(tk)

    if spot is None:
        return {"ticker": ticker, "spot": None,
                "error": f"Couldn't find ticker '{ticker}'. Double check the symbol."}

    dcf_data = dcf.compute_dcf(ticker, growth=dcf_growth, discount_rate=dcf_discount_rate)
    comps_data = compute_comps(tk, spot, comps_years)
    analyst_data = compute_analyst_target(tk, spot)
    news = get_news(tk, news_count)

    methods = []
    for key in ("dcf", "comps", "analyst"):
        sub = {"dcf": dcf_data, "comps": comps_data, "analyst": analyst_data}[key]
        if is_usable_price(sub.get("implied_price")):
            methods.append({
                "key": key, "label": METHOD_LABELS[key], "price": sub["implied_price"],
                "weight": weights[key], "upside": sub["upside"],
            })

    composite = None
    composite_upside = None
    agreement = None
    if methods:
        total_w = sum(m["weight"] for m in methods)
        composite = (sum(m["price"] * m["weight"] for m in methods) / total_w) if total_w > 0 \
            else statistics.mean(m["price"] for m in methods)
        composite_upside = (composite - spot) / spot
    if len(methods) >= 2:
        agreement = agreement_assessment([m["price"] for m in methods])

    return {
        "ticker": ticker, "spot": spot, "error": None,
        "dcf": dcf_data, "comps": comps_data, "analyst": analyst_data,
        "methods": methods, "weights": weights,
        "composite": composite, "composite_upside": composite_upside,
        "agreement": agreement, "news": news,
    }


# --------------------------------------------------------------------------
# Display
# --------------------------------------------------------------------------

def money(x) -> str:
    if x is None:
        return "N/A"
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.2f}"


def fmt_pct(x) -> str:
    if x is None:
        return "N/A"
    return f"{x*100:+.1f}%"


def print_report(data: dict):
    if data.get("spot") is None:
        print(data["error"])
        sys.exit(1)

    ticker = data["ticker"]
    print("=" * 78)
    print(f"{ticker} — COMPOSITE VALUATION MODEL".center(78))
    print("=" * 78)
    print(f"Current price: {money(data['spot'])}")

    print()
    print("VALUATION METHODS")
    print("-" * 78)
    print(f"{'Method':<26}{'Fair Value':>14}{'Upside/Downside':>20}{'Weight':>12}")
    for key in ("dcf", "comps", "analyst"):
        sub = data[key]
        label = METHOD_LABELS[key]
        price = sub.get("implied_price")
        if is_usable_price(price):
            w = data["weights"][key]
            print(f"{label:<26}{money(price):>14}{fmt_pct(sub['upside']):>20}{w*100:>11.0f}%")
        elif price is not None:
            reason = "base FCF is negative — company isn't cash-flow positive" \
                if key == "dcf" and sub.get("nonpositive_base") else "non-positive fair value isn't meaningful"
            print(f"{label:<26}{money(price):>14}{'excluded':>20}{'—':>12}   ({reason})")
        else:
            print(f"{label:<26}{'N/A':>14}{'—':>20}{'—':>12}   ({sub.get('error') or 'unavailable'})")

    if data["composite"] is not None:
        n_used = len(data["methods"])
        print()
        print("=" * 78)
        print(f"  WEIGHTED COMPOSITE FAIR VALUE: {money(data['composite'])}  (from {n_used} of 3 methods)")
        print(f"  CURRENT PRICE:                 {money(data['spot'])}")
        direction = "UPSIDE" if data["composite_upside"] >= 0 else "DOWNSIDE"
        print(f"  IMPLIED {direction}:{'':<16}{fmt_pct(data['composite_upside'])}")
        print("=" * 78)
    else:
        print("\nNo valuation method produced a usable estimate for this ticker.")

    print()
    print("METHOD AGREEMENT")
    print("-" * 78)
    ag = data["agreement"]
    if ag:
        print(f"  Range:      {money(ag['low'])} - {money(ag['high'])}")
        print(f"  Spread:     {fmt_pct(ag['spread_pct']).lstrip('+')} of the mean estimate")
        print(f"  Assessment: {ag['label']}")
        if ag["label"] == "Wide disagreement":
            print()
            print("  The methods are telling meaningfully different stories here — worth")
            print("  digging into WHY (a noisy DCF cash-flow base, a P/E multiple that's")
            print("  re-rated structurally, or analysts slow to update) rather than just")
            print("  trusting the average over the disagreement.")
    elif len(data["methods"]) == 1:
        print("  Only one method produced a usable estimate — no cross-check available.")
    else:
        print("  Not enough methods produced usable estimates to assess agreement.")

    comps = data["comps"]
    if comps.get("implied_price"):
        print()
        print("COMPS DETAIL")
        print("-" * 78)
        pe_txt = f"{comps['current_pe']:.1f}x" if comps.get("current_pe") else "N/A"
        print(f"  Current trailing P/E:          {pe_txt}")
        print(f"  {len(comps['historical_pe'])}-yr historical average P/E:  {comps['avg_historical_pe']:.1f}x")
        by_year = ", ".join(f"{x['period_end'][:4]}: {x['pe']:.1f}x" for x in comps["historical_pe"])
        print(f"  By fiscal year:                {by_year}")

    an = data["analyst"]
    if an.get("implied_price"):
        print()
        print("ANALYST DETAIL")
        print("-" * 78)
        print("  Targets are the ~12-month price targets Wall Street analysts publish —")
        print("  a Street convention, not a date guaranteed by this data. DCF and comps,")
        print("  by contrast, carry no timeline: they're fair-value-as-of-today, not a")
        print("  projected future price.")
        print(f"  Mean target:    {money(an['target_mean'])}")
        print(f"  Median target:  {money(an['target_median'])}")
        print(f"  Range:          {money(an['target_low'])} - {money(an['target_high'])}")
        print(f"  # Analysts:     {an['num_analysts'] if an['num_analysts'] else 'N/A'}")
        print(f"  Rating:         {an['recommendation'] or 'N/A'}")

    print()
    print("RECENT HEADLINES")
    print("-" * 78)
    if data["news"]:
        for n in data["news"]:
            print(f"  • {n['title']}  ({n['publisher']})")
    else:
        print("  No recent headlines found.")

    print()
    print("=" * 78)
    print("Note: this blends three independent, imperfect models — a cash-flow")
    print("projection (DCF), a mean-reversion-to-its-own-multiple model (comps),")
    print("and Wall Street's own targets (analyst consensus). None of them is")
    print("ground truth; where they disagree is often more informative than the")
    print("composite number itself. Not investment advice.")
    print("=" * 78)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_weights(s: str) -> dict:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--weights needs exactly 3 comma-separated numbers: dcf,comps,analyst")
    try:
        dcf_w, comps_w, analyst_w = (float(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError("--weights values must be numbers, e.g. 0.5,0.25,0.25")
    if any(w < 0 for w in (dcf_w, comps_w, analyst_w)):
        raise argparse.ArgumentTypeError("--weights values can't be negative")
    return {"dcf": dcf_w, "comps": comps_w, "analyst": analyst_w}


def main():
    parser = argparse.ArgumentParser(description="Composite stock valuation: DCF + comps (P/E) + analyst consensus.")
    parser.add_argument("ticker", help="Ticker symbol, e.g. NVDA")
    parser.add_argument("--weights", type=parse_weights, default=None,
                         help="Comma-separated weights for dcf,comps,analyst (need not sum to 1, "
                              "e.g. 0.5,0.25,0.25). Default: equal weight.")
    parser.add_argument("--dcf-growth", type=float, default=None,
                         help="Override the DCF's FCF growth rate, e.g. 0.15. Default: dcf.py's historical CAGR.")
    parser.add_argument("--dcf-discount-rate", type=float, default=None,
                         help="Override the DCF's discount rate (WACC), e.g. 0.10. Default: estimated WACC.")
    parser.add_argument("--comps-years", type=int, default=DEFAULT_COMPS_YEARS,
                         help=f"Years of history to average for the comps P/E model. Default: {DEFAULT_COMPS_YEARS}.")
    parser.add_argument("--news-count", type=int, default=DEFAULT_NEWS_COUNT,
                         help=f"Number of recent headlines to print. Default: {DEFAULT_NEWS_COUNT}.")
    args = parser.parse_args()

    data = compute_valuation(
        args.ticker, args.weights, args.dcf_growth, args.dcf_discount_rate,
        args.comps_years, args.news_count,
    )
    print_report(data)


if __name__ == "__main__":
    main()
