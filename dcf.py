#!/usr/bin/env python3
"""
dcf.py

Builds a Discounted Cash Flow valuation for a ticker: pulls historical
financials via yfinance, projects free cash flow forward, discounts it
back at an estimated WACC, and compares the resulting implied fair value
per share against the current market price.

This is an ANALYST TOOL, not an oracle. A DCF is only as good as its
assumptions (growth rate, discount rate, terminal growth) — small changes
to those inputs swing the output a lot. Treat the base case as a starting
point to argue with, not a final answer. Use --growth / --discount-rate /
--terminal-growth to test how sensitive the valuation is to your
assumptions (that sensitivity IS the point of the exercise).

USAGE
-----
    python3 dcf.py NVDA
    python3 dcf.py NVDA --growth 0.15 --years 5 --terminal-growth 0.03 --discount-rate 0.10
    python3 dcf.py NVDA --sensitivity      # prints a grid across growth x discount rate
    python3 dcf.py NVDA --flat-growth      # disable the growth taper, use flat growth instead

MODEL IMPROVEMENTS (v2)
------------------------
- Growth now TAPERS from your starting growth rate down to the terminal
  growth rate over the projection window, instead of running flat for N
  years and then falling off a cliff into the terminal value. This is
  closer to how growth actually decays for real companies. Use
  --flat-growth to go back to the old straight-line behavior.
- Base FCF is now a blend: it prints BOTH the most recent year and the
  trailing average (--base-years, default 3), and uses the average as the
  base (less noisy, less exposed to one weird year) unless you pass
  --use-latest-fcf.
- Pulls Wall Street's forward growth estimate (when available) alongside
  your historical CAGR, so you can see when the two disagree — a big gap
  is a signal to dig into why before trusting either number.
- Cost of debt is now estimated from actual interest expense / total debt
  when available, instead of a flat guess.
- Flags when the FCF base is small/noisy relative to market cap — this is
  exactly the situation that produces misleadingly extreme DCF outputs
  (small young companies, recent IPOs, cash-flow-negative growth names).
"""

import sys
import argparse
import logging
import statistics

import yfinance as yf

logging.getLogger("yfinance").setLevel(logging.CRITICAL)

DEFAULT_YEARS = 5
DEFAULT_BASE_YEARS = 3            # number of trailing years averaged for the base FCF
DEFAULT_TERMINAL_GROWTH = 0.025   # long-run GDP-ish growth assumption
EQUITY_RISK_PREMIUM = 0.05        # standard long-run assumption
DEFAULT_RISK_FREE_RATE = 0.045    # ~ current 10yr treasury, edit if needed
DEFAULT_COST_OF_DEBT = 0.05       # fallback if we can't derive it from financials
FALLBACK_GROWTH = 0.08            # used if historical FCF growth can't be computed
FALLBACK_BETA = 1.2               # used if beta isn't available
THIN_BASE_THRESHOLD = 0.015       # if base FCF / market cap < 1.5%, flag as a noisy/thin base


# --------------------------------------------------------------------------
# Data fetching
# --------------------------------------------------------------------------

def get_spot(tk: yf.Ticker) -> float | None:
    try:
        price = tk.fast_info["last_price"]
        if price:
            return float(price)
    except Exception:
        pass
    try:
        hist = tk.history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return None


def get_free_cash_flows(tk: yf.Ticker) -> list[float]:
    """Historical FCF = Operating Cash Flow - CapEx, oldest to newest."""
    cf = tk.cashflow
    if cf is None or cf.empty:
        return []

    def find_row(names):
        for name in names:
            if name in cf.index:
                return cf.loc[name]
        return None

    ocf = find_row(["Total Cash From Operating Activities", "Operating Cash Flow", "Cash Flow From Continuing Operating Activities"])
    capex = find_row(["Capital Expenditure", "Capital Expenditures", "Purchase Of PPE"])

    if ocf is None or capex is None:
        return []

    fcf = (ocf - capex.abs()).dropna()
    fcf = fcf.sort_index()  # oldest -> newest
    return [float(x) for x in fcf.tolist()]


def _safe_float(x) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if x != x else x  # NaN != NaN


def _real_debt_from_statement(stmt, col) -> float | None:
    """Long Term Debt + Current Debt, excluding lease liabilities. Falls back to
    (Current Debt And Capital Lease Obligation - Current Capital Lease Obligation)
    for the current-debt piece when 'Current Debt' itself is missing/NaN, which
    happens on some tickers/periods even though the split rows are present."""
    lt = _safe_float(stmt.loc["Long Term Debt", col]) if "Long Term Debt" in stmt.index else None
    cur = _safe_float(stmt.loc["Current Debt", col]) if "Current Debt" in stmt.index else None
    if cur is None and {"Current Debt And Capital Lease Obligation", "Current Capital Lease Obligation"} <= set(stmt.index):
        total_cur = _safe_float(stmt.loc["Current Debt And Capital Lease Obligation", col])
        cur_lease = _safe_float(stmt.loc["Current Capital Lease Obligation", col])
        if total_cur is not None and cur_lease is not None:
            cur = total_cur - cur_lease
    if lt is None and cur is None:
        return None
    return (lt or 0.0) + (cur or 0.0)


def get_real_debt(tk: yf.Ticker, reported_total_debt: float) -> tuple[float, str]:
    """.info's totalDebt often folds in capital/operating lease liabilities
    (ASC 842 puts these on the balance sheet as debt-like line items), which
    inflates leverage for any lease-heavy business — retailers, restaurants,
    airlines, REITs. That's normal store/aircraft/property lease obligations,
    not money owed to bondholders, and shouldn't be netted against enterprise
    value the same way. Prefers real interest-bearing debt (Long Term Debt +
    Current Debt) from the most recent QUARTERLY balance sheet — .info's
    totalDebt itself is quarterly-sourced, and falling back to the annual
    statement instead can pull a stale figure (e.g. a ticker that paid down
    debt mid-year would show its old, higher balance). Falls back to the
    annual balance sheet, then to .info's totalDebt (flagged as such), if
    fresher data isn't available."""
    for stmt_attr, source_label in [("quarterly_balance_sheet", "latest quarter"), ("balance_sheet", "latest annual filing")]:
        try:
            stmt = getattr(tk, stmt_attr)
        except Exception:
            stmt = None
        if stmt is None or stmt.empty:
            continue
        total = _real_debt_from_statement(stmt, stmt.columns[0])
        if total is not None:
            return total, f"long-term + current debt from {source_label}'s balance sheet (excludes lease liabilities)"
    return reported_total_debt, "totalDebt from .info (may include lease liabilities for lease-heavy businesses)"


def get_shares_outstanding(tk: yf.Ticker, info: dict) -> float | None:
    """.info's sharesOutstanding is occasionally None even for large, liquid
    tickers (a real yfinance data gap, not specific to any one ticker) —
    fast_info carries it in that case."""
    shares = info.get("sharesOutstanding")
    if shares:
        return float(shares)
    try:
        fi_shares = tk.fast_info.get("shares")
        if fi_shares:
            return float(fi_shares)
    except Exception:
        pass
    return None


def get_market_cap(tk: yf.Ticker, info: dict) -> float | None:
    mc = info.get("marketCap")
    if mc:
        return float(mc)
    try:
        fi_mc = tk.fast_info.get("market_cap")
        if fi_mc:
            return float(fi_mc)
    except Exception:
        pass
    return None


def get_balance_sheet_items(tk: yf.Ticker) -> dict:
    try:
        info = tk.info
    except Exception:
        info = {}

    reported_total_debt = info.get("totalDebt") or 0
    real_debt, debt_source = get_real_debt(tk, reported_total_debt)

    return {
        "shares_outstanding": get_shares_outstanding(tk, info),
        "total_debt": real_debt,
        "total_debt_reported": reported_total_debt,
        "debt_source": debt_source,
        "cash": info.get("totalCash") or 0,
        "market_cap": get_market_cap(tk, info),
        "beta": info.get("beta"),
        "analyst_growth": info.get("earningsGrowth") or info.get("revenueGrowth"),
    }


def get_interest_expense(tk: yf.Ticker) -> tuple[float, float | None] | tuple[None, None]:
    """Pulls the most recent annual interest expense from the income
    statement, along with real debt AS OF THAT SAME ANNUAL PERIOD (from the
    annual balance sheet, not the fresher quarterly one get_real_debt() uses
    for net-debt/WACC) — cost of debt = interest expense / debt only makes
    sense when both numbers describe the same point in time. Mixing a stale
    annual interest figure with a fresh post-paydown quarterly debt balance
    (or vice versa) produces a distorted rate."""
    try:
        fin = tk.financials
        if fin is None or fin.empty:
            return None, None
        for name in ["Interest Expense", "Interest Expense Non Operating"]:
            if name in fin.index:
                row = fin.loc[name].dropna()
                if not row.empty:
                    period = row.index[0]
                    interest = abs(float(row.iloc[0]))
                    debt_basis = None
                    try:
                        bs_stmt = tk.balance_sheet
                        if bs_stmt is not None and not bs_stmt.empty and period in bs_stmt.columns:
                            debt_basis = _real_debt_from_statement(bs_stmt, period)
                    except Exception:
                        pass
                    return interest, debt_basis
    except Exception:
        pass
    return None, None


# --------------------------------------------------------------------------
# Assumption estimation
# --------------------------------------------------------------------------

def historical_fcf_cagr(fcf_history: list[float]) -> float | None:
    """CAGR from the first to the last positive FCF year, using the actual
    number of periods elapsed between them (not just the count of positive
    entries) — a company with one negative-FCF year in the middle of an
    otherwise-growing history would otherwise get its CAGR computed over too
    few years and silently overstated."""
    positive = [(i, x) for i, x in enumerate(fcf_history) if x > 0]
    if len(positive) < 2:
        return None
    first_i, first = positive[0]
    last_i, last = positive[-1]
    years = last_i - first_i
    if years <= 0 or first <= 0:
        return None
    return (last / first) ** (1 / years) - 1


def estimate_cost_of_debt(interest_expense: float | None, debt_basis: float | None) -> tuple[float, bool]:
    """Returns (cost_of_debt, was_estimated_from_real_data). debt_basis must be
    the debt figure from the SAME period as interest_expense (see
    get_interest_expense) — not bs['total_debt'], which may be a fresher
    quarterly figure."""
    if interest_expense and debt_basis and debt_basis > 0:
        rate = interest_expense / debt_basis
        # sanity clamp — bad data can occasionally produce absurd ratios
        if 0.005 <= rate <= 0.20:
            return rate, True
    return DEFAULT_COST_OF_DEBT, False


def estimate_wacc(bs: dict, risk_free: float, cost_of_debt: float) -> float:
    beta = bs.get("beta") or FALLBACK_BETA
    cost_of_equity = risk_free + beta * EQUITY_RISK_PREMIUM

    market_cap = bs.get("market_cap") or 0
    total_debt = bs.get("total_debt") or 0
    total_capital = market_cap + total_debt

    if total_capital <= 0:
        return cost_of_equity  # no debt data, assume all-equity

    weight_equity = market_cap / total_capital
    weight_debt = total_debt / total_capital
    after_tax_cost_of_debt = cost_of_debt * (1 - 0.21)  # ~ US corp tax rate assumption

    return weight_equity * cost_of_equity + weight_debt * after_tax_cost_of_debt


# --------------------------------------------------------------------------
# DCF math
# --------------------------------------------------------------------------

def project_fcf_flat(base_fcf: float, growth: float, years: int) -> list[float]:
    """Old behavior: same growth rate every year, then a hard drop to terminal growth."""
    return [base_fcf * (1 + growth) ** i for i in range(1, years + 1)]


def project_fcf_tapered(base_fcf: float, start_growth: float, terminal_growth: float, years: int) -> list[float]:
    """Growth rate steps down linearly from start_growth (year 1) to
    terminal_growth (final year), so the transition into the terminal
    value isn't a cliff. This is closer to how growth actually decays
    for real companies as they scale and face tougher comps."""
    if years <= 1:
        return [base_fcf * (1 + start_growth)]
    projected = []
    prev = base_fcf
    for i in range(years):
        # linear interpolation of the growth rate used THIS year
        weight = i / (years - 1)
        g = start_growth + (terminal_growth - start_growth) * weight
        prev = prev * (1 + g)
        projected.append(prev)
    return projected


def discount_cash_flows(cash_flows: list[float], rate: float) -> list[float]:
    return [cf / (1 + rate) ** (i + 1) for i, cf in enumerate(cash_flows)]


def terminal_value(final_fcf: float, terminal_growth: float, discount_rate: float) -> float:
    if discount_rate <= terminal_growth:
        raise ValueError("Discount rate must exceed terminal growth rate.")
    return final_fcf * (1 + terminal_growth) / (discount_rate - terminal_growth)


def run_dcf(base_fcf: float, growth: float, years: int, discount_rate: float,
            terminal_growth: float, taper: bool = True):
    if taper:
        projected = project_fcf_tapered(base_fcf, growth, terminal_growth, years)
    else:
        projected = project_fcf_flat(base_fcf, growth, years)
    discounted = discount_cash_flows(projected, discount_rate)
    tv = terminal_value(projected[-1], terminal_growth, discount_rate)
    discounted_tv = tv / (1 + discount_rate) ** years
    enterprise_value = sum(discounted) + discounted_tv
    return {
        "projected_fcf": projected,
        "discounted_fcf": discounted,
        "terminal_value": tv,
        "discounted_terminal_value": discounted_tv,
        "enterprise_value": enterprise_value,
    }


# --------------------------------------------------------------------------
# Display
# --------------------------------------------------------------------------

def money(x) -> str:
    if x is None:
        return "N/A"
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.0f}"


def money_ps(x) -> str:
    if x is None:
        return "N/A"
    return f"${x:,.2f}"


def compute_dcf(ticker: str, growth: float | None = None, years: int = DEFAULT_YEARS,
                 discount_rate: float | None = None, terminal_growth: float = DEFAULT_TERMINAL_GROWTH,
                 use_latest_fcf: bool = False, flat_growth: bool = False,
                 base_years: int = DEFAULT_BASE_YEARS) -> dict:
    """Fetches + computes everything needed for a DCF valuation and returns
    it as plain data (not printed text), so callers other than the CLI
    (e.g. valuation_model.py) can reuse the model without scraping printed
    output — same decoupling pattern as outcomes.py's compute_outcomes()."""
    tk = yf.Ticker(ticker)
    spot = get_spot(tk)

    if spot is None:
        return {"ticker": ticker.upper(), "spot": None,
                "error": f"Couldn't find ticker '{ticker}'. Double check the symbol."}

    fcf_history = get_free_cash_flows(tk)
    bs = get_balance_sheet_items(tk)
    interest_expense, interest_debt_basis = get_interest_expense(tk)

    if not bs.get("market_cap") and bs.get("shares_outstanding"):
        bs["market_cap"] = bs["shares_outstanding"] * spot  # last-resort fallback when info/fast_info both lack it

    data = {
        "ticker": ticker.upper(), "spot": spot, "fcf_history": fcf_history, "bs": bs,
        "years": years, "terminal_growth": terminal_growth, "flat_growth": flat_growth, "error": None,
    }

    if not fcf_history:
        data["error"] = "Couldn't pull free cash flow history for this ticker — cannot build a DCF."
        return data

    latest_fcf = fcf_history[-1]
    recent_years = fcf_history[-base_years:] if len(fcf_history) >= base_years else fcf_history
    avg_fcf = statistics.mean(recent_years)
    base_fcf = latest_fcf if use_latest_fcf else avg_fcf
    base_label = "most recent year" if use_latest_fcf else f"{len(recent_years)}-yr average"

    market_cap = bs.get("market_cap")
    base_fcf_ratio = base_fcf / market_cap if market_cap and base_fcf > 0 else None

    auto_growth = historical_fcf_cagr(fcf_history)
    analyst_growth = bs.get("analyst_growth")
    used_growth = growth if growth is not None else (auto_growth if auto_growth else FALLBACK_GROWTH)

    cost_of_debt, cod_from_real_data = estimate_cost_of_debt(interest_expense, interest_debt_basis)
    wacc = estimate_wacc(bs, DEFAULT_RISK_FREE_RATE, cost_of_debt)
    used_discount = discount_rate if discount_rate is not None else wacc

    data.update({
        "latest_fcf": latest_fcf, "avg_fcf": avg_fcf, "avg_fcf_years": len(recent_years),
        "base_fcf": base_fcf, "base_label": base_label,
        "base_fcf_ratio": base_fcf_ratio,
        "thin_base": base_fcf_ratio is not None and base_fcf_ratio < THIN_BASE_THRESHOLD,
        "nonpositive_base": base_fcf <= 0,
        "auto_growth": auto_growth, "analyst_growth": analyst_growth,
        "used_growth": used_growth, "growth_from_arg": growth is not None,
        "cost_of_debt": cost_of_debt, "cod_from_real_data": cod_from_real_data,
        "wacc": wacc, "used_discount": used_discount, "discount_from_arg": discount_rate is not None,
    })

    if used_discount <= terminal_growth:
        data["error"] = "Discount rate must be greater than terminal growth rate — adjust --discount-rate or --terminal-growth."
        return data

    result = run_dcf(base_fcf, used_growth, years, used_discount, terminal_growth, taper=not flat_growth)
    enterprise_value = result["enterprise_value"]
    net_debt = (bs.get("total_debt") or 0) - (bs.get("cash") or 0)
    equity_value = enterprise_value - net_debt
    shares = bs.get("shares_outstanding")

    data.update({"result": result, "enterprise_value": enterprise_value, "net_debt": net_debt,
                 "equity_value": equity_value, "shares": shares})

    if shares:
        implied_price = equity_value / shares
        data["implied_price"] = implied_price
        data["upside"] = (implied_price - spot) / spot
    else:
        data["implied_price"] = None
        data["upside"] = None

    return data


def print_report(ticker: str, growth: float | None, years: int, discount_rate: float | None,
                  terminal_growth: float, run_sensitivity: bool, use_latest_fcf: bool, flat_growth: bool,
                  base_years: int = DEFAULT_BASE_YEARS):
    data = compute_dcf(ticker, growth, years, discount_rate, terminal_growth,
                        use_latest_fcf, flat_growth, base_years)

    if data["spot"] is None:
        print(data["error"])
        sys.exit(1)

    print("=" * 78)
    print(f"{data['ticker']} — DCF VALUATION".center(78))
    print("=" * 78)
    print(f"Current price: {money_ps(data['spot'])}")

    if not data["fcf_history"]:
        print(f"\n{data['error']}")
        return

    print(f"Most recent annual FCF:  {money(data['latest_fcf'])}")
    print(f"{data['avg_fcf_years']}-yr average FCF:      {money(data['avg_fcf'])}")
    print(f"Using as base ({data['base_label']}): {money(data['base_fcf'])}")
    print(f"FCF history (oldest -> newest): {', '.join(money(x) for x in data['fcf_history'])}")

    # ---- flag a thin/noisy base ----
    if data["thin_base"]:
        print()
        print(f"  ⚠ Base FCF is only {data['base_fcf_ratio']*100:.2f}% of market cap — small/noisy base.")
        print(f"    The model is very sensitive to growth assumptions here. A young or")
        print(f"    fast-scaling company's trailing CAGR can look extreme and won't")
        print(f"    necessarily continue — sanity-check --growth by hand before trusting")
        print(f"    the output.")
    elif data["nonpositive_base"]:
        print()
        print(f"  ⚠ Base FCF is negative or zero — this DCF will not produce a meaningful")
        print(f"    result for a company that isn't yet cash-flow positive. Consider a")
        print(f"    revenue-multiple or forward-looking approach instead.")

    # ---- assumptions ----
    bs = data["bs"]
    auto_growth = data["auto_growth"]
    analyst_growth = data["analyst_growth"]
    used_growth = data["used_growth"]
    used_discount = data["used_discount"]

    print()
    print("ASSUMPTIONS")
    print("-" * 78)
    print(f"  Projection years:         {years}")
    print(f"  FCF growth rate used:     {used_growth*100:.1f}%  "
          f"{'(from --growth)' if data['growth_from_arg'] else f'(historical CAGR: {auto_growth*100:.1f}%)' if auto_growth else '(fallback default, insufficient history)'}")
    if analyst_growth is not None:
        gap_flag = "  <- differs meaningfully from historical CAGR, worth digging into why" \
            if auto_growth and abs(analyst_growth - auto_growth) > 0.10 else ""
        print(f"  Analyst forward estimate: {analyst_growth*100:.1f}%  (cross-check only, not used automatically){gap_flag}")
    print(f"  Growth taper to terminal: {'ON — growth glides down to terminal rate by year ' + str(years) if not flat_growth else 'OFF (--flat-growth): flat rate then a cliff to terminal'}")
    print(f"  Discount rate used:       {used_discount*100:.1f}%  "
          f"{'(from --discount-rate)' if data['discount_from_arg'] else '(estimated WACC)'}")
    print(f"  Cost of debt:             {data['cost_of_debt']*100:.1f}%  "
          f"{'(from actual interest expense / debt)' if data['cod_from_real_data'] else '(fallback default, no interest expense data)'}")
    print(f"  Terminal growth rate:     {terminal_growth*100:.1f}%")
    if bs.get("beta"):
        print(f"  Beta (for WACC):          {bs['beta']:.2f}")
    print(f"  Debt used (net debt/WACC): {money(bs['total_debt'])}  ({bs['debt_source']})")
    if bs.get("total_debt_reported") and abs(bs["total_debt_reported"] - bs["total_debt"]) > max(bs["total_debt"] * 0.05, 1e6):
        print(f"  .info reported totalDebt: {money(bs['total_debt_reported'])}  "
              f"(differs from debt used above — likely includes lease liabilities)")

    if data["error"]:
        print(f"\n{data['error']}")
        return

    # ---- base case ----
    result = data["result"]

    print()
    print("PROJECTED FREE CASH FLOWS")
    print("-" * 78)
    print(f"{'Year':>6}{'Projected FCF':>22}{'Discounted (PV)':>22}")
    for i, (proj, disc) in enumerate(zip(result["projected_fcf"], result["discounted_fcf"]), start=1):
        print(f"{i:>6}{money(proj):>22}{money(disc):>22}")

    print()
    print(f"  Terminal value (undiscounted):   {money(result['terminal_value'])}")
    print(f"  Terminal value (discounted, PV): {money(result['discounted_terminal_value'])}")
    print(f"  Sum of discounted FCF:           {money(sum(result['discounted_fcf']))}")

    enterprise_value = data["enterprise_value"]
    net_debt = data["net_debt"]
    equity_value = data["equity_value"]
    shares = data["shares"]

    print()
    print("VALUATION SUMMARY")
    print("-" * 78)
    print(f"  Enterprise value:       {money(enterprise_value)}")
    print(f"  Less: net debt:         {money(net_debt)}")
    print(f"  Implied equity value:   {money(equity_value)}")

    if shares:
        implied_price = data["implied_price"]
        upside = data["upside"]
        print(f"  Shares outstanding:     {shares:,.0f}")
        print()
        print("=" * 78)
        print(f"  IMPLIED FAIR VALUE:     {money_ps(implied_price)}")
        print(f"  CURRENT PRICE:          {money_ps(data['spot'])}")
        direction = "UPSIDE" if upside >= 0 else "DOWNSIDE"
        print(f"  IMPLIED {direction}:{'':<8}{upside*100:+.1f}%")
        print("=" * 78)
    else:
        print("\n  Shares outstanding not available — can't compute per-share value.")

    # ---- sensitivity table ----
    if run_sensitivity and shares:
        print()
        print("SENSITIVITY: IMPLIED FAIR VALUE PER SHARE")
        print("(rows = growth rate, columns = discount rate)")
        print("-" * 78)
        growth_range = [used_growth - 0.04, used_growth - 0.02, used_growth, used_growth + 0.02, used_growth + 0.04]
        discount_range = [used_discount - 0.02, used_discount - 0.01, used_discount, used_discount + 0.01, used_discount + 0.02]

        header = "Growth \\ Disc".rjust(15) + "".join(f"{d*100:>10.1f}%" for d in discount_range)
        print(header)
        for g in growth_range:
            row = f"{g*100:>14.1f}%"
            for d in discount_range:
                if d <= terminal_growth:
                    row += f"{'N/A':>11}"
                    continue
                try:
                    r = run_dcf(data["base_fcf"], g, years, d, terminal_growth, taper=not flat_growth)
                    ev = r["enterprise_value"]
                    eq = ev - net_debt
                    price = eq / shares
                    row += f"{money_ps(price):>11}"
                except Exception:
                    row += f"{'N/A':>11}"
            print(row)

    print()
    if flat_growth:
        print("Note: this run used FLAT growth (--flat-growth) — same rate every year,")
        print("then a hard drop to terminal growth. Real models usually taper growth")
        print("down gradually instead; that's the tool's default (omit --flat-growth).")
    else:
        print("Note: growth tapers toward the terminal rate over the projection")
        print("window. Real models often vary growth/margins year by year with more")
        print("nuance. Treat this as a starting framework to argue with, not a")
        print("finished model.")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build a DCF valuation for a ticker.")
    parser.add_argument("ticker", help="Ticker symbol, e.g. NVDA")
    parser.add_argument("--growth", type=float, default=None, help="Annual FCF growth rate, e.g. 0.15 for 15%%. Default: historical CAGR.")
    parser.add_argument("--years", type=int, default=DEFAULT_YEARS, help="Number of projection years. Default: 5.")
    parser.add_argument("--discount-rate", type=float, default=None, help="Discount rate (WACC), e.g. 0.10 for 10%%. Default: estimated WACC.")
    parser.add_argument("--terminal-growth", type=float, default=DEFAULT_TERMINAL_GROWTH, help="Terminal growth rate. Default: 2.5%%.")
    parser.add_argument("--sensitivity", action="store_true", help="Print a growth x discount rate sensitivity grid.")
    parser.add_argument("--use-latest-fcf", action="store_true", help="Use most recent year FCF as base instead of the base-years average.")
    parser.add_argument("--flat-growth", action="store_true", help="Disable growth taper; use flat growth then a hard drop to terminal (old behavior).")
    parser.add_argument("--base-years", type=int, default=DEFAULT_BASE_YEARS,
                         help=f"Number of trailing years to average for the base FCF. Default: {DEFAULT_BASE_YEARS}. "
                              "Capped automatically to however many years of history are actually available.")
    args = parser.parse_args()

    if args.base_years < 1:
        print("--base-years must be at least 1.")
        sys.exit(1)

    print_report(
        args.ticker,
        args.growth,
        args.years,
        args.discount_rate,
        args.terminal_growth,
        args.sensitivity,
        args.use_latest_fcf,
        args.flat_growth,
        args.base_years,
    )


if __name__ == "__main__":
    main()
