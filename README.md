# Valuation Model

A composite stock valuation that combines a DCF, a historical-P/E comps model and the analyst consensus target into one weighted fair value, then checks how closely the three methods agree.

## Sample output

`python3 valuation_model.py MSFT`, run on 2026-10-07 (trimmed):

```
Method                        Fair Value     Upside/Downside      Weight
DCF Fair Value                   $140.87              -73.4%         33%
Comps (Historical P/E)           $579.74               +9.7%         33%
Analyst Target (~12mo)           $587.63              +11.1%         33%

  WEIGHTED COMPOSITE FAIR VALUE: $436.08  (from 3 of 3 methods)
  CURRENT PRICE:                 $528.70
  IMPLIED DOWNSIDE:                -17.5%

METHOD AGREEMENT
  Range:      $140.87 - $587.63
  Spread:     102.4% of the mean estimate
  Assessment: Wide disagreement

COMPS DETAIL
  Current trailing P/E:          29.5x
  4-yr historical average P/E:  32.3x
  By fiscal year:                2023: 34.3x, 2024: 38.0x, 2025: 36.1x, 2026: 20.7x
```

## What it does

- **DCF:** reuses `compute_dcf()` from the included `dcf.py`, which tapers FCF growth to a 2.5% terminal rate and discounts at an estimated WACC.
- **Comps against the stock's own history:** averages the stock's P/E at each fiscal year-end that yfinance reports, usually 4 to 5 years, and multiplies that average by trailing EPS. Loss years and P/Es above 200 are dropped. This is mean reversion to the stock's own multiple, not a peer-group comparison.
- **Analyst consensus:** the mean ~12-month price target (median if no mean), with the target range, number of analysts and rating.
- **Weighted composite and agreement check:** weights are equal by default, or set with `--weights`. A method with no fair value or a non-positive one is dropped and the rest are reweighted. The spread between methods, as a share of their mean, is rated strong (under 8%), moderate (under 20%) or wide disagreement.

## How to run it

Requires Python 3.10+.

```bash
git clone https://github.com/Hmelconian21/valuation-model.git
cd valuation-model
pip install -r requirements.txt

python3 valuation_model.py MSFT
python3 valuation_model.py MSFT --weights 0.5,0.25,0.25   # dcf,comps,analyst
python3 valuation_model.py MSFT --dcf-growth 0.10 --comps-years 8
python3 valuation_model.py --help
```

`dcf.py` (the DCF model) and `outcomes.py` (news helper) are supporting modules.

*Educational tool, not investment advice.*

Part of a set of Python finance tools — see [Stock Research System](https://github.com/Hmelconian21/Stock-Research-System) and the [live risk dashboard](https://henry-risk-dashboard.streamlit.app).
