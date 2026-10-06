# valuation-model

Composite stock valuation that blends a DCF, a historical-P/E comps model, and analyst consensus into one weighted fair value with a method-agreement check.

## Run

Requires Python 3.10+.

```bash
pip install -r requirements.txt
python3 valuation_model.py NVDA
python3 valuation_model.py NVDA --weights 0.5,0.25,0.25
python3 valuation_model.py --help
```

`dcf.py` (DCF model) and `outcomes.py` (news helper) are included as supporting modules. Educational tool, not investment advice.
