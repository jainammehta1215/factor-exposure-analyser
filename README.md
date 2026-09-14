# Multi-Factor Exposure Analyser

**Asset-allocation track · Project 2 of 6** · explains *why* assets co-move · pairs with
[Project 1, Risk & Performance Analytics](https://github.com/jainammehta1215/portfolio-risk-analytics)
(where the risk sits) and feeds the covariance thinking in
[Project 3, Portfolio Optimisation](https://github.com/jainammehta1215/portfolio-optimisation)

Regress any stock, ETF, fund or portfolio on the Fama-French factor models and find out how much
of its return is factor exposure you could buy for a few basis points, and how much is left.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jainammehta1215/factor-exposure-analyser/blob/main/notebooks/factor_exposure_analyser.ipynb)
[![tests](https://github.com/jainammehta1215/factor-exposure-analyser/actions/workflows/tests.yml/badge.svg)](https://github.com/jainammehta1215/factor-exposure-analyser/actions)

---

## 1. Project overview

The first question an allocator asks a manager who claims alpha is whether it survives a factor
regression. This project answers it with the standard toolkit of empirical asset pricing:

* five nested models (CAPM, Fama-French 3, Carhart 4, Fama-French 5, FF5 + momentum);
* Newey-West HAC standard errors, implemented from first principles and verified against
  `statsmodels` to machine precision;
* rolling loadings and a style map to detect drift;
* return attribution into alpha and factor contributions;
* residual diagnostics (Jarque-Bera, Ljung-Box, Durbin-Watson, ARCH-LM);
* a cross-sectional table across many funds with Benjamini-Hochberg control for multiple testing.

The default asset set is chosen so each name should load on a known factor (value, momentum,
quality, size, minimum volatility) plus two active names, Berkshire Hathaway and ARK Innovation,
where the interesting questions live.

## 2. Real-world finance use case

* **Manager due diligence** at pensions, endowments and fund-of-funds: this regression is run on
  every candidate manager before an allocation. Alpha that disappears under FF5 is not paid for.
* **Risk management:** factor loadings are the bridge from "what do we own" to "what are we
  exposed to". A book that is long value and short momentum has a risk profile no sector breakdown
  reveals.
* **Product design:** a smart-beta ETF is supposed to deliver a specific factor; the loadings and
  tracking R² show whether it does.
* **Performance attribution** for internal review: how much of last year came from the market,
  from style tilts, and from selection.

## 3. System architecture

```
config.py ─┬─► data.py (Yahoo prices, per-asset history) ──┐
           └─► french.py (Ken French library → factor sets) ─┤
                                                             ▼
                             factors.align() → excess returns on the factor calendar
                                                             │
        ┌──────────────┬──────────────┬──────────────┬───────┴──────┬──────────────┐
        ▼              ▼              ▼              ▼              ▼              ▼
   fit_models     coefficient    rolling_regression attribution  residual_    cross_section
   (5 models)     table + HAC CI + style_drift                   diagnostics  + BH adjustment
        └──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
                                                             ▼
                                              plots.py → notebook figures / CSV tables
```

## 4. Required APIs and data sources

| Source | What | Access |
|---|---|---|
| Kenneth French Data Library | US 3- and 5-factor sets (daily and monthly, 1963→), momentum, Developed ex-US and Emerging 5-factor + momentum (monthly) | Free, direct zip download |
| Yahoo Finance (`yfinance`) | Adjusted prices for any asset; FX crosses if needed | Free, no key |

French publishes with a lag of about six weeks; the sample ends at the last factor date.

## 5. Required Python libraries

`numpy pandas scipy matplotlib yfinance requests` · dev: `pytest statsmodels nbformat nbconvert ipykernel`
(`statsmodels` is used only in a test that cross-checks the HAC implementation.)

## 6. Folder / file structure

```
factor-exposure-analyser/
├── README.md · requirements.txt · pyproject.toml · build_notebook.py
├── notebooks/factor_exposure_analyser.ipynb
├── src/factorlens/
│   ├── config.py     assets, portfolio, French file registry, model definitions, settings
│   ├── data.py       shared loader (Project 1) with per-asset history retained
│   ├── french.py     download, robust parsing, factor-set assembly
│   ├── factors.py    OLS + Newey-West, comparison, rolling, attribution, diagnostics, cross-section
│   └── plots.py      figures
├── tests/test_factorlens.py     10 offline tests
├── .github/workflows/tests.yml
└── outputs/{figures,tables}
```

## 7. Step-by-step build guide

1. French loader first, with a parser that scans for date rows rather than trusting line counts
   (the header length varies between files and over time). Cache everything.
2. Alignment: compound daily asset returns to month-ends with `min_count=1` so unlisted months stay
   NaN rather than becoming 0% returns; subtract French's RF.
3. OLS via least squares; Newey-West covariance with Bartlett weights; the 4(T/100)^(2/9) lag rule.
   Test against `statsmodels` before building anything on top.
4. Model comparison table; coefficient table with HAC intervals.
5. Rolling regressions with HAC t-stats; style-drift statistics (range, sign flips).
6. Attribution as alpha + Σ β·E[f], and cumulative additive paths that sum exactly to the actual.
7. Diagnostics implemented directly (Ljung-Box, Durbin-Watson, ARCH-LM) to keep dependencies light.
8. Cross-section with Benjamini-Hochberg adjusted p-values.
9. Plots, tests, notebook assembled from the modules, executed top to bottom before publishing.

## 8. Data collection pipeline

Prices: shared loader from Project 1 (retry, cache, calendar alignment, bad-print repair, FX), with
one change: each asset keeps its own history instead of being truncated to the common start, so
Berkshire gets 259 monthly observations while ARKK gets 141. Factors: zipped CSV → first data block
→ percent to decimal → −99.99 masked → cached CSV; momentum merged from its own file; regional sets
assembled by name.

## 9. Data cleaning and feature engineering

Excess returns over the one-month T-bill from the same French file (not a separate FRED series,
so the risk-free is exactly the one used to build the factors). Monthly frequency by default; daily
optional for US assets. Rolling window 36 months. Portfolios are blended from constituent returns
and analysed as a single asset.

## 10. Core models / algorithms

| Component | Detail |
|---|---|
| Regression | rᵢₜ − rfₜ = αᵢ + Σₖ βᵢₖ fₖₜ + εᵢₜ, OLS |
| HAC covariance | (X′X)⁻¹ [Σ eₜ² xₜxₜ′ + Σₗ wₗ Σₜ eₜeₜ₋ₗ (xₜxₜ₋ₗ′ + xₜ₋ₗxₜ′)] (X′X)⁻¹, wₗ = 1 − l/(L+1), L = ⌊4(T/100)^{2/9}⌋ |
| Models | CAPM · FF3 · Carhart 4 · FF5 · FF5+Mom |
| Fit statistics | alpha (annualised), OLS and HAC t, p-value, R², adjusted R², residual vol, information ratio (α/σε), AIC, BIC |
| Rolling | window OLS with HAC t-stats; drift = std, range and sign flips of loadings |
| Attribution | ᾱ + Σ βₖ f̄ₖ annualised; cumulative additive paths |
| Diagnostics | Jarque-Bera, Ljung-Box(12), Durbin-Watson, ARCH-LM(4) |
| Multiple testing | Benjamini-Hochberg step-up on HAC p-values across assets |

## 11. Visualisations

Cumulative factor returns · alpha and adjusted R² across nested models with HAC error bars ·
coefficient plot with 95% HAC intervals and t-stats · rolling loadings with significant-alpha
shading and rolling R² · SMB/HML style map coloured by time · attribution bars and cumulative
stack · residual time series, Q-Q and ACF · cross-sectional alpha bars coloured by BH survival ·
exposure heat-map across assets · portfolio-as-asset coefficient plot.

## 12. Results on the default set (monthly, US factors, to July 2026)

| Asset | Obs | α p.a. (FF5+Mom) | HAC t | adj R² | Notable loadings |
|---|---|---|---|---|---|
| BRK-B | 259 | +3.0% | 1.10 | 0.38 | β 0.8, SMB −0.3, HML +0.5, RMW +0.1 |
| ARKK | 141 | +1.7% | 0.39 | 0.78 | β 1.5, HML −0.8, RMW −1.0 |
| VTV | 259 | −0.6% | −0.7 | 0.94 | HML +0.3, CMA +0.2 |
| MTUM | 160 | −0.4% | −0.2 | 0.89 | Mom +0.45 |
| QUAL | 157 | −0.8% | −0.9 | 0.96 | RMW +0.2 |
| IWM | 259 | −1.6% | −2.9 | 0.99 | SMB +0.8 |
| USMV | 178 | −1.3% | −0.8 | 0.79 | β 0.72, RMW +0.3 |

No alpha survives the Benjamini-Hochberg correction. Berkshire's CAPM alpha of 3.7% trims to 3.0%
under FF5+Mom and is never significant; ARKK's rolling alpha swings from +12% (2018–21 windows) to
−15% (2024) with a beta near 1.5 throughout. The style ETFs load where their names say, with R²
above 0.9, which is the check that the machinery is wired correctly.

## 13. Final deliverables

Executed notebook (39 cells, ~2 min in Colab) · `factorlens` package · 10 tests with CI · 10
figures · tables (model comparison, coefficient tables, rolling loadings, attribution, cross-section).

## 14. Potential upgrades

AQR factors (QMJ, BAB) and a bond/credit/commodity set for multi-asset portfolios · Bayesian
shrinkage of rolling loadings · regime-conditional betas (up/down markets, VIX regimes) ·
returns-based style analysis with investable indices (Sharpe 1992) · bootstrap and White's reality
check across the fund universe · point-in-time fund universe with dead funds included.

## 15. Using it on your own fund or portfolio

```python
assets = {"VFIAX": "an index fund", "FCNTX": "an active fund", "MYSTOCK": "a stock"}
portfolio = {"AAPL": 0.3, "MSFT": 0.3, "TLT": 0.4}       # analysed as one blended asset
REGION, FREQUENCY, FOCUS = "us", "M", "FCNTX"
```

For non-US assets use `REGION = "dev_ex_us"` or `"emerging"` (monthly only). Fund NAVs with stale
pricing bias betas toward zero; keep the frequency monthly and read the Ljung-Box statistic.

## Running it

```bash
pip install -r requirements.txt
PYTHONPATH=src pytest -q tests/            # 10 passed
python build_notebook.py
```
Or open the notebook in Colab and Run all.

## Asset-allocation track

Six projects that build one capability each and share a reporting layer. Completed ones are linked.

| # | Project | What it adds |
|---|---|---|
| 1 | [Portfolio Risk & Performance Analytics Dashboard](https://github.com/jainammehta1215/portfolio-risk-analytics) | Where the risk sits: Euler decomposition, benchmark-relative statistics, stress scenarios, HTML dashboard |
| 2 | **Multi-Factor Exposure Analyser (Fama-French)** (this repo) | Why assets co-move: factor betas, alpha after factor adjustment, style drift |
| 3 | [Portfolio Optimisation: Mean-Variance, Black-Litterman and Robust Methods](https://github.com/jainammehta1215/portfolio-optimisation) | What weights to hold: Markowitz and its fixes, tested out of sample net of costs |
| 4 | [Risk Parity and Hierarchical Risk Parity](https://github.com/jainammehta1215/risk-parity-hrp) | Allocating by risk instead of capital; clustering instead of matrix inversion |
| 5 | Macro Nowcasting and Recession Probability | The regime the allocation lives in: yield-curve probit, dynamic factor model, real-time vintages |
| 6 | Yield Curve Construction and Fixed-Income Immunisation | The rates side: bootstrapping, Nelson-Siegel, key-rate durations, liability matching |

A master repository will consolidate all six with a shared core once the track is complete.

## References

Fama & French (1993, 2015) · Carhart (1997) · Newey & West (1987, 1994) · Frazzini, Kabiller &
Pedersen (2018) *Buffett's Alpha* · Harvey, Liu & Zhu (2016) *...and the Cross-Section of Expected
Returns* · Benjamini & Hochberg (1995) · Sharpe (1992) returns-based style analysis.
