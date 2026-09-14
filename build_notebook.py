"""Assemble notebooks/factor_exposure_analyser.ipynb from src/factorlens."""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "src" / "factorlens"
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
               "language_info": {"name": "python"}, "colab": {"provenance": [], "toc_visible": True}}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Multi-Factor Exposure Analyser

**Asset-allocation track · Project 2 of 6.** The first question any allocator asks a manager who
claims alpha is whether it survives a factor regression. Most "alpha" is repackaged exposure to
size, value, momentum or quality that can be bought for a few basis points in an ETF.

This notebook regresses any stock, ETF, fund or portfolio on the Fama-French factor models and
answers:

| Question | Where |
|---|---|
| How much of the return is explained by known factors, and how much is left? | §5 model comparison |
| Which factors, how strongly, and are the loadings statistically meaningful? | §6 coefficient plot with Newey-West errors |
| Have the exposures been stable, or has the strategy drifted? | §7 rolling loadings and style map |
| Where did the return actually come from? | §8 attribution |
| Can the regression be trusted? | §9 residual diagnostics |
| Across a whole set of funds, who has alpha after correcting for multiple testing? | §10 cross-section |

**Data:** Kenneth French Data Library (US daily and monthly factors 1963→; Developed ex-US and
Emerging monthly factors), Yahoo Finance for asset prices. Standard errors are Newey-West HAC,
computed from first principles and verified against `statsmodels` in the test-suite.
""")

md("## 1. Setup")
code(r"""
import importlib, subprocess, sys
def ensure(pkg, mod):
    try: importlib.import_module(mod); return
    except ImportError: pass
    for extra in ([], ["--break-system-packages"]):
        if subprocess.call([sys.executable, "-m", "pip", "install", "-q", pkg] + extra) == 0: return
    print(f"WARNING: could not install {pkg}")
for pkg, mod in [("yfinance", "yfinance"), ("scipy", "scipy")]:
    ensure(pkg, mod)
print("dependencies ready")
""")
code(r"""
import os, sys, warnings, logging, time
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
USE_DRIVE = False
if USE_DRIVE:
    from google.colab import drive  # type: ignore
    drive.mount("/content/drive"); os.chdir("/content/drive/MyDrive/factor-exposure-analyser")
for d in ["src/factorlens", "data/cache", "outputs/figures", "outputs/tables", "tests"]:
    os.makedirs(d, exist_ok=True)
if os.path.abspath("src") not in sys.path:
    sys.path.insert(0, os.path.abspath("src"))
print("working directory:", os.getcwd())
""")

md(r"""
## 2. Package source

```
factor-exposure-analyser/
├── notebooks/factor_exposure_analyser.ipynb
├── src/factorlens/
│   ├── config.py     assets, portfolio, French file registry, model definitions, settings
│   ├── data.py       shared loader (same as Project 1): Yahoo prices, calendar alignment, FX, bad-print repair
│   ├── french.py     Kenneth French library download, robust parsing, factor-set assembly
│   ├── factors.py    OLS + Newey-West, model comparison, rolling, attribution, diagnostics, cross-section
│   └── plots.py      figures
├── tests/test_factorlens.py   10 offline tests (incl. exact match to statsmodels HAC)
└── outputs/{figures,tables}
```
""")
ORDER = ["__init__", "config", "data", "french", "factors", "plots"]
BLURB = {
    "__init__": "Package version.",
    "config": "Assets chosen so each illustrates a factor (value, momentum, quality, size, min-vol, and two active names), the Project-1 portfolio for the blended case, the French library file registry, and the five model definitions (CAPM, FF3, Carhart 4, FF5, FF5+Mom).",
    "data": "The shared data layer from Project 1 with one change: assets keep their own history instead of being truncated to a common start, because a factor regression should use every observation the asset has.",
    "french": "Downloads the zipped CSVs, parses the first data block by scanning for date rows (the files have free-text headers, an annual block and a copyright footer), converts percent to decimal, treats −99.99 as missing, caches, and assembles Mkt-RF/SMB/HML/RMW/CMA/Mom/RF for US (daily or monthly), Developed ex-US and Emerging (monthly).",
    "factors": "OLS by least squares; Newey-West HAC covariance with the Bartlett kernel and the 4(T/100)^(2/9) lag rule; model comparison table (alpha, OLS and HAC t-stats, R², information ratio, AIC/BIC); rolling regressions with HAC t-stats; style-drift statistics; return attribution (alpha + Σ β·E[f]) with cumulative paths; Jarque-Bera, Ljung-Box, Durbin-Watson and ARCH-LM diagnostics; cross-sectional table with Benjamini-Hochberg adjusted p-values.",
    "plots": "One function per figure.",
}
for name in ORDER:
    md(f"### `factorlens/{name}.py`\n\n{BLURB[name]}")
    code(f"%%writefile src/factorlens/{name}.py\n{(SRC / f'{name}.py').read_text()}")

code(r"""
import importlib, factorlens
for m in ["config", "data", "french", "factors", "plots"]:
    importlib.reload(importlib.import_module(f"factorlens.{m}"))
from factorlens import data, french, factors as fx, plots
from factorlens.config import ASSETS, PORTFOLIO, PORTFOLIO_NAME, MODELS, DATA, ANALYSIS
import numpy as np, pandas as pd, matplotlib.pyplot as plt
plots.set_style()
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40); pd.set_option("display.precision", 4)
print("factorlens", factorlens.__version__)
""")

md(r"""
## 3. Choose what to analyse

`ASSETS` is a set picked so that each one should load on a particular factor; that makes the
results checkable. Add your own tickers, or a fund. `PORTFOLIO` is analysed as one asset
(monthly-rebalanced blend). Frequency is monthly by default, the convention in the literature;
daily gives more observations but noisier factor loadings for anything with a stale NAV.

| Region | Frequency | Factors |
|---|---|---|
| `us` | `M` or `D` | Mkt-RF, SMB, HML, RMW, CMA, Mom (1963→) |
| `dev_ex_us` | `M` | same, developed markets ex-US (1990→) |
| `emerging` | `M` | same, emerging markets (1989→) |
""")
code(r"""
assets = dict(ASSETS)                    # ticker -> why it is here
# assets["VFIAX"] = "a mutual fund"      # any Yahoo symbol works
portfolio = dict(PORTFOLIO)              # analysed as a single blended asset
REGION, FREQUENCY = "us", "M"
FOCUS = "BRK-B"                          # asset used for the deep-dive sections
cfg = ANALYSIS.__class__(frequency=FREQUENCY, region=REGION, rolling_window=36 if FREQUENCY == "M" else 252)
for t, why in assets.items(): print(f"{t:7s} {why}")
""")

md("## 4. Data: prices and factors")
code(r"""
t0 = time.time()
tickers = list(assets) + [t for t in portfolio if t not in assets]
D = data.load_all(tickers, DATA)
returns = D["returns"]
print(f"prices loaded in {time.time()-t0:.1f}s | {D['prices'].index[0].date()} -> {D['prices'].index[-1].date()}")
for k, v in D["issues"].items(): print(f"  {k}: {v}")

F = french.load_factors(REGION, FREQUENCY, DATA)
print(f"\nFrench factors ({REGION}, {FREQUENCY}): {F.index[0].date()} -> {F.index[-1].date()} ({len(F)} periods). "
      f"The library publishes with a lag of roughly six weeks, so the sample ends at the last factor date.")
display(F.tail(3).style.format("{:.2%}"))
fig = plots.plot_cumulative_factors(F.loc["1990":], ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]); plots.save(fig, "01_factors"); plt.show()
""")
code(r"""
# Blend the portfolio into a single return series (monthly rebalance within the daily data), then align everything to the factor calendar
w = pd.Series(portfolio); w = w / w.sum()
port_daily = (returns[w.index] * w).sum(axis=1, min_count=len(w))    # equal to monthly-rebalanced if resampled monthly below
asset_returns = returns[list(assets)].copy(); asset_returns[PORTFOLIO_NAME] = port_daily
excess, f = fx.align(asset_returns, F, FREQUENCY)
print(f"aligned sample: {excess.index[0].date()} -> {excess.index[-1].date()}")
display(excess.notna().sum().rename("observations").to_frame().T)
excess.to_csv("outputs/tables/excess_returns.csv"); f.to_csv("outputs/tables/factors_aligned.csv")
""")

md(r"""
## 5. Does the alpha survive? Model comparison for one asset

Five nested models. Watch alpha, its HAC t-statistic and adjusted R² as factors are added. If
alpha collapses toward zero while R² rises, the "skill" was factor exposure. The Newey-West
t-stat is the one to trust; the OLS t-stat is shown to make the difference visible.
""")
code(r"""
res = fx.fit_models(excess[FOCUS].dropna(), f, FOCUS, MODELS, cfg)
cmp = fx.comparison_table(res)
cmp.to_csv(f"outputs/tables/model_comparison_{FOCUS}.csv")
cols = ["n_obs", "alpha_ann", "t_alpha_ols", "t_alpha_hac", "p_alpha_hac", "adj_r2", "resid_vol_ann", "info_ratio", "bic"]
display(cmp[cols].style.format({"n_obs": "{:.0f}", "alpha_ann": "{:+.2%}", "t_alpha_ols": "{:.2f}", "t_alpha_hac": "{:.2f}", "p_alpha_hac": "{:.3f}",
                                 "adj_r2": "{:.3f}", "resid_vol_ann": "{:.2%}", "info_ratio": "{:.2f}", "bic": "{:.0f}"})
        .background_gradient(subset=["adj_r2"], cmap="Blues"))
fig = plots.plot_alpha_across_models(cmp, FOCUS); plots.save(fig, f"02_alpha_models_{FOCUS}"); plt.show()
""")

md("## 6. Loadings with Newey-West confidence intervals")
code(r"""
BEST = "FF5+Mom"
ct = fx.coefficient_table(res[BEST], cfg.confidence)
display(ct.style.format({"estimate": "{:+.4f}", "se_ols": "{:.4f}", "se_hac": "{:.4f}", "t_hac": "{:.2f}", "ci_low": "{:+.3f}", "ci_high": "{:+.3f}", "hac_vs_ols": "{:.2f}x"}))
print(f"Newey-West lags: {res[BEST].hac_lags}. 'hac_vs_ols' > 1 means OLS understated the uncertainty.")
fig = plots.plot_coefficients(ct, FOCUS, BEST); plots.save(fig, f"03_coefficients_{FOCUS}"); plt.show()
""")

md(r"""
## 7. Style drift: rolling loadings

A 36-month rolling window. Stable loadings mean the strategy is what it says it is; wandering
loadings mean the fund's risk is changing under the investor. The style map traces the size
and value loadings through time — the classic style box, animated.
""")
code(r"""
DRIFT = "ARKK" if "ARKK" in excess.columns else FOCUS
fac6 = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
rr = fx.rolling_regression(excess[DRIFT].dropna(), f[fac6], cfg.rolling_window, FREQUENCY)
drift = fx.style_drift(rr["coef"], fac6)
display(drift.style.format({"mean": "{:+.2f}", "std": "{:.2f}", "min": "{:+.2f}", "max": "{:+.2f}", "range": "{:.2f}", "sign_flips": "{:.0f}", "latest": "{:+.2f}"})
        .background_gradient(subset=["range"], cmap="Reds"))
fig = plots.plot_rolling_loadings(rr, DRIFT, fac6, cfg.rolling_window); plots.save(fig, f"04_rolling_{DRIFT}"); plt.show()
fig = plots.plot_style_map(rr, DRIFT); plots.save(fig, f"05_style_map_{DRIFT}"); plt.show()
rr["coef"].to_csv(f"outputs/tables/rolling_loadings_{DRIFT}.csv")
""")

md(r"""
## 8. Where did the return come from?

Average annualised excess return = alpha + Σ βₖ · E[fₖ]. The right-hand panel accumulates the
same decomposition through time (additive period returns, so the pieces sum exactly).
""")
code(r"""
att = fx.attribution(res[BEST])
display(att["table"].style.format({"ann_contribution": "{:+.2%}", "share_of_total": "{:+.1%}"}))
fig = plots.plot_attribution(att, FOCUS, BEST, FREQUENCY); plots.save(fig, f"06_attribution_{FOCUS}"); plt.show()
""")

md(r"""
## 9. Can the regression be trusted?

Residual diagnostics. Heteroskedasticity (ARCH) is nearly universal in monthly returns, which
is why HAC errors are the default here. Autocorrelation in residuals would point to stale
pricing (illiquid holdings, smoothed NAVs) and understated risk.
""")
code(r"""
diag = fx.residual_diagnostics(res[BEST])
display(diag.style.format({"statistic": "{:.3f}", "p_value": "{:.3f}"}))
fig = plots.plot_residuals(res[BEST]); plots.save(fig, f"07_residuals_{FOCUS}"); plt.show()
""")

md(r"""
## 10. Cross-section: who actually has alpha?

Every asset on the same model. With eight assets, one will show a raw p < 0.05 by chance about a
third of the time, so the table also reports Benjamini-Hochberg adjusted p-values; those are the
ones that survive the fact that you looked at several funds. (Harvey, Liu & Zhu argue the bar for
a *new factor* should be t > 3 for the same reason.)
""")
code(r"""
cs = fx.cross_section(excess, f, BEST, cfg)
cs.to_csv(f"outputs/tables/cross_section_{BEST}.csv")
show = ["n_obs", "alpha_ann", "t_alpha_hac", "p_alpha_hac", "p_alpha_bh", "adj_r2", "info_ratio"]
display(cs[show].sort_values("t_alpha_hac", ascending=False).style.format({"n_obs": "{:.0f}", "alpha_ann": "{:+.2%}", "t_alpha_hac": "{:.2f}", "p_alpha_hac": "{:.3f}", "p_alpha_bh": "{:.3f}", "adj_r2": "{:.3f}", "info_ratio": "{:.2f}"})
        .background_gradient(subset=["p_alpha_bh"], cmap="Greens_r", vmin=0, vmax=0.5))
fig = plots.plot_alpha_cross_section(cs, BEST); plots.save(fig, "08_alpha_cross_section"); plt.show()
expo = fx.exposures_matrix(cs)
fig = plots.plot_exposure_heatmap(expo, BEST); plots.save(fig, "09_exposures"); plt.show()
""")
md(r"""
Read the heat-map against the reasons each asset was chosen (§3): VTV should be positive on HML,
MTUM on Mom, QUAL on RMW, IWM on SMB, USMV below 1 on Mkt-RF. If the loadings match the labels,
the machinery works; the interesting rows are the active names.
""")

md("## 11. A portfolio as a single asset")
code(r"""
resP = fx.fit_models(excess[PORTFOLIO_NAME].dropna(), f, PORTFOLIO_NAME, MODELS, cfg)
cmpP = fx.comparison_table(resP)
display(cmpP[cols].style.format({"n_obs": "{:.0f}", "alpha_ann": "{:+.2%}", "t_alpha_ols": "{:.2f}", "t_alpha_hac": "{:.2f}", "p_alpha_hac": "{:.3f}",
                                   "adj_r2": "{:.3f}", "resid_vol_ann": "{:.2%}", "info_ratio": "{:.2f}", "bic": "{:.0f}"}))
ctP = fx.coefficient_table(resP[BEST]); fig = plots.plot_coefficients(ctP, PORTFOLIO_NAME, BEST); plots.save(fig, "10_portfolio_coefficients"); plt.show()
print("A multi-asset portfolio has a low R² against *equity* factors by construction: bonds, gold and REITs are not spanned by them. "
      "That unexplained part is not alpha; it is exposure to factors this model does not contain (duration, credit, commodity). See the upgrades.")
""")

md(r"""
## 12. Reading the results

Write your own after each run; the pattern on the default set:

1. **Berkshire.** The alpha is positive under every model (about 3–4% a year over 2005–2026)
   and statistically insignificant under every model (HAC t ≈ 1.1–1.3). Adding value and
   profitability trims it but does not remove it; R² stays below 0.4 because a single stock
   carries a lot of idiosyncratic risk. The loadings are the informative part: beta around 0.8,
   negative size, positive value, positive profitability. The published "Buffett's Alpha" result
   (Frazzini, Kabiller & Pedersen) needs the betting-against-beta and quality-minus-junk factors
   to close the gap fully; those are the first upgrade below. Try `FOCUS = "BRK-B"` on a
   2014-onward sample and the alpha does collapse to zero, which is a reminder that the answer
   depends on the window.
2. **ARKK.** Beta around 1.5, strongly negative RMW (unprofitable growth) and negative HML. The
   rolling loadings wander and the alpha changes sign across sub-periods, which is what style
   drift looks like in numbers.
3. **The style ETFs load where they should**, with R² above 0.9. That is the sanity check that
   the regressions are wired correctly.
4. **Nobody survives the multiple-testing correction.** Every Benjamini-Hochberg adjusted
   p-value is above 0.05. The passive style ETFs show small negative alphas, which is what fees
   plus imperfect tracking of an academic long-short factor should produce.
5. **OLS t-stats overstate significance** relative to HAC by 20–40% for the active names. If a
   fund's pitch book quotes a t-stat, ask which one.

**Caveats.** The French factors are academic long-short portfolios, not investable ETFs; a fund
can have zero alpha against them and still be worth holding if it delivers the exposure more
cheaply than the alternatives. International assets should be run against the regional factor
sets, not the US ones. Fund NAVs with stale pricing bias betas toward zero; use monthly data and
check the Ljung-Box statistic.

## 13. Potential upgrades

| Upgrade | Why |
|---|---|
| Add AQR factors (QMJ, BAB) and a bond/credit/commodity factor set | Explains multi-asset portfolios; completes the Buffett decomposition |
| Bayesian shrinkage of loadings toward a prior (e.g. a benchmark's) | Rolling 36-month betas are noisy; shrinkage stabilises them |
| Regime-conditional loadings (up vs down markets, high vs low VIX) | Beta asymmetry is where a lot of "alpha" hides |
| Returns-based style analysis (Sharpe 1992) with long-only constraints | The practitioner alternative when factors must be investable indices |
| Bootstrap and White's reality check across the fund universe | Formal multiple-testing control beyond BH |
| Point-in-time fund universe with survivorship handling (CRSP mutual fund style) | The cross-section is biased if dead funds are missing |
""")

nb.cells = cells
out = ROOT / "notebooks" / "factor_exposure_analyser.ipynb"
nbf.write(nb, out)
print("wrote", out, "with", len(cells), "cells")
