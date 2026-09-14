"""
Factor regressions.

    r_it - rf_t = alpha_i + sum_k beta_ik f_kt + e_it

Implemented from first principles with numpy so the arithmetic is visible and
testable: OLS point estimates, Newey-West (1987) HAC standard errors with the
Bartlett kernel, model comparison, rolling regressions, return attribution
and residual diagnostics.

Why HAC errors
--------------
Monthly fund returns are mildly autocorrelated and strongly heteroskedastic
(volatility clusters). Plain OLS standard errors understate the uncertainty
of alpha, sometimes by a lot. Newey-West corrects both problems and is the
standard in the empirical asset-pricing literature.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .config import ANALYSIS, MODELS, AnalysisConfig

PERIODS = {"M": 12, "D": 252, "W": 52}


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #
def to_monthly(daily_returns: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """
    Compound daily simple returns to calendar-month returns (month-end index).
    Months with no observations stay NaN (min_count=1), otherwise pandas would
    report an empty product as a 0% return for assets not yet listed.
    """
    return (1 + daily_returns).resample("ME").prod(min_count=1) - 1


def align(asset_returns: pd.DataFrame, factors: pd.DataFrame, frequency: str = "M") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Put asset and factor returns on the same calendar. Daily asset returns are
    compounded to monthly when frequency == "M". Returns (excess_asset_returns, factors).
    """
    r = to_monthly(asset_returns) if frequency.upper() == "M" else asset_returns.copy()
    if frequency.upper() == "M":
        r.index = r.index + pd.offsets.MonthEnd(0)
    idx = r.index.intersection(factors.index)
    r, f = r.loc[idx], factors.loc[idx]
    excess = r.sub(f["RF"], axis=0)
    return excess, f


# --------------------------------------------------------------------------- #
# OLS + Newey-West
# --------------------------------------------------------------------------- #
@dataclass
class RegressionResult:
    asset: str
    model: str
    factors: List[str]
    n_obs: int
    frequency: str
    alpha: float                      # per period
    betas: pd.Series
    se_ols: pd.Series                 # incl. 'alpha'
    se_hac: pd.Series                 # incl. 'alpha'
    hac_lags: int
    r2: float
    adj_r2: float
    resid: pd.Series
    fitted: pd.Series
    X: pd.DataFrame
    y: pd.Series

    @property
    def periods(self) -> int:
        return PERIODS[self.frequency]

    @property
    def alpha_ann(self) -> float:
        return self.alpha * self.periods

    @property
    def t_alpha_hac(self) -> float:
        return self.alpha / self.se_hac["alpha"]

    @property
    def t_alpha_ols(self) -> float:
        return self.alpha / self.se_ols["alpha"]

    @property
    def p_alpha_hac(self) -> float:
        return 2 * (1 - stats.t.cdf(abs(self.t_alpha_hac), df=self.n_obs - len(self.factors) - 1))

    @property
    def t_betas_hac(self) -> pd.Series:
        return self.betas / self.se_hac[self.factors]

    @property
    def resid_vol_ann(self) -> float:
        return float(self.resid.std(ddof=len(self.factors) + 1) * np.sqrt(self.periods))

    @property
    def information_ratio(self) -> float:
        """Alpha per unit of idiosyncratic risk (appraisal ratio)."""
        return self.alpha_ann / self.resid_vol_ann if self.resid_vol_ann > 0 else np.nan

    @property
    def aic(self) -> float:
        k = len(self.factors) + 1
        rss = float((self.resid ** 2).sum())
        return self.n_obs * np.log(rss / self.n_obs) + 2 * k

    @property
    def bic(self) -> float:
        k = len(self.factors) + 1
        rss = float((self.resid ** 2).sum())
        return self.n_obs * np.log(rss / self.n_obs) + k * np.log(self.n_obs)

    def summary_row(self) -> Dict[str, float]:
        row = {"n_obs": self.n_obs, "alpha_ann": self.alpha_ann, "t_alpha_ols": self.t_alpha_ols,
               "t_alpha_hac": self.t_alpha_hac, "p_alpha_hac": self.p_alpha_hac,
               "r2": self.r2, "adj_r2": self.adj_r2, "resid_vol_ann": self.resid_vol_ann,
               "info_ratio": self.information_ratio, "aic": self.aic, "bic": self.bic}
        for f in self.factors:
            row[f"beta_{f}"] = self.betas[f]
            row[f"t_{f}"] = self.t_betas_hac[f]
        return row


def newey_west_lags(n: int) -> int:
    """Newey-West (1994) rule of thumb: floor(4 (T/100)^(2/9))."""
    return int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))


def _hac_cov(X: np.ndarray, e: np.ndarray, lags: int) -> np.ndarray:
    """Newey-West covariance of the OLS coefficients with a Bartlett kernel."""
    n, k = X.shape
    xe = X * e[:, None]
    S = xe.T @ xe
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        G = xe[l:].T @ xe[:-l]
        S += w * (G + G.T)
    XtX_inv = np.linalg.inv(X.T @ X)
    return XtX_inv @ S @ XtX_inv


def regress(y: pd.Series, X: pd.DataFrame, asset: str = "", model: str = "",
            frequency: str = "M", hac_lags: Optional[int] = None,
            min_obs: int = ANALYSIS.min_obs) -> RegressionResult:
    """OLS of excess returns on factors with an intercept; OLS and HAC standard errors."""
    df = pd.concat([y.rename("y"), X], axis=1).dropna()
    if len(df) < min_obs:
        raise ValueError(f"{asset}/{model}: only {len(df)} observations (< {min_obs})")
    yv = df["y"].values.astype(float)
    Xm = np.column_stack([np.ones(len(df)), df[X.columns].values.astype(float)])
    n, k = Xm.shape
    beta, *_ = np.linalg.lstsq(Xm, yv, rcond=None)
    fitted = Xm @ beta
    e = yv - fitted
    rss = float(e @ e)
    tss = float(((yv - yv.mean()) ** 2).sum())
    r2 = 1 - rss / tss if tss > 0 else np.nan
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k)
    sigma2 = rss / (n - k)
    cov_ols = sigma2 * np.linalg.inv(Xm.T @ Xm)
    lags = newey_west_lags(n) if hac_lags is None else int(hac_lags)
    cov_hac = _hac_cov(Xm, e, lags)
    names = ["alpha"] + list(X.columns)
    return RegressionResult(
        asset=asset, model=model, factors=list(X.columns), n_obs=n, frequency=frequency,
        alpha=float(beta[0]), betas=pd.Series(beta[1:], index=list(X.columns)),
        se_ols=pd.Series(np.sqrt(np.diag(cov_ols)), index=names),
        se_hac=pd.Series(np.sqrt(np.diag(cov_hac)), index=names), hac_lags=lags,
        r2=float(r2), adj_r2=float(adj_r2),
        resid=pd.Series(e, index=df.index), fitted=pd.Series(fitted, index=df.index),
        X=df[X.columns], y=df["y"])


# --------------------------------------------------------------------------- #
# Model comparison for one asset
# --------------------------------------------------------------------------- #
def fit_models(excess: pd.Series, factors: pd.DataFrame, asset: str,
               models: Dict[str, List[str]] = MODELS, cfg: AnalysisConfig = ANALYSIS
               ) -> Dict[str, RegressionResult]:
    out = {}
    for name, cols in models.items():
        missing = [c for c in cols if c not in factors.columns]
        if missing:
            continue
        out[name] = regress(excess, factors[cols], asset, name, cfg.frequency, cfg.hac_lags, cfg.min_obs)
    return out


def comparison_table(results: Dict[str, RegressionResult]) -> pd.DataFrame:
    return pd.DataFrame({m: r.summary_row() for m, r in results.items()}).T


def coefficient_table(res: RegressionResult, confidence: float = 0.95) -> pd.DataFrame:
    """Point estimates with OLS and HAC standard errors, t-stats and HAC confidence bands."""
    z = stats.t.ppf(0.5 + confidence / 2, df=res.n_obs - len(res.factors) - 1)
    est = pd.concat([pd.Series({"alpha": res.alpha}), res.betas])
    tbl = pd.DataFrame({"estimate": est, "se_ols": res.se_ols, "se_hac": res.se_hac})
    tbl["t_hac"] = tbl["estimate"] / tbl["se_hac"]
    tbl["ci_low"] = tbl["estimate"] - z * tbl["se_hac"]
    tbl["ci_high"] = tbl["estimate"] + z * tbl["se_hac"]
    tbl["hac_vs_ols"] = tbl["se_hac"] / tbl["se_ols"]
    return tbl


# --------------------------------------------------------------------------- #
# Rolling regressions and style drift
# --------------------------------------------------------------------------- #
def rolling_regression(excess: pd.Series, factors: pd.DataFrame, window: int,
                       frequency: str = "M", min_obs: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    """
    Rolling OLS with HAC t-stats. Returns dict of DataFrames: 'coef', 'tstat',
    'r2' (Series). Alpha is per period in 'coef' and annualised in 'alpha_ann'.
    """
    df = pd.concat([excess.rename("y"), factors], axis=1).dropna()
    min_obs = min_obs or window
    coefs, tstats, r2s, idx = [], [], [], []
    for end in range(window, len(df) + 1):
        seg = df.iloc[end - window: end]
        try:
            r = regress(seg["y"], seg[factors.columns], frequency=frequency, min_obs=min_obs)
        except (ValueError, np.linalg.LinAlgError):
            continue
        coefs.append(pd.concat([pd.Series({"alpha": r.alpha}), r.betas]))
        tstats.append(pd.concat([pd.Series({"alpha": r.t_alpha_hac}), r.t_betas_hac]))
        r2s.append(r.r2); idx.append(seg.index[-1])
    coef = pd.DataFrame(coefs, index=idx); coef["alpha_ann"] = coef["alpha"] * PERIODS[frequency]
    return dict(coef=coef, tstat=pd.DataFrame(tstats, index=idx), r2=pd.Series(r2s, index=idx, name="r2"))


def style_drift(rolling_coef: pd.DataFrame, factors: List[str]) -> pd.DataFrame:
    """Dispersion and range of rolling betas; sign flips flag genuine drift."""
    rows = {}
    for f in factors:
        s = rolling_coef[f].dropna()
        rows[f] = dict(mean=s.mean(), std=s.std(), min=s.min(), max=s.max(),
                       range=s.max() - s.min(),
                       sign_flips=int((np.sign(s).diff().abs() > 0).sum()),
                       latest=s.iloc[-1])
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- #
# Attribution
# --------------------------------------------------------------------------- #
def attribution(res: RegressionResult) -> Dict[str, object]:
    """
    Decompose average annualised excess return into alpha + beta_k * E[f_k] +
    residual (zero by construction in-sample), and build the cumulative
    contribution paths using additive period returns.
    """
    periods = res.periods
    fmeans = res.X.mean()
    contrib = (res.betas * fmeans) * periods
    total = float(res.y.mean() * periods)
    table = pd.concat([pd.Series({"alpha": res.alpha_ann}), contrib]).to_frame("ann_contribution")
    table["share_of_total"] = table["ann_contribution"] / total if total != 0 else np.nan
    table.loc["total_excess_return"] = [total, 1.0]
    cum = pd.DataFrame({f: (res.betas[f] * res.X[f]).cumsum() for f in res.factors})
    cum["alpha"] = res.alpha * np.arange(1, len(res.y) + 1)
    cum["residual"] = res.resid.cumsum()
    cum["actual"] = res.y.cumsum()
    return dict(table=table, cumulative=cum)


# --------------------------------------------------------------------------- #
# Residual diagnostics
# --------------------------------------------------------------------------- #
def ljung_box(x: np.ndarray, lags: int = 12) -> Tuple[float, float]:
    x = np.asarray(x) - np.mean(x)
    n = len(x)
    acf = np.array([np.sum(x[l:] * x[:-l]) / np.sum(x * x) for l in range(1, lags + 1)])
    q = n * (n + 2) * np.sum(acf ** 2 / (n - np.arange(1, lags + 1)))
    return float(q), float(1 - stats.chi2.cdf(q, lags))


def durbin_watson(e: np.ndarray) -> float:
    e = np.asarray(e)
    return float(np.sum(np.diff(e) ** 2) / np.sum(e ** 2))


def arch_lm(e: np.ndarray, lags: int = 4) -> Tuple[float, float]:
    """Engle's ARCH LM test on squared residuals (n R^2 ~ chi2(lags))."""
    e2 = np.asarray(e) ** 2
    y = e2[lags:]
    X = np.column_stack([np.ones(len(y))] + [e2[lags - l: -l] for l in range(1, lags + 1)])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - np.sum((y - X @ b) ** 2) / np.sum((y - y.mean()) ** 2)
    lm = len(y) * r2
    return float(lm), float(1 - stats.chi2.cdf(lm, lags))


def residual_diagnostics(res: RegressionResult) -> pd.DataFrame:
    e = res.resid.values
    jb, jb_p = stats.jarque_bera(e)
    lb, lb_p = ljung_box(e, 12 if res.frequency == "M" else 20)
    arch, arch_p = arch_lm(e)
    rows = [
        ("Jarque-Bera (normality)", jb, jb_p, "residuals are normal"),
        ("Ljung-Box Q (autocorrelation)", lb, lb_p, "no autocorrelation up to lag 12"),
        ("Durbin-Watson", durbin_watson(e), np.nan, "≈2 means no first-order autocorrelation"),
        ("ARCH LM (heteroskedasticity)", arch, arch_p, "constant residual variance"),
        ("Skew", stats.skew(e), np.nan, ""),
        ("Excess kurtosis", stats.kurtosis(e), np.nan, ""),
    ]
    return pd.DataFrame(rows, columns=["test", "statistic", "p_value", "null hypothesis"]).set_index("test")


# --------------------------------------------------------------------------- #
# Cross-section: many assets, one model
# --------------------------------------------------------------------------- #
def cross_section(excess: pd.DataFrame, factors: pd.DataFrame, model: str,
                  cfg: AnalysisConfig = ANALYSIS) -> pd.DataFrame:
    """One row per asset: alpha, HAC t, betas, R². Adds Benjamini-Hochberg adjusted p-values."""
    cols = MODELS[model]
    rows = {}
    for a in excess.columns:
        try:
            r = regress(excess[a], factors[cols], a, model, cfg.frequency, cfg.hac_lags, cfg.min_obs)
        except ValueError:
            continue
        rows[a] = r.summary_row()
    tbl = pd.DataFrame(rows).T
    if len(tbl):
        p = tbl["p_alpha_hac"].values
        m = len(p)
        order = np.argsort(p)
        adj = np.empty(m)
        prev = 1.0
        for rank, i in enumerate(order[::-1], start=0):
            j = m - rank
            prev = min(prev, p[i] * m / j)
            adj[i] = prev
        tbl["p_alpha_bh"] = adj
    return tbl


def exposures_matrix(cs: pd.DataFrame) -> pd.DataFrame:
    return cs[[c for c in cs.columns if c.startswith("beta_")]].rename(columns=lambda c: c[5:])
