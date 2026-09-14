"""Offline tests. Run: PYTHONPATH=src pytest -q tests/"""
import numpy as np
import pandas as pd
import pytest

from factorlens import factors as fx, french

FRENCH_SAMPLE = """This file was created using the 202607 CRSP database.
Some more header text.

,Mkt-RF,SMB,HML,RMW,CMA,RF
202401,   1.05,  -2.10,   0.30,   0.80,  -0.50,   0.45
202402,   5.00,  -1.00,   2.00,   1.00,   0.50,   0.44
202403,   2.90,   0.60,   4.00,   0.10,   1.20,   0.43

 Annual Factors: January-December
,Mkt-RF,SMB,HML,RMW,CMA,RF
  2024,  19.78,  -13.44,   -8.43,    5.11,  -10.04,    5.26

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""


def test_french_parser_first_block_only():
    df = french.parse_french_csv(FRENCH_SAMPLE)
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    assert len(df) == 3
    assert df.index[0] == pd.Timestamp("2024-01-31")
    assert abs(df.loc["2024-02-29", "Mkt-RF"] - 0.05) < 1e-12


@pytest.fixture(scope="module")
def simulated():
    """Excess returns generated from known betas so the regression can be checked exactly."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2005-01-31", periods=240, freq="ME")
    F = pd.DataFrame(rng.normal(0, 0.04, (240, 4)), index=idx, columns=["Mkt-RF", "SMB", "HML", "Mom"])
    F["RF"] = 0.002
    true = pd.Series({"Mkt-RF": 1.2, "SMB": -0.4, "HML": 0.5, "Mom": 0.1})
    alpha = 0.003
    y = alpha + F[true.index] @ true + rng.normal(0, 0.01, 240)
    return y, F, true, alpha


def test_ols_recovers_known_betas(simulated):
    y, F, true, alpha = simulated
    r = fx.regress(y, F[["Mkt-RF", "SMB", "HML", "Mom"]], "sim", "C4", "M")
    assert np.allclose(r.betas.values, true.values, atol=0.03)
    assert abs(r.alpha - alpha) < 0.003
    assert r.r2 > 0.9
    # HAC with zero lags equals White heteroskedasticity-robust SEs (sandwich), so both must be finite and positive
    assert (r.se_hac > 0).all() and (r.se_ols > 0).all()


def test_hac_matches_statsmodels_when_available(simulated):
    sm = pytest.importorskip("statsmodels.api")
    y, F, *_ = simulated
    r = fx.regress(y, F[["Mkt-RF", "SMB", "HML"]], "sim", "FF3", "M", hac_lags=3)
    X = sm.add_constant(r.X)
    m = sm.OLS(r.y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3, "use_correction": False})
    assert np.allclose(m.bse.values, r.se_hac.values, atol=1e-12)
    assert np.allclose(m.params.values, np.r_[r.alpha, r.betas.values], atol=1e-12)


def test_factor_on_itself_has_unit_beta_zero_alpha(simulated):
    _, F, *_ = simulated
    r = fx.regress(F["Mkt-RF"], F[["Mkt-RF"]], "mkt", "CAPM", "M")
    assert abs(r.betas["Mkt-RF"] - 1) < 1e-12 and abs(r.alpha) < 1e-12 and abs(r.r2 - 1) < 1e-12


def test_to_monthly_compounds_and_keeps_empty_months_nan():
    idx = pd.bdate_range("2024-01-01", "2024-03-29")
    r = pd.Series(0.01, index=idx)
    r[idx.month == 2] = np.nan
    m = fx.to_monthly(r)
    jan_days = int((idx.month == 1).sum())
    assert abs(m.iloc[0] - (1.01 ** jan_days - 1)) < 1e-12
    assert np.isnan(m.iloc[1])


def test_rolling_regression_shapes_and_drift(simulated):
    y, F, true, _ = simulated
    rr = fx.rolling_regression(y, F[["Mkt-RF", "SMB", "HML", "Mom"]], window=60, frequency="M")
    assert rr["coef"].shape[0] == 240 - 60 + 1
    assert set(["alpha", "alpha_ann", "Mkt-RF"]).issubset(rr["coef"].columns)
    drift = fx.style_drift(rr["coef"], ["Mkt-RF", "SMB", "HML", "Mom"])
    assert abs(drift.loc["Mkt-RF", "mean"] - 1.2) < 0.1


def test_attribution_sums_to_total(simulated):
    y, F, *_ = simulated
    r = fx.regress(y, F[["Mkt-RF", "SMB", "HML"]], "sim", "FF3", "M")
    att = fx.attribution(r)
    t = att["table"]
    parts = t.drop(index="total_excess_return")["ann_contribution"].sum()
    assert abs(parts - t.loc["total_excess_return", "ann_contribution"]) < 1e-10
    cum = att["cumulative"]
    recon = cum.drop(columns="actual").sum(axis=1)
    assert np.allclose(recon.values, cum["actual"].values, atol=1e-10)


def test_residual_diagnostics_run(simulated):
    y, F, *_ = simulated
    r = fx.regress(y, F[["Mkt-RF"]], "sim", "CAPM", "M")
    d = fx.residual_diagnostics(r)
    assert "Durbin-Watson" in d.index and 1.5 < d.loc["Durbin-Watson", "statistic"] < 2.5


def test_cross_section_bh_adjustment_monotone(simulated):
    y, F, *_ = simulated
    rng = np.random.default_rng(1)
    ex = pd.DataFrame({f"a{i}": F["Mkt-RF"] * rng.uniform(0.5, 1.5) + rng.normal(0, 0.02, len(F)) for i in range(6)}, index=F.index)
    ex["a0"] = ex["a0"] + 0.01                     # one asset with real alpha
    cs = fx.cross_section(ex, F, "CAPM")
    assert (cs["p_alpha_bh"] >= cs["p_alpha_hac"] - 1e-12).all()
    assert cs.loc["a0", "p_alpha_bh"] < 0.05


def test_newey_west_rule():
    assert fx.newey_west_lags(100) == 4 and fx.newey_west_lags(259) == 4 and fx.newey_west_lags(3000) == 8
