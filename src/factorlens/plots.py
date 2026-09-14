"""Figures for the factor analyser. Functions return matplotlib Figures."""
from __future__ import annotations

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from . import factors as fx

PALETTE = ["#1f4e79", "#c0392b", "#e67e22", "#27ae60", "#8e44ad", "#16a085", "#7f8c8d", "#d4ac0d"]
FACTOR_COLORS = {"Mkt-RF": "#1f4e79", "SMB": "#e67e22", "HML": "#27ae60", "RMW": "#8e44ad",
                 "CMA": "#16a085", "Mom": "#c0392b", "alpha": "#2c3e50", "residual": "#bdc3c7"}
_pct = PercentFormatter(1.0, decimals=0)


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "figure.facecolor": "white",
        "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "axes.titlesize": 12, "axes.labelsize": 10,
        "legend.frameon": False, "legend.fontsize": 9, "font.size": 10,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
    })


def save(fig: plt.Figure, name: str, outdir: str = "outputs/figures") -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{name}.png")
    fig.savefig(path, bbox_inches="tight")
    return path


def _c(f: str) -> str:
    return FACTOR_COLORS.get(f, "#7f8c8d")


# --------------------------------------------------------------------------- #
def plot_alpha_across_models(tbl: pd.DataFrame, asset: str) -> plt.Figure:
    """Alpha (with HAC 95% band) and adjusted R² as factors are added."""
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(tbl))
    se = tbl["alpha_ann"] / tbl["t_alpha_hac"]
    ax.bar(x, tbl["alpha_ann"], color=[_c("alpha")] * len(tbl), alpha=0.85, width=0.55,
           yerr=1.96 * se.abs(), capsize=4, ecolor="#c0392b", label="Annualised alpha ± 1.96 HAC s.e.")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(tbl.index); ax.yaxis.set_major_formatter(_pct)
    ax.set_ylabel("Alpha (annualised)")
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.plot(x, tbl["adj_r2"], "o-", color="#e67e22", lw=1.6, label="Adjusted R²")
    ax2.set_ylim(0, 1); ax2.set_ylabel("Adjusted R²")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right")
    ax.set_title(f"{asset}: alpha shrinks as factors are added")
    return fig


def plot_coefficients(ct: pd.DataFrame, asset: str, model: str) -> plt.Figure:
    """Point estimates with HAC confidence intervals; alpha shown per period on its own axis."""
    betas = ct.drop(index="alpha")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    y = np.arange(len(betas))
    ax.barh(y, betas["estimate"], color=[_c(f) for f in betas.index], alpha=0.85, height=0.55)
    ax.errorbar(betas["estimate"], y, xerr=[betas["estimate"] - betas["ci_low"], betas["ci_high"] - betas["estimate"]],
                fmt="none", ecolor="black", capsize=4, lw=1)
    for i, (f, row) in enumerate(betas.iterrows()):
        ax.text(row["ci_high"] + 0.03 if row["estimate"] >= 0 else row["ci_low"] - 0.03, i,
                f"t = {row['t_hac']:.1f}", va="center", ha="left" if row["estimate"] >= 0 else "right", fontsize=8)
    ax.axvline(0, color="black", lw=0.8); ax.set_yticks(y); ax.set_yticklabels(betas.index); ax.invert_yaxis()
    ax.set_xlabel("Factor loading (beta) with 95% HAC interval")
    a = ct.loc["alpha"]
    ax.set_title(f"{asset} · {model}   alpha {a['estimate']*12:+.2%} p.a. (t = {a['t_hac']:.2f})")
    return fig


def plot_rolling_loadings(rr: Dict[str, pd.DataFrame], asset: str, factors: List[str], window: int) -> plt.Figure:
    coef, r2 = rr["coef"], rr["r2"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.4, 1]})
    for f in factors:
        axes[0].plot(coef.index, coef[f], lw=1.6, color=_c(f), label=f)
    axes[0].axhline(0, color="black", lw=0.8); axes[0].axhline(1, color="grey", lw=0.6, ls=":")
    axes[0].set_title(f"{asset}: rolling {window}-period factor loadings"); axes[0].legend(ncol=len(factors), fontsize=8)
    axes[1].plot(coef.index, coef["alpha_ann"], color=_c("alpha"), lw=1.6, label="Rolling alpha (annualised)")
    sig = rr["tstat"]["alpha"].abs() > 1.96
    axes[1].fill_between(coef.index, coef["alpha_ann"].min(), coef["alpha_ann"].max(), where=sig, color="#c0392b", alpha=0.12,
                         label="|t| > 1.96")
    axes[1].axhline(0, color="black", lw=0.8); axes[1].yaxis.set_major_formatter(_pct); axes[1].legend(fontsize=8)
    axes[2].plot(r2.index, r2, color="#e67e22", lw=1.4); axes[2].set_ylim(0, 1); axes[2].set_title("Rolling R²", fontsize=10)
    fig.tight_layout()
    return fig


def plot_style_map(rr: Dict[str, pd.DataFrame], asset: str, x: str = "SMB", y: str = "HML") -> plt.Figure:
    """Trajectory of two rolling loadings; the classic size/value style box, with time as colour."""
    coef = rr["coef"]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    t = np.linspace(0, 1, len(coef))
    ax.plot(coef[x], coef[y], color="#bdc3c7", lw=0.8, zorder=1)
    sc = ax.scatter(coef[x], coef[y], c=t, cmap="viridis", s=22, zorder=2)
    ax.scatter(coef[x].iloc[-1], coef[y].iloc[-1], s=140, facecolors="none", edgecolors="#c0392b", lw=2, zorder=3, label="latest")
    ax.axhline(0, color="black", lw=0.8); ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(f"{x} loading  (← large · small →)"); ax.set_ylabel(f"{y} loading  (← growth · value →)")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046); cb.set_ticks([0, 1]); cb.set_ticklabels([str(coef.index[0].year), str(coef.index[-1].year)])
    ax.set_title(f"{asset}: style drift"); ax.legend(loc="upper left")
    return fig


def plot_attribution(att: Dict[str, object], asset: str, model: str, frequency: str) -> plt.Figure:
    table, cum = att["table"], att["cumulative"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.6]})
    rows = table.drop(index="total_excess_return")
    ax1.barh(rows.index, rows["ann_contribution"], color=[_c(f) for f in rows.index])
    ax1.axvline(0, color="black", lw=0.8); ax1.xaxis.set_major_formatter(_pct); ax1.invert_yaxis()
    ax1.set_title(f"Annualised excess return {table.loc['total_excess_return','ann_contribution']:.1%} decomposed")
    comps = [c for c in cum.columns if c not in ("actual",)]
    ax2.stackplot(cum.index, [cum[c].values for c in comps], labels=comps, colors=[_c(c) for c in comps], alpha=0.85)
    ax2.plot(cum.index, cum["actual"], color="black", lw=1.6, label="Actual (cumulative excess)")
    ax2.yaxis.set_major_formatter(_pct); ax2.set_title("Cumulative contribution (additive returns)"); ax2.legend(fontsize=7, ncol=2, loc="upper left")
    fig.suptitle(f"{asset} · {model}", fontweight="bold"); fig.tight_layout()
    return fig


def plot_residuals(res: fx.RegressionResult) -> plt.Figure:
    e = res.resid
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot(e.index, e, lw=0.9, color="#1f4e79"); axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_title("Residuals over time"); axes[0].yaxis.set_major_formatter(_pct)
    from scipy import stats
    (osm, osr), (slope, intercept, r) = stats.probplot(e.values, dist="norm")
    axes[1].scatter(osm, osr, s=10, color="#1f4e79"); axes[1].plot(osm, slope * np.array(osm) + intercept, color="#c0392b", lw=1)
    axes[1].set_title("Normal Q-Q"); axes[1].set_xlabel("Theoretical quantiles")
    n = len(e); lags = 20 if res.frequency == "D" else 12
    x = e.values - e.values.mean()
    acf = [1.0] + [float(np.sum(x[l:] * x[:-l]) / np.sum(x * x)) for l in range(1, lags + 1)]
    axes[2].bar(range(lags + 1), acf, color="#1f4e79", width=0.6)
    axes[2].axhline(1.96 / np.sqrt(n), color="#c0392b", ls="--", lw=0.8); axes[2].axhline(-1.96 / np.sqrt(n), color="#c0392b", ls="--", lw=0.8)
    axes[2].set_title("Residual autocorrelation"); axes[2].set_xlabel("Lag")
    fig.suptitle(f"{res.asset} · {res.model} residual diagnostics", fontweight="bold"); fig.tight_layout()
    return fig


def plot_exposure_heatmap(expo: pd.DataFrame, model: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(1.1 * expo.shape[1] + 3, 0.5 * len(expo) + 1.5))
    vmax = float(np.nanmax(np.abs(expo.values)))
    im = ax.imshow(expo.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(expo.shape[1])); ax.set_xticklabels(expo.columns); ax.set_yticks(range(len(expo))); ax.set_yticklabels(expo.index); ax.grid(False)
    for i in range(expo.shape[0]):
        for j in range(expo.shape[1]):
            ax.text(j, i, f"{expo.values[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(expo.values[i, j]) > 0.6 * vmax else "black")
    fig.colorbar(im, ax=ax, fraction=0.03); ax.set_title(f"Factor loadings across assets ({model})")
    return fig


def plot_alpha_cross_section(cs: pd.DataFrame, model: str) -> plt.Figure:
    """Alpha with HAC t-stats; highlights survivors of the Benjamini-Hochberg correction."""
    d = cs.sort_values("alpha_ann")
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(d) + 1.5))
    colors = ["#27ae60" if p < 0.05 else ("#e67e22" if q < 0.05 else "#7f8c8d") for p, q in zip(d["p_alpha_bh"], d["p_alpha_hac"])]
    ax.barh(d.index, d["alpha_ann"], color=colors, alpha=0.9)
    for i, (a, t) in enumerate(zip(d["alpha_ann"], d["t_alpha_hac"])):
        ax.text(a + (0.002 if a >= 0 else -0.002), i, f"t={t:.1f}", va="center", ha="left" if a >= 0 else "right", fontsize=8)
    ax.axvline(0, color="black", lw=0.8); ax.xaxis.set_major_formatter(_pct)
    ax.set_title(f"Annualised alpha vs {model}   (green: significant after BH correction · orange: raw p<0.05 only · grey: not significant)", fontsize=10)
    return fig


def plot_cumulative_factors(f: pd.DataFrame, factors: List[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for c in factors:
        ax.plot(f.index, (1 + f[c]).cumprod(), lw=1.5, color=_c(c), label=c)
    ax.set_yscale("log"); ax.set_title("Cumulative factor returns (long-short portfolios, log scale)"); ax.legend(ncol=len(factors))
    return fig
