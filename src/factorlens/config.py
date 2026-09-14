"""
Configuration for the multi-factor exposure analyser.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# --------------------------------------------------------------------------- #
# Assets to analyse. Chosen so each one tells a factor story.
# --------------------------------------------------------------------------- #
ASSETS: Dict[str, str] = {
    "BRK-B": "Berkshire Hathaway (is Buffett's alpha just quality + low beta?)",
    "ARKK": "ARK Innovation (growth, high beta, style drift)",
    "VTV": "Vanguard Value ETF (should load on HML)",
    "MTUM": "iShares Momentum ETF (should load on Mom)",
    "QUAL": "iShares Quality ETF (should load on RMW)",
    "IWM": "Russell 2000 ETF (should load on SMB)",
    "USMV": "iShares Min Vol ETF (beta well below 1)",
}

# A portfolio can be analysed as a single asset (monthly-rebalanced blend).
PORTFOLIO: Dict[str, float] = {"SPY": 0.30, "QQQ": 0.10, "EFA": 0.10, "EEM": 0.05, "IEF": 0.15,
                               "TLT": 0.10, "LQD": 0.05, "GLD": 0.10, "VNQ": 0.05}
PORTFOLIO_NAME = "Project-1 multi-asset portfolio"

# --------------------------------------------------------------------------- #
# Kenneth French library files. Each entry: (file stem, frequency, columns)
# --------------------------------------------------------------------------- #
FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{stem}_CSV.zip"

FACTOR_FILES: Dict[str, Dict[str, str]] = {
    # US
    "us_3_monthly": dict(stem="F-F_Research_Data_Factors", freq="M"),
    "us_3_daily": dict(stem="F-F_Research_Data_Factors_daily", freq="D"),
    "us_5_monthly": dict(stem="F-F_Research_Data_5_Factors_2x3", freq="M"),
    "us_5_daily": dict(stem="F-F_Research_Data_5_Factors_2x3_daily", freq="D"),
    "us_mom_monthly": dict(stem="F-F_Momentum_Factor", freq="M"),
    "us_mom_daily": dict(stem="F-F_Momentum_Factor_daily", freq="D"),
    # International (monthly only)
    "dev_ex_us_5_monthly": dict(stem="Developed_ex_US_5_Factors", freq="M"),
    "dev_ex_us_mom_monthly": dict(stem="Developed_ex_US_Mom_Factor", freq="M"),
    "emerging_5_monthly": dict(stem="Emerging_5_Factors", freq="M"),
    "emerging_mom_monthly": dict(stem="Emerging_MOM_Factor", freq="M"),
}

# Model definitions: name -> factor columns (excess market return is 'Mkt-RF').
MODELS: Dict[str, List[str]] = {
    "CAPM": ["Mkt-RF"],
    "FF3": ["Mkt-RF", "SMB", "HML"],
    "Carhart4": ["Mkt-RF", "SMB", "HML", "Mom"],
    "FF5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
    "FF5+Mom": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"],
}


@dataclass
class DataConfig:
    start: str = "2005-01-01"
    end: Optional[str] = None
    cache_dir: str = "data/cache"
    fred_series: str = "DGS3MO"          # unused for regressions (French supplies RF) but kept for the shared loader
    base_currency: str = "USD"
    trading_days: int = 252
    min_history_fraction: float = 0.0    # assets with short history are allowed; regressions use available overlap


@dataclass
class AnalysisConfig:
    frequency: str = "M"                 # "M" monthly (standard) or "D" daily
    region: str = "us"                   # us | dev_ex_us | emerging
    rolling_window: int = 36             # periods (36 months, or ~252 days if daily)
    min_obs: int = 24                    # minimum observations for any regression
    hac_lags: Optional[int] = None       # None -> Newey-West automatic rule 4(T/100)^(2/9)
    confidence: float = 0.95


DATA = DataConfig()
ANALYSIS = AnalysisConfig()
