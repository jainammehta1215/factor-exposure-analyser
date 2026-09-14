"""
Kenneth French Data Library loader.

The library publishes zipped CSVs with a free-text header, one or more data
blocks (monthly then annual, or a single daily block) and a copyright footer.
Dates are YYYYMM (monthly) or YYYYMMDD (daily). Values are in percent.

This module downloads (with caching), parses the *first* data block robustly
by scanning for date-like rows, converts to decimal returns and returns a
DataFrame indexed by period end. It also assembles the factor sets used by
the models (3, 5, 5+Mom) for a region and frequency.
"""
from __future__ import annotations

import io
import logging
import os
import re
import time
import zipfile
from typing import Dict, List, Optional

import pandas as pd
import requests

from .config import DATA, FACTOR_FILES, FRENCH_BASE, DataConfig

logger = logging.getLogger(__name__)

_DATE_ROW = re.compile(r"^\s*(\d{6}|\d{8})\s*,")


def parse_french_csv(text: str) -> pd.DataFrame:
    """
    Parse the first data block of a French CSV. Returns decimal returns with a
    DatetimeIndex at period end (month end for YYYYMM, the day for YYYYMMDD).
    """
    lines = text.splitlines()
    header_idx = None
    rows: List[str] = []
    for i, line in enumerate(lines):
        if _DATE_ROW.match(line):
            if header_idx is None:
                # the column header is the nearest previous non-empty line
                j = i - 1
                while j >= 0 and not lines[j].strip():
                    j -= 1
                header_idx = j
            rows.append(line)
        elif rows:
            break                        # first blank / non-date line after the block ends it
    if not rows or header_idx is None:
        raise ValueError("No data block found in French CSV")

    header = [h.strip() for h in lines[header_idx].split(",")]
    header[0] = "date"
    df = pd.read_csv(io.StringIO("\n".join(rows)), header=None, names=header, skipinitialspace=True)
    key = df["date"].astype(str).str.strip()
    if key.str.len().iloc[0] == 6:
        df.index = pd.to_datetime(key, format="%Y%m") + pd.offsets.MonthEnd(0)
    else:
        df.index = pd.to_datetime(key, format="%Y%m%d")
    df = df.drop(columns="date").astype(float) / 100.0
    df.index.name = "date"
    # French uses -99.99 / -999 as missing markers
    df = df.mask(df <= -0.99)
    return df


def _retry(fn, attempts=3, wait=2.0, label="download"):
    last = None
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            logger.warning("%s failed (attempt %d/%d): %s", label, i, attempts, exc)
            time.sleep(wait * i)
    raise last  # type: ignore[misc]


def load_french_file(key: str, cfg: DataConfig = DATA, force_refresh: bool = False) -> pd.DataFrame:
    """Download (or read from cache) one library file by its FACTOR_FILES key."""
    meta = FACTOR_FILES[key]
    os.makedirs(cfg.cache_dir, exist_ok=True)
    cache = os.path.join(cfg.cache_dir, f"french_{key}.csv")
    if os.path.exists(cache) and not force_refresh:
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    url = FRENCH_BASE.format(stem=meta["stem"])
    def _dl():
        r = requests.get(url, timeout=60, headers={"User-Agent": "factor-exposure-analyser/1.0"})
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        return z.read(name).decode("latin1")
    text = _retry(_dl, label=f"French {key}")
    df = parse_french_csv(text)
    df.to_csv(cache)
    logger.info("French %s: %d rows %s -> %s", key, len(df), df.index[0].date(), df.index[-1].date())
    return df


def load_factors(region: str = "us", frequency: str = "M", cfg: DataConfig = DATA,
                 include_momentum: bool = True) -> pd.DataFrame:
    """
    Assemble Mkt-RF, SMB, HML, RMW, CMA, (Mom), RF for a region and frequency.

    US: daily or monthly. International regions: monthly only (French does not
    publish daily international factors). Momentum is merged from its own file.
    """
    freq = frequency.upper()
    if region == "us":
        five = load_french_file("us_5_monthly" if freq == "M" else "us_5_daily", cfg)
        mom = load_french_file("us_mom_monthly" if freq == "M" else "us_mom_daily", cfg) if include_momentum else None
    elif region in ("dev_ex_us", "emerging"):
        if freq != "M":
            raise ValueError("International factors are monthly only")
        five = load_french_file(f"{region}_5_monthly", cfg)
        mom = load_french_file(f"{region}_mom_monthly", cfg) if include_momentum else None
    else:
        raise ValueError(f"Unknown region {region}; use us | dev_ex_us | emerging")

    df = five.copy()
    if mom is not None:
        mom_col = [c for c in mom.columns if c.lower().startswith("mom") or c.upper() == "WML"][0]
        df = df.join(mom[[mom_col]].rename(columns={mom_col: "Mom"}), how="left")
    # Standardise column names across files (international files use 'Mkt-RF' too, but be safe)
    df.columns = [c.replace(" ", "") for c in df.columns]
    if "WML" in df.columns:
        df = df.rename(columns={"WML": "Mom"})
    wanted = [c for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "RF"] if c in df.columns]
    df = df[wanted].dropna(subset=["Mkt-RF"])
    df.attrs["region"] = region
    df.attrs["frequency"] = freq
    return df


def factor_descriptions() -> Dict[str, str]:
    return {
        "Mkt-RF": "Market excess return (value-weighted market minus 1-month T-bill)",
        "SMB": "Small minus big: size premium",
        "HML": "High minus low book-to-market: value premium",
        "RMW": "Robust minus weak operating profitability: quality/profitability premium",
        "CMA": "Conservative minus aggressive investment: investment premium",
        "Mom": "Winners minus losers over months t-12 to t-2: momentum premium",
        "RF": "One-month Treasury bill rate",
    }
