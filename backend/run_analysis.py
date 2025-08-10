# backend/run_analysis.py
import os
import json
import re
import math
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===========================
# Helpers & parsing
# ===========================

MONTH_LETTERS = set(list("FGHJKMNQUVXZ"))

def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def _to_utc(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, errors="coerce", utc=True)

def _mid_from_quotes(df: pd.DataFrame, bid_col="close_bid", ask_col="close_ask") -> pd.Series:
    if {bid_col, ask_col}.issubset(df.columns):
        bid = _numeric(df[bid_col])
        ask = _numeric(df[ask_col])
        return (bid + ask) / 2.0
    return pd.Series(index=df.index, dtype="float64")

def _valid_price(s: pd.Series) -> pd.Series:
    s = _numeric(s)
    return s.where(s > 0)

def _sanitize_contract_for_path(c: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(c))

def _extract_df1_contract(raw_key: pd.Series) -> pd.Series:
    """Dataset 1 'key' like F.US.QOK25 -> QOK25."""
    pat = re.compile(r"F\.US\.([A-Z0-9]+)")
    return raw_key.astype(str).str.extract(pat, expand=False)

def _extract_df2_contract(alias_series: pd.Series) -> pd.Series:
    """Dataset 2 alias_underlying_ric like LCOK5 / LCOM5."""
    return alias_series.astype(str)

def _extract_month_year_tail(code: str) -> Tuple[Optional[str], Optional[int]]:
    """Extract (month_letter, base_year) from a vendor contract code."""
    if not isinstance(code, str):
        return (None, None)
    m = re.search(r"([FGHJKMNQUVXZ])(\d{1,2})$", code)
    if not m:
        return (None, None)
    mon = m.group(1)
    yr = m.group(2)
    base = 2000 + int(yr)  # '5'->2005 (we'll snap using a hint)
    return (mon, base)

def _normalize_contract(code: str, ts_year_hint: Optional[int] = None) -> Optional[str]:
    """Return normalized 'MYYYY' (e.g., 'K2025'). Resolve decade using ts_year_hint."""
    mon, year = _extract_month_year_tail(code)
    if mon is None or year is None:
        return None
    if ts_year_hint is not None:
        candidates = [year, year + 20, year + 40]
        year = min(candidates, key=lambda y: abs(ts_year_hint - y))
    if mon not in MONTH_LETTERS:
        return None
    return f"{mon}{year}"

# ===========================
# Active bar logic
# ===========================

def compute_active_masks(df: pd.DataFrame, prefer_mid: bool) -> pd.Series:
    """Active if (volume>0) OR (optionally) mid exists."""
    vol_active = (_numeric(df.get("volume", np.nan)) > 0)
    if prefer_mid:
        mid = _mid_from_quotes(df)
        price_active = _valid_price(df.get("close", np.nan)).notna() | mid.notna()
    else:
        price_active = _valid_price(df.get("close", np.nan)).notna()
    return (vol_active | price_active).astype("bool")

def compute_trade_active(df: pd.DataFrame) -> pd.Series:
    """Trade-active: volume>0 AND valid close>0."""
    vol = _numeric(df.get("volume", np.nan))
    px  = _valid_price(df.get("close", np.nan))
    return ((vol > 0) & px.notna()).astype("bool")

# ===========================
# Load & clean
# ===========================

def load_data(path1: str, path2: str):
    print("--- 1. Loading Raw Data ---")
    df1 = pd.read_csv(path1, low_memory=False)
    df2 = pd.read_csv(path2, low_memory=False)
    print("✅ Loaded.")
    return df1, df2

def clean_inputs(df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """UTC index, dedup, numeric coercion, contract extraction, active masks."""
    print("\n--- 2. Cleaning & Normalizing ---")
    d1 = df1.copy()
    d2 = df2.copy()

    d1["date_time"] = _to_utc(d1.get("date_time"))
    d2["date_time"] = _to_utc(d2.get("date_time"))
    d1 = d1.dropna(subset=["date_time"])
    d2 = d2.dropna(subset=["date_time"])

    if "gmt_offset" in d2.columns:
        gmt = pd.to_numeric(d2["gmt_offset"], errors="coerce").fillna(0)
        d2["date_time"] = d2["date_time"] - pd.to_timedelta(gmt, unit="h")

    d1 = d1.set_index("date_time").sort_index()
    d2 = d2.set_index("date_time").sort_index()

    d1 = d1[~d1.index.duplicated(keep="last")]
    d2 = d2[~d2.index.duplicated(keep="last")]

    ohlc = ["open","high","low","close"]
    for c in ohlc + ["volume","close_bid","close_ask"]:
        if c in d1.columns: d1[c] = _numeric(d1[c])
        if c in d2.columns: d2[c] = _numeric(d2[c])

    for c in ohlc:
        if c in d1.columns: d1[c] = d1[c].where(d1[c] > 0)
        if c in d2.columns: d2[c] = d2[c].where(d2[c] > 0)

    if "iscomplete" in d1.columns:
        bad = d1["iscomplete"] == False
        d1.loc[bad, ohlc] = np.nan

    alias_col = None
    for c in d2.columns:
        if c.lower().strip() in {"alias_underlying_ric", "alias_ric", "alias"}:
            alias_col = c
            break
    if alias_col is None:
        raise ValueError("Dataset 2 must contain alias_underlying_ric (or alias_ric/alias).")

    ts_year_hint_d1 = int(np.median(d1.index.year)) if len(d1) else 2025
    ts_year_hint_d2 = int(np.median(d2.index.year)) if len(d2) else 2025
    d1["contract_raw"]  = _extract_df1_contract(d1.get("key", "").fillna(""))
    d1["contract_norm"] = d1["contract_raw"].map(lambda x: _normalize_contract(x, ts_year_hint_d1))
    d2["contract_raw"]  = _extract_df2_contract(d2[alias_col].fillna(""))
    d2["contract_norm"] = d2["contract_raw"].map(lambda x: _normalize_contract(x, ts_year_hint_d2))

    d1["active_any"]   = compute_active_masks(d1, prefer_mid=False)
    d2["active_any"]   = compute_active_masks(d2, prefer_mid=True)
    d1["trade_active"] = compute_trade_active(d1)
    d2["trade_active"] = compute_trade_active(d2)

    print("✅ Cleaned and extracted contracts.")
    return d1, d2, alias_col

# ===========================
# Windows, scaling, metrics
# ===========================

def per_contract_windows(d1: pd.DataFrame, d2: pd.DataFrame) -> pd.DataFrame:
    """Intersection of first/last TRADE-ACTIVE times in both datasets per contract."""
    rows = []
    common = sorted(set(d1["contract_norm"].dropna()) & set(d2["contract_norm"].dropna()))
    for c in common:
        s1 = d1[d1["contract_norm"] == c]
        s2 = d2[d2["contract_norm"] == c]
        if s1.empty or s2.empty:
            continue
        t0_d1 = s1.index[s1["trade_active"]].min()
        t0_d2 = s2.index[s2["trade_active"]].min()
        t1_d1 = s1.index[s1["trade_active"]].max()
        t1_d2 = s2.index[s2["trade_active"]].max()
        if pd.isna(t0_d1) or pd.isna(t0_d2) or pd.isna(t1_d1) or pd.isna(t1_d2):
            continue
        t0 = max(t0_d1, t0_d2); t1 = min(t1_d1, t1_d2)
        if t0 < t1:
            rows.append(dict(
                contract=c,
                start_d1=str(t0_d1), start_d2=str(t0_d2),
                start=str(t0),
                end_d1=str(t1_d1), end_d2=str(t1_d2),
                end=str(t1)
            ))
    return pd.DataFrame(rows)

def winsorize(s: pd.Series, p_low=0.01, p_high=0.99) -> pd.Series:
    if s.empty:
        return s
    a, b = s.quantile([p_low, p_high])
    return s.clip(a, b)

def per_contract_scaling(d1: pd.DataFrame, d2: pd.DataFrame, windows: pd.DataFrame) -> Dict[str, float]:
    """Median(close_d1/close_d2) on matched + active bars within [T0,T1], winsorized."""
    factors = {}
    for _, r in windows.iterrows():
        c  = r["contract"]
        t0 = pd.to_datetime(r["start"]); t1 = pd.to_datetime(r["end"])
        s1 = d1[(d1["contract_norm"] == c) & (d1.index >= t0) & (d1.index <= t1)]
        s2 = d2[(d2["contract_norm"] == c) & (d2.index >= t0) & (d2.index <= t1)]
        idx = s1.index.intersection(s2.index)
        if len(idx) == 0:
            continue
        d1_close = _valid_price(s1.reindex(idx)["close"])
        d2_close = _valid_price(s2.reindex(idx)["close"])
        d2_mid   = _mid_from_quotes(s2.reindex(idx))
        d2_close = d2_close.fillna(d2_mid)
        act = (s1.reindex(idx)["trade_active"].astype("bool").reindex(idx, fill_value=False) &
               s2.reindex(idx)["active_any"].astype("bool").reindex(idx, fill_value=False))
        v = d1_close.notna() & d2_close.notna() & act
        ratios = (d1_close[v] / d2_close[v]).replace([np.inf, -np.inf], np.nan).dropna()
        if len(ratios) >= 20:
            ratios = winsorize(ratios, 0.01, 0.99)
        if len(ratios) >= 5 and np.isfinite(ratios.median()) and ratios.median() > 0:
            factors[c] = float(ratios.median())
    return factors

def apply_per_contract_scaling(d1: pd.DataFrame, factors: Dict[str, float]) -> pd.DataFrame:
    d1 = d1.copy()
    ohlc = ["open","high","low","close"]
    d1["scale_factor"] = d1["contract_norm"].map(factors).astype("float64")
    global_factor = np.nanmedian(list(factors.values())) if len(factors) else 1.0
    d1["scale_factor"] = d1["scale_factor"].fillna(global_factor)
    for c in ohlc:
        if c in d1.columns:
            d1[c + "_scaled"] = d1[c] / d1["scale_factor"]
    return d1

def _safe_log_returns(series: pd.Series) -> pd.Series:
    s = _valid_price(series)
    valid = s.gt(0) & s.shift(1).gt(0)
    out = pd.Series(index=s.index, dtype="float64")
    out.loc[valid] = np.log(s.loc[valid]) - np.log(s.shift(1).loc[valid])
    return out

def compute_metrics(d1s: pd.DataFrame, d2: pd.DataFrame, windows: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-contract metrics on matched+active bars within [T0,T1]."""
    rows = []
    details = []
    for _, r in windows.iterrows():
        c  = r["contract"]
        t0 = pd.to_datetime(r["start"]); t1 = pd.to_datetime(r["end"])
        s1 = d1s[(d1s["contract_norm"] == c) & (d1s.index >= t0) & (d1s.index <= t1)]
        s2 = d2[(d2["contract_norm"] == c) & (d2.index >= t0) & (d2.index <= t1)]
        idx = s1.index.intersection(s2.index)
        if len(idx) == 0:
            continue
        x = s1.reindex(idx)["close_scaled"]
        y = s2.reindex(idx)["close"]
        act = (s1.reindex(idx)["trade_active"].astype("bool").reindex(idx, fill_value=False) &
               s2.reindex(idx)["active_any"].astype("bool").reindex(idx, fill_value=False))
        m = x.notna() & y.notna() & act
        if not m.any():
            continue
        xs, ys = x[m], y[m]
        diff = xs - ys
        mae  = float(diff.abs().mean())
        rmse = float(np.sqrt((diff.pow(2)).mean()))
        mape = float((diff.abs() / ys).mean() * 100)
        corr = xs.pct_change().corr(ys.pct_change())
        corr_log = _safe_log_returns(xs).corr(_safe_log_returns(ys))
        rows.append(dict(contract=c, n=int(m.sum()), mae=mae, rmse=rmse, mape=mape,
                         median_spread=float(diff.median()),
                         corr=float(corr) if pd.notna(corr) else np.nan,
                         corr_log=float(corr_log) if pd.notna(corr_log) else np.nan))
        details.append(pd.DataFrame({
            "timestamp": xs.index,
            "contract": c,
            "close_d1_scaled": xs.values,
            "close_d2": ys.values,
            "spread": diff.values
        }))
    summary = pd.DataFrame(rows).sort_values("contract") if rows else pd.DataFrame(columns=["contract"])
    detailed = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    return summary, detailed

# ===========================
# Roll diagnostics
# ===========================

def compute_roll_dates(df: pd.DataFrame, label: str) -> pd.DataFrame:
    s = df["contract_norm"].dropna()
    changed = s.ne(s.shift(1))
    rolls = s[changed]
    return pd.DataFrame({
        "when": rolls.index,
        "new": rolls.values,
        "old": s.shift(1)[changed].values,
        "source": label
    }).reset_index(drop=True)

# ===========================
# Heatmaps
# ===========================

def plot_missing_heatmap(df: pd.DataFrame, dataset_name: str, value_col="close", freq="5min", tz=None, out_dir="."):
    """Days×Hours heatmap of missing bars (NaN or <=0) over dataset's span (canonical 5-min grid)."""
    if df.empty:
        return None
    idx = df.index
    if tz:
        idx = idx.tz_convert(tz); df = df.tz_convert(tz)
    elif idx.tz is None:
        idx = idx.tz_localize("UTC"); df = df.tz_localize("UTC")

    start = idx.min().floor(freq); end = idx.max().ceil(freq)
    grid  = pd.date_range(start=start, end=end, freq=freq, tz=idx.tz)
    s = _numeric(df.get(value_col)).reindex(grid)
    missing = s.isna() | (s <= 0)
    tmp = pd.DataFrame({"missing": missing.astype(int)}, index=grid)
    tmp["date"] = tmp.index.date; tmp["hour"] = tmp.index.hour
    heat = tmp.groupby(["date","hour"])["missing"].mean().unstack(fill_value=0)

    plt.figure(figsize=(12, max(4, len(heat)*0.25)))
    im = plt.imshow(heat, aspect="auto", origin="lower",
                    extent=[-0.5, 23.5, -0.5, len(heat)-0.5], vmin=0, vmax=1)
    plt.colorbar(im, label="Fraction missing (1 = 100% missing)")
    plt.xticks(range(24)); plt.yticks(range(len(heat)), [str(d) for d in heat.index])
    plt.xlabel("Hour of day" + (f" ({tz})" if tz else " (UTC)")); plt.ylabel("Date")
    plt.title(f"{dataset_name}: Missing bars heatmap")
    out_path = f"{out_dir}/missing_heatmap_{dataset_name}.png"
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()
    return out_path

def plot_contract_activity_heatmap(df: pd.DataFrame, dataset_name: str, contract_key: str,
                                   freq: str = "5min", tz: Optional[str] = None, out_dir: str = "."):
    """Per-contract heatmap: fraction INACTIVE (not trade-active) by day×hour over contract span."""
    if df.empty or "trade_active" not in df.columns or "contract_norm" not in df.columns:
        return None
    sub = df[df["contract_norm"] == contract_key]
    if sub.empty:
        return None
    idx = sub.index
    if tz:
        idx = idx.tz_convert(tz); sub = sub.tz_convert(tz)
    elif idx.tz is None:
        idx = idx.tz_localize("UTC"); sub = sub.tz_localize("UTC")

    start = idx.min().floor(freq); end = idx.max().ceil(freq)
    grid  = pd.date_range(start=start, end=end, freq=freq, tz=idx.tz)

    ta = sub["trade_active"].astype("bool").reindex(grid, fill_value=False)
    inactive = (~ta).astype(int)

    df_h = pd.DataFrame({"inactive": inactive}, index=grid)
    df_h["date"] = df_h.index.date; df_h["hour"] = df_h.index.hour
    heat = df_h.groupby(["date","hour"])["inactive"].mean().unstack(fill_value=0)

    plt.figure(figsize=(12, max(4, len(heat)*0.25)))
    im = plt.imshow(heat, aspect="auto", origin="lower",
                    extent=[-0.5, 23.5, -0.5, len(heat)-0.5], vmin=0, vmax=1)
    plt.colorbar(im, label="Fraction inactive (1 = 100% inactive)")
    plt.xticks(range(24)); plt.yticks(range(len(heat)), [str(d) for d in heat.index])
    plt.xlabel("Hour of day" + (f" ({tz})" if tz else " (UTC)")); plt.ylabel("Date")
    plt.title(f"{dataset_name} – {contract_key}: Trade inactivity heatmap")
    out_path = f"{out_dir}/heatmap_{dataset_name}_{_sanitize_contract_for_path(contract_key)}.png"
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()
    return out_path

def generate_per_contract_heatmaps(df: pd.DataFrame, dataset_name: str, out_dir: str,
                                   freq: str = "5min", tz: Optional[str] = "UTC", limit: Optional[int] = 8) -> List[str]:
    if df.empty or "contract_norm" not in df.columns:
        return []
    spans = []
    for c in sorted(set(df["contract_norm"].dropna())):
        seg = df[df["contract_norm"] == c]
        if seg.empty: continue
        spans.append((c, (seg.index.max() - seg.index.min()).total_seconds()))
    spans.sort(key=lambda x: x[1], reverse=True)
    if limit is not None:
        spans = spans[:limit]
    paths = []
    for c, _ in spans:
        p = plot_contract_activity_heatmap(df, dataset_name, c, freq=freq, tz=tz, out_dir=out_dir)
        if p: paths.append(p)
    return paths

def plot_overlap_heatmaps(d1: pd.DataFrame, d2: pd.DataFrame, windows: pd.DataFrame,
                          out_dir: str, freq: str = "5min", tz: Optional[str] = "UTC",
                          limit: Optional[int] = 8) -> List[str]:
    """Per-contract overlap: BOTH trade-active, ONLY D1, ONLY D2 (fraction by day×hour)."""
    paths = []
    if windows.empty:
        return paths
    w = windows.copy()
    w["span_sec"] = (pd.to_datetime(w["end"]) - pd.to_datetime(w["start"])).dt.total_seconds()
    w = w.sort_values("span_sec", ascending=False)
    if limit is not None:
        w = w.head(limit)
    for _, r in w.iterrows():
        c = r["contract"]; t0 = pd.to_datetime(r["start"]); t1 = pd.to_datetime(r["end"])
        s1 = d1[(d1["contract_norm"] == c) & (d1.index >= t0) & (d1.index <= t1)]
        s2 = d2[(d2["contract_norm"] == c) & (d2.index >= t0) & (d2.index <= t1)]
        if s1.empty or s2.empty: continue

        idx = s1.index.union(s2.index)
        if tz:
            idx = idx.tz_convert(tz)
            s1 = s1.tz_convert(tz); s2 = s2.tz_convert(tz)
        elif idx.tz is None:
            idx = idx.tz_localize("UTC"); s1 = s1.tz_localize("UTC"); s2 = s2.tz_localize("UTC")

        start = idx.min().floor(freq); end = idx.max().ceil(freq)
        grid  = pd.date_range(start=start, end=end, freq=freq, tz=idx.tz)

        d1_ta = s1["trade_active"].astype("bool").reindex(grid, fill_value=False)
        d2_ta = s2["trade_active"].astype("bool").reindex(grid, fill_value=False)

        both = (d1_ta & d2_ta).astype(int)
        only1 = (d1_ta & ~d2_ta).astype(int)
        only2 = (d2_ta & ~d1_ta).astype(int)

        def heat(series, title_suffix, fname_suffix):
            tmp = pd.DataFrame({"v": series}, index=grid)
            tmp["date"] = tmp.index.date; tmp["hour"] = tmp.index.hour
            heat = tmp.groupby(["date","hour"])["v"].mean().unstack(fill_value=0)
            plt.figure(figsize=(12, max(4, len(heat)*0.25)))
            im = plt.imshow(heat, aspect="auto", origin="lower",
                            extent=[-0.5, 23.5, -0.5, len(heat)-0.5], vmin=0, vmax=1)
            plt.colorbar(im, label="Fraction")
            plt.xticks(range(24)); plt.yticks(range(len(heat)), [str(d) for d in heat.index])
            plt.xlabel("Hour of day" + (f" ({tz})" if tz else " (UTC)")); plt.ylabel("Date")
            plt.title(f"{c}: {title_suffix}")
            path = f"{out_dir}/overlap_{_sanitize_contract_for_path(c)}_{fname_suffix}.png"
            plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()
            return path

        paths += [
            heat(both,  "BOTH trade-active (fraction)", "both"),
            heat(only1, "ONLY Dataset1 trade-active (fraction)", "only_d1"),
            heat(only2, "ONLY Dataset2 trade-active (fraction)", "only_d2"),
        ]
    return paths

# ===========================
# Uptime
# ===========================

def uptime_overall(df: pd.DataFrame, freq="5min") -> Dict[str, float]:
    if df.empty:
        return {"present_close_rate": np.nan, "trade_active_rate": np.nan}
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    idx = df.index
    start = idx.min().floor(freq); end = idx.max().ceil(freq)
    grid  = pd.date_range(start=start, end=end, freq=freq, tz=idx.tz)
    present_close = _valid_price(df.get("close")).reindex(grid).notna().mean()
    trade_rate = df.get("trade_active").astype("bool").reindex(grid, fill_value=False).mean()
    return {"present_close_rate": float(present_close), "trade_active_rate": float(trade_rate)}

def uptime_per_contract(d1: pd.DataFrame, d2: pd.DataFrame, windows: pd.DataFrame, freq="5min") -> pd.DataFrame:
    rows = []
    for _, r in windows.iterrows():
        c = r["contract"]; t0 = pd.to_datetime(r["start"]); t1 = pd.to_datetime(r["end"])
        s1 = d1[(d1["contract_norm"] == c) & (d1.index >= t0) & (d1.index <= t1)]
        s2 = d2[(d2["contract_norm"] == c) & (d2.index >= t0) & (d2.index <= t1)]
        if s1.empty or s2.empty:
            continue
        idx = s1.index.union(s2.index)
        if idx.tz is None:
            s1 = s1.tz_localize("UTC"); s2 = s2.tz_localize("UTC"); idx = idx.tz_localize("UTC")
        start = idx.min().floor(freq); end = idx.max().ceil(freq)
        grid  = pd.date_range(start=start, end=end, freq=freq, tz=idx.tz)

        d1_present = _valid_price(s1.get("close")).reindex(grid).notna().mean()
        d2_present = _valid_price(s2.get("close")).reindex(grid).notna().mean()
        d1_ta = s1.get("trade_active").astype("bool").reindex(grid, fill_value=False)
        d2_ta = s2.get("trade_active").astype("bool").reindex(grid, fill_value=False)
        both_ta = (d1_ta & d2_ta).mean()

        rows.append(dict(
            contract=c,
            d1_present_close_rate=float(d1_present),
            d2_present_close_rate=float(d2_present),
            d1_trade_active_rate=float(d1_ta.mean()),
            d2_trade_active_rate=float(d2_ta.mean()),
            both_trade_active_rate=float(both_ta),
            expected_bars=int(len(grid))
        ))
    return pd.DataFrame(rows).sort_values("contract")

# ===========================
# Contract panels (union grid)
# ===========================

def _ensure_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out

def build_contract_panel(d1: pd.DataFrame, d2: pd.DataFrame, contract: str, freq: str = "5min") -> pd.DataFrame:
    """Contract-centric panel on a union 5-min grid (preserves D1-only and D2-only bars)."""
    s1 = d1[d1["contract_norm"] == contract]
    s2 = d2[d2["contract_norm"] == contract]
    if s1.empty and s2.empty:
        return pd.DataFrame()

    idx_all = s1.index.union(s2.index)
    if idx_all.tz is None:
        s1 = s1.tz_localize("UTC") if s1.index.tz is None else s1
        s2 = s2.tz_localize("UTC") if s2.index.tz is None else s2
        idx_all = idx_all.tz_localize("UTC")
    start = idx_all.min().floor(freq); end = idx_all.max().ceil(freq)
    grid = pd.date_range(start=start, end=end, freq=freq, tz=idx_all.tz)

    base_cols = ["open","high","low","close","volume","close_bid","close_ask","trade_active","active_any"]
    s1 = _ensure_cols(s1, base_cols).reindex(grid)
    s2 = _ensure_cols(s2, base_cols).reindex(grid)

    q1 = s1["close_bid"].notna() & s1["close_ask"].notna()
    q2 = s2["close_bid"].notna() & s2["close_ask"].notna()

    panel = pd.DataFrame(index=grid)
    panel["open_d1"]   = s1["open"]
    panel["high_d1"]   = s1["high"]
    panel["low_d1"]    = s1["low"]
    panel["close_d1"]  = s1["close"]
    panel["volume_d1"] = s1["volume"]
    panel["bid_d1"]    = s1["close_bid"]
    panel["ask_d1"]    = s1["close_ask"]
    panel["trade_active_d1"]   = s1["trade_active"].astype("bool").reindex(grid, fill_value=False)
    panel["quote_active_d1"]   = q1.reindex(grid, fill_value=False)
    panel["present_close_d1"]  = panel["close_d1"].notna()

    panel["open_d2"]   = s2["open"]
    panel["high_d2"]   = s2["high"]
    panel["low_d2"]    = s2["low"]
    panel["close_d2"]  = s2["close"]
    panel["volume_d2"] = s2["volume"]
    panel["bid_d2"]    = s2["close_bid"]
    panel["ask_d2"]    = s2["close_ask"]
    panel["trade_active_d2"]   = s2["trade_active"].astype("bool").reindex(grid, fill_value=False)
    panel["quote_active_d2"]   = q2.reindex(grid, fill_value=False)
    panel["present_close_d2"]  = panel["close_d2"].notna()

    panel["both_trade_active"] = panel["trade_active_d1"] & panel["trade_active_d2"]
    panel["only_d1_active"]    = panel["trade_active_d1"] & ~panel["trade_active_d2"]
    panel["only_d2_active"]    = panel["trade_active_d2"] & ~panel["trade_active_d1"]

    panel["close_d1_scaled"] = np.nan  # filled if factor available
    return panel

def export_contract_panels(d1: pd.DataFrame, d2: pd.DataFrame, factors: Dict[str, float],
                           out_dir: str, freq: str = "5min", limit: Optional[int] = 8) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    spans = []
    all_contracts = sorted(set(d1["contract_norm"].dropna()) | set(d2["contract_norm"].dropna()))
    for c in all_contracts:
        seg1 = d1[d1["contract_norm"] == c]
        seg2 = d2[d2["contract_norm"] == c]
        if seg1.empty and seg2.empty: continue
        idx_all = seg1.index.union(seg2.index)
        span = (idx_all.max() - idx_all.min()).total_seconds()
        spans.append((c, span))
    spans.sort(key=lambda x: x[1], reverse=True)
    if limit is not None:
        spans = spans[:limit]

    paths = []
    for c, _ in spans:
        panel = build_contract_panel(d1, d2, c, freq=freq)
        if panel.empty: continue
        fac = factors.get(c)
        if fac and fac > 0 and np.isfinite(fac):
            panel["close_d1_scaled"] = panel["close_d1"] / fac
        out_path = os.path.join(out_dir, f"panel_{c}.csv")
        panel.to_csv(out_path, index_label="timestamp")
        paths.append(out_path)
    return paths

# ===========================
# Extra diagnostics
# ===========================

def lead_lag_diag(d1s: pd.DataFrame, d2: pd.DataFrame, windows: pd.DataFrame, max_lag=2) -> pd.DataFrame:
    rows = []
    for _, r in windows.iterrows():
        c=r["contract"]; t0=pd.to_datetime(r["start"]); t1=pd.to_datetime(r["end"])
        s1=d1s[(d1s["contract_norm"]==c)&(d1s.index>=t0)&(d1s.index<=t1)]
        s2=d2[(d2["contract_norm"]==c)&(d2.index>=t0)&(d2.index<=t1)]
        idx=s1.index.intersection(s2.index)
        x=np.log(s1.reindex(idx)["close_scaled"]).diff()
        y=np.log(s2.reindex(idx)["close"]).diff()
        act=(s1.reindex(idx)["trade_active"].astype("bool").reindex(idx, fill_value=False) &
             s2.reindex(idx)["active_any"].astype("bool").reindex(idx, fill_value=False))
        x, y = x[act], y[act]
        best_lag, best_corr = 0, np.nan
        for lag in range(-max_lag, max_lag+1):
            if lag<0: corr = x.shift(-lag).corr(y)  # x leads
            elif lag>0: corr = x.corr(y.shift(lag)) # y leads
            else: corr = x.corr(y)
            if pd.notna(corr) and (pd.isna(best_corr) or corr>best_corr):
                best_corr, best_lag = corr, lag
        rows.append({"contract": c, "best_lag": int(best_lag), "best_corr": float(best_corr)})
    return pd.DataFrame(rows).sort_values("contract")

def error_by_hour(d1s: pd.DataFrame, d2: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for _, r in windows.iterrows():
        c=r["contract"]; t0=pd.to_datetime(r["start"]); t1=pd.to_datetime(r["end"])
        s1=d1s[(d1s["contract_norm"]==c)&(d1s.index>=t0)&(d1s.index<=t1)]
        s2=d2[(d2["contract_norm"]==c)&(d2.index>=t0)&(d2.index<=t1)]
        idx=s1.index.intersection(s2.index)
        e=(s1.reindex(idx)["close_scaled"]-s2.reindex(idx)["close"])
        act=(s1.reindex(idx)["trade_active"].astype("bool").reindex(idx, fill_value=False) &
             s2.reindex(idx)["active_any"].astype("bool").reindex(idx, fill_value=False))
        df=pd.DataFrame({"err":e[act]}); df["hour"]=df.index.hour
        g=df.groupby("hour")["err"].agg(["count","mean","median","std"]).reset_index()
        g.insert(0,"contract",c); rows.append(g)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def rolling_ratio(d1s: pd.DataFrame, d2: pd.DataFrame, windows: pd.DataFrame, win: int = 24) -> pd.DataFrame:
    """Rolling median of (d1_scaled/d2) on matched+active bars (~2h if 5m bars and win=24)."""
    rows=[]
    for _, r in windows.iterrows():
        c=r["contract"]; t0=pd.to_datetime(r["start"]); t1=pd.to_datetime(r["end"])
        s1=d1s[(d1s["contract_norm"]==c)&(d1s.index>=t0)&(d1s.index<=t1)]
        s2=d2[(d2["contract_norm"]==c)&(d2.index>=t0)&(d2.index<=t1)]
        idx=s1.index.intersection(s2.index)
        x=s1.reindex(idx)["close_scaled"]; y=s2.reindex(idx)["close"]
        act=(s1.reindex(idx)["trade_active"].astype("bool").reindex(idx, fill_value=False) &
             s2.reindex(idx)["active_any"].astype("bool").reindex(idx, fill_value=False))
        z=(x/y)[act].replace([np.inf,-np.inf], np.nan).dropna()
        if z.empty: 
            continue
        rr=z.rolling(win, min_periods=max(5, win//2)).median()
        rows.append(pd.DataFrame({"timestamp": rr.index, "contract": c, "rolling_ratio": rr.values}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def plot_rolling_ratio(rr_df: pd.DataFrame, out_dir: str, limit: Optional[int] = 6):
    if rr_df.empty:
        return []
    paths=[]
    # choose contracts by number of points
    counts = rr_df["contract"].value_counts().sort_values(ascending=False)
    if limit is not None:
        counts = counts.head(limit)
    for c in counts.index:
        sub = rr_df[rr_df["contract"]==c].dropna(subset=["rolling_ratio"])
        if sub.empty: continue
        plt.figure(figsize=(12,4))
        plt.plot(sub["timestamp"], sub["rolling_ratio"])
        plt.title(f"{c}: Rolling median of (D1_scaled / D2)")
        plt.xlabel("Time"); plt.ylabel("Rolling ratio")
        path = f"{out_dir}/rolling_ratio_{_sanitize_contract_for_path(c)}.png"
        plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()
        paths.append(path)
    return paths

# ===========================
# Orchestrate
# ===========================

def run_pipeline(
    path1="raw_data/Data-1.csv",
    path2="raw_data/Data-2.csv",
    out_dir="artifacts",
    freq="5min",
):
    os.makedirs(out_dir, exist_ok=True)

    # 1) Load & clean
    df1_raw, df2_raw = load_data(path1, path2)
    d1, d2, alias_col = clean_inputs(df1_raw, df2_raw)

    # 2) Contract windows (intersection of first/last TRADE-ACTIVE)
    windows = per_contract_windows(d1, d2)
    windows_path = f"{out_dir}/contract_windows.csv"; windows.to_csv(windows_path, index=False)

    # 3) Per-contract scaling on matched+active bars
    factors = per_contract_scaling(d1, d2, windows)
    scaling_tbl = pd.DataFrame([{"contract":k, "scale_factor":v} for k,v in factors.items()]).sort_values("contract")
    scaling_path = f"{out_dir}/per_contract_scaling.csv"; scaling_tbl.to_csv(scaling_path, index=False)

    # 4) Apply scaling to df1
    d1s = apply_per_contract_scaling(d1, factors)

    # 5) Metrics on matched windows
    summary, detailed = compute_metrics(d1s, d2, windows)
    summary_path = f"{out_dir}/metrics_summary.csv"; summary.to_csv(summary_path, index=False)
    detailed_path = f"{out_dir}/matched_detail.csv"; detailed.to_csv(detailed_path, index=False)

    # 6) Roll diagnostics
    roll1 = compute_roll_dates(d1, "dataset1"); roll2 = compute_roll_dates(d2, "dataset2")
    rolls = pd.concat([roll1, roll2], ignore_index=True)
    rolls_path = f"{out_dir}/roll_dates.csv"; rolls.to_csv(rolls_path, index=False)

    # 7) Merged snapshot (compact intersection only; for quick inspection)
    idx = d1s.index.intersection(d2.index)
    merged = pd.DataFrame(index=idx)
    merged["contract_d1"] = d1s.reindex(idx)["contract_norm"]
    merged["contract_d2"] = d2.reindex(idx)["contract_norm"]
    merged["same_contract"] = merged["contract_d1"].eq(merged["contract_d2"])
    merged["close_d1_scaled"] = d1s.reindex(idx)["close_scaled"]
    merged["close_d2"] = d2.reindex(idx)["close"]
    merged["spread"] = merged["close_d1_scaled"] - merged["close_d2"]
    merged["vol_d1"] = d1s.reindex(idx)["volume"]
    merged["vol_d2"] = d2.reindex(idx)["volume"]
    merged_path = f"{out_dir}/merged_snapshot.csv"; merged.to_csv(merged_path)

    # 8) Heatmaps
    hm1 = plot_missing_heatmap(d1, "dataset1_futures", out_dir=out_dir, freq=freq, tz="UTC")
    hm2 = plot_missing_heatmap(d2, "dataset2_futures", out_dir=out_dir, freq=freq, tz="UTC")
    per_contract_hm1 = generate_per_contract_heatmaps(d1, "dataset1_futures", out_dir=out_dir, tz="UTC", limit=8)
    per_contract_hm2 = generate_per_contract_heatmaps(d2, "dataset2_futures", out_dir=out_dir, tz="UTC", limit=8)
    overlap_hms = plot_overlap_heatmaps(d1, d2, windows, out_dir=out_dir, tz="UTC", limit=8)

    # 9) Uptime
    up1 = uptime_overall(d1, freq=freq); up2 = uptime_overall(d2, freq=freq)
    up_overall = pd.DataFrame([dict(dataset="dataset1", **up1), dict(dataset="dataset2", **up2)])
    up_overall_path = f"{out_dir}/uptime_overall.csv"; up_overall.to_csv(up_overall_path, index=False)
    up_contract = uptime_per_contract(d1, d2, windows, freq=freq)
    up_contract_path = f"{out_dir}/uptime_per_contract.csv"; up_contract.to_csv(up_contract_path, index=False)

    # 10) Contract-centric union panels (preserve one-sided bars)
    panel_paths = export_contract_panels(d1, d2, factors, out_dir=out_dir, freq=freq, limit=8)

    # 11) Extra diagnostics
    ll = lead_lag_diag(d1s, d2, windows, max_lag=2)
    ll_path = f"{out_dir}/lead_lag.csv"; ll.to_csv(ll_path, index=False)
    e_hour = error_by_hour(d1s, d2, windows)
    e_hour_path = f"{out_dir}/error_by_hour.csv"; e_hour.to_csv(e_hour_path, index=False)
    rr = rolling_ratio(d1s, d2, windows, win=24)
    rr_path = f"{out_dir}/rolling_ratio.csv"; rr.to_csv(rr_path, index=False)
    rr_plots = plot_rolling_ratio(rr, out_dir=out_dir, limit=6)

    # 12) Metadata index
    meta = {
        "outputs": {
            "contract_windows": windows_path,
            "per_contract_scaling": scaling_path,
            "metrics_summary": summary_path,
            "matched_detail": detailed_path,
            "roll_dates": rolls_path,
            "merged_snapshot": merged_path,
            "heatmap_dataset1": hm1,
            "heatmap_dataset2": hm2,
            "heatmaps_per_contract_dataset1": per_contract_hm1,
            "heatmaps_per_contract_dataset2": per_contract_hm2,
            "overlap_heatmaps": overlap_hms,
            "uptime_overall": up_overall_path,
            "uptime_per_contract": up_contract_path,
            "contract_panels": panel_paths,
            "lead_lag": ll_path,
            "error_by_hour": e_hour_path,
            "rolling_ratio_csv": rr_path,
            "rolling_ratio_plots": rr_plots,
        },
        "n_rows": {"dataset1": int(len(d1)), "dataset2": int(len(d2))},
        "time_span": {
            "dataset1": [str(d1.index.min()), str(d1.index.max())] if len(d1) else None,
            "dataset2": [str(d2.index.min()), str(d2.index.max())] if len(d2) else None,
        },
        "notes": [
            "Per-contract scaling via median(d1_close/d2_close) on matched + active bars, winsorized (1–99%).",
            "Windows are intersection of first/last TRADE-ACTIVE times in both datasets per contract.",
            "Global heatmaps use canonical 5-min grid; per-contract heatmaps focus on inactivity; overlap heatmaps show both/only-one active fractions.",
            "Uptime = fraction of expected grid bars present/active; contract panels preserve one-sided activity.",
            "Diagnostics: lead/lag (best lag, corr), hour-of-day error profile, rolling ratio drift (plots)."
        ]
    }
    with open(f"{out_dir}/analysis_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Artifacts saved to:", out_dir)
    for k, v in meta["outputs"].items():
        print(f" - {k}: {v}")

if __name__ == "__main__":
    run_pipeline(
        path1=os.environ.get("DATASET1_PATH", "raw_data/Data-1.csv"),
        path2=os.environ.get("DATASET2_PATH", "raw_data/Data-2.csv"),
        out_dir=os.environ.get("OUT_DIR", "artifacts"),
        freq=os.environ.get("BAR_FREQ", "5min"),
    )
