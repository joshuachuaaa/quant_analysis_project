# backend/api_server.py
import os
import json
from functools import lru_cache
from pathlib import Path
from typing import Optional, Dict

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

ARTIFACTS_DIR = Path(os.getenv("OUT_DIR", "artifacts")).resolve()
TABLES_DIR = ARTIFACTS_DIR / "tables"
PANELS_DIR = ARTIFACTS_DIR / "panels"
HEATMAPS_DIR = ARTIFACTS_DIR / "heatmaps"
DIAG_DIR = ARTIFACTS_DIR / "diagnostics"

app = FastAPI(title="Quant Analysis API", version="1.0")

# CORS (adjust for your front-end origin if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static artifacts (images, xlsx, csvs) directly
if ARTIFACTS_DIR.exists():
    app.mount("/artifacts", StaticFiles(directory=str(ARTIFACTS_DIR)), name="artifacts")


def _csv_to_json(path: Path):
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Not found: {path.name}")
    try:
        df = pd.read_csv(path)
        return json.loads(df.to_json(orient="records", date_format="iso"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed reading {path.name}: {e}")

@lru_cache(maxsize=1)
def _load_meta() -> Dict:
    p = ARTIFACTS_DIR / "analysis_metadata.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="analysis_metadata.json not found: run analysis first.")
    with p.open("r") as f:
        return json.load(f)

def _contract_list_from_windows():
    win = TABLES_DIR / "contract_windows.csv"
    if not win.exists():
        return []
    df = pd.read_csv(win)
    return sorted(list({c for c in df.get("contract", []) if isinstance(c, str)}))

def _sanitize_contract_for_path(c: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(c))

@app.get("/api/meta")
def get_meta():
    return _load_meta()

@app.get("/api/contracts")
def get_contracts():
    # prefer windows, fallback to scaling table
    contracts = _contract_list_from_windows()
    if not contracts:
        scaling = TABLES_DIR / "per_contract_scaling.csv"
        if scaling.exists():
            df = pd.read_csv(scaling)
            contracts = sorted(list(df["contract"].dropna().unique()))
    return {"contracts": contracts}

@app.get("/api/metrics")
def get_metrics():
    return _csv_to_json(TABLES_DIR / "metrics_summary.csv")

@app.get("/api/windows")
def get_windows():
    return _csv_to_json(TABLES_DIR / "contract_windows.csv")

@app.get("/api/uptime/overall")
def get_uptime_overall():
    return _csv_to_json(TABLES_DIR / "uptime_overall.csv")

@app.get("/api/uptime/contracts")
def get_uptime_contracts():
    return _csv_to_json(TABLES_DIR / "uptime_per_contract.csv")

@app.get("/api/leadlag")
def get_leadlag():
    return _csv_to_json(TABLES_DIR / "lead_lag.csv")

@app.get("/api/error_by_hour")
def get_error_by_hour():
    return _csv_to_json(TABLES_DIR / "error_by_hour.csv")

@app.get("/api/metrics/rolling_ratio")
def get_rolling_ratio():
    return _csv_to_json(TABLES_DIR / "rolling_ratio.csv")

@app.get("/api/panel")
def get_panel(contract: str = Query(..., description="Normalized contract, e.g. K2025")):
    safe = _sanitize_contract_for_path(contract)
    path = PANELS_DIR / f"panel_{safe}.csv"
    return _csv_to_json(path)

@app.get("/api/report.xlsx")
def get_report_xlsx():
    path = TABLES_DIR / "report.xlsx"
    if not path.exists():
        raise HTTPException(status_code=404, detail="report.xlsx not found")
    return FileResponse(str(path), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename="report.xlsx")

@app.get("/health")
def health():
    ok = ARTIFACTS_DIR.exists()
    return {"status": "ok" if ok else "missing_artifacts_dir", "artifacts_dir": str(ARTIFACTS_DIR)}
