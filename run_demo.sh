#!/usr/bin/env bash
set -e
source venv/bin/activate
echo "[*] Starting ZeroGate v4 Engine & SOC Telemetry Dashboard on http://localhost:8000 ..."
uvicorn backend.main:app --host 0.0.0.0 --port 8000
