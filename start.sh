#!/usr/bin/env bash
set -euo pipefail

echo "Checking /app contents..."
ls -la /app || true

echo "Starting Streamlit..."
exec bash -lc "streamlit run app.py --server.port 8501 --server.address 0.0.0.0"
