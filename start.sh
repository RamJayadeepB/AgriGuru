#!/usr/bin/env bash
set -euo pipefail

# Path to expected model file(s)
MODEL_PATH="/app/model.joblib"
ENCODER_PATH="/app/encoder.joblib"
COLUMNS_PATH="/app/model_columns.joblib"

echo "Checking model files in /app..."
ls -la /app || true

# Helper to download a file if missing
download_if_missing() {
  local target="$1"
  local url="$2"

  if [ -f "$target" ]; then
    echo "$target already exists — skip download."
    return 0
  fi

  if [ -z "$url" ]; then
    echo "ERROR: $target not found and no URL provided."
    return 1
  fi

  echo "Downloading $target from $url ..."
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail -o "$target" "$url"
  else
    wget -q -O "$target" "$url"
  fi
  echo "Downloaded $target"
}

# If model artifacts are missing, try to fetch via env vars (not used for Option A)
if [ ! -f "$MODEL_PATH" ] || [ ! -f "$ENCODER_PATH" ] || [ ! -f "$COLUMNS_PATH" ]; then
  echo "One or more model artifacts missing."
  if [ -n "${MODEL_BUNDLE_URL:-}" ]; then
    echo "MODEL_BUNDLE_URL provided — downloading bundle..."
    tmp="/tmp/model_bundle.zip"
    if command -v curl >/dev/null 2>&1; then
      curl -L --fail -o "$tmp" "$MODEL_BUNDLE_URL"
    else
      wget -q -O "$tmp" "$MODEL_BUNDLE_URL"
    fi
    echo "Unzipping bundle..."
    unzip -o "$tmp" -d /app || true
    rm -f "$tmp"
  else
    # Individual URLs fallback
    if [ -n "${MODEL_URL:-}" ] && [ ! -f "$MODEL_PATH" ]; then
      download_if_missing "$MODEL_PATH" "$MODEL_URL" || exit 1
    fi
    if [ -n "${ENCODER_URL:-}" ] && [ ! -f "$ENCODER_PATH" ]; then
      download_if_missing "$ENCODER_PATH" "$ENCODER_URL" || exit 1
    fi
    if [ -n "${COLUMNS_URL:-}" ] && [ ! -f "$COLUMNS_PATH" ]; then
      download_if_missing "$COLUMNS_PATH" "$COLUMNS_URL" || exit 1
    fi
  fi
fi

# Final check
if [ ! -f "$MODEL_PATH" ] || [ ! -f "$ENCODER_PATH" ] || [ ! -f "$COLUMNS_PATH" ]; then
  echo "ERROR: one or more model artifacts are still missing after attempted download:"
  ls -la /app
  exit 1
fi

echo "Starting Streamlit on port ${STREAMLIT_SERVER_PORT:-8501} ..."
exec bash -lc "streamlit run app.py --server.port ${STREAMLIT_SERVER_PORT:-8501} --server.address ${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}"
