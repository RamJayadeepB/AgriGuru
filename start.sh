#!/usr/bin/env bash
set -e

# Path to expected model file(s)
MODEL_PATH="/app/model.joblib"
ENCODER_PATH="/app/encoder.joblib"
COLUMNS_PATH="/app/model_columns.joblib"

# MODEL_URL environment variable expected if model is not in repo.
# e.g. MODEL_URL="https://my-bucket.s3.amazonaws.com/models/crop-model.joblib"
MODEL_URL="${MODEL_URL:-}"

download_if_missing() {
  local target=$1
  local url=$2

  if [ -f "$target" ]; then
    echo "$target already exists — skip download."
    return 0
  fi

  if [ -z "$url" ]; then
    echo "ERROR: $target not found and no URL provided."
    return 1
  fi

  echo "Downloading $target from $url ..."
  # Try curl then wget
  if command -v curl >/dev/null 2>&1; then
    curl -L -o "$target" "$url"
  else
    wget -O "$target" "$url"
  fi
  echo "Downloaded $target"
}

# If model files are missing but MODEL_URL is set, download
# Expect MODEL_URL to point to a zip containing model.joblib, encoder.joblib, model_columns.joblib
# or point directly to model.joblib (in which case you should also set ENCODER_URL, COLUMNS_URL)
if [ ! -f "$MODEL_PATH" ] || [ ! -f "$ENCODER_PATH" ] || [ ! -f "$COLUMNS_PATH" ]; then
  echo "One or more model artifact(s) missing."

  # If MODEL_BUNDLE_URL provided (zip), download and unzip
  if [ -n "${MODEL_BUNDLE_URL:-}" ]; then
    echo "MODEL_BUNDLE_URL is set. Downloading bundle..."
    tmp="/tmp/model_bundle.zip"
    if command -v curl >/dev/null 2>&1; then
      curl -L -o "$tmp" "$MODEL_BUNDLE_URL"
    else
      wget -O "$tmp" "$MODEL_BUNDLE_URL"
    fi
    echo "Unzipping bundle..."
    unzip -o "$tmp" -d /app || true
    rm -f "$tmp"
  else
    # Fallback: download individually if specific URLs provided
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

# Final presence check
if [ ! -f "$MODEL_PATH" ] || [ ! -f "$ENCODER_PATH" ] || [ ! -f "$COLUMNS_PATH" ]; then
  echo "ERROR: one or more model artifacts still missing after attempted download:"
  ls -la /app
  exit 1
fi

# Start streamlit
echo "Starting Streamlit..."
exec streamlit run app.py --server.port $STREAMLIT_SERVER_PORT --server.address $STREAMLIT_SERVER_ADDRESS
