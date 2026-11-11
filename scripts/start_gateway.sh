#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="/home/aida/vllm/private-llm-platform"
PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"

exec "${PYTHON_BIN}" -m uvicorn private_llm_platform.gateway.app:app \
  --app-dir "${PROJECT_DIR}/src" \
  --host 127.0.0.1 \
  --port 8080
