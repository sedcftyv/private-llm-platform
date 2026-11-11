#!/usr/bin/env bash

set -euo pipefail

VLLM_BIN="/home/aida/vllm/.venv/bin/vllm"
MODEL_DIR="/home/aida/vllm/Qwen3-4B"

export VLLM_USE_FLASHINFER_SAMPLER=0

exec "${VLLM_BIN}" serve "${MODEL_DIR}" \
  --served-model-name qwen3-4b \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype half \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.75 \
  -O0
