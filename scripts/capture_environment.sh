#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="/home/aida/vllm/private-llm-platform"
PYTHON_BIN="/home/aida/vllm/.venv/bin/python"
VLLM_BIN="/home/aida/vllm/.venv/bin/vllm"
MODEL_DIR="/home/aida/vllm/Qwen3-4B"
OUTPUT_FILE="${PROJECT_DIR}/artifacts/day1/environment.txt"

mkdir -p "${PROJECT_DIR}/artifacts/day1"

{
  date --iso-8601=seconds
  uname -a
  nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap,power.limit \
    --format=csv,noheader
  "${PYTHON_BIN}" -c 'import json, platform, sys, torch, vllm; print(json.dumps({"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__, "torch_cuda": torch.version.cuda, "vllm": vllm.__version__, "cuda_available": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None, "memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2) if torch.cuda.is_available() else None}, ensure_ascii=False, indent=2))'
  "${VLLM_BIN}" --version
  "${PYTHON_BIN}" -c 'from transformers import AutoConfig, AutoTokenizer; p="/home/aida/vllm/Qwen3-4B"; c=AutoConfig.from_pretrained(p, trust_remote_code=False); t=AutoTokenizer.from_pretrained(p, trust_remote_code=False); print({"config_class": type(c).__name__, "model_type": c.model_type, "architectures": c.architectures, "declared_dtype": str(c.torch_dtype), "tokenizer_class": type(t).__name__, "vocab_size": t.vocab_size})'
  du -sh "${MODEL_DIR}"
  df -h "${PROJECT_DIR}"
} >"${OUTPUT_FILE}" 2>&1

echo "Environment report written to ${OUTPUT_FILE}"
