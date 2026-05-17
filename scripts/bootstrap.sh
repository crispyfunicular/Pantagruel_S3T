#!/usr/bin/env bash
# Bootstrap environment for S3T (Phase 1 — PRD.md)
# Creates venv, installs requirements.txt, verifies torch/CUDA.

set -euo pipefail

EXIT_NOT_IMPLEMENTED=7

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="python3"
VENV_DIR="${PROJECT_ROOT}/.venv"
CUDA_INDEX_URL=""
LOCK_DEPS=false

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --python PATH           Python interpreter (default: python3)
  --venv PATH             Virtualenv directory (default: .venv)
  --with-cuda-index-url URL   PyTorch CUDA wheel index (optional)
  --lock                  Write requirements.lock.txt after install
  -h, --help              Show this help

Example:
  ./scripts/bootstrap.sh
  ./scripts/bootstrap.sh --with-cuda-index-url https://download.pytorch.org/whl/cu124 --lock
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --venv)
      VENV_DIR="$2"
      shift 2
      ;;
    --with-cuda-index-url)
      CUDA_INDEX_URL="$2"
      shift 2
      ;;
    --lock)
      LOCK_DEPS=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "${PROJECT_ROOT}"

echo "==> S3T bootstrap"
echo "    Project root: ${PROJECT_ROOT}"
echo "    Python:       ${PYTHON_BIN}"
echo "    Venv:         ${VENV_DIR}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: Python not found: ${PYTHON_BIN}" >&2
  exit 3
fi

# Create virtual environment
if [[ ! -d "${VENV_DIR}" ]]; then
  echo "==> Creating virtual environment..."
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
  echo "==> Virtual environment already exists."
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

echo "==> Upgrading pip..."
pip install --upgrade pip

echo "==> Installing dependencies from requirements.txt..."
if [[ -n "${CUDA_INDEX_URL}" ]]; then
  pip install torch --index-url "${CUDA_INDEX_URL}"
fi
pip install -r requirements.txt

if [[ "${LOCK_DEPS}" == true ]]; then
  echo "==> Writing requirements.lock.txt..."
  pip freeze > requirements.lock.txt
fi

echo "==> Verifying torch / CUDA..."
python -c "
import sys
import torch
print('torch version:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('cuda device:', torch.cuda.get_device_name(0))
sys.exit(0)
"

# Placeholder: Hugging Face Pantagruel weights download (not implemented yet)
download_pantagruel_weights() {
  echo "NotYetImplemented: download Pantagruel weights from Hugging Face (PantagrueLLM/)" >&2
  return ${EXIT_NOT_IMPLEMENTED}
}

# Uncomment when implemented:
# download_pantagruel_weights

echo ""
echo "Bootstrap complete. Activate with:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "Next steps:"
echo "  python scripts/pipeline.py preflight"
echo "  python scripts/pipeline.py run --langpair fr-es --run-id run_001"
