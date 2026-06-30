#!/usr/bin/env bash
# Modyco — speechLLM L-14k 2e seed (run_050, piste F/H).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
SCRIPT="${ROOT}/2_speechLLM/scripts/run_050_b1_utterance_large_14k_seed2_nohup.sh"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

if ! pgrep -af "^python.*pipeline\.py (train|run)" >/dev/null 2>&1; then
  echo "OK: GPU libre."
else
  echo "ERROR: GPU occupée" >&2
  pgrep -af "^python.*pipeline\.py (train|run)" >&2 || true
  exit 2
fi

chmod +x "$SCRIPT"
python 2_speechLLM/pipeline.py train \
  --config 2_speechLLM/configs/fr-en/b1_utterance_large_14k_seed2.yaml \
  --run-id run_050_speechllm_b1_utterance_large_14k_seed2 \
  --dry-run
exec bash "$SCRIPT"
