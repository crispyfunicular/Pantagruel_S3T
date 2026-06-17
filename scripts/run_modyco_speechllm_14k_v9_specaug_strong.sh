#!/usr/bin/env bash
# Modyco — speechLLM L-14k + SpecAugment fort (run_045).
#
# Budget : ≤ 3 h GPU. Durée attendue ~2 h.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NOHUP_SCRIPT="${ROOT}/2_speechLLM/scripts/run_045_b1_utterance_large_14k_v9_specaug_strong_nohup.sh"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

require_gpu_free() {
  if pgrep -af "^python.*pipeline\.py (train|run)" >/dev/null 2>&1; then
    echo "ERROR: entraînement GPU encore actif sur Modyco :" >&2
    pgrep -af "^python.*pipeline\.py (train|run)" >&2 || true
    exit 2
  fi
  echo "OK: aucun pipeline.py train/run Python actif."
}

FORCE=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    -h|--help|help)
      sed -n '1,8p' "$0" | tail -n +2
      exit 0
      ;;
    *)
      echo "Argument inconnu: ${arg}" >&2
      exit 2
      ;;
  esac
done

if [[ "$FORCE" != "1" ]]; then
  require_gpu_free
fi

chmod +x "$NOHUP_SCRIPT"

echo "=== $(date -Is) Pré-vol speechLLM L-14k SpecAugment fort (Modyco, run_045) ==="
python 2_speechLLM/pipeline.py train \
  --config 2_speechLLM/configs/fr-en/b1_utterance_large_14k_v9_specaug_strong.yaml \
  --run-id run_045_speechllm_b1_utterance_large_14k_v9_specaug_strong \
  --dry-run

echo "=== $(date -Is) Délégation → ${NOHUP_SCRIPT} ==="
exec bash "$NOHUP_SCRIPT"
