#!/usr/bin/env bash
# Modyco — speechLLM B1 B-1k dégel utterance (run_006, piste H).
#
# Budget : ~3–4 h GPU.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SPEECHLLM_SCRIPT="${ROOT}/2_speechLLM/scripts/run_006_b1_utterance_unfreeze_nohup.sh"

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
  echo "OK: aucun pipeline.py train/run actif."
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

chmod +x "$SPEECHLLM_SCRIPT"

echo "=== $(date -Is) Pré-vol speechLLM B-1k unfreeze utterance (Modyco, run_006) ==="
python 2_speechLLM/pipeline.py train \
  --config 2_speechLLM/configs/fr-en/b1_utterance_unfreeze.yaml \
  --run-id run_006_speechllm_b1_utterance_unfreeze \
  --dry-run

echo "=== $(date -Is) Délégation → ${SPEECHLLM_SCRIPT} ==="
exec bash "$SPEECHLLM_SCRIPT"
