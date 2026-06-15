#!/usr/bin/env bash
# Modyco — Pantagruel Speech_Text utterance v2 (run_040).
#
# Budget : ~2–3 h GPU.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NOHUP_SCRIPT="${ROOT}/5_Pantagruel_multimodal/scripts/run_040_base_utterance_v2_nohup.sh"

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

echo "=== $(date -Is) Pré-vol Speech_Text utterance v2 (Modyco, run_040) ==="
python 5_Pantagruel_multimodal/pipeline.py train \
  --config 5_Pantagruel_multimodal/configs/fr-en/base_utterance_v2.yaml \
  --run-id run_040_pantagruel_multimodal_utterance_v2 \
  --dry-run

echo "=== $(date -Is) Délégation → ${NOHUP_SCRIPT} ==="
exec bash "$NOHUP_SCRIPT"
