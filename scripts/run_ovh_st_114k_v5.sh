#!/usr/bin/env bash
# OVH — ST Pantagruel-L-114k v5 SpecAugment (run_028).
#
# Durée attendue : ~9–12 h GPU. À enchaîner après run_025 v4 (collapse probable).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ST_SCRIPT="${ROOT}/1_Transformer/scripts/run_028_baseline_utterance_large_114k_v5_nohup.sh"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

require_gpu_free() {
  if pgrep -af "^python.*pipeline\.py (train|run)" >/dev/null 2>&1; then
    echo "ERROR: entraînement GPU encore actif sur OVH :" >&2
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
      sed -n '1,10p' "$0" | tail -n +2
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

chmod +x "$ST_SCRIPT"

echo "=== $(date -Is) Pré-vol ST L-114k v5 SpecAugment (OVH, run_028) ==="
python 1_Transformer/pipeline.py train \
  --config 1_Transformer/configs/fr-en/base_utterance_large_114k_v5.yaml \
  --run-id run_028_transformer_baseline_utterance_large_114k_v5 \
  --dry-run

echo "=== $(date -Is) Délégation → ${ST_SCRIPT} ==="
exec bash "$ST_SCRIPT"
