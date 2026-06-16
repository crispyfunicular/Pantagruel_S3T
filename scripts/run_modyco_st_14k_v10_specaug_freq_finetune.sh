#!/usr/bin/env bash
# Modyco — finetune ST L-14k v10 SpecAugment freq depuis run_026 (run_041, ~3,5 h GPU).
#
# Piste amélioration run_026 : masquage fréquentiel en plus du temporel v5,
# initialisé depuis best.pt run_026 (26,12 BLEU test).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NOHUP_SCRIPT="${ROOT}/1_Transformer/scripts/run_041_finetune_utterance_large_14k_v10_specaug_freq_nohup.sh"
INIT_CKPT="${ROOT}/runs/fr-en/run_026_transformer_baseline_utterance_large_14k_v5/checkpoints/best.pt"

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
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
  if [[ -n "${used}" && "${used}" -ge 8192 ]]; then
    echo "ERROR: VRAM ${used} MiB occupée (≥ 8192) — GPU non disponible." >&2
    exit 2
  fi
  echo "OK: GPU libre (VRAM ${used:-?} MiB)."
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

if [[ ! -f "$INIT_CKPT" ]]; then
  echo "ERROR: checkpoint run_026 manquant : ${INIT_CKPT}" >&2
  exit 2
fi

if [[ "$FORCE" != "1" ]]; then
  require_gpu_free
fi

chmod +x "$NOHUP_SCRIPT"

echo "=== $(date -Is) Pré-vol finetune L-14k v10 SpecAugment freq (Modyco, run_041) ==="
python 1_Transformer/pipeline.py train \
  --config 1_Transformer/configs/fr-en/base_utterance_large_14k_v10_specaug_freq_finetune.yaml \
  --run-id run_041_transformer_finetune_utterance_large_14k_v10_specaug_freq_from_run026 \
  --resume \
  --resume-from "$INIT_CKPT" \
  --dry-run

echo "=== $(date -Is) Délégation → ${NOHUP_SCRIPT} ==="
exec bash "$NOHUP_SCRIPT"
