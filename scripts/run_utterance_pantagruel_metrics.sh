#!/usr/bin/env bash
# Métriques SacreBLEU sur segmentation utterance (protocole Pantagruel / Table 8).
#
# Évaluations sans ré-entraînement (GPU ou API) :
#   - Cascade ASR→MT (Whisper large-v3 + Marian)
#   - Gemini 2.5 Flash (si GEMINI_API_KEY)
#
# L'entraînement ST / speechLLM utterance est lancé à part (long) :
#   - 1_Transformer/scripts/run_002_baseline_utterance_nohup.sh
#   - 2_speechLLM : b1_utterance_long.yaml (run_003_…)
#
# Usage :
#   cd ~/S3T && source .venv/bin/activate
#   bash scripts/run_utterance_pantagruel_metrics.sh          # cascade + gemini
#   bash scripts/run_utterance_pantagruel_metrics.sh cascade  # cascade seul
#   bash scripts/run_utterance_pantagruel_metrics.sh gemini   # gemini seul

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

MANIFESTS="datasets/manifests/fr-en"
LOG_DIR="${ROOT}/logs"
mkdir -p "$LOG_DIR"

require_utterance_data() {
  for split in valid test; do
    if [[ ! -f "${MANIFESTS}/${split}.tsv" ]]; then
      echo "ERROR: ${MANIFESTS}/${split}.tsv absent." >&2
      echo "Préparer les données utterance (défaut m-TEDx) :" >&2
      echo "  python scripts_communs/pipeline.py prepare --langpair fr-en" >&2
      echo "  # ou : python scripts_communs/2_prepare.py --langpair fr-en --segment-mode utterance" >&2
      exit 2
    fi
  done
  echo "OK: manifests utterance présents (${MANIFESTS})."
}

run_cascade() {
  local log="${LOG_DIR}/run_001_cascade_utterance_evaluate.log"
  {
    echo "=== $(date -Is) CASCADE utterance run_001_cascade_utterance ==="
    python 4_cascade/pipeline.py evaluate \
      --config 4_cascade/configs/fr-en/cascade.yaml \
      --run-id run_001_cascade_utterance \
      -v
    echo "=== $(date -Is) DONE cascade utterance ==="
  } 2>&1 | tee -a "$log"
}

run_gemini() {
  if [[ -z "${GEMINI_API_KEY:-}" ]]; then
    echo "SKIP Gemini : GEMINI_API_KEY non définie." >&2
    return 0
  fi
  local log="${LOG_DIR}/run_002_gemini_flash_utterance_evaluate.log"
  {
    echo "=== $(date -Is) GEMINI utterance run_002_gemini_flash_utterance ==="
    python 3_Gemini/pipeline.py evaluate \
      --config 3_Gemini/configs/fr-en/gemini_flash_utterance.yaml \
      --run-id run_002_gemini_flash_utterance \
      -v
    echo "=== $(date -Is) DONE gemini utterance ==="
  } 2>&1 | tee -a "$log"
}

require_utterance_data

MODE="${1:-all}"
case "$MODE" in
  all)
    run_cascade
    run_gemini
    ;;
  cascade)
    run_cascade
    ;;
  gemini)
    run_gemini
    ;;
  *)
    echo "Usage: $0 [all|cascade|gemini]" >&2
    exit 2
    ;;
esac

echo "=== $(date -Is) Fin évaluations utterance (hors ST/speechLLM train) ==="
echo "Baseline ST Table 8 : nohup bash 1_Transformer/scripts/run_002_baseline_utterance_nohup.sh"
