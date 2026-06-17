#!/usr/bin/env bash
# Bench d'évaluation multi-variantes — protocole figé documentation/protocole_evaluation.md
#
# Lance uniquement les étapes ``evaluate`` (pas de train). Chaque run doit déjà
# disposer d'un checkpoint (ST / speechLLM) ou d'une clé API (Gemini).
#
# Usage :
#   cd S3T && source .venv/bin/activate
#   bash scripts/bench_evaluate_variants.sh              # sentence_like
#   bash scripts/bench_evaluate_variants.sh utterance
#   bash scripts/bench_evaluate_variants.sh --dry-run

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SEGMENT_MODE="sentence_like"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    utterance|sentence_like)
      SEGMENT_MODE="$1"
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      sed -n '1,20p' "$0"
      exit 0
      ;;
    *)
      echo "Argument inconnu: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

DRY_FLAG=()
if [[ "$DRY_RUN" -eq 1 ]]; then
  DRY_FLAG=(--dry-run)
fi

echo "=== Bench evaluate (segment_mode=${SEGMENT_MODE}, protocole 2026-06-02-v1) ==="

if [[ "$SEGMENT_MODE" == "utterance" ]]; then
  MANIFESTS="${ROOT}/datasets/manifests/fr-en"
  ST_CFG="1_Transformer/configs/fr-en/base_utterance.yaml"
  ST_RUN="run_002_transformer_baseline_utterance"
  SL_CFG="2_speechLLM/configs/fr-en/b1_utterance_long.yaml"
  SL_RUN="run_003_speechllm_b1_utterance_long"
  GEM_CFG="3_Gemini/configs/fr-en/gemini_flash_utterance.yaml"
  GEM_RUN="run_002_gemini_flash_utterance"
  CAS_CFG="4_cascade/configs/fr-en/cascade.yaml"
  CAS_RUN="run_001_cascade_utterance"
else
  MANIFESTS="${ROOT}/datasets/manifests_sentence/fr-en"
  ST_CFG="1_Transformer/configs/fr-en/base_sentence_like.yaml"
  ST_RUN="run_001_transformer_baseline_sentence_like"
  SL_CFG="2_speechLLM/configs/fr-en/b1_sentence_long.yaml"
  SL_RUN="run_002_speechllm_b1_sentence_long"
  GEM_CFG="3_Gemini/configs/fr-en/gemini_flash_sentence.yaml"
  GEM_RUN="run_001_gemini_flash_sentence_like_v2"
  CAS_CFG="4_cascade/configs/fr-en/cascade_sentence.yaml"
  CAS_RUN="run_001_cascade_sentence_like"
fi

for split in valid test; do
  if [[ ! -f "${MANIFESTS}/${split}.tsv" ]]; then
    echo "ERROR: ${MANIFESTS}/${split}.tsv absent (prepare --segment-mode ${SEGMENT_MODE})." >&2
    exit 2
  fi
done

run() {
  echo "--- $* ---"
  "$@"
}

# ST Table 8
run python 1_Transformer/pipeline.py evaluate \
  --config "${ST_CFG}" --run-id "${ST_RUN}" "${DRY_FLAG[@]}"

# speechLLM B1 (encodeur gelé — ajuster run/config pour run_005)
run python 2_speechLLM/pipeline.py evaluate \
  --config "${SL_CFG}" --run-id "${SL_RUN}" "${DRY_FLAG[@]}"

# Gemini 2.5 (nécessite GEMINI_API_KEY sauf --dry-run)
if [[ -z "${GEMINI_API_KEY:-}" && "$DRY_RUN" -eq 0 ]]; then
  echo "SKIP Gemini : GEMINI_API_KEY non définie."
else
  run python 3_Gemini/pipeline.py evaluate \
    --config "${GEM_CFG}" --run-id "${GEM_RUN}" "${DRY_FLAG[@]}"
fi

# Cascade
run python 4_cascade/pipeline.py evaluate \
  --config "${CAS_CFG}" --run-id "${CAS_RUN}" "${DRY_FLAG[@]}"

if [[ "$DRY_RUN" -eq 0 ]]; then
  run python scripts_communs/update_experiments_tracking.py --all
fi

echo "=== Bench evaluate terminé ==="
