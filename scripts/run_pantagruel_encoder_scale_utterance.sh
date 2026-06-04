#!/usr/bin/env bash
# Bench encodeurs Pantagruel 14k / 114k — protocole utterance (Table 8).
#
# « 14k » et « 114k » = heures de pré-entraînement (LeBenchmark / INA), pas des corpus m-TEDx
# séparés : on réutilise les manifests utterance fr-en déjà préparés.
#
# Usage (depuis la racine S3T, venv activé) :
#   bash scripts/run_pantagruel_encoder_scale_utterance.sh smoke
#   bash scripts/run_pantagruel_encoder_scale_utterance.sh dry-run
#   bash scripts/run_pantagruel_encoder_scale_utterance.sh st-14k      # train+eval ST L-14k
#   bash scripts/run_pantagruel_encoder_scale_utterance.sh st-114k
#   bash scripts/run_pantagruel_encoder_scale_utterance.sh speechllm-14k
#   bash scripts/run_pantagruel_encoder_scale_utterance.sh speechllm-114k
#   bash scripts/run_pantagruel_encoder_scale_utterance.sh eval-all     # si checkpoints OK
#
# Entraînements longs (tour GPU, nohup) :
#   nohup bash 1_Transformer/scripts/run_010_baseline_utterance_14k_nohup.sh \
#     > logs/run_010_wrapper.log 2>&1 &
#   nohup bash 1_Transformer/scripts/run_011_baseline_utterance_114k_nohup.sh \
#     > logs/run_011_wrapper.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

MANIFESTS="${ROOT}/datasets/manifests/fr-en"

require_utterance_data() {
  for split in train valid test; do
    if [[ ! -f "${MANIFESTS}/${split}.tsv" ]]; then
      echo "ERROR: ${MANIFESTS}/${split}.tsv absent." >&2
      echo "  python scripts_communs/pipeline.py prepare --langpair fr-en" >&2
      exit 2
    fi
  done
  echo "OK: données utterance (${MANIFESTS})."
}

cmd_smoke() {
  python scripts/smoke_pantagruel_encoders.py --encoders 14k,114k
}

cmd_dry_run() {
  require_utterance_data
  python 1_Transformer/pipeline.py train \
    --config 1_Transformer/configs/fr-en/base_utterance_large_14k.yaml \
    --run-id run_010_transformer_baseline_utterance_large_14k --dry-run
  python 1_Transformer/pipeline.py train \
    --config 1_Transformer/configs/fr-en/base_utterance_large_114k.yaml \
    --run-id run_011_transformer_baseline_utterance_large_114k --dry-run
  python 2_speechLLM/pipeline.py train \
    --config 2_speechLLM/configs/fr-en/b1_utterance_large_14k.yaml \
    --run-id run_012_speechllm_b1_utterance_large_14k --dry-run
  python 2_speechLLM/pipeline.py train \
    --config 2_speechLLM/configs/fr-en/b1_utterance_large_114k.yaml \
    --run-id run_013_speechllm_b1_utterance_large_114k --dry-run
}

run_st_14k() {
  bash 1_Transformer/scripts/run_010_baseline_utterance_14k_nohup.sh
}

run_st_114k() {
  bash 1_Transformer/scripts/run_011_baseline_utterance_114k_nohup.sh
}

run_speechllm_14k() {
  require_utterance_data
  python 2_speechLLM/pipeline.py run \
    --config 2_speechLLM/configs/fr-en/b1_utterance_large_14k.yaml \
    --run-id run_012_speechllm_b1_utterance_large_14k -v
}

run_speechllm_114k() {
  require_utterance_data
  python 2_speechLLM/pipeline.py run \
    --config 2_speechLLM/configs/fr-en/b1_utterance_large_114k.yaml \
    --run-id run_013_speechllm_b1_utterance_large_114k -v
}

cmd_eval_all() {
  require_utterance_data
  for pair in \
    "1_Transformer/configs/fr-en/base_utterance_large_14k.yaml run_010_transformer_baseline_utterance_large_14k" \
    "1_Transformer/configs/fr-en/base_utterance_large_114k.yaml run_011_transformer_baseline_utterance_large_114k" \
    "2_speechLLM/configs/fr-en/b1_utterance_large_14k.yaml run_012_speechllm_b1_utterance_large_14k" \
    "2_speechLLM/configs/fr-en/b1_utterance_large_114k.yaml run_013_speechllm_b1_utterance_large_114k"; do
    # shellcheck disable=SC2086
    set -- $pair
    cfg="$1"
    run_id="$2"
    ckpt="${ROOT}/runs/fr-en/${run_id}/checkpoints/best.pt"
    if [[ ! -f "$ckpt" ]]; then
      echo "SKIP evaluate ${run_id} : pas de checkpoint ${ckpt}"
      continue
    fi
    if [[ "$cfg" == 1_Transformer/* ]]; then
      python 1_Transformer/pipeline.py evaluate --config "$cfg" --run-id "$run_id" --beam-size 5 -v
    else
      python 2_speechLLM/pipeline.py evaluate --config "$cfg" --run-id "$run_id" -v
    fi
  done
  python scripts_communs/update_experiments_tracking.py --all
}

usage() {
  sed -n '1,22p' "$0"
}

MODE="${1:-}"
case "$MODE" in
  smoke) cmd_smoke ;;
  dry-run) cmd_dry_run ;;
  st-14k) run_st_14k ;;
  st-114k) run_st_114k ;;
  speechllm-14k) run_speechllm_14k ;;
  speechllm-114k) run_speechllm_114k ;;
  eval-all) cmd_eval_all ;;
  -h|--help|help|"") usage ;;
  *)
    echo "Commande inconnue: ${MODE}" >&2
    usage >&2
    exit 2
    ;;
esac
