#!/usr/bin/env bash
# OVH — ST L-114k v13 : SpecAugment temporel fort 0.15 (run_064, piste A+C).
#
# Base run_033 (**25,10** test). Changement unique : mask_time_prob 0.05 → 0.15.
# ~10–20 h GPU sur V100 32 Go ; timeout 20 h + éval si coupure.
#
# Usage :
#   cd ~/S3T && source .venv/bin/activate && mkdir -p logs
#   nohup bash scripts/run_ovh_st_064_l114k_heavy_specaug.sh \
#     > logs/run_064_launch.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

MAX_HOURS_ST="${MAX_HOURS_ST:-20}"

CFG="1_Transformer/configs/fr-en/base_utterance_large_114k_v13_heavy_specaug.yaml"
RUN_ID="run_064_transformer_baseline_utterance_large_114k_v13_heavy_specaug"
LOG="logs/${RUN_ID}_train_eval.log"
SPM_MODEL="datasets/processed/spm/fr-en_5000.model"

if pgrep -af "^python.*pipeline\.py (train|run)" >/dev/null 2>&1; then
  echo "ERROR: GPU occupé :" >&2
  pgrep -af "^python.*pipeline\.py (train|run)" >&2 || true
  exit 2
fi

ensure_spm_5k() {
  if [[ -f "$SPM_MODEL" ]]; then
    return 0
  fi
  local target_txt="datasets/manifests/fr-en/train.target.txt"
  python 1_Transformer/3_spm.py \
    --langpair fr-en \
    --vocab-size 5000 \
    --manifests-root datasets/manifests \
    --train-text "$target_txt" \
    --overwrite
}

echo "=== $(date -Is) Préparation SPM 5k ==="
ensure_spm_5k

echo "=== $(date -Is) Dry-run ${RUN_ID} ==="
python 1_Transformer/pipeline.py train --config "$CFG" --run-id "$RUN_ID" --dry-run

echo "=== $(date -Is) LANCEMENT ${RUN_ID} (max ${MAX_HOURS_ST}h) ==="
{
  timeout "$((MAX_HOURS_ST * 3600))" \
    python 1_Transformer/pipeline.py train \
      --config "$CFG" \
      --run-id "$RUN_ID" \
      -v
  ec=$?
  if [[ "$ec" -eq 124 ]]; then
    echo "=== $(date -Is) TIMEOUT ${MAX_HOURS_ST}h — éval best checkpoint ==="
    python 1_Transformer/pipeline.py evaluate \
      --config "$CFG" \
      --run-id "$RUN_ID" \
      --beam-size 5 -v || true
  elif [[ "$ec" -ne 0 ]]; then
    exit "$ec"
  else
    echo "=== $(date -Is) EVALUATE ${RUN_ID} (beam 5) ==="
    python 1_Transformer/pipeline.py evaluate \
      --config "$CFG" \
      --run-id "$RUN_ID" \
      --beam-size 5 -v
  fi
} 2>&1 | tee -a "$LOG"

BLEU="n/a"
[[ -f "runs/fr-en/${RUN_ID}/eval/sacrebleu_test.txt" ]] \
  && BLEU="$(head -1 "runs/fr-en/${RUN_ID}/eval/sacrebleu_test.txt")"
echo "=== $(date -Is) ${RUN_ID} — BLEU test : ${BLEU} ==="
