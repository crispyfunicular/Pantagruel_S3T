#!/usr/bin/env bash
# Baseline ST Table 8 utterance v2 — gel encodeur 5k + early stopping + LR 1e-4.
#
# Successeur de run_002 (collapse mode, BLEU test 3,79). Durée estimée : ~2–4 h
# (early stop attendu vers ~20–24k updates vs 80k forcés).
#
# Lancement (tour GPU) :
#   cd ~/S3T && source .venv/bin/activate
#   mkdir -p logs
#   nohup bash 1_Transformer/scripts/run_004_baseline_utterance_v2_nohup.sh \
#     > logs/run_004_transformer_utterance_v2_wrapper.log 2>&1 &
#   echo $! > logs/run_004_transformer_utterance_v2.pid
#   tail -f logs/run_004_transformer_baseline_utterance_v2_spm_train_eval.log

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

CFG="1_Transformer/configs/fr-en/base_utterance_v2.yaml"
RUN="run_004_transformer_baseline_utterance_v2"
MANIFESTS="datasets/manifests/fr-en"
LOG_DIR="${ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/${RUN}_spm_train_eval.log"
SPM_MODEL="datasets/processed/spm/fr-en_1000.model"

for split in train valid test; do
  if [[ ! -f "${MANIFESTS}/${split}.tsv" ]]; then
    echo "ERROR: manquant ${MANIFESTS}/${split}.tsv" >&2
    exit 2
  fi
done

{
  if [[ ! -f "$SPM_MODEL" ]]; then
    echo "=== $(date -Is) SPM utterance (vocab 1k) ==="
    TARGET_TXT="${MANIFESTS}/train.target.txt"
    python 1_Transformer/3_spm.py \
      --langpair fr-en \
      --vocab-size 1000 \
      --manifests-root datasets/manifests \
      --train-text "${TARGET_TXT}" \
      --overwrite
  else
    echo "=== $(date -Is) SPM existant : ${SPM_MODEL} ==="
  fi

  echo "=== $(date -Is) TRAIN ${RUN} (v2: freeze 5k, early stop, LR 1e-4) ==="
  python 1_Transformer/pipeline.py train --config "$CFG" --run-id "$RUN" -v

  echo "=== $(date -Is) EVALUATE ${RUN} (greedy, beam 5 journalisé) ==="
  python 1_Transformer/pipeline.py evaluate --config "$CFG" --run-id "$RUN" --beam-size 5 -v

  echo "=== $(date -Is) DONE ==="
} 2>&1 | tee -a "$LOG"
