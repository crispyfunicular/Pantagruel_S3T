#!/usr/bin/env bash
# Run 040 — Pantagruel Speech_Text utterance v2 (aligné run_026).
#
# Budget Modyco : ~2–3 h GPU.
#
# Lancement nohup (Modyco) :
#   nohup bash 5_Pantagruel_multimodal/scripts/run_040_base_utterance_v2_nohup.sh \
#     > logs/run_040_multimodal_utterance_v2_wrapper.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

CFG="5_Pantagruel_multimodal/configs/fr-en/base_utterance_v2.yaml"
RUN="run_040_pantagruel_multimodal_utterance_v2"
MANIFESTS="datasets/manifests/fr-en"
LOG_DIR="${ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/${RUN}_spm_train_eval.log"
SPM_MODEL="datasets/processed/spm/fr-en_1000.model"

for split in train valid test; do
  if [[ ! -f "${MANIFESTS}/${split}.tsv" ]]; then
    echo "ERROR: manquant ${MANIFESTS}/${split}.tsv — prepare utterance d'abord." >&2
    exit 2
  fi
done

{
  if [[ ! -f "$SPM_MODEL" ]]; then
    echo "=== $(date -Is) SPM utterance (vocab 1k) — absent, entraînement SPM ==="
    python 5_Pantagruel_multimodal/pipeline.py spm --config "$CFG" -v
  else
    echo "=== $(date -Is) SPM existant : ${SPM_MODEL} ==="
  fi

  echo "=== $(date -Is) TRAIN ${RUN} (Speech_Text utterance, recette run_026) ==="
  python 5_Pantagruel_multimodal/pipeline.py train --config "$CFG" --run-id "$RUN" -v

  echo "=== $(date -Is) EVALUATE ${RUN} (beam 5) ==="
  python 5_Pantagruel_multimodal/pipeline.py evaluate --config "$CFG" --run-id "$RUN" --beam-size 5 -v

  echo "=== $(date -Is) DONE ==="
} 2>&1 | tee -a "$LOG"
