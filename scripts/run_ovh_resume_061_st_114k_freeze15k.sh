#!/usr/bin/env bash
# OVH — Reprise entraînement run_061 (ST L-114k gel 15k) après timeout 14 h @ ~48–50k/80k.
#
# Le checkpoint ``last.pt`` conserve update, optimiseur, best BLEU dev et compteur
# early-stop. Reprendre avec ``--resume`` (pas ``--overwrite``).
#
# Estimation : ~30k updates restants × ~1 s/update ≈ 8–10 h GPU ; marge 20 h.
#
# Usage (GPU libre) :
#   cd ~/S3T && source .venv/bin/activate && mkdir -p logs
#   nohup bash scripts/run_ovh_resume_061_st_114k_freeze15k.sh \
#     > logs/run_061_resume_train_eval.log 2>&1 &
#
# Après éval finale, enchaîner run_062 Mistral (GPU libre) :
#   nohup bash scripts/run_ovh_chain_061_st_114k_freeze15k_then_062_mistral.sh \
#     > logs/run_062_only.log 2>&1 &
#   # (échec si run_061 checkpoints présents sans --resume — préférer un script run_062 seul)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

MAX_HOURS_ST="${MAX_HOURS_ST:-20}"

CFG_ST="1_Transformer/configs/fr-en/base_utterance_large_114k_v12_spm5k_freeze15k.yaml"
RUN_ST="run_061_transformer_baseline_utterance_large_114k_v12_spm5k_freeze15k"
LOG_ST="logs/${RUN_ST}_resume_train_eval.log"
CKPT_LAST="runs/fr-en/${RUN_ST}/checkpoints/last.pt"

if pgrep -af "^python.*pipeline\.py (train|run)" >/dev/null 2>&1; then
  echo "ERROR: entraînement GPU encore actif :" >&2
  pgrep -af "^python.*pipeline\.py (train|run)" >&2 || true
  exit 2
fi

if [[ ! -f "$CKPT_LAST" ]]; then
  echo "ERROR: checkpoint manquant : ${CKPT_LAST}" >&2
  exit 2
fi

echo "=== $(date -Is) Reprise ${RUN_ST} depuis last.pt (max ${MAX_HOURS_ST}h) ==="
python 1_Transformer/pipeline.py train \
  --config "$CFG_ST" \
  --run-id "$RUN_ST" \
  --resume \
  --dry-run

echo "=== $(date -Is) LANCEMENT train --resume ==="
{
  timeout "$((MAX_HOURS_ST * 3600))" \
    python 1_Transformer/pipeline.py train \
      --config "$CFG_ST" \
      --run-id "$RUN_ST" \
      --resume \
      -v
  ec=$?
  if [[ "$ec" -eq 124 ]]; then
    echo "=== $(date -Is) TIMEOUT ${MAX_HOURS_ST}h — éval best checkpoint ==="
    python 1_Transformer/pipeline.py evaluate \
      --config "$CFG_ST" \
      --run-id "$RUN_ST" \
      --beam-size 5 -v || true
  elif [[ "$ec" -ne 0 ]]; then
    exit "$ec"
  else
    echo "=== $(date -Is) EVALUATE ${RUN_ST} (beam 5) ==="
    python 1_Transformer/pipeline.py evaluate \
      --config "$CFG_ST" \
      --run-id "$RUN_ST" \
      --beam-size 5 -v
  fi
} 2>&1 | tee -a "$LOG_ST"

BLEU_ST="n/a"
if [[ -f "runs/fr-en/${RUN_ST}/eval/sacrebleu_test.txt" ]]; then
  BLEU_ST="$(head -1 "runs/fr-en/${RUN_ST}/eval/sacrebleu_test.txt")"
fi
echo "=== $(date -Is) ${RUN_ST} — BLEU test : ${BLEU_ST} ==="
