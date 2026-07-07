#!/usr/bin/env bash
# OVH — Reprise après timeout run_061 (14 h) sans éval ni run_062.
#
# Contexte (5 juil. 2026) : la chaîne 061→062 s'est arrêtée ~49,7k/80k updates
# (best dev ~20,44 @ 44k) ; run_052_transformer terminé (~46k, early stop) sans
# éval sur le serveur ; GPU libre.
#
# Étapes :
#   1. Éval beam 5 run_052_transformer (si pas déjà faite)
#   2. Éval beam 5 run_061 (best.pt)
#   3. run_062 speechLLM Mistral-7B 4-bit (~12–15 h)
#
# Usage :
#   cd ~/S3T && source .venv/bin/activate && mkdir -p logs
#   nohup bash scripts/run_ovh_recover_eval_052_061_then_062_mistral.sh \
#     > logs/run_ovh_recover_052_061_062.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

MAX_HOURS_MISTRAL="${MAX_HOURS_MISTRAL:-15}"

require_gpu_free() {
  if pgrep -af "^python.*pipeline\.py (train|run)" >/dev/null 2>&1; then
    echo "ERROR: entraînement GPU encore actif :" >&2
    pgrep -af "^python.*pipeline\.py (train|run)" >&2 || true
    exit 2
  fi
  echo "OK: GPU libre."
}

require_gpu_free
mkdir -p logs

CFG_052="1_Transformer/configs/fr-en/base_utterance_large_14k_v12_spm5k_freeze15k.yaml"
RUN_052="run_052_transformer_baseline_utterance_large_14k_v12_spm5k_freeze15k"

CFG_061="1_Transformer/configs/fr-en/base_utterance_large_114k_v12_spm5k_freeze15k.yaml"
RUN_061="run_061_transformer_baseline_utterance_large_114k_v12_spm5k_freeze15k"

CFG_SLM="2_speechLLM/configs/fr-en/b1_utterance_large_14k_mistral_7b_ovh.yaml"
RUN_SLM="run_062_speechllm_b2bis_utterance_large_14k_mistral_7b_ovh"
LOG_SLM="logs/${RUN_SLM}_train_eval.log"

eval_if_needed() {
  local cfg="$1"
  local run="$2"
  local label="$3"
  if [[ -f "runs/fr-en/${run}/eval/sacrebleu_test.txt" ]]; then
    echo "=== $(date -Is) ${label} — éval déjà présente : $(head -1 "runs/fr-en/${run}/eval/sacrebleu_test.txt") ==="
    return 0
  fi
  if [[ ! -f "runs/fr-en/${run}/checkpoints/best.pt" ]]; then
    echo "ERROR: checkpoint manquant pour ${run}" >&2
    exit 2
  fi
  echo "=== $(date -Is) ${label} — EVALUATE (beam 5) ==="
  python 1_Transformer/pipeline.py evaluate \
    --config "$cfg" \
    --run-id "$run" \
    --beam-size 5 -v
}

echo "=== $(date -Is) Reprise OVH — eval 052 + 061 puis run_062 ==="

eval_if_needed "$CFG_052" "$RUN_052" "run_052_transformer"
eval_if_needed "$CFG_061" "$RUN_061" "run_061 ST L-114k"

BLEU_052="n/a"
BLEU_061="n/a"
[[ -f "runs/fr-en/${RUN_052}/eval/sacrebleu_test.txt" ]] \
  && BLEU_052="$(head -1 "runs/fr-en/${RUN_052}/eval/sacrebleu_test.txt")"
[[ -f "runs/fr-en/${RUN_061}/eval/sacrebleu_test.txt" ]] \
  && BLEU_061="$(head -1 "runs/fr-en/${RUN_061}/eval/sacrebleu_test.txt")"
echo "=== $(date -Is) run_052 BLEU test : ${BLEU_052} ==="
echo "=== $(date -Is) run_061 BLEU test : ${BLEU_061} ==="

echo "=== $(date -Is) Dry-run ${RUN_SLM} ==="
python 2_speechLLM/pipeline.py train --config "$CFG_SLM" --run-id "$RUN_SLM" --dry-run

echo "=== $(date -Is) LANCEMENT ${RUN_SLM} (max ${MAX_HOURS_MISTRAL}h) ==="
{
  timeout "$((MAX_HOURS_MISTRAL * 3600))" \
    python 2_speechLLM/pipeline.py run \
      --config "$CFG_SLM" \
      --run-id "$RUN_SLM" \
      -v
  ec=$?
  if [[ "$ec" -eq 124 ]]; then
    echo "=== $(date -Is) TIMEOUT ${MAX_HOURS_MISTRAL}h — éval best checkpoint ==="
    python 2_speechLLM/pipeline.py evaluate \
      --config "$CFG_SLM" \
      --run-id "$RUN_SLM" -v || true
  elif [[ "$ec" -ne 0 ]]; then
    exit "$ec"
  fi
} 2>&1 | tee -a "$LOG_SLM"

BLEU_SLM="n/a"
[[ -f "runs/fr-en/${RUN_SLM}/eval/sacrebleu_test.txt" ]] \
  && BLEU_SLM="$(head -1 "runs/fr-en/${RUN_SLM}/eval/sacrebleu_test.txt")"

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  REPRISE OVH 052/061/062 TERMINÉE                                    ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║  run_052  ST L-14k gel 15k            : ${BLEU_052}"
echo "║  run_061  ST L-114k gel 15k (timeout) : ${BLEU_061}"
echo "║  run_062  speechLLM Mistral L-14k OVH  : ${BLEU_SLM}"
echo "╚══════════════════════════════════════════════════════════════════════╝"
