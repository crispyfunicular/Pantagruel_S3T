#!/usr/bin/env bash
# Exécuter sur Modyco : run_021 speechLLM L-14k v3 (~2–4 h GPU).
set -euo pipefail
cd ~/S3T
mkdir -p logs

if pgrep -af "python.*pipeline.py (train|run)" | grep -v pgrep >/dev/null 2>&1; then
  echo "Pipeline actif — abandon"
  pgrep -af pipeline.py | grep -v pgrep || true
  exit 3
fi

chmod +x scripts/run_modyco_speechllm_14k_v3.sh \
  2_speechLLM/scripts/run_021_b1_utterance_large_14k_v3_nohup.sh

RUN_DIR=~/S3T/runs/fr-en/run_021_speechllm_b1_utterance_large_14k_v3
EXTRA_ENV=()
if [[ -f "${RUN_DIR}/checkpoints/last.pt" || -f "${RUN_DIR}/checkpoints/best.pt" ]]; then
  if [[ -f "${RUN_DIR}/eval/sacrebleu_test.txt" ]]; then
    echo "Eval test déjà présente — rien à faire"
    cat "${RUN_DIR}/eval/sacrebleu_test.txt"
    exit 0
  fi
  echo "Checkpoint existant sans eval test — reprise (--resume)"
  EXTRA_ENV=(env RESUME=1)
else
  echo "Pas de checkpoint — run complet"
fi

echo "=== $(date -Is) Lancement run_021 speechLLM L-14k v3 (budget ≤4 h) ==="
nohup "${EXTRA_ENV[@]}" bash scripts/run_modyco_speechllm_14k_v3.sh \
  > logs/run_021_speechllm_14k_v3_chain_wrapper.log 2>&1 &
echo "PID=$!"
sleep 5
tail -20 logs/run_021_speechllm_14k_v3_chain_wrapper.log 2>/dev/null || true
tail -5 logs/run_021_speechllm_b1_utterance_large_14k_v3_train_eval.log 2>/dev/null || true
