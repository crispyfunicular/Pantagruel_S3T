#!/usr/bin/env bash
# Exécuter sur Modyco : lance la chaîne 066→023 si idle (repart de zéro).
set -euo pipefail
cd ~/S3T
mkdir -p logs

is_running() {
  pgrep -af "$1" 2>/dev/null | grep -v "pgrep -af" | grep -v "bash -s" | grep -q .
}

if is_running "run_modyco_wait_chain_066_llama_then_replicate.sh"; then
  echo "Waiter chain déjà actif :"
  pgrep -af "run_modyco_wait_chain_066_llama_then_replicate.sh" | grep -v pgrep || true
  tail -15 logs/run_waiter_chain_066_replicate_modyco.log 2>/dev/null || true
  exit 0
fi

if is_running "python.*pipeline.py (train|run)"; then
  echo "Pipeline actif — abandon"
  pgrep -af pipeline.py | grep -v pgrep || true
  exit 3
fi

chmod +x \
  scripts/run_modyco_wait_chain_066_llama_then_replicate.sh \
  scripts/run_modyco_wait_gpu_then_speechllm_066_llama_unfreeze.sh \
  scripts/run_modyco_speechllm_066_llama_unfreeze.sh \
  scripts/run_modyco_speechllm_14k_replicate.sh \
  2_speechLLM/scripts/run_023_b1_utterance_large_14k_replicate_nohup.sh

MAX_H="${MAX_HOURS_SLM:-10}"
echo "=== $(date -Is) Lancement chaîne 066→023 (MAX_HOURS_SLM=${MAX_H}, OVERWRITE=1, ~12h total) ==="
nohup env MAX_HOURS_SLM="${MAX_H}" OVERWRITE=1 bash scripts/run_modyco_wait_chain_066_llama_then_replicate.sh \
  > logs/run_waiter_chain_066_replicate_modyco.log 2>&1 &
echo "WAITER_PID=$!"
sleep 5
tail -25 logs/run_waiter_chain_066_replicate_modyco.log
pgrep -af "run_modyco_wait_chain_066_llama_then_replicate.sh" | grep -v pgrep || true
