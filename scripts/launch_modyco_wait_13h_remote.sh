#!/usr/bin/env bash
# Exécuter sur Modyco : waiter GPU 13 h (run_044 → run_046).
set -euo pipefail
cd ~/S3T
mkdir -p logs

is_running() {
  pgrep -af "$1" 2>/dev/null | grep -v "pgrep -af" | grep -v "bash -s" | grep -q .
}

if is_running "run_modyco_wait_gpu_13h_chain.sh"; then
  echo "Waiter 13h déjà actif :"
  pgrep -af "run_modyco_wait_gpu_13h_chain.sh" | grep -v pgrep || true
  tail -15 logs/run_waiter_13h_modyco.log 2>/dev/null || true
  exit 0
fi

if is_running "python.*pipeline.py (train|run)"; then
  echo "Pipeline actif — abandon"
  pgrep -af pipeline.py | grep -v pgrep || true
  exit 3
fi

chmod +x \
  scripts/run_modyco_wait_gpu_13h_chain.sh \
  scripts/run_modyco_speechllm_114k_v5_specaug.sh \
  scripts/run_modyco_st_14k_v11_batch32.sh \
  2_speechLLM/scripts/run_044_b1_utterance_large_114k_v5_specaug_nohup.sh \
  1_Transformer/scripts/run_046_baseline_utterance_large_14k_v11_batch32_nohup.sh

MAX_H="${MAX_HOURS:-13}"
echo "=== $(date -Is) Lancement waiter GPU ${MAX_H}h (run_044 → run_046) ==="
nohup env MAX_HOURS="${MAX_H}" bash scripts/run_modyco_wait_gpu_13h_chain.sh \
  > logs/run_waiter_13h_modyco.log 2>&1 &
echo "WAITER_PID=$!"
sleep 5
tail -25 logs/run_waiter_13h_modyco.log
pgrep -af "run_modyco_wait_gpu_13h_chain.sh" | grep -v pgrep || true
