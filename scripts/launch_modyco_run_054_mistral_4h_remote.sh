#!/usr/bin/env bash
# Exécuter sur Modyco : run_054 speechLLM Mistral-7B (cap 4 h GPU).
set -euo pipefail
cd ~/S3T
mkdir -p logs

if pgrep -af "python.*pipeline.py (train|run)" | grep -v pgrep >/dev/null 2>&1; then
  echo "Pipeline actif — abandon"
  pgrep -af pipeline.py | grep -v pgrep || true
  exit 3
fi

chmod +x scripts/run_modyco_speechllm_14k_mistral_7b.sh \
  2_speechLLM/scripts/run_054_b1_utterance_large_14k_mistral_7b_nohup.sh

echo "=== $(date -Is) Lancement run_054 Mistral-7B (MAX_RUN_HOURS=4, OVERWRITE=1) ==="
nohup env MAX_RUN_HOURS=4 OVERWRITE=1 bash scripts/run_modyco_speechllm_14k_mistral_7b.sh \
  > logs/run_054_mistral_4h_chain_wrapper.log 2>&1 &
echo "PID=$!"
sleep 8
tail -25 logs/run_054_mistral_4h_chain_wrapper.log 2>/dev/null || true
