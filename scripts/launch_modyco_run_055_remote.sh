#!/usr/bin/env bash
# Exécuter sur Modyco : run_055 Llama seed 2 (GPU libre requis).
set -euo pipefail
cd ~/S3T
mkdir -p logs

if pgrep -af "python.*pipeline.py (train|run)" | grep -v pgrep >/dev/null 2>&1; then
  echo "Pipeline actif — abandon"
  pgrep -af pipeline.py | grep -v pgrep || true
  exit 3
fi

chmod +x scripts/run_modyco_wait_gpu_then_speechllm_055_llama_seed2.sh \
  scripts/run_modyco_speechllm_14k_llama32_3b_seed2.sh

echo "=== $(date -Is) Lancement run_055 Llama seed 2 (waiter GPU, OVERWRITE=1) ==="
nohup env OVERWRITE=1 bash scripts/run_modyco_wait_gpu_then_speechllm_055_llama_seed2.sh \
  > logs/run_waiter_055_modyco.log 2>&1 &
echo "WAITER_PID=$!"
sleep 5
tail -15 logs/run_waiter_055_modyco.log
