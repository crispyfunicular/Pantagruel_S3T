#!/usr/bin/env bash
# OVH — Open baseline SeamlessM4T v2 large (run_073, piste K).
#
# ST directe Meta fr→en, zero-shot ; ~4–8 h GPU sur V100 32 Go.
#
# Usage :
#   cd ~/S3T && source .venv/bin/activate && mkdir -p logs
#   nohup bash scripts/run_ovh_open_seamless_073.sh \
#     > logs/run_073_launch.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

CFG="6_open_baselines/configs/fr-en/seamless_m4t_v2_large.yaml"
RUN_ID="run_073_open_seamlessm4t_v2_utterance"
LOG="logs/${RUN_ID}_eval.log"

echo "=== $(date -Is) GPU ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "=== $(date -Is) Dry-run ${RUN_ID} ==="
python 6_open_baselines/pipeline.py evaluate \
  --config "$CFG" \
  --run-id "$RUN_ID" \
  --dry-run

echo "=== $(date -Is) LANCEMENT ${RUN_ID} ==="
python 6_open_baselines/pipeline.py evaluate \
  --config "$CFG" \
  --run-id "$RUN_ID" \
  -v 2>&1 | tee -a "$LOG"

BLEU="n/a"
[[ -f "runs/fr-en/${RUN_ID}/eval/sacrebleu_test.txt" ]] \
  && BLEU="$(head -1 "runs/fr-en/${RUN_ID}/eval/sacrebleu_test.txt")"
echo "=== $(date -Is) ${RUN_ID} — BLEU test : ${BLEU} ==="
