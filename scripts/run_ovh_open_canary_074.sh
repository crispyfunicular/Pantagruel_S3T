#!/usr/bin/env bash
# OVH — Open baseline Canary-1B v2 (run_074, piste K).
#
# ST NVIDIA fr→en via NeMo ; installe nemo_toolkit si absent.
# ~4–8 h GPU sur V100 32 Go.
#
# Usage :
#   cd ~/S3T && source .venv/bin/activate && mkdir -p logs
#   nohup bash scripts/run_ovh_open_canary_074.sh \
#     > logs/run_074_launch.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

CFG="6_open_baselines/configs/fr-en/canary_1b.yaml"
RUN_ID="run_074_open_canary_1b_utterance"
LOG="logs/${RUN_ID}_eval.log"

ensure_nemo() {
  if python -c "import nemo.collections.asr" 2>/dev/null; then
    echo "NeMo déjà installé."
    return 0
  fi
  echo "=== Installation nemo_toolkit[asr] (Canary-1B) ==="
  pip install 'nemo_toolkit[asr]>=2.0'
}

echo "=== $(date -Is) GPU ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

ensure_nemo

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
