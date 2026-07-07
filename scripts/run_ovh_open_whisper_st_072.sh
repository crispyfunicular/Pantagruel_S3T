#!/usr/bin/env bash
# OVH — Open baseline Whisper-ST (run_072, piste K).
#
# Whisper large-v3 task=translate : ST directe fr→en, zero-shot, open weights.
# ~2–6 h GPU sur V100 32 Go (éval valid+test utterance).
#
# Usage :
#   cd ~/S3T && source .venv/bin/activate && mkdir -p logs
#   nohup bash scripts/run_ovh_open_whisper_st_072.sh \
#     > logs/run_072_launch.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

CFG="6_open_baselines/configs/fr-en/whisper_large_v3_st.yaml"
RUN_ID="run_072_open_whisper_st_utterance"
LOG="logs/${RUN_ID}_eval.log"

if pgrep -af "^python.*pipeline\.py (train|run)" >/dev/null 2>&1; then
  echo "WARNING: entraînement ST/speechLLM actif — éval open baseline en parallèle OK" >&2
fi

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
