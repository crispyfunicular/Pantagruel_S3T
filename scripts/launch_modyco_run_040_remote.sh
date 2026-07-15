#!/usr/bin/env bash
# Exécuter sur Modyco : run_040 Pantagruel Speech_Text utterance v2 (~2–3 h GPU).
set -euo pipefail
cd ~/S3T
mkdir -p logs

if pgrep -af "python.*pipeline.py (train|run)" | grep -v pgrep >/dev/null 2>&1; then
  echo "Pipeline actif — abandon"
  pgrep -af pipeline.py | grep -v pgrep || true
  exit 3
fi

RUN_GLOB="run_040_pantagruel_multimodal_utterance_v2"
EVAL="$(ls -d runs/fr-en/${RUN_GLOB}/eval/sacrebleu_test.txt 2>/dev/null | head -1 || true)"
if [[ -n "${EVAL}" && -s "${EVAL}" ]]; then
  echo "Eval déjà présente : ${EVAL}"
  cat "${EVAL}"
  exit 0
fi

chmod +x scripts/run_modyco_multimodal_utterance_v2.sh \
  5_Pantagruel_multimodal/scripts/run_040_base_utterance_v2_nohup.sh

echo "=== $(date -Is) Lancement run_040 multimodal utterance v2 (budget ~3 h) ==="
nohup bash scripts/run_modyco_multimodal_utterance_v2.sh \
  > logs/run_040_multimodal_chain_wrapper.log 2>&1 &
echo "PID=$!"
sleep 5
tail -20 logs/run_040_multimodal_chain_wrapper.log 2>/dev/null || true
