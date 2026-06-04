#!/usr/bin/env bash
# Cascade ASR→MT — évaluation nocturne : sentence_like puis utterance (même protocole SacreBLEU).
#
# Durée estimée : ~3–6 h (Whisper large-v3 + Marian, ~960 + ~1900 segments valid/test).
# Prérequis : GPU libre (~6–10 Go VRAM), manifests + WAV prepare OK.
#
# Lancement détachable (tour GPU) :
#   cd ~/S3T && source .venv/bin/activate
#   mkdir -p logs
#   nohup bash 4_cascade/scripts/run_night_cascade_nohup.sh \
#     > logs/run_night_cascade_nohup_wrapper.log 2>&1 &
#   echo $! > logs/run_night_cascade_nohup.pid
#   tail -f logs/run_night_cascade_sentence_like.log

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

LOG_DIR="${ROOT}/logs"
mkdir -p "$LOG_DIR"

run_eval() {
  local cfg="$1"
  local run_id="$2"
  local log="${LOG_DIR}/${run_id}_evaluate.log"
  {
    echo "=== $(date -Is) EVALUATE cascade ${run_id} ==="
    echo "Config: ${cfg}"
    python 4_cascade/pipeline.py evaluate \
      --config "$cfg" \
      --run-id "$run_id" \
      -v
    echo "=== $(date -Is) DONE ${run_id} ==="
  } 2>&1 | tee -a "$log"
}

{
  echo "=== $(date -Is) NIGHT CASCADE START ==="
  run_eval "4_cascade/configs/fr-en/cascade_sentence.yaml" \
    "run_001_cascade_sentence_like"

  if [[ -f "${ROOT}/datasets/manifests/fr-en/valid.tsv" ]]; then
    run_eval "4_cascade/configs/fr-en/cascade.yaml" \
      "run_001_cascade_utterance"
  else
    echo "=== $(date -Is) SKIP utterance (manifests absents — lancer prepare sans sentence_like) ==="
  fi

  echo "=== $(date -Is) NIGHT CASCADE ALL DONE ==="
} 2>&1 | tee -a "${LOG_DIR}/run_night_cascade_all.log"
