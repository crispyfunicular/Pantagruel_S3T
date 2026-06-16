#!/usr/bin/env bash
# Run 005 — encodeur dégelé (fr-en, sentence_like) : train 20k updates puis evaluate.
# Prérequis : patch ``_speechllm_checkpoint_prefixes`` dans speechllm_lib.py (persiste encoder.*).
#
# Lancement détachable (fermer le laptop) :
#   cd ~/S3T && source .venv/bin/activate
#   mkdir -p logs
#   nohup bash 2_speechLLM/scripts/run_005_unfreeze_train_eval_nohup.sh \
#     > logs/run_005_nohup_wrapper.log 2>&1 &
#   echo $! > logs/run_005_nohup.pid
#   tail -f logs/run_005_speechllm_b1_sentence_long_unfreeze_encoder_train_eval.log

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

CFG="2_speechLLM/configs/fr-en/b1_sentence_long_unfreeze_encoder.yaml"
RUN="run_005_speechllm_b1_sentence_long_unfreeze_encoder"
LOG_DIR="${ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/${RUN}_train_eval.log"

if ! grep -q '_speechllm_checkpoint_prefixes' "${ROOT}/2_speechLLM/speechllm_lib.py"; then
  echo "ERROR: patch checkpoint manquant dans 2_speechLLM/speechllm_lib.py" >&2
  exit 1
fi

{
  echo "=== $(date -Is) TRAIN ${RUN} ==="
  python 2_speechLLM/pipeline.py train --config "$CFG" --run-id "$RUN" -v

  echo "=== $(date -Is) Vérification checkpoint (encoder.*) ==="
  python -c "
import torch
from pathlib import Path
p = Path('runs/fr-en/${RUN}/checkpoints/best.pt')
payload = torch.load(p, map_location='cpu', weights_only=False)
enc = [k for k in payload['trainable_state'] if k.startswith('encoder.')]
print(f'encoder tensors in best.pt: {len(enc)}')
if not enc:
    raise SystemExit('ERREUR: best.pt ne contient pas encoder.* — ne pas évaluer.')
"

  echo "=== $(date -Is) EVALUATE ${RUN} ==="
  python 2_speechLLM/pipeline.py evaluate --config "$CFG" --run-id "$RUN" --beam-size 1 -v

  echo "=== $(date -Is) DONE ==="
} 2>&1 | tee -a "$LOG"
