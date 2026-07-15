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

ensure_torch_stack_compat() {
  # OVH = Tesla V100 (CC 7.0) : utiliser l'index cu124 (cu130 / torch 2.12+ exclut le V100).
  # NeMo 2.7 requiert torch>=2.6 ; aligner torchvision/torchaudio sur la même série.
  local cuda_tag="cu124"
  if python -c "
import torch
import torchvision
import torchaudio
boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]])
scores = torch.tensor([0.9, 0.8])
torchvision.ops.nms(boxes, scores, 0.5)
_ = torch.tensor([1.0], device='cuda') + 1
print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'torchaudio', torchaudio.__version__)
" 2>/dev/null; then
    echo "Stack torch/torchvision/torchaudio compatible (V100)."
    return 0
  fi
  echo "=== Réalignement torch 2.6 + torchvision/torchaudio (${cuda_tag}, V100) ==="
  pip install "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" \
    --index-url "https://download.pytorch.org/whl/${cuda_tag}"
  python -c "
import torch
import torchvision
import torchaudio
boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]])
scores = torch.tensor([0.9, 0.8])
torchvision.ops.nms(boxes, scores, 0.5)
_ = torch.tensor([1.0], device='cuda') + 1
print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'torchaudio', torchaudio.__version__, 'OK')
"
}

ensure_nemo() {
  ensure_torch_stack_compat
  if python -c "import nemo.collections.asr" 2>/dev/null; then
    echo "NeMo déjà installé."
    return 0
  fi
  echo "=== Installation nemo_toolkit[asr] (Canary-1B) ==="
  pip install 'nemo_toolkit[asr]>=2.0'
  ensure_torch_stack_compat
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
