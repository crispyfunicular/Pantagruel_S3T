#!/usr/bin/env bash
# Modyco — Open baseline Canary-1B v2 sentence_like (run_081, taille de contexte).
#
# Compare à run_074 utterance (40,04 BLEU test). Même venv NeMo (.venv-canary).
# Budget : ≤ 15 h GPU. Prérequis : manifests datasets/manifests_sentence/fr-en/.
#
# NE PAS LANCER sans feu vert utilisateur.
#
# Usage (Modyco, GPU libre) :
#   cd ~/S3T && mkdir -p logs
#   nohup bash scripts/run_modyco_open_canary_sentence_081.sh \
#     > logs/run_081_modyco_launch.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MAX_HOURS="${MAX_HOURS_CANARY:-15}"
CANARY_VENV="${CANARY_VENV:-${ROOT}/.venv-canary}"
CFG="6_open_baselines/configs/fr-en/canary_1b_sentence.yaml"
RUN_ID="run_081_modyco_open_canary_1b_sentence"
LOG="logs/${RUN_ID}_eval.log"

ensure_canary_venv() {
  if [[ ! -d "${CANARY_VENV}" ]]; then
    echo "=== Création venv Canary : ${CANARY_VENV} ==="
    python3 -m venv "${CANARY_VENV}"
  fi
  # shellcheck source=/dev/null
  source "${CANARY_VENV}/bin/activate"
  pip install -q -U pip
  pip install -q -r "${ROOT}/requirements.txt"
}

ensure_torch_stack() {
  python -c "
import torch
import torchvision
import torchaudio
boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]])
scores = torch.tensor([0.9, 0.8])
torchvision.ops.nms(boxes, scores, 0.5)
_ = torch.tensor([1.0], device='cuda') + 1
print('stack OK', torch.__version__)
" 2>/dev/null && return 0

  local cuda_tag
  cuda_tag="$(python -c "import torch; c=torch.version.cuda; print(f'cu{c.replace(\".\", \"\")}' if c else 'cu124')")"
  echo "=== Alignement torch 2.6 + torchvision/torchaudio (${cuda_tag}) ==="
  pip install "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" \
    --index-url "https://download.pytorch.org/whl/${cuda_tag}"
}

ensure_nemo() {
  ensure_torch_stack
  if python -c "import nemo.collections.asr" 2>/dev/null; then
    echo "NeMo déjà installé (venv Canary)."
    return 0
  fi
  echo "=== Installation nemo_toolkit[asr] dans venv Canary ==="
  pip install 'nemo_toolkit[asr]>=2.0'
  ensure_torch_stack
}

gpu_vram_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

require_gpu_free() {
  local vram
  vram="$(gpu_vram_used_mib || echo 99999)"
  if pgrep -af "^python.*pipeline\.py (train|run|evaluate)" >/dev/null 2>&1; then
    echo "ERROR: job GPU actif sur Modyco :" >&2
    pgrep -af "^python.*pipeline\.py (train|run|evaluate)" >&2 || true
    exit 2
  fi
  if [[ "${vram}" =~ ^[0-9]+$ ]] && (( vram > 4096 )); then
    echo "ERROR: VRAM occupée (${vram} MiB > 4096) :" >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | head -5 >&2 || true
    exit 2
  fi
  echo "OK: GPU libre (VRAM ${vram} MiB)."
}

FORCE=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    -h|--help|help)
      sed -n '1,20p' "$0" | tail -n +2
      exit 0
      ;;
    *)
      echo "Argument inconnu: ${arg}" >&2
      exit 2
      ;;
  esac
done

[[ "$FORCE" != "1" ]] && require_gpu_free
mkdir -p logs

if [[ ! -f datasets/manifests_sentence/fr-en/test.tsv ]]; then
  echo "ERROR: manifests sentence_like absents (datasets/manifests_sentence/fr-en/)." >&2
  exit 2
fi

ensure_canary_venv
ensure_nemo

echo "=== $(date -Is) GPU ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

echo "=== $(date -Is) Dry-run ${RUN_ID} ==="
python 6_open_baselines/pipeline.py evaluate \
  --config "$CFG" \
  --run-id "$RUN_ID" \
  --dry-run

echo "=== $(date -Is) LANCEMENT ${RUN_ID} (max ${MAX_HOURS}h) ==="
{
  timeout "$((MAX_HOURS * 3600))" \
    python 6_open_baselines/pipeline.py evaluate \
      --config "$CFG" \
      --run-id "$RUN_ID" \
      -v
  ec=$?
  if [[ "$ec" -eq 124 ]]; then
    echo "=== $(date -Is) TIMEOUT ${MAX_HOURS}h — éval partielle conservée ==="
  elif [[ "$ec" -ne 0 ]]; then
    exit "$ec"
  fi
} 2>&1 | tee -a "$LOG"

BLEU="n/a"
[[ -f "runs/fr-en/${RUN_ID}/eval/sacrebleu_test.txt" ]] \
  && BLEU="$(head -1 "runs/fr-en/${RUN_ID}/eval/sacrebleu_test.txt")"
echo "=== $(date -Is) ${RUN_ID} — BLEU test : ${BLEU} ==="
