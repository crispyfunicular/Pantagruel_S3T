#!/usr/bin/env bash
# Modyco — Open baselines ST : Whisper-ST puis SeamlessM4T v2 (piste K).
#
# Réplication cross-machine OVH/IMAG sur RTX 5000 Ada (32 Go).
# Whisper ~1–3 h ; Seamless ~2–6 h selon VRAM libre (processus tiers possibles).
#
# Usage (Modyco, GPU libre ou --force) :
#   cd ~/S3T && mkdir -p logs
#   nohup bash scripts/run_modyco_open_whisper_seamless_075.sh \
#     > logs/run_075_modyco_open_baselines_launch.log 2>&1 &
#
# Via waiter :
#   nohup bash scripts/run_modyco_wait_gpu_then_open_sentence_080.sh \
#     > logs/run_waiter_open_080_modyco.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

CFG_WHISPER="6_open_baselines/configs/fr-en/whisper_large_v3_st_sentence.yaml"
CFG_SEAMLESS="6_open_baselines/configs/fr-en/seamless_m4t_v2_large_sentence.yaml"
RUN_WHISPER="run_080_modyco_open_whisper_st_sentence"
RUN_SEAMLESS="run_080b_modyco_open_seamlessm4t_v2_sentence"
LOG="logs/run_080_modyco_open_baselines_sentence_eval.log"

gpu_vram_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

gpu_vram_free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

require_gpu_ok() {
  local vram_used vram_free
  vram_used="$(gpu_vram_used_mib || echo 99999)"
  vram_free="$(gpu_vram_free_mib || echo 0)"
  if pgrep -af "^python.*pipeline\.py (train|run|evaluate)" >/dev/null 2>&1; then
    echo "ERROR: pipeline S3T actif sur Modyco :" >&2
    pgrep -af "^python.*pipeline\.py (train|run|evaluate)" >&2 || true
    exit 2
  fi
  if [[ "${1:-}" == "--force" ]]; then
    echo "OK: --force (VRAM used ${vram_used} MiB, free ${vram_free} MiB)."
    return 0
  fi
  if [[ "${vram_used}" =~ ^[0-9]+$ ]] && (( vram_used > 4096 )); then
    if [[ "${vram_free}" =~ ^[0-9]+$ ]] && (( vram_free >= 18000 )); then
      echo "OK: VRAM partiellement occupée (${vram_used} MiB) mais ${vram_free} MiB libres — éval open baseline."
      return 0
    fi
    echo "ERROR: VRAM insuffisante (used ${vram_used} MiB, free ${vram_free} MiB) :" >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | head -5 >&2 || true
    exit 2
  fi
  echo "OK: GPU libre (VRAM ${vram_used} MiB)."
}

FORCE=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    -h|--help|help)
      sed -n '1,18p' "$0" | tail -n +2
      exit 0
      ;;
    *)
      echo "Argument inconnu: ${arg}" >&2
      exit 2
      ;;
  esac
done

[[ "$FORCE" == "1" ]] && require_gpu_ok --force || require_gpu_ok
mkdir -p logs

echo "=== $(date -Is) start open baselines Modyco on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

{
  echo "=== $(date -Is) [1/2] Whisper-ST ${RUN_WHISPER} ==="
  python 6_open_baselines/pipeline.py evaluate \
    --config "${CFG_WHISPER}" \
    --run-id "${RUN_WHISPER}" \
    -v

  echo "=== $(date -Is) [2/2] SeamlessM4T v2 ${RUN_SEAMLESS} ==="
  python 6_open_baselines/pipeline.py evaluate \
    --config "${CFG_SEAMLESS}" \
    --run-id "${RUN_SEAMLESS}" \
    -v
} 2>&1 | tee -a "${LOG}"

for run in "${RUN_WHISPER}" "${RUN_SEAMLESS}"; do
  path="runs/fr-en/${run}/eval/sacrebleu_test.txt"
  if [[ -f "$path" ]]; then
    echo "=== ${run} : $(head -1 "$path") ==="
  fi
done

echo "=== $(date -Is) done ==="
