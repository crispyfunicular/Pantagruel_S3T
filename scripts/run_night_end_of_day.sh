#!/usr/bin/env bash
# Fin de journée — jobs GPU enchaînés (tour Modyco), sans session SSH ouverte.
#
# Enchaîne (mode all, défaut) :
#   1. Cascade evaluate : sentence_like puis utterance (~3–6 h)
#   2. Baseline ST Table 8 utterance : SPM → train 80k → eval beam 5 (~8 h)
#
# Modes :
#   cascade   — évaluations cascade seulement (~3–6 h)
#   st        — baseline ST utterance run_002 (legacy, ~8 h)
#   st-v2     — baseline ST utterance v2 run_004 (gel 5k + early stop, ~2–4 h)
#   all       — cascade puis ST run_002 (défaut, ~11–14 h, une seule GPU)
#
# Lancement (sur la tour, dans ~/S3T) :
#   cd ~/S3T && source .venv/bin/activate
#   mkdir -p logs
#   chmod +x scripts/run_night_end_of_day.sh
#   nohup bash scripts/run_night_end_of_day.sh all \
#     > logs/run_night_end_of_day_wrapper.log 2>&1 &
#   echo $! > logs/run_night_end_of_day.pid
#   tail -f logs/run_night_end_of_day.log
#
# Demain :
#   tail -50 logs/run_night_end_of_day.log
#   cat logs/run_night_end_of_day.pid && ps -p "$(cat logs/run_night_end_of_day.pid)" -o pid,cmd

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODE="${1:-all}"
LOG_DIR="${ROOT}/logs"
MAIN_LOG="${LOG_DIR}/run_night_end_of_day.log"
mkdir -p "$LOG_DIR"
# Créer le log tout de suite (tail -f possible dès le lancement nohup).
: >"$MAIN_LOG"

if [[ ! -f "${BASH_SOURCE[0]}" ]]; then
  echo "ERROR: script introuvable: ${BASH_SOURCE[0]}" | tee -a "$MAIN_LOG" >&2
  exit 2
fi

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

warn_if_gpu_busy() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "=== GPU ($(date -Is)) ==="
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
  fi
  local busy=""
  busy="$(pgrep -af "python.*(4_train|train\.py|evaluate\.py|pipeline\.py)" 2>/dev/null | grep -v "pgrep" | head -5 || true)"
  if [[ -n "$busy" ]]; then
    echo "WARNING: processus Python train/eval déjà actifs — risque de conflit GPU." >&2
    printf '%s\n' "$busy" >&2
    echo "  Arrêter ou attendre la fin avant de lancer ce script." >&2
  fi
}

run_cascade() {
  echo "=== $(date -Is) PHASE cascade ==="
  bash "${ROOT}/4_cascade/scripts/run_night_cascade_nohup.sh"
}

run_st_utterance() {
  echo "=== $(date -Is) PHASE ST utterance (Table 8, run_002 legacy) ==="
  bash "${ROOT}/1_Transformer/scripts/run_002_baseline_utterance_nohup.sh"
}

run_st_utterance_v2() {
  echo "=== $(date -Is) PHASE ST utterance v2 (run_004, early stop) ==="
  bash "${ROOT}/1_Transformer/scripts/run_004_baseline_utterance_v2_nohup.sh"
}

update_tracking() {
  if [[ -f "${ROOT}/scripts_communs/update_experiments_tracking.py" ]]; then
    echo "=== $(date -Is) update_experiments_tracking ==="
    python scripts_communs/update_experiments_tracking.py --all || true
  fi
}

{
  echo "=== $(date -Is) NIGHT END OF DAY START (mode=${MODE}) ==="
  echo "Host: $(hostname)  ROOT=${ROOT}"
  warn_if_gpu_busy

  case "$MODE" in
    cascade)
      run_cascade
      ;;
    st)
      run_st_utterance
      ;;
    st-v2)
      run_st_utterance_v2
      ;;
    all)
      run_cascade
      run_st_utterance
      ;;
    *)
      echo "Usage: $0 [all|cascade|st|st-v2]" >&2
      exit 2
      ;;
  esac

  update_tracking
  echo "=== $(date -Is) NIGHT END OF DAY DONE (mode=${MODE}) ==="
} 2>&1 | tee -a "$MAIN_LOG"
