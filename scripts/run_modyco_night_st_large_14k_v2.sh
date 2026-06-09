#!/usr/bin/env bash
# Modyco — lancement nocturne ST Pantagruel-L-14k v2 (retry après collapse run_010).
#
# Ne pas lancer en journée si le GPU est partagé : ce script vérifie qu'aucun
# entraînement n'est actif sur la tour avant de déléguer au nohup v2.
#
# Pas de doublon OVH : speechLLM Large (run_012/013) reste sur la VM OVH.
#
# Prérequis : st_common.py avec _resolve_encoder_output_dim (fix dimension Large).
#
# Lancement ce soir (depuis le poste local) :
#   ./scripts/tour.sh ssh 'cd ~/S3T && source .venv/bin/activate && mkdir -p logs && \
#     nohup bash scripts/run_modyco_night_st_large_14k_v2.sh \
#     > logs/run_014_st_large_14k_v2_chain_wrapper.log 2>&1 &'
#
# Lancement direct sur Modyco :
#   cd ~/S3T && source .venv/bin/activate
#   mkdir -p logs
#   nohup bash scripts/run_modyco_night_st_large_14k_v2.sh \
#     > logs/run_014_st_large_14k_v2_chain_wrapper.log 2>&1 &
#   tail -f logs/run_014_transformer_baseline_utterance_large_14k_v2_spm_train_eval.log
#
# Forcer malgré un job GPU actif (déconseillé) :
#   bash scripts/run_modyco_night_st_large_14k_v2.sh --force

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ST_SCRIPT="${ROOT}/1_Transformer/scripts/run_014_baseline_utterance_large_14k_v2_nohup.sh"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

require_gpu_free() {
  if pgrep -af "pipeline.py train" >/dev/null 2>&1; then
    echo "ERROR: entraînement GPU encore actif sur Modyco :" >&2
    pgrep -af "pipeline.py train" >&2 || true
    echo "  Attendre la fin (ex. run_010 en cours) ou --force." >&2
    exit 2
  fi
  echo "OK: aucun pipeline.py train actif."
}

usage() {
  sed -n '1,26p' "$0" | tail -n +2
}

FORCE=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    -h|--help|help) usage; exit 0 ;;
    *)
      echo "Argument inconnu: ${arg}" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$FORCE" != "1" ]]; then
  require_gpu_free
fi

if [[ ! -x "$ST_SCRIPT" ]]; then
  chmod +x "$ST_SCRIPT"
fi

echo "=== $(date -Is) Pré-vol ST L-14k v2 (Modyco, run_014) ==="
echo "=== $(date -Is) Délégation → ${ST_SCRIPT} ==="
exec bash "$ST_SCRIPT"
