#!/usr/bin/env bash
# Modyco — speechLLM Pantagruel-L-14k v3 (run_021).
#
# Suite run_020 ST v3 (21,22 test). speechLLM Phi-2 gelé, max 128 tok, eval dev complet.
# Budget utilisateur : ≤ 4 h GPU. Durée attendue ~2–4 h.
#
# Depuis le poste local :
#   rsync -az --exclude '.venv' --exclude 'runs/' --exclude 'datasets/' \
#     -e "ssh -i ~/.ssh/id_ed25519" ./ mpellissier@10.8.0.2:~/S3T/
#   ./scripts/tour.sh ssh 'cd ~/S3T && source .venv/bin/activate && mkdir -p logs && \
#     nohup bash scripts/run_modyco_speechllm_14k_v3.sh \
#     > logs/run_021_speechllm_14k_v3_chain_wrapper.log 2>&1 &'
#
# Sur Modyco :
#   cd ~/S3T && source .venv/bin/activate
#   mkdir -p logs
#   nohup bash scripts/run_modyco_speechllm_14k_v3.sh > logs/run_021_speechllm_14k_v3_chain_wrapper.log 2>&1 &
#   tail -f logs/run_021_speechllm_b1_utterance_large_14k_v3_train_eval.log
#
# Forcer malgré un job GPU actif (déconseillé) :
#   bash scripts/run_modyco_speechllm_14k_v3.sh --force

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SPEECHLLM_SCRIPT="${ROOT}/2_speechLLM/scripts/run_021_b1_utterance_large_14k_v3_nohup.sh"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

require_gpu_free() {
  if pgrep -af "^python.*pipeline\.py (train|run)" >/dev/null 2>&1; then
    echo "ERROR: entraînement GPU encore actif sur Modyco :" >&2
    pgrep -af "^python.*pipeline\.py (train|run)" >&2 || true
    exit 2
  fi
  echo "OK: aucun pipeline.py train/run Python actif."
}

FORCE=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    -h|--help|help)
      sed -n '1,22p' "$0" | tail -n +2
      exit 0
      ;;
    *)
      echo "Argument inconnu: ${arg}" >&2
      exit 2
      ;;
  esac
done

if [[ "$FORCE" != "1" ]]; then
  require_gpu_free
fi

chmod +x "$SPEECHLLM_SCRIPT"

echo "=== $(date -Is) Pré-vol speechLLM L-14k v3 (Modyco, run_021) ==="
python 2_speechLLM/pipeline.py train \
  --config 2_speechLLM/configs/fr-en/b1_utterance_large_14k_v3.yaml \
  --run-id run_021_speechllm_b1_utterance_large_14k_v3 \
  --dry-run

echo "=== $(date -Is) Délégation → ${SPEECHLLM_SCRIPT} ==="
exec bash "$SPEECHLLM_SCRIPT"
