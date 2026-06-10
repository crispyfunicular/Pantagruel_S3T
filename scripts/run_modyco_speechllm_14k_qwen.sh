#!/usr/bin/env bash
# Modyco — speechLLM B2bis Pantagruel-L-14k + Qwen2.5-3B utterance (run_018).
#
# Ablation LLM vs run_012 (Phi-2 @ 15,03 test). Budget utilisateur : ≤ 14 h GPU.
# Durée attendue ~5–8 h. Pas de doublon OVH : run_017 (Phi-2 L-114k v2) sur la VM OVH.
#
# Depuis le poste local (nohup sur la tour) :
#   rsync -az --exclude '.venv' --exclude 'runs/' --exclude 'datasets/' \
#     -e "ssh -i ~/.ssh/id_ed25519" ./ mpellissier@10.8.0.2:~/S3T/
#   ./scripts/tour.sh ssh 'cd ~/S3T && source .venv/bin/activate && mkdir -p logs && \
#     nohup bash scripts/run_modyco_speechllm_14k_qwen.sh \
#     > logs/run_018_speechllm_qwen_chain_wrapper.log 2>&1 &'
#
# Lancement direct sur Modyco :
#   cd ~/S3T && source .venv/bin/activate
#   mkdir -p logs
#   nohup bash scripts/run_modyco_speechllm_14k_qwen.sh \
#     > logs/run_018_speechllm_qwen_chain_wrapper.log 2>&1 &
#   tail -f logs/run_018_speechllm_b2bis_utterance_large_14k_qwen25_3b_train_eval.log
#
# Forcer malgré un job GPU actif (déconseillé) :
#   bash scripts/run_modyco_speechllm_14k_qwen.sh --force

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SPEECHLLM_SCRIPT="${ROOT}/2_speechLLM/scripts/run_018_b1_utterance_large_14k_qwen25_3b_nohup.sh"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

require_gpu_free() {
  # Ne matcher que les processus Python (pas les commandes ssh/bash contenant « pipeline.py »).
  if pgrep -af "^python.*pipeline\.py (train|run)" >/dev/null 2>&1; then
    echo "ERROR: entraînement GPU encore actif sur Modyco :" >&2
    pgrep -af "python.*pipeline.py (train|run)" >&2 || true
    echo "  Attendre la fin ou relancer avec --force." >&2
    exit 2
  fi
  echo "OK: aucun pipeline.py train/run Python actif."
}

FORCE=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    -h|--help|help)
      sed -n '1,24p' "$0" | tail -n +2
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

echo "=== $(date -Is) Pré-vol speechLLM L-14k + Qwen2.5-3B (Modyco, run_018) ==="
python 2_speechLLM/pipeline.py train \
  --config 2_speechLLM/configs/fr-en/b1_utterance_large_14k_qwen25_3b.yaml \
  --run-id run_018_speechllm_b2bis_utterance_large_14k_qwen25_3b \
  --dry-run

echo "=== $(date -Is) Délégation → ${SPEECHLLM_SCRIPT} ==="
exec bash "$SPEECHLLM_SCRIPT"
