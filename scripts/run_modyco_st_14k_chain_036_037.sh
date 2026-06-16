#!/usr/bin/env bash
# Modyco — enchaînement run_036 (warmup 10k) puis run_037 (SpecAugment fort) sur L-14k.
#
# run_036 : recette v5 (SpecAugment léger 0.05) + warmup 10k — ~8–10 h GPU
# run_037 : recette v5 + mask_time_prob 0.10 — ~8–10 h GPU
# Budget total : ~16–20 h GPU (lancer en début de nuit).
#
# Depuis le poste local :
#   rsync -az --exclude '.venv' --exclude 'runs/' --exclude 'datasets/' \
#     -e "ssh -i ~/.ssh/id_ed25519" ./ mpellissier@10.8.0.2:~/S3T/
#   ./scripts/tour.sh ssh 'cd ~/S3T && source .venv/bin/activate && mkdir -p logs && \
#     nohup bash scripts/run_modyco_st_14k_chain_036_037.sh \
#     > logs/chain_036_037_wrapper.log 2>&1 &'
#
# Sur Modyco :
#   nohup bash scripts/run_modyco_st_14k_chain_036_037.sh > logs/chain_036_037_wrapper.log 2>&1 &
#   tail -f logs/chain_036_037_wrapper.log
#
# Forcer :
#   bash scripts/run_modyco_st_14k_chain_036_037.sh --force

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

require_gpu_free() {
  if pgrep -af "^python.*pipeline\.py (train|run)" >/dev/null 2>&1; then
    echo "ERROR: entraînement GPU actif sur Modyco :" >&2
    pgrep -af "^python.*pipeline\.py (train|run)" >&2 || true
    exit 2
  fi
  echo "OK: aucun train actif."
}

FORCE=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    -h|--help|help)
      sed -n '1,26p' "$0" | tail -n +2
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

chmod +x \
  "${ROOT}/1_Transformer/scripts/run_036_baseline_utterance_large_14k_v9_warmup10k_nohup.sh" \
  "${ROOT}/1_Transformer/scripts/run_037_baseline_utterance_large_14k_v9_specaug_strong_nohup.sh"

echo "=== $(date -Is) CHAÎNE run_036 → run_037 (Modyco, L-14k) ==="

echo "--- run_036 : warmup 10k + SpecAugment ---"
bash "${ROOT}/1_Transformer/scripts/run_036_baseline_utterance_large_14k_v9_warmup10k_nohup.sh"

echo "--- run_037 : SpecAugment fort (prob 0.10) ---"
bash "${ROOT}/1_Transformer/scripts/run_037_baseline_utterance_large_14k_v9_specaug_strong_nohup.sh"

echo "=== $(date -Is) CHAÎNE TERMINÉE ==="
