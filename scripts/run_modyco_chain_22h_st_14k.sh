#!/usr/bin/env bash
# Modyco — enchaînement prévu ~22h : ST baseline Pantagruel-L-14k (Table 8, utterance).
#
# À lancer sur la tour Modyco APRÈS la fin de run_010_speechllm_b2bis_qwen25_3b (Qwen B2bis).
# Pas de doublon avec OVH : les runs speechLLM Large (run_012/013) tournent sur la VM OVH ;
# ici variante 1_Transformer (SPM + décodeur 6L), run_010_transformer_baseline_utterance_large_14k.
#
# Durée estimée : ~12–18 h GPU (80k updates, encodeur Large, batch 1).
#
# Préparation (depuis le poste local, avant 22h) :
#   rsync -az -e "ssh -i ~/.ssh/id_ed25519" scripts/run_modyco_chain_22h_st_14k.sh \
#     mpellissier@10.8.0.2:~/S3T/scripts/
#   # ou : ./scripts/tour.sh ssh 'chmod +x ~/S3T/scripts/run_modyco_chain_22h_st_14k.sh'
#
# Lancement ce soir sur Modyco (22h, venv activé) :
#   cd ~/S3T && source .venv/bin/activate
#   mkdir -p logs
#   nohup bash scripts/run_modyco_chain_22h_st_14k.sh \
#     > logs/run_010_st_14k_chain_wrapper.log 2>&1 &
#   echo $! > logs/run_010_st_14k_chain.pid
#   tail -f logs/run_010_transformer_baseline_utterance_large_14k_spm_train_eval.log
#
# Depuis le poste local (équivalent) :
#   ./scripts/tour.sh ssh 'cd ~/S3T && source .venv/bin/activate && mkdir -p logs && \
#     nohup bash scripts/run_modyco_chain_22h_st_14k.sh \
#     > logs/run_010_st_14k_chain_wrapper.log 2>&1 &'

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

QWEN_RUN="run_010_speechllm_b2bis_qwen25_3b"
ST_SCRIPT="${ROOT}/1_Transformer/scripts/run_010_baseline_utterance_14k_nohup.sh"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

require_qwen_finished() {
  if pgrep -af "pipeline.py run.*b2bis_qwen25_3b" >/dev/null 2>&1; then
    echo "ERROR: Qwen B2bis encore en cours (run_010_speechllm_b2bis_qwen25_3b)." >&2
    echo "  Attendre la fin ou lancer avec --force si tu assumes le chevauchement GPU." >&2
    exit 2
  fi
  local ckpt="${ROOT}/runs/fr-en/${QWEN_RUN}/checkpoints/best.pt"
  if [[ ! -f "$ckpt" ]]; then
    echo "WARN: pas de checkpoint ${ckpt} — Qwen peut avoir échoué ou être incomplet." >&2
    if [[ "${FORCE:-0}" != "1" ]]; then
      echo "  Relancer avec --force pour ignorer cette vérification." >&2
      exit 2
    fi
  else
    echo "OK: Qwen terminé (checkpoint présent)."
  fi
}

usage() {
  sed -n '1,28p' "$0" | tail -n +2
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

if [[ ! -x "$ST_SCRIPT" ]]; then
  chmod +x "$ST_SCRIPT"
fi

echo "=== $(date -Is) Pré-vol enchaînement ST L-14k (Modyco) ==="
require_qwen_finished
echo "=== $(date -Is) Délégation → ${ST_SCRIPT} ==="
exec bash "$ST_SCRIPT"
