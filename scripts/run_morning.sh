#!/usr/bin/env bash
# Lancement manuel au réveil — état GPU OVH + Modyco et jobs prévus.
#
# Usage (depuis la racine S3T, PC réveillé) :
#   ./scripts/run_morning.sh              # bilan + commande suggérée
#   ./scripts/run_morning.sh status       # idem
#   ./scripts/run_morning.sh ovh          # run_022 speechLLM L-114k v3 (si GPU libre)
#   ./scripts/run_morning.sh modyco       # run_021 speechLLM L-14k v3 (si GPU libre)
#   ./scripts/run_morning.sh ovh --stop-waiter   # stoppe l'attente auto avant lancement manuel
#
# Jobs attendus cette nuit (sans votre PC) :
#   OVH    : run_019 ST L-114k v3 → puis run_022 (waiter nohup si toujours actif)
#   Modyco : run_021 speechLLM L-14k v3 (nohup)
#
# Variables : OVH_HOST AKER_USER TOUR_HOST TOUR_USER SSH_IDENTITY_FILE

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OVH_HOST="${OVH_HOST:-ubuntu@145.239.52.158}"
TOUR_HOST="${TOUR_HOST:-mpellissier@10.8.0.2}"
SSH_ID="${SSH_IDENTITY_FILE:-$HOME/.ssh/id_ed25519}"

SSH_OVH=(ssh -o BatchMode=yes -o ConnectTimeout=15)
SSH_TOUR=(ssh -i "${SSH_ID}" -o BatchMode=yes -o ConnectTimeout=15 -o IdentitiesOnly=yes)

RUN_OVH_ST="run_019_transformer_baseline_utterance_large_114k_v3"
RUN_OVH_SLM="run_022_speechllm_b1_utterance_large_114k_v3"
RUN_MODY_SLM="run_021_speechllm_b1_utterance_large_14k_v3"

usage() {
  sed -n '1,16p' "$0" | tail -n +2
}

remote_ovh() {
  "${SSH_OVH[@]}" "${OVH_HOST}" "$@"
}

remote_modyco() {
  "${SSH_TOUR[@]}" "${TOUR_HOST}" "$@"
}

bleu_test_line() {
  local path="$1"
  if [[ -f "$path" ]]; then
    head -1 "$path" 2>/dev/null || true
  else
    echo "(pas encore)"
  fi
}

status_ovh() {
  echo "════════ OVH (${OVH_HOST}) ════════"
  remote_ovh bash -s <<REMOTE || echo "(connexion OVH impossible)"
set -euo pipefail
S3T=~/S3T
echo "GPU: \$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo n/a)"
if pgrep -af 'python.*pipeline.py (train|run)' >/dev/null 2>&1; then
  echo "Processus:"
  pgrep -af 'python.*pipeline.py (train|run)' | head -3
else
  echo "Processus pipeline: aucun"
fi
if pgrep -af 'run_ovh_wait_st_then_speechllm' >/dev/null 2>&1; then
  echo "Waiter run_022: ACTIF"
else
  echo "Waiter run_022: inactif"
fi
for run in ${RUN_OVH_ST} ${RUN_OVH_SLM}; do
  evalf="\${S3T}/runs/fr-en/\${run}/eval/sacrebleu_test.txt"
  echo "--- \${run} ---"
  if [[ -f "\$evalf" ]]; then
    head -1 "\$evalf"
  elif [[ -d "\${S3T}/runs/fr-en/\${run}" ]]; then
    echo "en cours ou train terminé sans eval"
    tail -1 "\${S3T}/runs/fr-en/\${run}/train.log" 2>/dev/null | head -c 120 || true
    echo
  else
    echo "pas démarré"
  fi
done
REMOTE
}

status_modyco() {
  echo ""
  echo "════════ Modyco (${TOUR_HOST}) ════════"
  remote_modyco bash -s <<REMOTE || echo "(connexion Modyco impossible — VPN/réseau ?)"
set -euo pipefail
S3T=~/S3T
echo "GPU: \$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo n/a)"
if pgrep -af '^python.*pipeline\.py (train|run)' >/dev/null 2>&1; then
  echo "Processus:"
  pgrep -af '^python.*pipeline\.py (train|run)' | head -3
else
  echo "Processus pipeline: aucun"
fi
run=${RUN_MODY_SLM}
evalf="\${S3T}/runs/fr-en/\${run}/eval/sacrebleu_test.txt"
echo "--- \${run} ---"
if [[ -f "\$evalf" ]]; then
  head -1 "\$evalf"
elif [[ -d "\${S3T}/runs/fr-en/\${run}" ]]; then
  echo "en cours ou train terminé sans eval"
  python3 -c "
import json
p='\${S3T}/runs/fr-en/\${run}/train.log'
try:
  r=json.loads(open(p).read().strip().split(chr(10))[-1])
  print('  update', r['update'], 'best_bleu_dev', round(r.get('best_bleu_dev') or 0, 2))
except Exception:
  print('  (train.log absent ou vide)')
" 2>/dev/null || true
else
  echo "pas démarré"
fi
REMOTE
}

suggest_next() {
  echo ""
  echo "════════ Suggestion ════════"
  echo "  ./scripts/run_morning.sh ovh      → speechLLM L-114k v3 (run_022) si OVH libre"
  echo "  ./scripts/run_morning.sh modyco   → speechLLM L-14k v3 (run_021) si Modyco libre"
  echo ""
  echo "Suivi logs :"
  echo "  ssh ${OVH_HOST} 'tail -f ~/S3T/logs/run_022_*train_eval.log'"
  echo "  ./scripts/tour.sh ssh 'tail -f ~/S3T/logs/run_021_*train_eval.log'"
}

cmd_status() {
  status_ovh
  status_modyco
  suggest_next
}

launch_ovh() {
  local stop_waiter=0
  for arg in "$@"; do
    [[ "$arg" == "--stop-waiter" ]] && stop_waiter=1
  done

  echo "Déploiement scripts OVH (rsync léger)…"
  rsync -az \
    "${ROOT}/2_speechLLM/configs/fr-en/b1_utterance_large_114k_v3.yaml" \
    "${ROOT}/2_speechLLM/scripts/run_022_b1_utterance_large_114k_v3_nohup.sh" \
    "${ROOT}/scripts/run_ovh_speechllm_114k_v3.sh" \
    -e ssh "${OVH_HOST}:~/S3T/" 2>/dev/null || true
  rsync -az "${ROOT}/2_speechLLM/configs/fr-en/b1_utterance_large_114k_v3.yaml" \
    "${OVH_HOST}:~/S3T/2_speechLLM/configs/fr-en/" 2>/dev/null || true
  rsync -az "${ROOT}/2_speechLLM/scripts/run_022_b1_utterance_large_114k_v3_nohup.sh" \
    "${OVH_HOST}:~/S3T/2_speechLLM/scripts/" 2>/dev/null || true
  rsync -az "${ROOT}/scripts/run_ovh_speechllm_114k_v3.sh" \
    "${OVH_HOST}:~/S3T/scripts/" 2>/dev/null || true

  remote_ovh bash -s "$stop_waiter" <<'REMOTE'
set -euo pipefail
STOP_WAITER="$1"
cd ~/S3T
chmod +x scripts/run_ovh_speechllm_114k_v3.sh 2_speechLLM/scripts/run_022_b1_utterance_large_114k_v3_nohup.sh

if [[ "$STOP_WAITER" == "1" ]]; then
  pkill -f run_ovh_wait_st_then_speechllm_114k_v3 || true
  echo "Waiter arrêté."
fi

if pgrep -af 'python.*pipeline.py (train|run)' >/dev/null 2>&1; then
  echo "ERROR: GPU OVH encore occupée — attendez la fin ou vérifiez avec: ./scripts/run_morning.sh status" >&2
  pgrep -af 'python.*pipeline.py (train|run)' >&2 || true
  exit 2
fi

if [[ -f runs/fr-en/run_022_speechllm_b1_utterance_large_114k_v3/eval/sacrebleu_test.txt ]]; then
  echo "run_022 déjà terminé — rien à lancer."
  head -1 runs/fr-en/run_022_speechllm_b1_utterance_large_114k_v3/eval/sacrebleu_test.txt
  exit 0
fi

source .venv/bin/activate
mkdir -p logs
nohup bash scripts/run_ovh_speechllm_114k_v3.sh \
  > logs/run_022_ovh_chain_wrapper.log 2>&1 &
echo "Lancé PID $! — tail -f logs/run_022_speechllm_b1_utterance_large_114k_v3_train_eval.log"
REMOTE
}

launch_modyco() {
  echo "Déploiement scripts Modyco (rsync léger)…"
  rsync -az --exclude '.venv' --exclude 'runs/' --exclude 'datasets/' \
    -e "ssh -i ${SSH_ID} -o IdentitiesOnly=yes" \
    "${ROOT}/2_speechLLM/configs/fr-en/b1_utterance_large_14k_v3.yaml" \
    "${ROOT}/2_speechLLM/scripts/run_021_b1_utterance_large_14k_v3_nohup.sh" \
    "${ROOT}/scripts/run_modyco_speechllm_14k_v3.sh" \
    "${TOUR_HOST}:~/S3T/" 2>/dev/null || true
  rsync -az -e "ssh -i ${SSH_ID} -o IdentitiesOnly=yes" \
    "${ROOT}/2_speechLLM/configs/fr-en/b1_utterance_large_14k_v3.yaml" \
    "${TOUR_HOST}:~/S3T/2_speechLLM/configs/fr-en/" 2>/dev/null || true
  rsync -az -e "ssh -i ${SSH_ID} -o IdentitiesOnly=yes" \
    "${ROOT}/2_speechLLM/scripts/run_021_b1_utterance_large_14k_v3_nohup.sh" \
    "${TOUR_HOST}:~/S3T/2_speechLLM/scripts/" 2>/dev/null || true
  rsync -az -e "ssh -i ${SSH_ID} -o IdentitiesOnly=yes" \
    "${ROOT}/scripts/run_modyco_speechllm_14k_v3.sh" \
    "${TOUR_HOST}:~/S3T/scripts/" 2>/dev/null || true

  remote_modyco bash -s <<'REMOTE'
set -euo pipefail
cd ~/S3T
chmod +x scripts/run_modyco_speechllm_14k_v3.sh 2_speechLLM/scripts/run_021_b1_utterance_large_14k_v3_nohup.sh

if pgrep -af '^python.*pipeline\.py (train|run)' >/dev/null 2>&1; then
  echo "ERROR: GPU Modyco encore occupée — attendez la fin ou: ./scripts/run_morning.sh status" >&2
  pgrep -af '^python.*pipeline\.py (train|run)' >&2 || true
  exit 2
fi

if [[ -f runs/fr-en/run_021_speechllm_b1_utterance_large_14k_v3/eval/sacrebleu_test.txt ]]; then
  echo "run_021 déjà terminé — rien à lancer."
  head -1 runs/fr-en/run_021_speechllm_b1_utterance_large_14k_v3/eval/sacrebleu_test.txt
  exit 0
fi

# Run partiel : archiver avant relance propre
if [[ -d runs/fr-en/run_021_speechllm_b1_utterance_large_14k_v3 ]]; then
  mv runs/fr-en/run_021_speechllm_b1_utterance_large_14k_v3 \
    "runs/fr-en/run_021_speechllm_b1_utterance_large_14k_v3_partial_$(date +%Y%m%d_%H%M)"
  echo "Run partiel archivé."
fi

source .venv/bin/activate
mkdir -p logs
nohup bash scripts/run_modyco_speechllm_14k_v3.sh \
  > logs/run_021_speechllm_14k_v3_chain_wrapper.log 2>&1 &
echo "Lancé PID $! — tail -f logs/run_021_speechllm_b1_utterance_large_14k_v3_train_eval.log"
REMOTE
}

main() {
  local cmd="${1:-status}"
  shift || true
  case "${cmd}" in
    status|"") cmd_status ;;
    ovh) launch_ovh "$@" ;;
    modyco) launch_modyco "$@" ;;
    -h|--help|help) usage ;;
    *)
      echo "Commande inconnue: ${cmd}" >&2
      usage >&2
      exit 2
      ;;
  esac
}

main "$@"
