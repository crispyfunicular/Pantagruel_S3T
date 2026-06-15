#!/usr/bin/env bash
# Connexion et déploiement S3T vers le serveur IMAG aker (login node).
#
# Le code vit sur aker ; les jobs GPU passent en général par un nœud compute
# (ex. lig-gpu1) avec le même $HOME NFS — vérifier sur site.
#
# Prérequis : accès SSH imbriqué ligone → aker (aker n'est pas joignable en direct
# depuis l'extérieur). Définir LIGONE_JUMP si besoin (défaut : bonapelm@ligone.imag.fr).
#
#   ssh bonapelm@ligone.imag.fr
#   ssh bonapelm@aker.imag.fr
#
# Usage :
#   ./scripts/aker.sh check
#   ./scripts/aker.sh ssh
#   ./scripts/aker.sh ssh 'hostname && df -h ~'
#   ./scripts/aker.sh rsync-code          # pipelines + configs (sans données lourdes)
#   ./scripts/aker.sh rsync-checkpoint RUN_ID
#   ./scripts/aker.sh rsync-eval RUN_ID   # eval/ local → aker

set -euo pipefail

LIGONE_JUMP="${LIGONE_JUMP:-bonapelm@ligone.imag.fr}"
AKER_USER="${AKER_USER:-bonapelm}"
AKER_HOST="${AKER_HOST:-aker.imag.fr}"
AKER_S3T="${AKER_S3T:-~/S3T}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_RUNS="${ROOT}/runs/fr-en"
REMOTE_RUNS="${AKER_S3T}/runs/fr-en"

SSH_ID="${SSH_IDENTITY_FILE:-$HOME/.ssh/id_ed25519}"
if [[ ! -f "${SSH_ID}" ]]; then
  echo "ERROR: clé SSH locale introuvable: ${SSH_ID}" >&2
  exit 2
fi
SSH_LOCAL=(
  ssh
  -i "${SSH_ID}"
  -o IdentitiesOnly=yes
  -o BatchMode=yes
  -o PreferredAuthentications=publickey
  -o PasswordAuthentication=no
  -o KbdInteractiveAuthentication=no
  -o StrictHostKeyChecking=accept-new
)
SSH_NESTED=(
  -o BatchMode=yes
  -o PreferredAuthentications=publickey
  -o PasswordAuthentication=no
  -o KbdInteractiveAuthentication=no
  -o StrictHostKeyChecking=accept-new
)
RSYNC_SSH="ssh -i ${SSH_ID} -o IdentitiesOnly=yes -o BatchMode=yes -o PreferredAuthentications=publickey -o PasswordAuthentication=no -o KbdInteractiveAuthentication=no -o StrictHostKeyChecking=accept-new ${LIGONE_JUMP} ssh -o BatchMode=yes -o PreferredAuthentications=publickey -o PasswordAuthentication=no -o KbdInteractiveAuthentication=no -o StrictHostKeyChecking=accept-new"

remote() {
  "${SSH_LOCAL[@]}" "${LIGONE_JUMP}" ssh "${SSH_NESTED[@]}" "${AKER_USER}@${AKER_HOST}" "$@"
}

usage() {
  sed -n '1,16p' "$0" | tail -n +2
  echo ""
  echo "Variables optionnelles : LIGONE_JUMP AKER_USER AKER_HOST AKER_S3T SSH_IDENTITY_FILE"
}

cmd_check() {
  echo "Test connexion ${AKER_USER}@${AKER_HOST}…"
  if remote "echo OK — \$(hostname) — ${AKER_S3T}"; then
    echo "Connexion OK."
    return 0
  fi
  echo "ÉCHEC : configure une clé SSH (ssh-copy-id ${AKER_USER}@${AKER_HOST})" >&2
  return 1
}

cmd_ssh() {
  if [[ $# -eq 0 ]]; then
    remote -t "mkdir -p ${AKER_S3T} && cd ${AKER_S3T} && exec bash -l"
  else
    remote "$@"
  fi
}

cmd_rsync_code() {
  echo "Déploiement pipelines S3T → ${AKER_USER}@${AKER_HOST}:${AKER_S3T}/"
  remote "mkdir -p ${AKER_S3T}"
  rsync -avz --progress -e "${RSYNC_SSH}" \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude '__pycache__/' \
    --exclude '.pytest_cache/' \
    --exclude '.ruff_cache/' \
    --exclude 'datasets/raw/' \
    --exclude 'datasets/processed/' \
    --exclude 'datasets/processed_sentence/' \
    --exclude 'corpus_audio/' \
    --exclude 'runs/' \
    --exclude 'inference/' \
    "${ROOT}/scripts_communs/" \
    "${ROOT}/1_Transformer/" \
    "${ROOT}/2_speechLLM/" \
    "${ROOT}/3_Gemini/" \
    "${ROOT}/4_cascade/" \
    "${ROOT}/5_Pantagruel_multimodal/" \
    "${ROOT}/scripts/" \
    "${ROOT}/tests/" \
    "${ROOT}/docs/" \
    "${ROOT}/requirements.txt" \
    "${ROOT}/pyproject.toml" \
    "${ROOT}/README.md" \
    "${ROOT}/rapport.md" \
    "${AKER_USER}@${AKER_HOST}:${AKER_S3T}/"
  echo "Terminé. Sur aker : cd ${AKER_S3T} && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
}

cmd_rsync_checkpoint() {
  local run_id="${1:?RUN_ID requis (ex. run_005_speechllm_b1_sentence_long_unfreeze_encoder)}"
  local ckpt="${LOCAL_RUNS}/${run_id}/checkpoints/best.pt"
  if [[ ! -f "${ckpt}" ]]; then
    echo "ERROR: checkpoint local absent: ${ckpt}" >&2
    exit 2
  fi
  remote "mkdir -p ${REMOTE_RUNS}/${run_id}/checkpoints"
  rsync -avz --progress -e "${RSYNC_SSH}" \
    "${ckpt}" \
    "${AKER_USER}@${AKER_HOST}:${REMOTE_RUNS}/${run_id}/checkpoints/"
  echo "OK: ${REMOTE_RUNS}/${run_id}/checkpoints/best.pt"
}

cmd_rsync_eval() {
  local run_id="${1:?RUN_ID requis}"
  local src="${LOCAL_RUNS}/${run_id}/eval/"
  if [[ ! -d "${src}" ]]; then
    echo "ERROR: dossier local absent: ${src}" >&2
    exit 2
  fi
  remote "mkdir -p ${REMOTE_RUNS}/${run_id}/eval"
  rsync -avz --progress -e "${RSYNC_SSH}" \
    "${src}" \
    "${AKER_USER}@${AKER_HOST}:${REMOTE_RUNS}/${run_id}/eval/"
  echo "OK: ${REMOTE_RUNS}/${run_id}/eval/"
}

main() {
  local cmd="${1:-}"
  shift || true
  case "${cmd}" in
    check) cmd_check "$@" ;;
    ssh) cmd_ssh "$@" ;;
    rsync-code) cmd_rsync_code "$@" ;;
    rsync-checkpoint) cmd_rsync_checkpoint "$@" ;;
    rsync-eval) cmd_rsync_eval "$@" ;;
    -h|--help|help|"") usage ;;
    *)
      echo "Commande inconnue: ${cmd}" >&2
      usage >&2
      exit 2
      ;;
  esac
}

main "$@"
