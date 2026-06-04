#!/usr/bin/env bash
# Connexion et rsync vers la tour GPU Modyco — sans invite de mot de passe.
#
# Équivalent à l'alias ~/.bashrc :  alias modyco="ssh mpellissier@10.8.0.2"
#   ./scripts/tour.sh ssh          ≈  modyco
#   ./scripts/tour.sh ssh '…'      ≈  modyco puis commande (une seule session)
#
# Prérequis : clé SSH (pas de mot de passe). Si ``modyco`` marche, ``tour.sh check`` aussi.
#   ssh-copy-id mpellissier@10.8.0.2
# Voir aussi scripts/tour.ssh.config.example et scripts/tour.bashrc.snippet
#
# Usage :
#   ./scripts/tour.sh ssh                    # shell interactif sur la tour
#   ./scripts/tour.sh ssh 'nvidia-smi'       # commande distante
#   ./scripts/tour.sh rsync-eval             # tous les eval/ locaux → tour
#   ./scripts/tour.sh rsync-eval RUN_ID      # un seul run
#   ./scripts/tour.sh check                  # test clé (échec rapide si pas de clé)

set -euo pipefail

# --- Cible tour (modifier ici si besoin) ---
TOUR_USER="${TOUR_USER:-mpellissier}"
TOUR_HOST="${TOUR_HOST:-10.8.0.2}"
TOUR_S3T="${TOUR_S3T:-/home/mpellissier/S3T}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_RUNS="${ROOT}/runs/fr-en"
REMOTE_RUNS="${TOUR_S3T}/runs/fr-en"

# Pas de mot de passe : clé SSH uniquement ; échec explicite si la clé manque.
# Ne pas utiliser REMOTE_DIR=~/S3T dans vos scripts : ~ est développé en LOCAL (/home/morgane/…).
# Clé sur **cette machine** (ex. /home/morgane/.ssh/id_ed25519), pas /home/mpellissier/… sur la tour.
SSH_ID="${SSH_IDENTITY_FILE:-$HOME/.ssh/id_ed25519}"
if [[ ! -f "${SSH_ID}" ]]; then
  echo "ERROR: clé SSH locale introuvable: ${SSH_ID}" >&2
  echo "  (Ne pas mettre le chemin distant mpellissier@10.8.0.2:/home/mpellissier/.ssh/…)" >&2
  echo "  ls -la ~/.ssh/id_*" >&2
  exit 2
fi
SSH_BASE=(
  ssh
  -i "${SSH_ID}"
  -o IdentitiesOnly=yes
  -o BatchMode=yes
  -o PreferredAuthentications=publickey
  -o PasswordAuthentication=no
  -o KbdInteractiveAuthentication=no
  -o StrictHostKeyChecking=accept-new
)
RSYNC_SSH="ssh -i ${SSH_ID} -o IdentitiesOnly=yes -o BatchMode=yes -o PreferredAuthentications=publickey -o PasswordAuthentication=no -o KbdInteractiveAuthentication=no -o StrictHostKeyChecking=accept-new"

remote() {
  "${SSH_BASE[@]}" "${TOUR_USER}@${TOUR_HOST}" "$@"
}

usage() {
  sed -n '1,18p' "$0" | tail -n +2
  echo ""
  echo "Variables optionnelles : TOUR_USER TOUR_HOST TOUR_S3T SSH_IDENTITY_FILE"
  echo "  SSH_IDENTITY_FILE = clé locale (défaut: ~/.ssh/id_ed25519)"
}

cmd_check() {
  echo "Test connexion ${TOUR_USER}@${TOUR_HOST} (clé SSH uniquement)…"
  if remote "echo OK — $(hostname) — ${TOUR_S3T}"; then
    echo "Connexion OK."
    return 0
  fi
  echo "ÉCHEC : aucune clé acceptée. Configure une clé :" >&2
  echo "  ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ''" >&2
  echo "  ssh-copy-id -i ${SSH_ID}.pub ${TOUR_USER}@${TOUR_HOST}" >&2
  echo "  eval \"\$(ssh-agent -s)\" && ssh-add ${SSH_ID}   # si la clé a une passphrase" >&2
  echo "  # ou copie manuelle de ~/.ssh/id_ed25519.pub dans ~/.ssh/authorized_keys sur la tour" >&2
  return 1
}

cmd_ssh() {
  if [[ $# -eq 0 ]]; then
    remote -t "cd ${TOUR_S3T} && exec bash -l"
  else
    remote "$@"
  fi
}

cmd_mkdir_run() {
  local run_id="${1:?RUN_ID requis}"
  remote "mkdir -p ${REMOTE_RUNS}/${run_id}/eval"
  echo "OK: ${REMOTE_RUNS}/${run_id}/eval"
}

cmd_rsync_eval() {
  local run_filter="${1:-}"
  if [[ -n "${run_filter}" ]]; then
    local src="${LOCAL_RUNS}/${run_filter}/eval/"
    if [[ ! -d "${src}" ]]; then
      echo "ERROR: dossier local absent: ${src}" >&2
      exit 2
    fi
    cmd_mkdir_run "${run_filter}"
    rsync -avz --progress -e "${RSYNC_SSH}" \
      "${src}" \
      "${TOUR_USER}@${TOUR_HOST}:${REMOTE_RUNS}/${run_filter}/eval/"
    return 0
  fi

  echo "Rsync de tous les eval/ vers ${TOUR_USER}@${TOUR_HOST}:${REMOTE_RUNS}/"
  local count=0
  for eval_dir in "${LOCAL_RUNS}"/run_*/eval; do
    [[ -d "${eval_dir}" ]] || continue
    run_id="$(basename "$(dirname "${eval_dir}")")"
    cmd_mkdir_run "${run_id}"
    rsync -avz -e "${RSYNC_SSH}" \
      "${eval_dir}/" \
      "${TOUR_USER}@${TOUR_HOST}:${REMOTE_RUNS}/${run_id}/eval/"
    echo "  → ${run_id}"
    count=$((count + 1))
  done
  echo "Terminé (${count} run(s))."
}

main() {
  local cmd="${1:-}"
  shift || true
  case "${cmd}" in
    check) cmd_check "$@" ;;
    ssh) cmd_ssh "$@" ;;
    mkdir-run) cmd_mkdir_run "$@" ;;
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
