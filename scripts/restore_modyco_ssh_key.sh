#!/usr/bin/env bash
# Vérifie que ~/.ssh/id_ed25519 correspond à ~/.ssh/id_ed25519.pub (Modyco).
#
# Usage (après avoir recopié votre clé privée d'origine) :
#   cp /chemin/vers/votre/sauvegarde/id_ed25519 ~/.ssh/id_ed25519
#   chmod 600 ~/.ssh/id_ed25519
#   ./scripts/restore_modyco_ssh_key.sh
#   ./scripts/tour.sh check

set -euo pipefail

PRIV="${HOME}/.ssh/id_ed25519"
PUB="${HOME}/.ssh/id_ed25519.pub"

if [[ ! -f "${PRIV}" ]]; then
  echo "ERROR: clé privée absente: ${PRIV}" >&2
  echo "Recopiez votre sauvegarde id_ed25519 (même paire qu'avant), puis relancez." >&2
  exit 2
fi

if [[ ! -f "${PUB}" ]]; then
  echo "ERROR: clé publique absente: ${PUB}" >&2
  exit 2
fi

chmod 600 "${PRIV}"

if ! head -1 "${PRIV}" | grep -q 'BEGIN OPENSSH PRIVATE KEY'; then
  echo "ERROR: ${PRIV} n'est pas une clé OpenSSH (fichier invalide)." >&2
  exit 2
fi

derived="$(ssh-keygen -y -f "${PRIV}" 2>/dev/null || true)"
expected="$(tr -d '\r\n' < "${PUB}" | awk '{print $1" "$2}')"
got="$(echo "${derived}" | tr -d '\r\n' | awk '{print $1" "$2}')"

if [[ "${got}" != "${expected}" ]]; then
  echo "ERROR: la clé privée ne correspond pas à ${PUB}" >&2
  echo "  attendu : ${expected}" >&2
  echo "  obtenu  : ${got}" >&2
  echo "Utilisez la sauvegarde de la paire d'origine (morgane@ThinkPad)." >&2
  exit 2
fi

echo "OK: paire de clés cohérente."
echo "Test Modyco :"
ssh -i "${PRIV}" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15 \
  mpellissier@10.8.0.2 'echo OK — $(hostname)'
