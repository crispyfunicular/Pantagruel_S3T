#!/usr/bin/env bash
# Smoke test serveur IMAG — ligone (bastion) → aker (login) → lig-gpu1 (GPU).
#
# Run léger pour valider déploiement, dépendances Python, accès HF et GPU.
# Par défaut : encodeur Pantagruel 1k (forward minimal, ~quelques minutes).
# Option --with-cascade : évaluation cascade ASR→MT sur 5 segments (~10–20 min GPU).
#
# IMPORTANT : aker (login) a un ulimit mémoire ~80 Mo — PyTorch ne tourne pas dessus.
# Le smoke GPU s'exécute sur lig-gpu1 (3e saut SSH, même $HOME NFS).
#
# Depuis le poste local :
#   ./scripts/run_aker_smoke.sh check
#   ./scripts/run_aker_smoke.sh prepare-gpu-access   # clé SSH aker → lig-gpu1
#   ./scripts/run_aker_smoke.sh all
#
# Manuel (ligone → aker → lig-gpu1) :
#   cd ~/S3T && source .venv/bin/activate
#   bash scripts/run_aker_smoke.sh run-local

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

LIGONE_JUMP="${LIGONE_JUMP:-bonapelm@ligone.imag.fr}"
AKER_USER="${AKER_USER:-bonapelm}"
AKER_HOST="${AKER_HOST:-aker.imag.fr}"
AKER_S3T="${AKER_S3T:-~/S3T}"
remote_s3t() {
  local aker_home
  aker_home="$(aker_home_dir)"
  ssh_aker "cd ${aker_home}/S3T && $*"
}
GPU_HOST="${GPU_HOST:-lig-gpu1.imag.fr}"

RUN_ID_ENCODER="${RUN_ID_ENCODER:-run_000_aker_encoder_smoke}"
RUN_ID_CASCADE="${RUN_ID_CASCADE:-run_000_aker_cascade_smoke5}"
CASCADE_LIMIT="${CASCADE_LIMIT:-5}"

SSH_ID="${SSH_IDENTITY_FILE:-$HOME/.ssh/id_ed25519}"
if [[ ! -f "${SSH_ID}" ]]; then
  echo "ERROR: clé SSH locale introuvable: ${SSH_ID}" >&2
  exit 2
fi

# Clé locale : uniquement pour le 1er saut (poste → ligone). Les sauts suivants
# réutilisent les clés déjà configurées sur ligone / aker.
SSH_LOCAL=(
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

ssh_ligone() {
  ssh "${SSH_LOCAL[@]}" "${LIGONE_JUMP}" "$@"
}

aker_home_dir() {
  ssh_ligone ssh "${SSH_NESTED[@]}" "${AKER_USER}@${AKER_HOST}" 'printf %s "$HOME"'
}

ssh_aker() {
  # Ligone doit recevoir une seule commande imbriquée (sinon le cd s'exécute sur ligone).
  local nested=(ssh "${SSH_NESTED[@]}" "${AKER_USER}@${AKER_HOST}")
  ssh_ligone "$(printf '%q ' "${nested[@]}")$(printf '%q' "$*")"
}

ssh_gpu() {
  local gpu_host="${1:?GPU_HOST requis}"
  shift
  local inner=(ssh "${SSH_NESTED[@]}" -o ConnectTimeout=15 "${AKER_USER}@${gpu_host}")
  local outer=(ssh "${SSH_NESTED[@]}" "${AKER_USER}@${AKER_HOST}")
  ssh_ligone "$(printf '%q ' "${outer[@]}")$(printf '%q ' "${inner[@]}")$(printf '%q' "$*")"
}

rsync_to_aker() {
  local src="${1:?}"
  local dst="${2:?}"
  rsync -az --progress \
    -e "ssh ${SSH_LOCAL[*]} ${LIGONE_JUMP} ssh ${SSH_NESTED[*]}" \
    "${src}" "${AKER_USER}@${AKER_HOST}:${dst}"
}

usage() {
  sed -n '1,20p' "$0" | tail -n +2
  echo ""
  echo "Commandes : check | deploy | prepare-gpu-access | setup-gpu | rsync-smoke-data"
  echo "            preflight | run | run-local | all"
  echo ""
  echo "Variables : LIGONE_JUMP AKER_USER AKER_HOST AKER_S3T GPU_HOST SSH_IDENTITY_FILE"
  echo "Options run/all : --with-cascade  --gpu-host HOST"
}

cmd_check() {
  echo "=== ligone ==="
  ssh_ligone "echo OK — \$(hostname) — \$(whoami)"
  echo "=== aker (via ligone) ==="
  ssh_aker "echo OK — \$(hostname) — \$(whoami) — ${AKER_S3T}"
  echo "=== lig-gpu1 (via aker) ==="
  if ssh_gpu "${GPU_HOST}" "echo OK — \$(hostname) — \$(whoami)"; then
    echo "GPU: connexion OK."
  else
    echo "WARN: lig-gpu1 inaccessible (clé SSH ou politique labo) — le smoke encodeur devra être lancé manuellement depuis aker." >&2
    return 0
  fi
}

cmd_deploy() {
  echo "Déploiement code S3T → ${AKER_USER}@${AKER_HOST}:${AKER_S3T}/"
  local aker_home
  aker_home="$(aker_home_dir)"
  ssh_aker "mkdir -p ${aker_home}/S3T"
  rsync_to_aker \
    "${ROOT}/scripts_communs/" \
    "${AKER_S3T}/scripts_communs/"
  rsync_to_aker "${ROOT}/scripts/" "${AKER_S3T}/scripts/"
  rsync_to_aker "${ROOT}/4_cascade/" "${AKER_S3T}/4_cascade/"
  rsync_to_aker "${ROOT}/requirements.txt" "${AKER_S3T}/requirements.txt"
  rsync_to_aker "${ROOT}/pyproject.toml" "${AKER_S3T}/pyproject.toml"
  echo "OK: code déployé."
}

cmd_rsync_smoke_data() {
  # 5 WAV + manifest valid (segments sentence_like) pour cascade --limit 5.
  local manifest="${ROOT}/datasets/manifests_sentence/fr-en/valid.tsv"
  local wav_dir="${ROOT}/datasets/processed_sentence/fr-en/valid"
  if [[ ! -f "${manifest}" ]]; then
    echo "ERROR: manifest local absent: ${manifest}" >&2
    echo "  Lancez d'abord prepare (sentence_like) en local." >&2
    exit 2
  fi
  local aker_home
  aker_home="$(aker_home_dir)"
  ssh_aker "mkdir -p ${aker_home}/S3T/datasets/manifests_sentence/fr-en ${aker_home}/S3T/datasets/processed_sentence/fr-en/valid"
  rsync_to_aker "${manifest}" "${AKER_S3T}/datasets/manifests_sentence/fr-en/valid.tsv"
  local ids=(9fxo9YJhnG8_m0 9fxo9YJhnG8_m1 9fxo9YJhnG8_m2 9fxo9YJhnG8_m3 9fxo9YJhnG8_m4)
  for id in "${ids[@]}"; do
    rsync_to_aker "${wav_dir}/${id}.wav" "${AKER_S3T}/datasets/processed_sentence/fr-en/valid/${id}.wav"
  done
  echo "OK: données smoke (5 WAV + valid.tsv)."
}

remote_setup_script() {
  cat <<'SETUP'
set -euo pipefail
cd "${HOME}/S3T"
export PATH="${HOME}/.local/bin:${PATH}"
# aker (login) a peu de RAM : installs séquentielles, sans cache pip.
export PIP_NO_CACHE_DIR=1

if [[ ! -x .venv/bin/python ]]; then
  echo "==> Création .venv (virtualenv si python3-venv absent)"
  if ! python3 -m venv .venv 2>/dev/null; then
    if ! command -v virtualenv >/dev/null 2>&1; then
      python3 -m pip install --user --break-system-packages virtualenv
      export PATH="${HOME}/.local/bin:${PATH}"
    fi
    virtualenv .venv
  fi
fi
# shellcheck source=/dev/null
source .venv/bin/activate
python -m pip install -U pip wheel
echo "==> Dépendances smoke (torch CUDA si disponible)"
pip install -q pyyaml sacrebleu soundfile
if python -c "import torch" 2>/dev/null; then
  echo "torch déjà installé"
else
  pip install -q torch torchaudio || pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cpu
fi
pip install -q transformers
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
SETUP
}

cmd_prepare_gpu_access() {
  echo "Préparation clé SSH aker → ${GPU_HOST}…"
  ssh_aker "bash -s" <<'KEY'
set -euo pipefail
mkdir -m 700 -p ~/.ssh
if [[ ! -f ~/.ssh/id_ed25519 ]]; then
  ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519 -C "bonapelm@aker-smoke"
  echo "Clé créée sur aker."
else
  echo "Clé existante sur aker."
fi
echo ""
echo "=== Clé publique à autoriser sur lig-gpu1 (~/.ssh/authorized_keys) ==="
cat ~/.ssh/id_ed25519.pub
echo ""
echo "Puis tester depuis aker : ssh ${USER}@lig-gpu1.imag.fr hostname"
KEY
}

cmd_setup_gpu() {
  echo "Setup Python sur ${GPU_HOST} (NFS partagé : ~/S3T)…"
  ssh_gpu "${GPU_HOST}" "bash -s" <<<"$(remote_setup_script)"
  echo "OK: .venv prêt sur ${GPU_HOST}."
}

cmd_preflight() {
  echo "Preflight léger sur aker (login — pas de torch, ulimit ~80 Mo)…"
  remote_s3t 'python3 scripts_communs/0_preflight.py --dry-run --min-disk-gb 5'
}

remote_run_encoder() {
  cat <<RUN
set -euo pipefail
cd "\${HOME}/S3T"
source .venv/bin/activate
echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "=== smoke encodeur Pantagruel 1k ==="
python scripts/smoke_pantagruel_encoders.py --encoders 1k
RUN
}

remote_run_cascade() {
  cat <<RUN
set -euo pipefail
cd "\${HOME}/S3T"
source .venv/bin/activate
echo "=== cascade smoke (${CASCADE_LIMIT} segments) ==="
python 4_cascade/pipeline.py evaluate \
  --config 4_cascade/configs/fr-en/cascade_sentence.yaml \
  --run-id ${RUN_ID_CASCADE} \
  --limit ${CASCADE_LIMIT} -v
RUN
}

cmd_run() {
  local with_cascade=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --with-cascade) with_cascade=1 ;;
      --gpu-host) GPU_HOST="$2"; shift ;;
      -h|--help) usage; exit 0 ;;
      *) echo "Option inconnue: $1" >&2; exit 2 ;;
    esac
    shift
  done

  echo "Lancement smoke encodeur sur ${GPU_HOST}…"
  if ! ssh_gpu "${GPU_HOST}" "$(remote_run_encoder)"; then
    echo "" >&2
    echo "ÉCHEC GPU. Connexion manuelle :" >&2
    echo "  ssh ${LIGONE_JUMP}" >&2
    echo "  ssh ${AKER_USER}@${AKER_HOST}" >&2
    echo "  ssh ${GPU_HOST}" >&2
    echo "  cd ~/S3T && source .venv/bin/activate && bash scripts/run_aker_smoke.sh run-local" >&2
    exit 1
  fi

  if [[ "${with_cascade}" -eq 1 ]]; then
    echo "Lancement cascade smoke sur ${GPU_HOST}…"
    ssh_gpu "${GPU_HOST}" "$(remote_run_cascade)"
  fi
  echo "OK: smoke terminé."
}

cmd_run_local() {
  # À exécuter déjà connecté sur un nœud GPU (même \$HOME NFS que aker).
  cd "${ROOT}"
  # shellcheck source=/dev/null
  source .venv/bin/activate
  echo "=== nvidia-smi ==="
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
  python scripts/smoke_pantagruel_encoders.py --encoders 1k
}

cmd_all() {
  local extra=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --with-cascade) extra+=("--with-cascade") ;;
      --gpu-host)
        extra+=("--gpu-host" "$2")
        shift
        ;;
      -h|--help) usage; exit 0 ;;
      *) echo "Option inconnue: $1" >&2; exit 2 ;;
    esac
    shift
  done

  cmd_check
  cmd_deploy
  cmd_prepare_gpu_access
  cmd_preflight
  local aker_home
  aker_home="$(aker_home_dir)"
  if ! ssh_gpu "${GPU_HOST}" "test -x ${aker_home}/S3T/.venv/bin/python" 2>/dev/null; then
    echo "Setup GPU requis (lig-gpu1 doit accepter la clé aker)…" >&2
    cmd_setup_gpu
  fi
  if [[ " ${extra[*]} " == *" --with-cascade "* ]]; then
    cmd_rsync_smoke_data
  fi
  cmd_run "${extra[@]}"
}

main() {
  local cmd="${1:-}"
  shift || true
  case "${cmd}" in
    check) cmd_check "$@" ;;
    deploy) cmd_deploy "$@" ;;
    prepare-gpu-access) cmd_prepare_gpu_access "$@" ;;
    setup-gpu) cmd_setup_gpu "$@" ;;
    rsync-smoke-data) cmd_rsync_smoke_data "$@" ;;
    preflight) cmd_preflight "$@" ;;
    run) cmd_run "$@" ;;
    run-local) cmd_run_local "$@" ;;
    all) cmd_all "$@" ;;
    -h|--help|help|"") usage ;;
    *)
      echo "Commande inconnue: ${cmd}" >&2
      usage >&2
      exit 2
      ;;
  esac
}

main "$@"
