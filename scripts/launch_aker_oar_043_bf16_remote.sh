#!/usr/bin/env bash
# Exécuter sur aker (login) : relance OAR run_043 ST replicate (bf16 overwrite).
set -euo pipefail
cd ~/S3T
mkdir -p logs

resolve_script() {
  local name="$1"
  if [[ -f "scripts/${name}" ]]; then
    echo "scripts/${name}"
  elif [[ -f "${name}" ]]; then
    echo "${name}"
  else
    echo "ERROR: script introuvable : ${name}" >&2
    exit 2
  fi
}

if oarstat 2>/dev/null | grep -q 'run_043_st_l14k_v5_replicate'; then
  echo "Job run_043 déjà actif :"
  oarstat 2>/dev/null | grep run_043_st_l14k_v5_replicate || true
  exit 0
fi

SCRIPT="$(resolve_script run_oar_st_l14k_v5_replicate_aker.sh)"
# OAR exige un chemin absolu.
SCRIPT="$(cd "$(dirname "${SCRIPT}")" && pwd)/$(basename "${SCRIPT}")"
chmod +x "${SCRIPT}"
# Garantir shebang exécutable pour oarsub -S
head -1 "${SCRIPT}" | grep -q '^#!' || { echo "ERROR: shebang manquant" >&2; exit 2; }

OAR_HOST="${OAR_HOST:-lig-gpu10.imag.fr}"
WALLTIME="${WALLTIME:-16:00:00}"
echo "=== $(date -Is) oarsub run_043_st_l14k_v5_replicate walltime=${WALLTIME} host=${OAR_HOST} (bf16 overwrite) ==="
OUT="$(oarsub -S -l "/gpu=1,walltime=${WALLTIME}" -p "host='${OAR_HOST}'" -n run_043_st_l14k_v5_replicate "${SCRIPT}" 2>&1)"
echo "${OUT}"
JOB_ID="$(echo "${OUT}" | grep -oE 'OAR_JOB_ID=[0-9]+' | cut -d= -f2 || true)"
if [[ -z "${JOB_ID}" ]]; then
  echo "ERROR: oarsub échoué" >&2
  exit 1
fi
echo "OAR_JOB_ID=${JOB_ID}"
sleep 3
oarstat -j "${JOB_ID}" 2>/dev/null || oarstat | grep "${JOB_ID}" || true
