#!/usr/bin/env bash
set -euo pipefail
cd /home/getalp/bonapelm/S3T
chmod +x run_oar_eval_043_st_replicate_aker.sh
# Libérer le job en attente (bloqué sur lig-gpu10) et relancer sans contrainte d'hôte.
oarstat -j 129743 >/dev/null 2>&1 && oardel 129743 || true
sleep 2
OUT="$(oarsub -S -l /gpu=1,walltime=02:00:00 -n run_043_eval_st_replicate \
  /home/getalp/bonapelm/S3T/run_oar_eval_043_st_replicate_aker.sh 2>&1)"
echo "${OUT}"
oarstat -u bonapelm | head -10
