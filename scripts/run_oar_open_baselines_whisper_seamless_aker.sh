#!/usr/bin/env bash
# Job OAR IMAG — open baselines ST (variante 6) : Whisper-ST puis SeamlessM4T v2.
#
# Réplication cross-machine des runs OVH run_072 (**36,60**) et run_073 (**38,02**).
# Éval only (~2–4 h GPU sur 2080 Ti) — pas de Canary (NeMo → Modyco).
#
# Soumission depuis aker (login shell) :
#   oarsub -l /gpu=1,walltime=12:00:00 -n run_075_open_baselines_aker \
#     /home/getalp/bonapelm/S3T/scripts/run_oar_open_baselines_whisper_seamless_aker.sh

#OAR -l /gpu=1,walltime=12:00:00
#OAR -n run_075_open_baselines_aker
set -euo pipefail

cd "${HOME}/S3T"
source .venv/bin/activate
mkdir -p logs

CFG_WHISPER="6_open_baselines/configs/fr-en/whisper_large_v3_st.yaml"
CFG_SEAMLESS="6_open_baselines/configs/fr-en/seamless_m4t_v2_large.yaml"
RUN_WHISPER="run_075_aker_open_whisper_st_utterance"
RUN_SEAMLESS="run_075b_aker_open_seamlessm4t_v2_utterance"
LOG="logs/run_075_aker_open_baselines_eval.log"

echo "=== $(date -u -Iseconds) start open baselines IMAG on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

{
  echo "=== $(date -u -Iseconds) [1/2] Whisper-ST ${RUN_WHISPER} ==="
  python 6_open_baselines/pipeline.py evaluate \
    --config "${CFG_WHISPER}" \
    --run-id "${RUN_WHISPER}" \
    -v

  echo "=== $(date -u -Iseconds) [2/2] SeamlessM4T v2 ${RUN_SEAMLESS} ==="
  python 6_open_baselines/pipeline.py evaluate \
    --config "${CFG_SEAMLESS}" \
    --run-id "${RUN_SEAMLESS}" \
    -v
} 2>&1 | tee -a "${LOG}"

for run in "${RUN_WHISPER}" "${RUN_SEAMLESS}"; do
  path="runs/fr-en/${run}/eval/sacrebleu_test.txt"
  if [[ -f "$path" ]]; then
    echo "=== ${run} : $(head -1 "$path") ==="
  fi
done

echo "=== $(date -u -Iseconds) done exit=$? ==="
