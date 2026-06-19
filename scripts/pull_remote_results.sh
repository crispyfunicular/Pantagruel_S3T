#!/usr/bin/env bash
# Rappatrie eval/ + metrics depuis Modyco et OVH vers runs/fr-en/ local.
#
# Usage :
#   ./scripts/pull_remote_results.sh
#   ./scripts/pull_remote_results.sh run_020_transformer_baseline_utterance_large_14k_v3

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OVH_HOST="${OVH_HOST:-ubuntu@145.239.52.158}"
TOUR_USER="${TOUR_USER:-mpellissier}"
TOUR_HOST="${TOUR_HOST:-10.8.0.2}"
SSH_ID="${SSH_IDENTITY_FILE:-$HOME/.ssh/id_ed25519}"
SSH_M="ssh -i ${SSH_ID} -o IdentitiesOnly=yes -o BatchMode=yes"
LOCAL_RUNS="${ROOT}/runs/fr-en"

DEFAULT_MODY=(
  run_004_transformer_baseline_utterance_v2
  run_014_transformer_baseline_utterance_large_14k_v2
  run_020_transformer_baseline_utterance_large_14k_v3
  run_021_speechllm_b1_utterance_large_14k_v3
  run_023_speechllm_b1_utterance_large_14k_replicate
  run_024_transformer_baseline_utterance_large_14k_v4
  run_026_transformer_baseline_utterance_large_14k_v5
  run_027_transformer_baseline_utterance_large_14k_v6_long
  run_031_transformer_baseline_utterance_large_14k_v7_spm5k
  run_034_transformer_baseline_utterance_large_14k_v8_spm8k
  run_035_transformer_baseline_utterance_b1k_v5
  run_037_transformer_baseline_utterance_large_14k_v9_specaug_strong
  run_041_finetune_utterance_large_14k_v10_specaug_freq
  run_043_transformer_baseline_utterance_large_14k_v5_replicate
  run_049_transformer_baseline_utterance_large_14k_v5_seed2
  run_046_transformer_baseline_utterance_large_14k_v11_batch32
  run_006_speechllm_b1_utterance_unfreeze
)
DEFAULT_OVH=(
  run_012_speechllm_b1_utterance_large_14k
  run_013_speechllm_b1_utterance_large_114k
  run_016_transformer_baseline_utterance_large_114k_v2
  run_017_speechllm_b1_utterance_large_114k_v2
  run_019_transformer_baseline_utterance_large_114k_v3
  run_022_speechllm_b1_utterance_large_114k_v3
  run_025_transformer_baseline_utterance_large_114k_v4
  run_028_transformer_baseline_utterance_large_114k_v5
  run_032_speechllm_b1_utterance_large_114k_replicate
  run_033_transformer_baseline_utterance_large_114k_v7_spm5k
)

pull_one() {
  local host="$1"
  local run="$2"
  local dest="${LOCAL_RUNS}/${run}"
  mkdir -p "${dest}/eval"
  rsync -az "${host}:~/S3T/runs/fr-en/${run}/eval/" "${dest}/eval/" 2>/dev/null || true
  rsync -az "${host}:~/S3T/runs/fr-en/${run}/metrics.json" \
    "${host}:~/S3T/runs/fr-en/${run}/config.yaml" \
    "${host}:~/S3T/runs/fr-en/${run}/train.log" \
    "${dest}/" 2>/dev/null || true
  if [[ -f "${dest}/eval/sacrebleu_test.txt" ]]; then
    echo "  ${run}: $(head -1 "${dest}/eval/sacrebleu_test.txt")"
  else
    echo "  ${run}: (pas d'éval test)"
  fi
}

if [[ $# -ge 1 ]]; then
  run="$1"
  echo "Pull ${run}…"
  pull_one "${TOUR_USER}@${TOUR_HOST}" "$run"
  pull_one "${OVH_HOST}" "$run"
  exit 0
fi

echo "=== Modyco ==="
for run in "${DEFAULT_MODY[@]}"; do
  pull_one "${TOUR_USER}@${TOUR_HOST}" "$run"
done
echo "=== OVH ==="
for run in "${DEFAULT_OVH[@]}"; do
  pull_one "${OVH_HOST}" "$run"
done

echo "=== Logs distants (légers) ==="
mkdir -p "${ROOT}/logs/remote_modyco" "${ROOT}/logs/remote_ovh"
rsync -az -e "${SSH_M}" "${TOUR_USER}@${TOUR_HOST}:~/S3T/logs/run_02"* \
  "${TOUR_USER}@${TOUR_HOST}:~/S3T/logs/run_03"* \
  "${ROOT}/logs/remote_modyco/" 2>/dev/null || true
rsync -az -e "ssh -o BatchMode=yes" "${OVH_HOST}:~/S3T/logs/run_02"* \
  "${OVH_HOST}:~/S3T/logs/run_03"* \
  "${ROOT}/logs/remote_ovh/" 2>/dev/null || true
echo "  logs → logs/remote_modyco/ et logs/remote_ovh/"
