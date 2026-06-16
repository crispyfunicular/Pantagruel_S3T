#!/usr/bin/env bash
# OVH — attend la fin de run_033 (L-114k SPM 5k) puis enchaîne run_038 (SpecAugment freq)
# puis run_042 (warmup 10k + SpecAugment).
#
# Usage (sur OVH) :
#   cd ~/S3T && source .venv/bin/activate
#   nohup bash scripts/run_ovh_wait_spm5k_then_st_114k_v10.sh \
#     > logs/chain_038_042_ovh_wait.log 2>&1 &
#   tail -f logs/chain_038_042_ovh_wait.log

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

POLL_SEC="${POLL_SEC:-300}"

echo "=== $(date -Is) Attente fin run_033 (poll ${POLL_SEC}s) ==="
while pgrep -af "^python.*pipeline\.py (train|run)" >/dev/null 2>&1; do
  pgrep -af "^python.*pipeline\.py (train|run)" | head -1 || true
  echo "$(date -Is) GPU occupée — nouvelle vérif dans ${POLL_SEC}s"
  sleep "$POLL_SEC"
done

echo "=== $(date -Is) GPU libre — run_038 (L-114k SpecAugment freq) ==="
chmod +x \
  "${ROOT}/1_Transformer/scripts/run_038_baseline_utterance_large_114k_v9_specaug_freq_nohup.sh" \
  "${ROOT}/1_Transformer/scripts/run_042_baseline_utterance_large_114k_v10_warmup10k_nohup.sh"

bash "${ROOT}/1_Transformer/scripts/run_038_baseline_utterance_large_114k_v9_specaug_freq_nohup.sh"

echo "=== $(date -Is) GPU libre — run_042 (L-114k warmup 10k + SpecAugment) ==="
bash "${ROOT}/1_Transformer/scripts/run_042_baseline_utterance_large_114k_v10_warmup10k_nohup.sh"

echo "=== $(date -Is) CHAÎNE OVH TERMINÉE ==="
