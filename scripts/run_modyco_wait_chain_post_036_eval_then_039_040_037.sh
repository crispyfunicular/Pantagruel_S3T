#!/usr/bin/env bash
# Modyco — reprise plan amélioration run_026 (après run_036 interrompu).
#
# Ordre (plan § run_039 → run_040 → run_037 ; run_036 déjà tenté) :
#   1. Éval beam 5 de run_036 si best.pt sans eval test
#   2. run_039 speechLLM L-14k + SpecAugment (~2 h)
#   3. run_040 Speech_Text utterance v2 (~2–3 h)
#   4. run_037 ST L-14k SpecAugment fort (~8–10 h)
#
# Usage (Modyco, nohup soir / week-end) :
#   cd ~/S3T && source .venv/bin/activate
#   nohup bash scripts/run_modyco_wait_chain_post_036_eval_then_039_040_037.sh \
#     > logs/run_036_eval_039_040_037_modyco_wait_chain.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

POLL_SEC="${POLL_SEC:-300}"

wait_gpu_free() {
  echo "=== $(date -Is) Attente GPU libre (poll ${POLL_SEC}s) ==="
  while pgrep -af "^python.*pipeline\.py (train|run)" >/dev/null 2>&1; do
    pgrep -af "^python.*pipeline\.py (train|run)" | head -1 || true
    echo "$(date -Is) GPU occupée — nouvelle vérif dans ${POLL_SEC}s"
    sleep "$POLL_SEC"
  done
}

wait_gpu_free
echo "=== $(date -Is) Évaluation run_036 (best.pt) si nécessaire ==="
bash "${ROOT}/scripts/run_modyco_eval_st_14k_v9_warmup10k.sh"

wait_gpu_free
echo "=== $(date -Is) Lancement run_039 speechLLM L-14k SpecAugment ==="
bash "${ROOT}/scripts/run_modyco_speechllm_14k_v5_specaug.sh"

wait_gpu_free
echo "=== $(date -Is) Lancement run_040 Speech_Text utterance v2 ==="
bash "${ROOT}/scripts/run_modyco_multimodal_utterance_v2.sh"

wait_gpu_free
echo "=== $(date -Is) Lancement run_037 ST L-14k v9 SpecAugment fort ==="
bash "${ROOT}/scripts/run_modyco_st_14k_v9_specaug_strong.sh"

echo "=== $(date -Is) Chaîne Modyco post-036 (eval→039→040→037) terminée ==="
