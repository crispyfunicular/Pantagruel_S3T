#!/usr/bin/env bash
# OVH — Chaîne open baselines (variante 6, piste K) :
#   1. run_072 : Whisper large-v3 ST (task=translate) — filet de sécurité
#   2. run_073 : SeamlessM4T v2 large — candidat #1 vs Gemini
#   3. run_074 : Canary-1B v2 (NeMo) — alternative NVIDIA
#
# Total estimé : ~6–18 h GPU (éval valid+test utterance, séquentiel).
# Prérequis : manifests fr-en, transformers ; NeMo installé par run_074 si absent.
#
# Lancement direct (GPU libre) :
#   nohup bash scripts/run_ovh_chain_open_072_then_073_then_074.sh \
#     > logs/run_chain_open_072_074_wrapper.log 2>&1 &
#
# Via waiter (après run_064 ou autre entraînement) :
#   nohup bash scripts/run_ovh_wait_064_done_then_open_072_074.sh \
#     > logs/run_waiter_open_072_074.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

MANIFESTS="datasets/manifests/fr-en"
SCRIPT_072="${ROOT}/scripts/run_ovh_open_whisper_st_072.sh"
SCRIPT_073="${ROOT}/scripts/run_ovh_open_seamless_073.sh"
SCRIPT_074="${ROOT}/scripts/run_ovh_open_canary_074.sh"

require_gpu_free() {
  if pgrep -af "^python.*pipeline\.py (train|run)" >/dev/null 2>&1; then
    echo "ERROR: entraînement GPU encore actif :" >&2
    pgrep -af "^python.*pipeline\.py (train|run)" >&2 || true
    exit 2
  fi
  echo "OK: GPU libre (pas de train/run actif)."
}

FORCE=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    -h|--help|help)
      sed -n '1,20p' "$0" | tail -n +2
      exit 0
      ;;
    *)
      echo "Argument inconnu: ${arg}" >&2
      exit 2
      ;;
  esac
done

[[ "$FORCE" != "1" ]] && require_gpu_free

for split in valid test; do
  if [[ ! -f "${MANIFESTS}/${split}.tsv" ]]; then
    echo "ERROR: manquant ${MANIFESTS}/${split}.tsv" >&2
    exit 2
  fi
done

mkdir -p logs

bleu_from_run() {
  local run_id="$1"
  local path="runs/fr-en/${run_id}/eval/sacrebleu_test.txt"
  if [[ -f "$path" ]]; then
    head -1 "$path"
  else
    echo "n/a"
  fi
}

# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 — Whisper-ST  (run_072)
# ══════════════════════════════════════════════════════════════════════════════
echo "=== $(date -Is) [1/3] LANCEMENT run_072 Whisper-ST ==="
bash "$SCRIPT_072"
BLEU_072="$(bleu_from_run run_072_open_whisper_st_utterance)"
echo "=== $(date -Is) [1/3] run_072 — BLEU test : ${BLEU_072} ==="

# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 — SeamlessM4T v2  (run_073)
# ══════════════════════════════════════════════════════════════════════════════
echo "=== $(date -Is) [2/3] LANCEMENT run_073 SeamlessM4T v2 ==="
bash "$SCRIPT_073"
BLEU_073="$(bleu_from_run run_073_open_seamlessm4t_v2_utterance)"
echo "=== $(date -Is) [2/3] run_073 — BLEU test : ${BLEU_073} ==="

# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 3 — Canary-1B  (run_074)
# ══════════════════════════════════════════════════════════════════════════════
echo "=== $(date -Is) [3/3] LANCEMENT run_074 Canary-1B ==="
bash "$SCRIPT_074"
BLEU_074="$(bleu_from_run run_074_open_canary_1b_utterance)"
echo "=== $(date -Is) [3/3] run_074 — BLEU test : ${BLEU_074} ==="

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  CHAÎNE OPEN BASELINES 072→073→074 TERMINÉE                         ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║  run_072  Whisper-ST large-v3          : ${BLEU_072}"
echo "║  run_073  SeamlessM4T v2 large         : ${BLEU_073}"
echo "║  run_074  Canary-1B v2                 : ${BLEU_074}"
echo "╚══════════════════════════════════════════════════════════════════════╝"
