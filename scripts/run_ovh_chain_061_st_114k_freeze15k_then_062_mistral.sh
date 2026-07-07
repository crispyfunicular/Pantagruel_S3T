#!/usr/bin/env bash
# OVH — Chaîne en deux étapes (GPU libre supposé) :
#   1. run_061 : ST L-114k SPM 5k + gel encodeur 15k (~10–14 h GPU)
#      Piste E sur encodeur gated — vs run_033 gel 5k (**25,10** test).
#   2. run_062 : speechLLM L-14k + Mistral-7B 4-bit (~12–15 h GPU)
#      Réplication OVH de run_054 Modyco (**14,22**) ; trop lourd pour IMAG 11 Go.
#
# Total estimé : ~22–29 h GPU.
#
# Prérequis : hf auth login (L-114k gated), manifests fr-en, bitsandbytes.
#
# Lancement direct (GPU libre) :
#   nohup bash scripts/run_ovh_chain_061_st_114k_freeze15k_then_062_mistral.sh \
#     > logs/run_chain_061_062_wrapper.log 2>&1 &
#
# Via waiter (après chaîne 055→052) :
#   nohup bash scripts/run_ovh_wait_chain_055_052_done_then_061_062.sh \
#     > logs/run_chain_061_062_waiter.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

MAX_HOURS_ST="${MAX_HOURS_ST:-14}"
MAX_HOURS_MISTRAL="${MAX_HOURS_MISTRAL:-15}"
MANIFESTS="datasets/manifests/fr-en"
SPM_MODEL="datasets/processed/spm/fr-en_5000.model"

require_gpu_free() {
  if pgrep -af "^python.*pipeline\.py (train|run)" >/dev/null 2>&1; then
    echo "ERROR: entraînement GPU encore actif :" >&2
    pgrep -af "^python.*pipeline\.py (train|run)" >&2 || true
    exit 2
  fi
  echo "OK: GPU libre."
}

FORCE=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    -h|--help|help)
      sed -n '1,22p' "$0" | tail -n +2
      exit 0
      ;;
    *)
      echo "Argument inconnu: ${arg}" >&2
      exit 2
      ;;
  esac
done

[[ "$FORCE" != "1" ]] && require_gpu_free

if [[ -z "${HF_TOKEN:-}" ]] && ! hf auth whoami >/dev/null 2>&1; then
  echo "WARN: HF non connecté — L-114k gated." >&2
fi

for split in train valid test; do
  if [[ ! -f "${MANIFESTS}/${split}.tsv" ]]; then
    echo "ERROR: manquant ${MANIFESTS}/${split}.tsv" >&2
    exit 2
  fi
done

mkdir -p logs

ensure_spm_5k() {
  if [[ -f "$SPM_MODEL" ]]; then
    echo "SPM existant : ${SPM_MODEL}"
    return 0
  fi
  local target_txt="${MANIFESTS}/train.target.txt"
  if [[ ! -f "$target_txt" ]]; then
    python -c "
import csv
from pathlib import Path
manifest = Path('${MANIFESTS}/train.tsv')
target   = Path('${target_txt}')
with manifest.open(encoding='utf-8') as fi, target.open('w', encoding='utf-8') as fo:
    for row in csv.DictReader(fi, delimiter='\t'):
        fo.write(row['tgt_text'].strip() + '\n')
"
  fi
  python 1_Transformer/3_spm.py \
    --langpair fr-en \
    --vocab-size 5000 \
    --manifests-root datasets/manifests \
    --train-text "$target_txt" \
    --overwrite
}

# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 — ST L-114k SPM 5k + gel 15k  (run_061)
# ══════════════════════════════════════════════════════════════════════════════

CFG_ST="1_Transformer/configs/fr-en/base_utterance_large_114k_v12_spm5k_freeze15k.yaml"
RUN_ST="run_061_transformer_baseline_utterance_large_114k_v12_spm5k_freeze15k"
LOG_ST="logs/${RUN_ST}_spm_train_eval.log"

echo "=== $(date -Is) [1/2] Préparation SPM 5k ==="
ensure_spm_5k

echo "=== $(date -Is) [1/2] Dry-run ${RUN_ST} ==="
python 1_Transformer/pipeline.py train --config "$CFG_ST" --run-id "$RUN_ST" --dry-run

echo "=== $(date -Is) [1/2] LANCEMENT ${RUN_ST} (max ${MAX_HOURS_ST}h) ==="
{
  timeout "$((MAX_HOURS_ST * 3600))" \
    python 1_Transformer/pipeline.py train \
      --config "$CFG_ST" \
      --run-id "$RUN_ST" \
      -v
  ec=$?
  if [[ "$ec" -eq 124 ]]; then
    echo "=== $(date -Is) [1/2] TIMEOUT ${MAX_HOURS_ST}h — éval best checkpoint ==="
    python 1_Transformer/pipeline.py evaluate \
      --config "$CFG_ST" \
      --run-id "$RUN_ST" \
      --beam-size 5 -v || true
  elif [[ "$ec" -ne 0 ]]; then
    exit "$ec"
  else
    echo "=== $(date -Is) [1/2] EVALUATE ${RUN_ST} (beam 5) ==="
    python 1_Transformer/pipeline.py evaluate \
      --config "$CFG_ST" \
      --run-id "$RUN_ST" \
      --beam-size 5 -v
  fi
} 2>&1 | tee -a "$LOG_ST"

BLEU_ST="n/a"
if [[ -f "runs/fr-en/${RUN_ST}/eval/sacrebleu_test.txt" ]]; then
  BLEU_ST="$(head -1 "runs/fr-en/${RUN_ST}/eval/sacrebleu_test.txt")"
fi
echo "=== $(date -Is) [1/2] ${RUN_ST} — BLEU test : ${BLEU_ST} ==="

# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 — speechLLM L-14k + Mistral-7B 4-bit  (run_062)
# ══════════════════════════════════════════════════════════════════════════════

CFG_SLM="2_speechLLM/configs/fr-en/b1_utterance_large_14k_mistral_7b_ovh.yaml"
RUN_SLM="run_062_speechllm_b2bis_utterance_large_14k_mistral_7b_ovh"
LOG_SLM="logs/${RUN_SLM}_train_eval.log"

echo "=== $(date -Is) [2/2] Dry-run ${RUN_SLM} ==="
python 2_speechLLM/pipeline.py train --config "$CFG_SLM" --run-id "$RUN_SLM" --dry-run

echo "=== $(date -Is) [2/2] LANCEMENT ${RUN_SLM} (max ${MAX_HOURS_MISTRAL}h) ==="
{
  timeout "$((MAX_HOURS_MISTRAL * 3600))" \
    python 2_speechLLM/pipeline.py run \
      --config "$CFG_SLM" \
      --run-id "$RUN_SLM" \
      -v
  ec=$?
  if [[ "$ec" -eq 124 ]]; then
    echo "=== $(date -Is) [2/2] TIMEOUT ${MAX_HOURS_MISTRAL}h — éval best checkpoint ==="
    python 2_speechLLM/pipeline.py evaluate \
      --config "$CFG_SLM" \
      --run-id "$RUN_SLM" -v || true
  elif [[ "$ec" -ne 0 ]]; then
    exit "$ec"
  fi
} 2>&1 | tee -a "$LOG_SLM"

BLEU_SLM="n/a"
if [[ -f "runs/fr-en/${RUN_SLM}/eval/sacrebleu_test.txt" ]]; then
  BLEU_SLM="$(head -1 "runs/fr-en/${RUN_SLM}/eval/sacrebleu_test.txt")"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  CHAÎNE OVH 061→062 TERMINÉE                                        ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║  run_061  ST L-114k SPM 5k gel 15k    : ${BLEU_ST}"
echo "║  run_062  speechLLM Mistral L-14k OVH  : ${BLEU_SLM}"
echo "╚══════════════════════════════════════════════════════════════════════╝"
