#!/usr/bin/env bash
# OVH — Chaîne en deux étapes (GPU libre supposé) :
#   1. run_055 : speechLLM L-14k + Llama-3.2-3B seed 2 (~3 h GPU)
#      Réplicabilité run_052 (16,31 test, seed 42) — piste F.
#   2. run_052_transformer : ST L-14k v12 SPM 5k + gel encodeur 15k (~8–12 h GPU)
#      Hypothèse piste E : gel 15k aide l'espace 5k sous-mots (run_031 gel 5k = 24,02).
#
# Total estimé : ~11–15 h GPU. Budget OVH : ≤ 14 h par étape (MAX_RUN_HOURS).
#
# Prérequis :
#   - HF_TOKEN ou `hf auth login` (Llama-3.2-3B gated)
#   - manifests utterance fr-en présents (datasets/manifests/fr-en/*.tsv)
#
# Depuis le ThinkPad :
#   rsync -avz --exclude '.venv' --exclude 'runs/' --exclude 'datasets/' \
#     -e ssh ./ ubuntu@145.239.52.158:~/S3T/
#   ssh ubuntu@145.239.52.158 'cd ~/S3T && source .venv/bin/activate && mkdir -p logs && \
#     nohup bash scripts/run_ovh_chain_055_speechllm_llama_seed2_then_052_st_spm5k_freeze15k.sh \
#     > logs/run_chain_055_052_wrapper.log 2>&1 &'
#
# Surveillance :
#   ssh ubuntu@145.239.52.158 'tail -f ~/S3T/logs/run_chain_055_052_wrapper.log'
#   ssh ubuntu@145.239.52.158 'tail -f ~/S3T/logs/run_055_speechllm_b2bis_utterance_large_14k_llama32_3b_seed2_train_eval.log'
#   ssh ubuntu@145.239.52.158 'tail -f ~/S3T/logs/run_052_transformer_baseline_utterance_large_14k_v12_spm5k_freeze15k_spm_train_eval.log'
#
# Stopper la chaîne sans tuer un run en cours :
#   pkill -f run_ovh_chain_055_speechllm_llama_seed2_then_052_st_spm5k_freeze15k.sh
#
# Forcer (GPU occupé, déconseillé) :
#   bash scripts/run_ovh_chain_055_speechllm_llama_seed2_then_052_st_spm5k_freeze15k.sh --force

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

MAX_HOURS_SPEECHLLM="${MAX_HOURS_SPEECHLLM:-5}"
MAX_HOURS_ST="${MAX_HOURS_ST:-13}"

# ── Vérification GPU libre ────────────────────────────────────────────────────

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
      sed -n '1,30p' "$0" | tail -n +2
      exit 0
      ;;
    *)
      echo "Argument inconnu: ${arg}" >&2
      exit 2
      ;;
  esac
done

[[ "$FORCE" != "1" ]] && require_gpu_free

# ── Vérification HF (Llama gated) ────────────────────────────────────────────

if [[ -z "${HF_TOKEN:-}" ]] && ! hf auth whoami >/dev/null 2>&1; then
  echo "WARN: HF_TOKEN absent et hf non connecté — Llama-3.2-3B est gated." >&2
  echo "  → hf auth login  (puis relancer ce script)" >&2
fi

# ── Vérification manifests ────────────────────────────────────────────────────

MANIFESTS="datasets/manifests/fr-en"
for split in train valid test; do
  if [[ ! -f "${MANIFESTS}/${split}.tsv" ]]; then
    echo "ERROR: manquant ${MANIFESTS}/${split}.tsv" >&2
    echo "  → python scripts_communs/pipeline.py prepare --langpair fr-en" >&2
    exit 2
  fi
done

mkdir -p logs

# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 — speechLLM L-14k + Llama-3.2-3B seed 2  (run_055)
# ══════════════════════════════════════════════════════════════════════════════

CFG_SPEECHLLM="2_speechLLM/configs/fr-en/b1_utterance_large_14k_llama32_3b_seed2.yaml"
RUN_SPEECHLLM="run_055_speechllm_b2bis_utterance_large_14k_llama32_3b_seed2"
LOG_SPEECHLLM="logs/${RUN_SPEECHLLM}_train_eval.log"

echo "=== $(date -Is) [1/2] Dry-run ${RUN_SPEECHLLM} ==="
python 2_speechLLM/pipeline.py train \
  --config "$CFG_SPEECHLLM" \
  --run-id "$RUN_SPEECHLLM" \
  --dry-run

echo "=== $(date -Is) [1/2] LANCEMENT ${RUN_SPEECHLLM} (max ${MAX_HOURS_SPEECHLLM}h) ==="
{
  timeout "$((MAX_HOURS_SPEECHLLM * 3600))" \
    python 2_speechLLM/pipeline.py run \
      --config "$CFG_SPEECHLLM" \
      --run-id "$RUN_SPEECHLLM" \
      -v
  ec=$?
  if [[ "$ec" -eq 124 ]]; then
    echo "=== $(date -Is) [1/2] TIMEOUT ${MAX_HOURS_SPEECHLLM}h — éval du meilleur checkpoint ==="
    python 2_speechLLM/pipeline.py evaluate \
      --config "$CFG_SPEECHLLM" \
      --run-id "${RUN_SPEECHLLM}_timeout_eval" || true
    exit 124
  fi
  [[ "$ec" -ne 0 ]] && exit "$ec"
  echo "=== $(date -Is) [1/2] DONE ${RUN_SPEECHLLM} ==="
} 2>&1 | tee -a "$LOG_SPEECHLLM"

# Résumé BLEU
BLEU_TEST_1="n/a"
BLEU_TEST_FILE="runs/fr-en/${RUN_SPEECHLLM}/eval/sacrebleu_test.txt"
if [[ -f "$BLEU_TEST_FILE" ]]; then
  BLEU_TEST_1="$(head -1 "$BLEU_TEST_FILE")"
fi
echo "=== $(date -Is) [1/2] ${RUN_SPEECHLLM} — BLEU test : ${BLEU_TEST_1} ==="

# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 — ST L-14k v12 SPM 5k + gel encodeur 15k  (run_052_transformer)
# ══════════════════════════════════════════════════════════════════════════════

CFG_ST="1_Transformer/configs/fr-en/base_utterance_large_14k_v12_spm5k_freeze15k.yaml"
RUN_ST="run_052_transformer_baseline_utterance_large_14k_v12_spm5k_freeze15k"
LOG_ST="logs/${RUN_ST}_spm_train_eval.log"
SPM_MODEL="datasets/processed/spm/fr-en_5000.model"

echo "=== $(date -Is) [2/2] Préparation SPM 5k (si absent) ==="
if [[ ! -f "$SPM_MODEL" ]]; then
  TARGET_TXT="${MANIFESTS}/train.target.txt"
  if [[ ! -f "$TARGET_TXT" ]]; then
    python -c "
import csv
from pathlib import Path
manifest = Path('${MANIFESTS}/train.tsv')
target   = Path('${TARGET_TXT}')
with manifest.open(encoding='utf-8') as fi, target.open('w', encoding='utf-8') as fo:
    for row in csv.DictReader(fi, delimiter='\t'):
        fo.write(row['tgt_text'].strip() + '\n')
"
  fi
  python 1_Transformer/3_spm.py \
    --langpair fr-en \
    --vocab-size 5000 \
    --manifests-root datasets/manifests \
    --train-text "$TARGET_TXT" \
    --overwrite
else
  echo "SPM existant : ${SPM_MODEL}"
fi

echo "=== $(date -Is) [2/2] Dry-run ${RUN_ST} ==="
python 1_Transformer/pipeline.py train \
  --config "$CFG_ST" \
  --run-id "$RUN_ST" \
  --dry-run

echo "=== $(date -Is) [2/2] LANCEMENT ${RUN_ST} (max ${MAX_HOURS_ST}h) ==="
{
  timeout "$((MAX_HOURS_ST * 3600))" \
    python 1_Transformer/pipeline.py train \
      --config "$CFG_ST" \
      --run-id "$RUN_ST" \
      -v
  ec=$?
  if [[ "$ec" -eq 124 ]]; then
    echo "=== $(date -Is) [2/2] TIMEOUT ${MAX_HOURS_ST}h — éval du meilleur checkpoint ==="
    python 1_Transformer/pipeline.py evaluate \
      --config "$CFG_ST" \
      --run-id "$RUN_ST" \
      --beam-size 5 -v || true
  elif [[ "$ec" -ne 0 ]]; then
    exit "$ec"
  else
    echo "=== $(date -Is) [2/2] EVALUATE ${RUN_ST} (beam 5) ==="
    python 1_Transformer/pipeline.py evaluate \
      --config "$CFG_ST" \
      --run-id "$RUN_ST" \
      --beam-size 5 -v
    echo "=== $(date -Is) [2/2] DONE ${RUN_ST} ==="
  fi
} 2>&1 | tee -a "$LOG_ST"

# Résumé final
BLEU_TEST_2="n/a"
BLEU_TEST_FILE2="runs/fr-en/${RUN_ST}/eval/sacrebleu_test.txt"
if [[ -f "$BLEU_TEST_FILE2" ]]; then
  BLEU_TEST_2="$(head -1 "$BLEU_TEST_FILE2")"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  CHAÎNE OVH TERMINÉE                                                ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║  run_055  speechLLM Llama L-14k seed 2 : ${BLEU_TEST_1}"
echo "║  run_052  ST L-14k SPM 5k gel 15k       : ${BLEU_TEST_2}"
echo "╚══════════════════════════════════════════════════════════════════════╝"
