#!/usr/bin/env bash
# Modyco — ST Pantagruel-L-14k v7 SPM 5k (run_031, piste 3 Table 8).
#
# Recette run_026 v5 + vocab SentencePiece 5000.
# Budget : ~8–10 h GPU. À enchaîner après run_027 v6 long.
#
# Depuis le poste local :
#   ./scripts/tour.sh ssh 'cd ~/S3T && source .venv/bin/activate && mkdir -p logs && \
#     nohup bash scripts/run_modyco_wait_st_then_st_14k_v7_spm5k.sh \
#     > logs/run_031_modyco_wait_chain.log 2>&1 &'
#
# Sur Modyco (GPU libre) :
#   cd ~/S3T && source .venv/bin/activate
#   mkdir -p logs
#   nohup bash scripts/run_modyco_st_14k_v7_spm5k.sh \
#     > logs/run_031_st_14k_v7_chain_wrapper.log 2>&1 &
#
# Forcer malgré un job GPU actif (déconseillé) :
#   bash scripts/run_modyco_st_14k_v7_spm5k.sh --force

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ST_SCRIPT="${ROOT}/1_Transformer/scripts/run_031_baseline_utterance_large_14k_v7_spm5k_nohup.sh"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

require_gpu_free() {
  if pgrep -af "^python.*pipeline\.py (train|run)" >/dev/null 2>&1; then
    echo "ERROR: entraînement GPU encore actif sur Modyco :" >&2
    pgrep -af "^python.*pipeline\.py (train|run)" >&2 || true
    exit 2
  fi
  echo "OK: aucun pipeline.py train/run Python actif."
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

if [[ "$FORCE" != "1" ]]; then
  require_gpu_free
fi

chmod +x "$ST_SCRIPT"

ensure_spm_5k() {
  local spm_model="datasets/processed/spm/fr-en_5000.model"
  if [[ -f "$spm_model" ]]; then
    echo "=== $(date -Is) SPM existant : ${spm_model} ==="
    return 0
  fi
  echo "=== $(date -Is) SPM 5k absent — entraînement avant pré-vol ==="
  local manifests="datasets/manifests/fr-en"
  local target_txt="${manifests}/train.target.txt"
  if [[ ! -f "$target_txt" && -f "${manifests}/train.tsv" ]]; then
    python -c "
import csv
from pathlib import Path
manifest = Path('${manifests}/train.tsv')
target = Path('${target_txt}')
with manifest.open(encoding='utf-8') as handle_in, target.open('w', encoding='utf-8') as handle_out:
    for row in csv.DictReader(handle_in, delimiter='\t'):
        handle_out.write(row['tgt_text'].strip() + '\n')
"
  fi
  python 1_Transformer/3_spm.py \
    --langpair fr-en \
    --vocab-size 5000 \
    --manifests-root datasets/manifests \
    --train-text "${target_txt}" \
    --overwrite
}

ensure_spm_5k

echo "=== $(date -Is) Pré-vol ST L-14k v7 SPM 5k (Modyco, run_031) ==="
python 1_Transformer/pipeline.py train \
  --config 1_Transformer/configs/fr-en/base_utterance_large_14k_v7_spm5k.yaml \
  --run-id run_031_transformer_baseline_utterance_large_14k_v7_spm5k \
  --dry-run

echo "=== $(date -Is) Délégation → ${ST_SCRIPT} ==="
exec bash "$ST_SCRIPT"
