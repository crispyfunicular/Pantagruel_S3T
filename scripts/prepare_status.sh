#!/usr/bin/env bash
# Utilitaire — afficher la progression d'un job prepare étape 2 long (hors pipeline).
#
# Usage : ./scripts/prepare_status.sh [langpair]
# Lit artifacts/prepare_<langpair>.progress.json et compte les WAV sur disque.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LANGPAIR="${1:-fr-en}"

cd "${ROOT}"

count_wavs() {
  find "datasets/processed/${LANGPAIR}" -name '*.wav' 2>/dev/null | wc -l
}

echo "=== Prepare status: ${LANGPAIR} ==="
if pgrep -af "scripts/2_prepare.py --langpair ${LANGPAIR}" 2>/dev/null; then
  echo "Process: RUNNING"
else
  echo "Process: not running"
fi

echo "WAV on disk: $(count_wavs)"
if [[ -f "artifacts/prepare_${LANGPAIR}.progress.json" ]]; then
  echo "--- progress (artifacts/prepare_${LANGPAIR}.progress.json) ---"
  cat "artifacts/prepare_${LANGPAIR}.progress.json"
fi
if [[ -f "artifacts/prepare_${LANGPAIR}.json" ]]; then
  echo "--- last report summary ---"
  python -c "
import json
from pathlib import Path
p = Path('artifacts/prepare_${LANGPAIR}.json')
d = json.loads(p.read_text())
print('exit_code:', d.get('exit_code'))
print('splits:', d.get('splits'))
w = d.get('wav_validation_summary')
if w:
    print('wav_validation:', w)
"

fi
if [[ -d "datasets/manifests/${LANGPAIR}" ]]; then
  echo "--- manifests ---"
  wc -l "datasets/manifests/${LANGPAIR}"/*.tsv 2>/dev/null || true
fi
