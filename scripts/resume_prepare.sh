#!/usr/bin/env bash
# Resume m-TEDx prepare (WAV 16 kHz mono PCM16 + manifests).
# Safe to run multiple times: skips valid WAV files already on disk.
#
# Usage:
#   ./scripts/resume_prepare.sh              # default fr-en
#   ./scripts/resume_prepare.sh fr-es
#   ./scripts/resume_prepare.sh fr-en --verify-only

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

LANGPAIR="${1:-fr-en}"
shift || true

if [[ ! -d ".venv" ]]; then
  echo "ERROR: .venv missing. Run ./scripts/bootstrap.sh first." >&2
  exit 2
fi

# shellcheck source=/dev/null
source ".venv/bin/activate"

mkdir -p logs

if pgrep -f "scripts/2_prepare.py --langpair ${LANGPAIR}" >/dev/null 2>&1; then
  echo "ERROR: prepare already running for ${LANGPAIR}." >&2
  echo "  Check: pgrep -af '2_prepare.py'" >&2
  echo "  Stop:  pkill -f 'scripts/2_prepare.py --langpair ${LANGPAIR}'" >&2
  exit 2
fi

RAW_CORPUS="${ROOT}/datasets/raw/${LANGPAIR}"
RAW_MTEDX="${ROOT}/datasets/raw/mtedx_${LANGPAIR}"
if [[ ! -d "${RAW_CORPUS}/data/train" && ! -d "${RAW_MTEDX}/data/train" ]]; then
  echo "ERROR: raw corpus not found. Run download first:" >&2
  echo "  python scripts/1_download.py --langpairs ${LANGPAIR}" >&2
  exit 2
fi

LOG="logs/prepare_${LANGPAIR}.log"
echo "==> Prepare ${LANGPAIR} (resume on, log: ${LOG})"

python scripts/2_prepare.py \
  --langpair "${LANGPAIR}" \
  --resume \
  --verbose \
  "$@" \
  2>&1 | tee -a "${LOG}"

echo "==> Final verification"
python scripts/2_prepare.py --langpair "${LANGPAIR}" --verify-only

echo "==> Done. Report: artifacts/prepare_${LANGPAIR}.json"
