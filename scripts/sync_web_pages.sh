#!/usr/bin/env bash
# Copie le site statique web/ → docs/ pour GitHub Pages (source « Deploy from /docs »).
# Source unique à éditer : web/ — ne pas modifier les *.html dans docs/ à la main.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEB="${ROOT}/web"
DOCS="${ROOT}/docs"

for f in "${WEB}"/*.html; do
  cp -f "$f" "${DOCS}/$(basename "$f")"
done
cp -f "${WEB}/.nojekyll" "${DOCS}/.nojekyll"
echo "Synchronisé : web/*.html + .nojekyll → docs/"
