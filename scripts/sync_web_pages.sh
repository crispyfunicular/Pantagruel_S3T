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
cp -f "${WEB}/theme.css" "${DOCS}/theme.css"
cp -f "${WEB}/theme.js" "${DOCS}/theme.js"
cp -f "${WEB}/.nojekyll" "${DOCS}/.nojekyll"
if [[ -d "${WEB}/audio" ]]; then
  mkdir -p "${DOCS}/audio"
  shopt -s nullglob
  for wav in "${WEB}/audio/"*.wav; do
    cp -f "$wav" "${DOCS}/audio/"
  done
  shopt -u nullglob
  echo "Synchronisé : web/audio/*.wav → docs/audio/"
fi
echo "Synchronisé : web/*.html + .nojekyll → docs/"
