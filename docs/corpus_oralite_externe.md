# Corpus oralité pluriTAL — tests ST hors m-TEDx

Corpus local (projet pluriTAL) :  
`~/git/perso/pluriTAL/oralite/projet_final/corpus/`

## Contenu

| Fichier | Rôle |
|---------|------|
| `N-M.wav` | Audio français lu (16 kHz mono, ~19 s) |
| `N-M.lab` | Transcription **française** de référence (pas d’anglais) |
| `*.lab` sans `.wav` | Items phonétiques isolés (hors scope ST batch) |

Environ **18** paires WAV+lab numérotées (`1-1` … `10-2`, etc.). Métadonnées détaillées : `audio_metadata_1-20.csv`, alignements MFA dans `aligned_one/`.

## Intégration S3T

Ce corpus sert à une **évaluation qualitative** (oralité, style lu) : les modèles S3T produisent de l’**anglais** ; on compare à la ref FR pour juger sens / fidélité, **sans SacreBLEU** (pas de `tgt_text` EN).

### 1. Construire le manifest

```bash
cd ~/git/GETALP/S3T
source .venv/bin/activate
python scripts/build_oralite_manifest.py
# ou : --corpus-dir /chemin/vers/corpus
```

Sortie : `datasets/external/oralite_fr/manifest.tsv` (chemins audio **absolus** vers pluriTAL).

### 2. Inférence multi-variantes

```bash
export GEMINI_API_KEY=...   # si variant gemini

# speechLLM (run_005 par défaut) + Gemini
python scripts/run_external_corpus_infer.py --variants speechllm,gemini

# + cascade (GPU)
python scripts/run_external_corpus_infer.py --variants speechllm,gemini,cascade

# Smoke 3 clips
python scripts/run_external_corpus_infer.py --variants speechllm --limit 3 -v
```

Sorties : `datasets/external/oralite_fr/predictions/{speechllm,gemini,cascade}.tsv`  
Colonnes : `id`, `src_text_fr`, `en_hypothesis`, `pipeline`, `audio`.

### 3. Checkpoints / configs par défaut

| Variante | Défaut |
|----------|--------|
| speechLLM | `run_005` (encodeur dégelé), `b1_sentence_long.yaml` |
| Gemini | `gemini_flash_sentence.yaml` (`gemini-2.5-flash`) |
| cascade | `cascade_sentence.yaml` (Whisper large-v3 + Marian) |

Surcharge : `--speechllm-checkpoint`, `--gemini-config`, etc.

## Limites

- Modèles entraînés sur **m-TEDx** (`sentence_like`) : transfert oralité = test de **généralisation**, pas benchmark chiffré.
- Pas de protocole Pantagruel / Table 8 sur ce corpus.
- ST Transformer (`1_Transformer`) : inférence unitaire via `6_infer.py` si besoin (non inclus dans le script batch actuel).

## Piste rapport / démo Félix (SLAM / speechLLM)

Montrer côte à côte pour 2–3 clips : `src_text_fr`, `en_hypothesis` speechLLM vs Gemini ; commenter erreurs d’oralité (liaisons, débit, lexique).
